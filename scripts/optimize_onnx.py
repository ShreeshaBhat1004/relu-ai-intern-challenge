import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

SEED = 42
MAX_LENGTH = 256
BASELINE_DIR = Path("./models/baseline")
OUTPUT_ROOT = Path("./models/onnx")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def find_onnx_file(directory: Path) -> Path:
    matches = sorted(directory.glob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"No ONNX file found in {directory}")
    return matches[0]


def export_and_optimize() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_dir = OUTPUT_ROOT / "baseline"
    optimized_dir = OUTPUT_ROOT / "optimized"
    quantized_dir = OUTPUT_ROOT / "quantized"

    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            str(BASELINE_DIR),
            export=True,
        )
        model.save_pretrained(str(baseline_dir))

        optimizer = ORTOptimizer.from_pretrained(model)
        optimization_config = OptimizationConfig(
            optimization_level=99,
            optimize_for_gpu=False,
            fp16=False,
            enable_transformers_specific_optimizations=True,
        )
        optimizer.optimize(
            save_dir=str(optimized_dir),
            optimization_config=optimization_config,
        )
    except Exception as exc:
        print(f"Optimum export/optimization failed, using torch.onnx.export fallback: {exc}")
        export_with_torch_onnx(baseline_dir, optimized_dir)

    quantizer = ORTQuantizer.from_pretrained(str(baseline_dir))
    try:
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
        )
    except Exception:
        qconfig = AutoQuantizationConfig.arm64(
            is_static=False,
            per_channel=False,
        )
    quantizer.quantize(
        save_dir=str(quantized_dir),
        quantization_config=qconfig,
    )


def export_with_torch_onnx(baseline_dir: Path, optimized_dir: Path) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    optimized_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained(str(BASELINE_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(BASELINE_DIR))
    model.eval()
    sample = tokenizer(
        "This movie was absolutely fantastic!",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    baseline_path = baseline_dir / "model.onnx"
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(baseline_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    tokenizer.save_pretrained(str(baseline_dir))

    optimized_path = optimized_dir / "model_optimized.onnx"
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        optimized_model = optimize_model(
            str(baseline_path),
            model_type="bert",
            num_heads=12,
            hidden_size=768,
        )
        optimized_model.save_model_to_file(str(optimized_path))
    except Exception as exc:
        print(f"ONNX graph optimization fallback failed, copying baseline ONNX instead: {exc}")
        optimized_path.write_bytes(baseline_path.read_bytes())
    tokenizer.save_pretrained(str(optimized_dir))


def evaluate_variant(model_dir: Path, tokenizer: DistilBertTokenizer) -> dict:
    session = ort.InferenceSession(str(find_onnx_file(model_dir)), session_options())
    dataset = load_dataset("imdb", split="test")

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    predictions = []
    labels = []

    for item in tqdm(tokenized, desc=f"Evaluating {model_dir.name}"):
        inputs = {
            "input_ids": np.asarray([item["input_ids"]], dtype=np.int64),
            "attention_mask": np.asarray([item["attention_mask"]], dtype=np.int64),
        }
        logits = session.run(None, inputs)[0]
        predictions.append(int(np.argmax(logits, axis=-1)[0]))
        labels.append(int(item["label"]))

    onnx_size_mb = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()) / (1024 * 1024)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
        "size_mb": round(onnx_size_mb, 3),
    }


def main() -> None:
    set_seed()
    export_and_optimize()
    tokenizer = DistilBertTokenizer.from_pretrained(str(BASELINE_DIR))

    metrics = {
        "onnx_baseline": evaluate_variant(OUTPUT_ROOT / "baseline", tokenizer),
        "onnx_optimized": evaluate_variant(OUTPUT_ROOT / "optimized", tokenizer),
        "onnx_quantized": evaluate_variant(OUTPUT_ROOT / "quantized", tokenizer),
    }

    with (OUTPUT_ROOT / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
