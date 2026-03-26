import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tabulate import tabulate
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from distill_to_lstm import LSTMStudent

MAX_LENGTH = 256
DEFAULT_TEXT = "This movie was a complete waste of time. Terrible acting and plot."
BASELINE_DIR = Path("./models/baseline")
QUANTIZED_DIR = Path("./models/quantized")
ONNX_DIR = Path("./models/onnx")
DISTILLED_DIR = Path("./models/distilled")
LABELS = ["NEGATIVE", "POSITIVE"]


def has_baseline_artifacts() -> bool:
    required = [
        BASELINE_DIR / "config.json",
        BASELINE_DIR / "tokenizer_config.json",
    ]
    return all(path.exists() for path in required)


def find_onnx_file(directory: Path) -> Path:
    matches = sorted(directory.glob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"No ONNX file found in {directory}")
    return matches[0]


def session_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def encode(tokenizer, text):
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def probabilities_from_logits(logits):
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[0]
    index = int(np.argmax(probs))
    return LABELS[index], float(probs[index])


def benchmark(fn, iterations=100):
    for _ in range(100):
        fn()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.mean(timings))


def run_pytorch(model, tokenizer, text, do_benchmark):
    encoded = encode(tokenizer, text)
    model.eval()
    with torch.no_grad():
        logits = model(encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits.cpu().numpy()
    label, confidence = probabilities_from_logits(logits)
    latency = benchmark(
        lambda: model(encoded["input_ids"], attention_mask=encoded["attention_mask"]),
    ) if do_benchmark else None
    return label, confidence, latency


def run_onnx(session, tokenizer, text, do_benchmark):
    encoded = encode(tokenizer, text)
    inputs = {
        "input_ids": encoded["input_ids"].numpy().astype(np.int64),
        "attention_mask": encoded["attention_mask"].numpy().astype(np.int64),
    }
    logits = session.run(None, inputs)[0]
    label, confidence = probabilities_from_logits(logits)
    latency = benchmark(lambda: session.run(None, inputs)) if do_benchmark else None
    return label, confidence, latency


def available_models():
    if not has_baseline_artifacts():
        raise FileNotFoundError(
            "Baseline tokenizer/model directory is missing. Run scripts/train_baseline.py first."
        )

    tokenizer = DistilBertTokenizer.from_pretrained(str(BASELINE_DIR))
    models = {}

    if BASELINE_DIR.exists():
        models["baseline"] = ("Baseline", DistilBertForSequenceClassification.from_pretrained(str(BASELINE_DIR)).cpu(), "pytorch")
    if (QUANTIZED_DIR / "distilbert_imdb_int8_full.pt").exists():
        models["quantized"] = ("Quantized INT8", torch.load(QUANTIZED_DIR / "distilbert_imdb_int8_full.pt", map_location="cpu"), "pytorch")
    if (ONNX_DIR / "baseline").exists():
        models["onnx"] = ("ONNX Baseline", ort.InferenceSession(str(find_onnx_file(ONNX_DIR / "baseline")), session_options()), "onnx")
    if (ONNX_DIR / "optimized").exists():
        models["onnx_optimized"] = ("ONNX Optimized", ort.InferenceSession(str(find_onnx_file(ONNX_DIR / "optimized")), session_options()), "onnx")
    if (ONNX_DIR / "quantized").exists():
        models["onnx_quantized"] = ("ONNX Quantized", ort.InferenceSession(str(find_onnx_file(ONNX_DIR / "quantized")), session_options()), "onnx")
        models["optimized"] = models["onnx_quantized"]
    if (DISTILLED_DIR / "lstm_student.pt").exists():
        student = LSTMStudent()
        student.load_state_dict(torch.load(DISTILLED_DIR / "lstm_student.pt", map_location="cpu"))
        student.cpu()
        models["distilled"] = ("LSTM Student", student, "pytorch")

    return tokenizer, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="optimized",
        choices=["baseline", "quantized", "onnx", "onnx_optimized", "onnx_quantized", "distilled", "optimized", "all"],
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    try:
        tokenizer, models = available_models()
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
        return
    requested = ["baseline", "quantized", "onnx", "onnx_optimized", "onnx_quantized", "distilled"] if args.model == "all" else [args.model]

    rows = []
    for key in requested:
        if key not in models:
            print(f"Warning: model '{key}' is unavailable. Train or export it first.")
            continue
        display_name, model, kind = models[key]
        if kind == "pytorch":
            prediction, confidence, latency = run_pytorch(model, tokenizer, args.text, args.benchmark)
        else:
            prediction, confidence, latency = run_onnx(model, tokenizer, args.text, args.benchmark)
        rows.append(
            [
                display_name,
                prediction,
                f"{confidence * 100:.1f}%",
                f"{latency:.2f}" if latency is not None else "-",
            ]
        )

    print("ReLU AI Intern Challenge - Inference Demo")
    print(f'Input: "{args.text}"')
    print(
        tabulate(
            rows,
            headers=["Model", "Prediction", "Confidence", "Latency (ms)"],
            tablefmt="rounded_grid",
        )
    )
    if not rows:
        print("Warning: no requested models are currently available.")
        return
    if args.model == "all":
        print("Recommendation: ONNX Quantized offers the best accuracy/speed tradeoff.")


if __name__ == "__main__":
    main()
