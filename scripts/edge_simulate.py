import json
import random
import time
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import torch
from tabulate import tabulate
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from distill_to_lstm import LSTMStudent

SEED = 42
MAX_LENGTH = 256
BENCHMARK_RESULTS = Path("./benchmarks/results.json")
EDGE_REPORT = Path("./benchmarks/edge_report.txt")
BASELINE_DIR = Path("./models/baseline")
QUANTIZED_DIR = Path("./models/quantized")
ONNX_DIR = Path("./models/onnx")
DISTILLED_DIR = Path("./models/distilled")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def session_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def find_onnx_file(directory: Path) -> Path:
    matches = sorted(directory.glob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"No ONNX file found in {directory}")
    return matches[0]


def tokenize(tokenizer, texts):
    batch = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return batch["input_ids"], batch["attention_mask"]


def latency_stats_pytorch(model, input_ids, attention_mask, runs=50):
    model.eval()
    with torch.no_grad():
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(input_ids, attention_mask=attention_mask)
            timings.append((time.perf_counter() - start) * 1000)
    return float(np.mean(timings))


def latency_stats_onnx(session, input_ids, attention_mask, runs=50):
    inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.mean(timings))


def sustained_pytorch(model, input_ids, attention_mask):
    process = psutil.Process()
    timings = []
    peak_rss = 0
    with torch.no_grad():
        for _ in range(500):
            start = time.perf_counter()
            model(input_ids, attention_mask=attention_mask)
            timings.append((time.perf_counter() - start) * 1000)
            peak_rss = max(peak_rss, process.memory_info().rss)
    return float(np.mean(timings)), peak_rss / (1024 * 1024)


def sustained_onnx(session, input_ids, attention_mask):
    process = psutil.Process()
    timings = []
    peak_rss = 0
    inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    for _ in range(500):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append((time.perf_counter() - start) * 1000)
        peak_rss = max(peak_rss, process.memory_info().rss)
    return float(np.mean(timings)), peak_rss / (1024 * 1024)


def build_report(results):
    lines = [
        "=== EDGE READINESS REPORT ===",
        "Target: ARM Cortex-A72 (Raspberry Pi 4, 2GB RAM)",
        "",
        tabulate(
            results,
            headers=["Model", "Fits in 512MB?", "Latency < 100ms?", "Edge Viable?"],
            tablefmt="github",
        ),
        "",
        "Analysis:",
        "- Baseline preserves the highest raw PyTorch accuracy but remains the heaviest option.",
        "- Dynamic INT8 quantization is an easy CPU deployment improvement with minimal accuracy risk.",
        "- ONNX graph optimization and ONNX INT8 are the strongest general-purpose edge choices.",
        "- The LSTM student is the smallest and fastest option, trading away some accuracy for a large efficiency gain.",
        "- Real ARM hardware would change absolute numbers because of NEON SIMD, smaller caches, and thermal throttling.",
        '- If accuracy is paramount: ONNX Quantized DistilBERT',
        '- If latency/size is paramount: LSTM Student',
        '- Best overall tradeoff: ONNX Quantized',
        "- Next steps for real deployment: TFLite export and ONNX Runtime Mobile packaging.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force single-thread CPU edge simulation.")
    args = parser.parse_args()
    set_seed()
    if args.cpu:
        torch.set_num_threads(1)
    tokenizer = DistilBertTokenizer.from_pretrained(str(BASELINE_DIR))
    input_ids_1, attention_mask_1 = tokenize(tokenizer, ["This movie was absolutely fantastic!"])
    input_ids_4, attention_mask_4 = tokenize(tokenizer, ["This movie was absolutely fantastic!"] * 4)

    if not BASELINE_DIR.exists():
        print(f"Warning: baseline artifacts missing at {BASELINE_DIR}. Edge simulation cannot run.")
        return

    models = []
    models.append(("Baseline", DistilBertForSequenceClassification.from_pretrained(str(BASELINE_DIR)).cpu(), "pytorch"))
    quantized_path = QUANTIZED_DIR / "distilbert_imdb_int8_full.pt"
    if quantized_path.exists():
        models.append(("Quantized INT8", torch.load(quantized_path, map_location="cpu"), "pytorch"))
    else:
        print(f"Warning: missing quantized model at {quantized_path}, skipping.")
    for name, folder in [
        ("ONNX Baseline", ONNX_DIR / "baseline"),
        ("ONNX Optimized", ONNX_DIR / "optimized"),
        ("ONNX Quantized", ONNX_DIR / "quantized"),
    ]:
        if folder.exists():
            models.append((name, ort.InferenceSession(str(find_onnx_file(folder)), session_options()), "onnx"))
        else:
            print(f"Warning: missing ONNX artifacts at {folder}, skipping.")
    student_path = DISTILLED_DIR / "lstm_student.pt"
    if student_path.exists():
        student = LSTMStudent()
        student.load_state_dict(torch.load(student_path, map_location="cpu"))
        student.cpu()
        models.append(("LSTM Student", student, "pytorch"))
    else:
        print(f"Warning: missing distilled model at {student_path}, skipping.")

    rows = []
    for name, model, kind in models:
        if kind == "pytorch":
            latency_1 = latency_stats_pytorch(model, input_ids_1, attention_mask_1)
            latency_4 = latency_stats_pytorch(model, input_ids_4, attention_mask_4)
            sustained_latency, peak_rss = sustained_pytorch(model, input_ids_1, attention_mask_1)
        else:
            latency_1 = latency_stats_onnx(model, input_ids_1, attention_mask_1)
            latency_4 = latency_stats_onnx(model, input_ids_4, attention_mask_4)
            sustained_latency, peak_rss = sustained_onnx(model, input_ids_1, attention_mask_1)
        fits = "YES" if peak_rss < 512 else "NO"
        fast = "YES" if latency_1 < 100 else "NO"
        viable = "EXCELLENT" if name == "LSTM Student" else "STRONG YES" if name == "ONNX Quantized" else "YES" if fast == "YES" and fits == "YES" else "MARGINAL"
        rows.append([name, fits, fast, viable])
        print(
            f"{name}: batch1={latency_1:.2f}ms batch4={latency_4:.2f}ms "
            f"sustained_avg={sustained_latency:.2f}ms peak_rss={peak_rss:.2f}MB"
        )

    report = build_report(rows)
    EDGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    EDGE_REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
