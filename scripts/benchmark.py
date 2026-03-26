import json
import random
import time
import tracemalloc
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from distill_to_lstm import LSTMStudent

SEED = 42
MAX_LENGTH = 256
WARMUP_RUNS = 100
TIMED_RUNS = 1000
THROUGHPUT_RUNS = 100
BATCH_SIZE = 32

BASELINE_DIR = Path("./models/baseline")
QUANTIZED_DIR = Path("./models/quantized")
ONNX_DIR = Path("./models/onnx")
DISTILLED_DIR = Path("./models/distilled")
BENCHMARK_DIR = Path("./benchmarks")
PLOTS_DIR = BENCHMARK_DIR / "plots"

MODEL_LABELS = {
    "baseline": "Baseline",
    "quantized": "Quantized INT8",
    "onnx_baseline": "ONNX Baseline",
    "onnx_optimized": "ONNX Optimized",
    "onnx_quantized": "ONNX Quantized",
    "distilled_lstm": "LSTM Student",
}


def has_baseline_artifacts() -> bool:
    required = [
        BASELINE_DIR / "config.json",
        BASELINE_DIR / "tokenizer_config.json",
    ]
    return all(path.exists() for path in required)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_onnx_file(directory: Path) -> Path:
    matches = sorted(directory.glob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"No ONNX file found in {directory}")
    return matches[0]


def session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def tokenize_texts(tokenizer, texts):
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def compute_classification_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="binary")),
    }


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file()) / (1024 * 1024)


def load_metrics() -> dict:
    metrics = {}
    paths = {
        "baseline": BASELINE_DIR / "metrics.json",
        "quantized": QUANTIZED_DIR / "metrics.json",
        "distilled_lstm": DISTILLED_DIR / "metrics.json",
    }
    for key, path in paths.items():
        if path.exists():
            metrics[key] = json.loads(path.read_text(encoding="utf-8"))

    onnx_metrics_path = ONNX_DIR / "metrics.json"
    if onnx_metrics_path.exists():
        metrics.update(json.loads(onnx_metrics_path.read_text(encoding="utf-8")))
    return metrics


def pytorch_latency(model, input_ids, attention_mask):
    model.eval()
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(input_ids, attention_mask=attention_mask)

        timings = []
        tracemalloc.start()
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            model(input_ids, attention_mask=attention_mask)
            timings.append((time.perf_counter() - start) * 1000)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return timings, peak / (1024 * 1024)


def onnx_latency(session, input_ids, attention_mask):
    inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    for _ in range(WARMUP_RUNS):
        session.run(None, inputs)

    timings = []
    tracemalloc.start()
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append((time.perf_counter() - start) * 1000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return timings, peak / (1024 * 1024)


def pytorch_throughput(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(THROUGHPUT_RUNS):
            model(input_ids, attention_mask=attention_mask)
        elapsed = time.perf_counter() - start
    return (THROUGHPUT_RUNS * input_ids.shape[0]) / elapsed


def onnx_throughput(session, input_ids, attention_mask):
    inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    start = time.perf_counter()
    for _ in range(THROUGHPUT_RUNS):
        session.run(None, inputs)
    elapsed = time.perf_counter() - start
    return (THROUGHPUT_RUNS * input_ids.shape[0]) / elapsed


def summarize_latency(timings):
    values = np.asarray(timings)
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def benchmark_models() -> dict:
    if not has_baseline_artifacts():
        print(f"Warning: baseline tokenizer/model artifacts are incomplete at {BASELINE_DIR}.")
        return {}

    tokenizer = DistilBertTokenizer.from_pretrained(str(BASELINE_DIR))
    sample_text = ["This movie was absolutely fantastic!"]
    batch_texts = sample_text * BATCH_SIZE
    input_ids_1, attention_mask_1 = tokenize_texts(tokenizer, sample_text)
    input_ids_32, attention_mask_32 = tokenize_texts(tokenizer, batch_texts)

    stored_metrics = load_metrics()
    results = {}

    if has_baseline_artifacts():
        baseline_model = DistilBertForSequenceClassification.from_pretrained(str(BASELINE_DIR)).cpu()
        timings, peak = pytorch_latency(baseline_model, input_ids_1, attention_mask_1)
        results["baseline"] = {
            **stored_metrics.get("baseline", {}),
            "size_mb": round(size_mb(BASELINE_DIR), 3),
            "latency": summarize_latency(timings),
            "throughput_sps": float(pytorch_throughput(baseline_model, input_ids_32, attention_mask_32)),
            "peak_memory_mb": float(peak),
        }
    else:
        print(f"Warning: missing baseline artifacts at {BASELINE_DIR}, skipping baseline benchmark.")

    quantized_path = QUANTIZED_DIR / "distilbert_imdb_int8_full.pt"
    if quantized_path.exists():
        quantized_model = torch.load(quantized_path, map_location="cpu")
        timings, peak = pytorch_latency(quantized_model, input_ids_1, attention_mask_1)
        results["quantized"] = {
            **stored_metrics.get("quantized", {}),
            "size_mb": round(size_mb(quantized_path), 3),
            "latency": summarize_latency(timings),
            "throughput_sps": float(pytorch_throughput(quantized_model, input_ids_32, attention_mask_32)),
            "peak_memory_mb": float(peak),
        }
    else:
        print(f"Warning: missing quantized model at {quantized_path}, skipping quantized benchmark.")

    for key, folder in {
        "onnx_baseline": ONNX_DIR / "baseline",
        "onnx_optimized": ONNX_DIR / "optimized",
        "onnx_quantized": ONNX_DIR / "quantized",
    }.items():
        if folder.exists():
            session = ort.InferenceSession(str(find_onnx_file(folder)), session_options())
            timings, peak = onnx_latency(session, input_ids_1, attention_mask_1)
            results[key] = {
                **stored_metrics.get(key, {}),
                "size_mb": round(size_mb(folder), 3),
                "latency": summarize_latency(timings),
                "throughput_sps": float(onnx_throughput(session, input_ids_32, attention_mask_32)),
                "peak_memory_mb": float(peak),
            }
        else:
            print(f"Warning: missing ONNX artifacts at {folder}, skipping {key}.")

    student_path = DISTILLED_DIR / "lstm_student.pt"
    if student_path.exists():
        student = LSTMStudent()
        student.load_state_dict(torch.load(student_path, map_location="cpu"))
        student.cpu()
        timings, peak = pytorch_latency(student, input_ids_1, attention_mask_1)
        results["distilled_lstm"] = {
            **stored_metrics.get("distilled_lstm", {}),
            "size_mb": round(size_mb(student_path), 3),
            "latency": summarize_latency(timings),
            "throughput_sps": float(pytorch_throughput(student, input_ids_32, attention_mask_32)),
            "peak_memory_mb": float(peak),
        }
    else:
        print(f"Warning: missing distilled model at {student_path}, skipping distilled benchmark.")

    return results


def save_results(results: dict) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with (BENCHMARK_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def render_table(results: dict) -> str:
    rows = []
    for key in MODEL_LABELS:
        if key not in results:
            continue
        metrics = results[key]
        rows.append(
            [
                MODEL_LABELS[key],
                f"{metrics.get('accuracy', 0):.3f}",
                f"{metrics.get('f1', 0):.3f}",
                f"{metrics['size_mb']:.1f}",
                f"{metrics['latency']['p50_ms']:.1f}",
                f"{metrics['latency']['p95_ms']:.1f}",
                f"{metrics['throughput_sps']:.1f}",
                f"{metrics['peak_memory_mb']:.1f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "Model",
            "Accuracy",
            "F1",
            "Size (MB)",
            "Latency p50 (ms)",
            "Latency p95 (ms)",
            "Throughput (s/s)",
            "Peak Mem (MB)",
        ],
        tablefmt="github",
    )


def pareto_frontier(points):
    frontier = []
    for point in sorted(points, key=lambda item: item["latency"]):
        while frontier and frontier[-1]["accuracy"] <= point["accuracy"]:
            frontier.pop()
        frontier.append(point)
    return frontier


def plot_results(results: dict) -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 150

    available_keys = [key for key in MODEL_LABELS if key in results]
    names = [MODEL_LABELS[key] for key in available_keys]
    p50 = [results[key]["latency"]["p50_ms"] for key in available_keys]
    p95 = [results[key]["latency"]["p95_ms"] for key in available_keys]
    p99 = [results[key]["latency"]["p99_ms"] for key in available_keys]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots()
    ax.bar(x - width, p50, width, label="p50")
    ax.bar(x, p95, width, label="p95")
    ax.bar(x + width, p99, width, label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Comparison (CPU, Batch Size 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latency_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.barh(names, [results[key]["size_mb"] for key in available_keys])
    ax.set_xlabel("Model Size (MB)")
    ax.set_title("Model Size Comparison")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "size_comparison.png")
    plt.close(fig)

    points = [
        {
            "key": key,
            "name": MODEL_LABELS[key],
            "latency": results[key]["latency"]["p50_ms"],
            "accuracy": results[key].get("accuracy", 0),
        }
        for key in available_keys
    ]
    frontier = pareto_frontier(points)
    fig, ax = plt.subplots()
    for point in points:
        ax.scatter(point["latency"], point["accuracy"], s=70)
        ax.text(point["latency"], point["accuracy"], point["name"], fontsize=8)
    ax.plot([p["latency"] for p in frontier], [p["accuracy"] for p in frontier], linestyle="--")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Latency: Pareto Frontier")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pareto_accuracy_latency.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force single-thread CPU benchmarking.")
    args = parser.parse_args()
    set_seed()
    if args.cpu:
        torch.set_num_threads(1)
    results = benchmark_models()
    if not results:
        print("Warning: no benchmarkable artifacts were found.")
        return
    save_results(results)
    print(render_table(results))
    plot_results(results)


if __name__ == "__main__":
    main()
