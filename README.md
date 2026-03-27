# Efficient AI Inference for Edge Devices

ReLU Technologies AI Systems Internship Challenge

This project compares a strong NLP baseline with smaller and faster deployment-ready alternatives for IMDB sentiment classification. The idea is simple: start with a fine-tuned DistilBERT model, then optimize it in different ways and measure the tradeoff between accuracy, model size, and CPU latency.

## What This Project Does

The repository builds and compares these model variants:

- `Baseline`: fine-tuned DistilBERT for sentiment classification
- `Quantized INT8`: PyTorch dynamic quantization of the baseline model
- `ONNX Baseline`: baseline model exported to ONNX Runtime
- `ONNX Optimized`: ONNX model with graph optimization
- `ONNX Quantized`: ONNX Runtime quantized model
- `LSTM Student`: a much smaller distilled student model

The goal is not just to get the best accuracy. It is to find the most practical model for deployment on CPU or edge-like environments.

## Main Takeaway

- Best raw accuracy: `Baseline DistilBERT`
- Best overall deployment tradeoff: `ONNX Quantized`
- Best ultra-light option: `LSTM Student`

In this project, `ONNX Quantized` turned out to be the best balance between quality and efficiency. The distilled LSTM was by far the smallest and fastest model, but with a larger drop in accuracy.

## Final Benchmark Summary

| Model | Accuracy | F1 | Size (MB) | Latency p50 (ms) |
|---|---:|---:|---:|---:|
| Baseline | 0.913 | 0.913 | 1788.8 | 283.1 |
| Quantized INT8 | 0.897 | 0.902 | 132.3 | 221.5 |
| ONNX Baseline | 0.913 | 0.913 | 255.5 | 300.0 |
| ONNX Optimized | 0.913 | 0.913 | 256.3 | 295.1 |
| ONNX Quantized | 0.910 | 0.910 | 64.2 | 234.2 |
| LSTM Student | 0.822 | 0.828 | 16.0 | 1.9 |

## Repository Layout

```text
.
├── benchmarks/
├── data/
├── implementation.md
├── models/
├── report/
│   ├── report.pdf
│   └── report.tex
├── requirements.txt
├── Makefile
├── README.md
└── scripts/
    ├── benchmark.py
    ├── distill_to_lstm.py
    ├── edge_simulate.py
    ├── model_loading.py
    ├── optimize_onnx.py
    ├── optimize_quantize.py
    ├── run_inference.py
    └── train_baseline.py
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Or use the Makefile:

```bash
make setup
```

## How To Run

### 1. Train the baseline model

```bash
python scripts/train_baseline.py
```

### 2. Create optimized versions

```bash
python scripts/optimize_quantize.py
python scripts/optimize_onnx.py
python scripts/distill_to_lstm.py
```

Or run them together:

```bash
make optimize
```

### 3. Benchmark all models

```bash
python scripts/benchmark.py
```

This generates benchmark outputs and plots under `benchmarks/`.

### 4. Run the edge simulation

```bash
python scripts/edge_simulate.py
```

This gives a deployment-style summary for CPU / edge scenarios.

### 5. Try inference yourself

```bash
python scripts/run_inference.py --model all
```

Examples:

```bash
python scripts/run_inference.py --model baseline --text "This movie was excellent."
python scripts/run_inference.py --model onnx_quantized --text "This movie was boring and too long."
python scripts/run_inference.py --model distilled --text "A surprisingly good film." --benchmark
```

Available model choices:

- `baseline`
- `quantized`
- `onnx`
- `onnx_optimized`
- `onnx_quantized`
- `distilled`
- `optimized` which aliases the recommended deployment model
- `all`

## Script Guide

- `scripts/train_baseline.py`
  Fine-tunes DistilBERT on IMDB and saves the baseline model.

- `scripts/optimize_quantize.py`
  Applies PyTorch dynamic INT8 quantization and evaluates it.

- `scripts/optimize_onnx.py`
  Exports the model to ONNX, applies optimization, and benchmarks ONNX variants.

- `scripts/distill_to_lstm.py`
  Trains a distilled bidirectional LSTM student from the DistilBERT teacher.

- `scripts/benchmark.py`
  Benchmarks all available models on CPU and writes comparison outputs.

- `scripts/edge_simulate.py`
  Produces an edge-readiness style report.

- `scripts/run_inference.py`
  Lets you run predictions interactively on one or more model variants.

## Notes

- All benchmarking is CPU-focused.
- The project uses IMDB sentiment classification as the benchmark task.
- PyTorch 2.6 changed `torch.load` defaults, so quantized model loading is handled carefully through `scripts/model_loading.py`.
- Memory numbers from a shared process should be interpreted carefully, especially in edge simulation.

## Report

The short technical report is available in:

- `report/report.tex`
- `report/report.pdf`

## Kaggle Links

- Training notebook: <https://www.kaggle.com/code/shreeshabhat1004/training>
- Benchmark notebook: <https://www.kaggle.com/code/shreeshabhat1004/benchmark>
- Inference notebook: <https://www.kaggle.com/code/shreeshabhat1004/inference?scriptVersionId=306837239>
