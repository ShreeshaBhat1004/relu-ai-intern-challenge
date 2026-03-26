# Efficient AI Inference for Edge Devices

> ReLU Technologies — AI Systems Internship Challenge

## Results at a Glance

Run `python scripts/benchmark.py` after training and optimization to populate the benchmark table and plots.

![Pareto](benchmarks/plots/pareto_accuracy_latency.png)

## Key Findings

- **ONNX Runtime + INT8 quantization** reduces latency while typically keeping accuracy within a narrow margin of the baseline.
- **Knowledge distillation to LSTM** targets a dramatic size reduction with an intentional accuracy tradeoff.
- The distilled LSTM is designed to be the smallest and fastest CPU-friendly model in the project.

## Quick Start

```bash
git clone https://github.com/[username]/relu-ai-intern-challenge.git
cd relu-ai-intern-challenge
make setup
make all
make demo
```

## Project Structure

```text
relu-ai-intern-challenge/
├── data/
├── models/
│   ├── baseline/
│   │   └── distilbert_imdb.pt
│   ├── quantized/
│   │   └── distilbert_imdb_int8.pt
│   ├── onnx/
│   │   ├── distilbert_imdb.onnx
│   │   └── distilbert_imdb_optimized.onnx
│   └── distilled/
│       └── lstm_student.pt
├── scripts/
│   ├── train_baseline.py
│   ├── optimize_quantize.py
│   ├── optimize_onnx.py
│   ├── distill_to_lstm.py
│   ├── benchmark.py
│   ├── edge_simulate.py
│   └── run_inference.py
├── benchmarks/
│   ├── results.json
│   └── plots/
│       ├── latency_comparison.png
│       ├── size_comparison.png
│       └── pareto_accuracy_latency.png
├── report/
│   └── report.pdf
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

## Approach

### Dataset

IMDB sentiment classification (25K train / 25K test). Binary classification. Chose this over CIFAR-10 because transformer optimization for NLP is more challenging and demonstrates deeper understanding of the inference pipeline.

### Baseline

DistilBERT (66M parameters) fine-tuned for 2 epochs. Sequence length 256.

### Optimization Techniques

1. **Dynamic INT8 Quantization** — PyTorch `quantize_dynamic` on all Linear layers
2. **ONNX Runtime Optimization** — Graph-level operator fusion (attention + LayerNorm), constant folding, plus ONNX-level INT8 quantization
3. **Knowledge Distillation** — Distilled DistilBERT into a bidirectional LSTM student using KL-divergence soft targets (`T=4`, `alpha=0.7`)

### Edge Simulation

All benchmarks run on CPU with `torch.set_num_threads(1)` to simulate single-core edge deployment. Batch size 1 is used for latency, batch size 32 for throughput.

## Benchmark Details

Full benchmark output is written to `benchmarks/results.json`.

## Lessons Learned

- Dynamic quantization should be considered a default CPU deployment pass for transformer inference.
- Graph-level ONNX optimization can materially improve transformer latency beyond raw export.
- Distillation exposes the clearest accuracy-versus-efficiency tradeoff in the project.
- The Pareto frontier is the cleanest way to compare deployment-ready model variants.

## Author

[Your name] — UVCE B.Tech
