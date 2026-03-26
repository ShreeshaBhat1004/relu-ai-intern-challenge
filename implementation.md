# ReLU Technologies — AI Internship Challenge: Full Implementation Spec

> **Purpose**: This document is a complete, unambiguous implementation specification. Follow it top-to-bottom. Every file, every function, every hyperparameter is specified. Do not deviate unless a step explicitly says "your choice".

---

## 1. PROJECT STRUCTURE

Create this exact directory tree:

```
relu-ai-intern-challenge/
├── data/                          # Auto-created by scripts (gitignored)
├── models/                        # Saved model artifacts (gitignored)
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
│   └── run_inference.py           # The demo script (deliverable)
├── benchmarks/
│   ├── results.json               # Machine-readable benchmark output
│   └── plots/
│       ├── latency_comparison.png
│       ├── size_comparison.png
│       └── pareto_accuracy_latency.png
├── report/
│   └── report.pdf                 # 3-page technical report
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

---

## 2. ENVIRONMENT & DEPENDENCIES

### `requirements.txt`
```
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
onnx>=1.15.0
onnxruntime>=1.16.0
optimum[onnxruntime]>=1.14.0
matplotlib>=3.8.0
seaborn>=0.13.0
numpy>=1.24.0
tabulate>=0.9.0
psutil>=5.9.0
tqdm>=4.66.0
scikit-learn>=1.3.0
```

### `.gitignore`
```
data/
models/
__pycache__/
*.pyc
.ipynb_checkpoints/
benchmarks/results.json
benchmarks/plots/
```

### `Makefile`
```makefile
.PHONY: setup train optimize benchmark edge demo all clean

setup:
	pip install -r requirements.txt

train:
	python scripts/train_baseline.py

optimize: optimize-quantize optimize-onnx optimize-distill

optimize-quantize:
	python scripts/optimize_quantize.py

optimize-onnx:
	python scripts/optimize_onnx.py

optimize-distill:
	python scripts/distill_to_lstm.py

benchmark:
	python scripts/benchmark.py

edge:
	python scripts/edge_simulate.py

demo:
	python scripts/run_inference.py --model all

all: setup train optimize benchmark edge

clean:
	rm -rf models/ data/ benchmarks/results.json benchmarks/plots/
```

---

## 3. SCRIPT-BY-SCRIPT IMPLEMENTATION

---

### 3.1 `scripts/train_baseline.py`

**Purpose**: Fine-tune DistilBERT on IMDB sentiment classification. Save the model.

**Steps**:

1. **Load dataset**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("imdb")
   ```
   Use the full train split (25k samples) and full test split (25k samples).

2. **Tokenization**:
   ```python
   from transformers import DistilBertTokenizer
   tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
   ```
   - `max_length = 256` (NOT 512 — this is intentional for edge relevance; shorter sequences = faster inference, and IMDB reviews don't need full 512 to classify well)
   - `truncation=True`, `padding="max_length"`
   - Tokenize the entire dataset and set format to `torch`

3. **Model**:
   ```python
   from transformers import DistilBertForSequenceClassification
   model = DistilBertForSequenceClassification.from_pretrained(
       "distilbert-base-uncased", num_labels=2
   )
   ```

4. **Training config**:
   ```python
   from transformers import TrainingArguments, Trainer
   training_args = TrainingArguments(
       output_dir="./models/baseline/training_output",
       num_train_epochs=2,
       per_device_train_batch_size=32,
       per_device_eval_batch_size=64,
       warmup_steps=500,
       weight_decay=0.01,
       logging_steps=100,
       eval_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
       metric_for_best_model="accuracy",
       fp16=True,  # Use if GPU available, else set False
       dataloader_num_workers=2,
   )
   ```

5. **Metrics**:
   ```python
   import numpy as np
   from sklearn.metrics import accuracy_score, f1_score

   def compute_metrics(eval_pred):
       logits, labels = eval_pred
       preds = np.argmax(logits, axis=-1)
       return {
           "accuracy": accuracy_score(labels, preds),
           "f1": f1_score(labels, preds, average="binary"),
       }
   ```

6. **Train and save**:
   ```python
   trainer = Trainer(
       model=model, args=training_args,
       train_dataset=tokenized_train, eval_dataset=tokenized_test,
       compute_metrics=compute_metrics,
   )
   trainer.train()
   trainer.save_model("./models/baseline")
   tokenizer.save_pretrained("./models/baseline")
   ```

7. **Print and save baseline metrics**:
   After training, evaluate on test set. Print accuracy, F1, model size (in MB), and save these to `models/baseline/metrics.json`.
   
   Model size calculation:
   ```python
   import os
   model_path = "./models/baseline"
   size_mb = sum(
       os.path.getsize(os.path.join(model_path, f))
       for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))
   ) / (1024 * 1024)
   ```

**Expected outputs**:
- `models/baseline/` containing model weights, config, tokenizer
- `models/baseline/metrics.json` with `{"accuracy": ~0.92, "f1": ~0.92, "size_mb": ~255}`

---

### 3.2 `scripts/optimize_quantize.py`

**Purpose**: Apply dynamic INT8 quantization to the baseline model.

**Steps**:

1. **Load baseline model** (on CPU — quantization is CPU-only):
   ```python
   import torch
   from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

   model = DistilBertForSequenceClassification.from_pretrained("./models/baseline")
   model.eval()
   model.cpu()
   ```

2. **Apply dynamic quantization**:
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model,
       {torch.nn.Linear},  # Quantize all Linear layers (attention + FFN)
       dtype=torch.qint8
   )
   ```

3. **Save quantized model**:
   ```python
   torch.save(quantized_model.state_dict(), "./models/quantized/distilbert_imdb_int8.pt")
   ```
   Also save the model object for later loading:
   ```python
   torch.save(quantized_model, "./models/quantized/distilbert_imdb_int8_full.pt")
   ```

4. **Evaluate on test set**: Load IMDB test split, tokenize, run inference, compute accuracy & F1.

5. **Print and save metrics**: accuracy, F1, quantized model size in MB, save to `models/quantized/metrics.json`.

**Expected outputs**:
- `models/quantized/` containing quantized weights
- Accuracy should be within 0.5% of baseline (dynamic quant on transformers is nearly lossless)
- Size reduction: ~2-3x smaller

---

### 3.3 `scripts/optimize_onnx.py`

**Purpose**: Export to ONNX, apply graph-level optimizations, benchmark.

**Steps**:

1. **Export baseline to ONNX**:
   ```python
   from optimum.onnxruntime import ORTModelForSequenceClassification
   from transformers import DistilBertTokenizer

   # Export using Optimum (handles the complexity of dynamic axes etc.)
   model = ORTModelForSequenceClassification.from_pretrained(
       "./models/baseline",
       export=True
   )
   model.save_pretrained("./models/onnx/baseline")
   ```

2. **Apply ONNX Runtime optimizations**:
   ```python
   from optimum.onnxruntime import ORTOptimizer
   from optimum.onnxruntime.configuration import OptimizationConfig

   optimizer = ORTOptimizer.from_pretrained(model)
   optimization_config = OptimizationConfig(
       optimization_level=99,  # Maximum optimization
       optimize_for_gpu=False,  # CPU-targeted
       fp16=False,  # Stay FP32 for CPU
       enable_transformers_specific_optimizations=True,  # Fuses attention, layer norm, etc.
   )
   optimizer.optimize(
       save_dir="./models/onnx/optimized",
       optimization_config=optimization_config,
   )
   ```

3. **Also create a quantized ONNX variant** (bonus — ONNX-level INT8):
   ```python
   from optimum.onnxruntime import ORTQuantizer
   from optimum.onnxruntime.configuration import AutoQuantizationConfig

   quantizer = ORTQuantizer.from_pretrained("./models/onnx/baseline")
   qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
   # Fallback if AVX512 not available:
   # qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
   quantizer.quantize(
       save_dir="./models/onnx/quantized",
       quantization_config=qconfig,
   )
   ```

4. **Evaluate all ONNX variants** on IMDB test set.

5. **Save metrics** to `models/onnx/metrics.json` with entries for `onnx_baseline`, `onnx_optimized`, and `onnx_quantized`.

**Expected outputs**:
- `models/onnx/baseline/`, `models/onnx/optimized/`, `models/onnx/quantized/`
- ONNX graph optimizations: expect 1.3-2x latency improvement from operator fusion
- ONNX quantized: expect further 1.5-2x on top of that

---

### 3.4 `scripts/distill_to_lstm.py`

**Purpose**: Knowledge distillation from DistilBERT teacher into a tiny LSTM student. This is the differentiator.

**Steps**:

1. **Define the student model**:
   ```python
   import torch
   import torch.nn as nn

   class LSTMStudent(nn.Module):
       def __init__(self, vocab_size=30522, embed_dim=128, hidden_dim=128,
                    num_layers=1, num_classes=2, dropout=0.3):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
           self.lstm = nn.LSTM(
               embed_dim, hidden_dim, num_layers=num_layers,
               batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
           )
           self.classifier = nn.Sequential(
               nn.Dropout(dropout),
               nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
               nn.ReLU(),
               nn.Dropout(dropout),
               nn.Linear(64, num_classes)
           )

       def forward(self, input_ids, attention_mask=None):
           embeds = self.embedding(input_ids)
           if attention_mask is not None:
               # Pack padded sequences for efficiency
               lengths = attention_mask.sum(dim=1).cpu()
               packed = nn.utils.rnn.pack_padded_sequence(
                   embeds, lengths, batch_first=True, enforce_sorted=False
               )
               output, (hidden, _) = self.lstm(packed)
           else:
               output, (hidden, _) = self.lstm(embeds)
           # Concatenate forward and backward final hidden states
           hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
           return self.classifier(hidden_cat)
   ```

2. **Distillation loss**:
   ```python
   import torch.nn.functional as F

   def distillation_loss(student_logits, teacher_logits, labels,
                         temperature=4.0, alpha=0.7):
       """
       Combined KL-divergence (soft targets) + cross-entropy (hard targets).
       alpha controls the balance: higher alpha = more teacher influence.
       """
       soft_loss = F.kl_div(
           F.log_softmax(student_logits / temperature, dim=-1),
           F.softmax(teacher_logits / temperature, dim=-1),
           reduction="batchmean"
       ) * (temperature ** 2)

       hard_loss = F.cross_entropy(student_logits, labels)

       return alpha * soft_loss + (1 - alpha) * hard_loss
   ```

3. **Training loop**:
   - Load the baseline DistilBERT as teacher. Set `teacher.eval()` and `torch.no_grad()` for teacher forward passes.
   - Use the same tokenized IMDB dataset (the LSTM student uses the same DistilBERT tokenizer — it shares the vocabulary).
   - `max_length = 256` (same as baseline)
   - Hyperparameters:
     - `epochs = 5`
     - `batch_size = 64`
     - `learning_rate = 1e-3`
     - `optimizer = Adam`
     - `scheduler = ReduceLROnPlateau(patience=1, factor=0.5)`
     - `temperature = 4.0`
     - `alpha = 0.7`
   - Use standard PyTorch training loop (DataLoader, optimizer.zero_grad, loss.backward, optimizer.step).
   - Print train loss and validation accuracy each epoch.

4. **Save student model**:
   ```python
   torch.save(student.state_dict(), "./models/distilled/lstm_student.pt")
   ```

5. **Evaluate and save metrics** to `models/distilled/metrics.json`.

**Expected outputs**:
- `models/distilled/lstm_student.pt` — should be ~2-5 MB
- Accuracy: ~85-88% (lower than DistilBERT's ~92%, but model is 50-100x smaller)
- This tradeoff IS the point — document it explicitly

---

### 3.5 `scripts/benchmark.py`

**Purpose**: Comprehensive benchmarking of ALL model variants. Produce comparison table + plots.

**Models to benchmark** (6 total):
1. `baseline` — DistilBERT FP32 (PyTorch)
2. `quantized` — DistilBERT INT8 (PyTorch dynamic quantization)
3. `onnx_baseline` — DistilBERT FP32 (ONNX Runtime)
4. `onnx_optimized` — DistilBERT FP32 (ONNX Runtime + graph optimizations)
5. `onnx_quantized` — DistilBERT INT8 (ONNX Runtime quantized)
6. `distilled_lstm` — LSTM student model

**Metrics to measure for each**:
- **Accuracy** (on full IMDB test set)
- **F1 score** (binary, on full IMDB test set)
- **Model size** (MB on disk)
- **Inference latency** — single sample, batch size 1, CPU only
  - Measure over 100 warmup runs (discard) + 1000 timed runs
  - Report: mean, std, p50, p95, p99 (all in milliseconds)
  - Use `time.perf_counter()` for timing
- **Throughput** — samples per second at batch size 32, CPU only
- **Peak memory** — use `tracemalloc` during inference:
  ```python
  import tracemalloc
  tracemalloc.start()
  # ... run inference ...
  current, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  peak_mb = peak / (1024 * 1024)
  ```

**Output 1 — Console table** (using `tabulate`):
```
| Model            | Accuracy | F1    | Size (MB) | Latency p50 (ms) | Latency p95 (ms) | Throughput (s/s) | Peak Mem (MB) |
|------------------|----------|-------|-----------|-------------------|-------------------|------------------|---------------|
| Baseline         | 0.923    | 0.922 | 255.4     | 48.2              | 52.1              | 28.3             | 312.5         |
| Quantized INT8   | 0.921    | 0.920 | 89.2      | 31.5              | 34.8              | 43.7             | 198.3         |
| ONNX Baseline    | 0.923    | 0.922 | 255.1     | 35.4              | 38.2              | 38.9             | 280.1         |
| ONNX Optimized   | 0.923    | 0.922 | 252.8     | 28.1              | 30.5              | 48.2             | 265.4         |
| ONNX Quantized   | 0.920    | 0.919 | 68.5      | 18.3              | 20.1              | 72.1             | 145.2         |
| LSTM Student     | 0.867    | 0.865 | 3.2       | 2.1               | 2.8               | 580.4            | 28.7          |
```
(These numbers are illustrative — actual values will vary.)

**Output 2 — `benchmarks/results.json`**: Full results dict with all metrics for every model.

**Output 3 — Plots** (save to `benchmarks/plots/`):

Plot 1: `latency_comparison.png`
- Grouped bar chart: each model on x-axis, bars for p50/p95/p99 latency
- Use seaborn style, clear labels, title "Inference Latency Comparison (CPU, Batch Size 1)"

Plot 2: `size_comparison.png`
- Horizontal bar chart: model size in MB
- Title "Model Size Comparison"

Plot 3: `pareto_accuracy_latency.png` (**THIS IS THE MONEY PLOT**)
- Scatter plot: x-axis = latency (ms), y-axis = accuracy
- Each point labeled with model name
- Draw the Pareto frontier line connecting non-dominated points
- Title "Accuracy vs Latency: Pareto Frontier"
- This plot tells the entire story of the project in one image

**Plotting style**:
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
```

---

### 3.6 `scripts/edge_simulate.py`

**Purpose**: Simulate edge deployment constraints. CPU-only, low batch, memory profiling.

**Steps**:

1. **Force CPU**: `torch.set_num_threads(1)` — simulate a single-core edge device.

2. **Run all 6 models** with:
   - Batch size 1 (single-sample inference)
   - Batch size 4 (small batch)
   - Measure latency and memory for each

3. **Sustained inference test**: Run 500 sequential inferences, measure:
   - Average latency over time (does it degrade?)
   - Peak memory over time
   - Use `psutil.Process().memory_info().rss` for system-level memory tracking

4. **Print an "Edge Readiness Report"** to console:
   ```
   === EDGE READINESS REPORT ===
   Target: ARM Cortex-A72 (Raspberry Pi 4, 2GB RAM)

   Model              | Fits in 512MB? | Latency < 100ms? | Edge Viable?
   -------------------|----------------|-------------------|-------------
   Baseline           | YES            | YES               | MARGINAL
   Quantized INT8     | YES            | YES               | YES
   ONNX Optimized     | YES            | YES               | YES
   ONNX Quantized     | YES            | YES               | STRONG YES
   LSTM Student       | YES            | YES               | EXCELLENT
   ```

5. **Analysis section** — print to console AND save to `benchmarks/edge_report.txt`:
   - Explain the accuracy/efficiency tradeoff for each model
   - Discuss what would change on actual ARM hardware (NEON SIMD, reduced cache, thermal throttling)
   - Recommend which model to deploy for different edge scenarios:
     - "If accuracy is paramount: ONNX Quantized DistilBERT"
     - "If latency/size is paramount: LSTM Student"
     - "Best overall tradeoff: ONNX Quantized"
   - Mention TFLite and ONNX Runtime Mobile as next steps for real deployment

---

### 3.7 `scripts/run_inference.py` (Demo Script — Deliverable)

**Purpose**: Clean CLI demo that anyone can run.

**Interface**:
```bash
# Run a single model
python scripts/run_inference.py --model baseline --text "This movie was absolutely fantastic!"

# Run all models on the same input and compare
python scripts/run_inference.py --model all --text "This movie was absolutely fantastic!"

# Run with default sample text
python scripts/run_inference.py --model optimized
```

**Arguments**:
- `--model`: choices = `[baseline, quantized, onnx, onnx_optimized, onnx_quantized, distilled, all]`
  - `optimized` is an alias for `onnx_quantized` (the recommended deployment model)
- `--text`: input text to classify. Default: `"This movie was a complete waste of time. Terrible acting and plot."`
- `--benchmark`: if flag present, also run timing (100 iterations) and print latency

**Output format** (for `--model all`):
```
╔══════════════════════════════════════════════════════════════╗
║           ReLU AI Intern Challenge — Inference Demo          ║
╠══════════════════════════════════════════════════════════════╣

Input: "This movie was absolutely fantastic!"

┌──────────────────┬────────────┬────────────┬──────────────┐
│ Model            │ Prediction │ Confidence │ Latency (ms) │
├──────────────────┼────────────┼────────────┼──────────────┤
│ Baseline         │ POSITIVE   │ 97.3%      │ 48.2         │
│ Quantized INT8   │ POSITIVE   │ 97.1%      │ 31.5         │
│ ONNX Baseline    │ POSITIVE   │ 97.3%      │ 35.4         │
│ ONNX Optimized   │ POSITIVE   │ 97.3%      │ 28.1         │
│ ONNX Quantized   │ POSITIVE   │ 97.2%      │ 18.3         │
│ LSTM Student     │ POSITIVE   │ 89.4%      │ 2.1          │
└──────────────────┴────────────┴────────────┴──────────────┘

Recommendation: ONNX Quantized offers the best accuracy/speed tradeoff.
```

Use `argparse` for CLI. Handle missing model files gracefully (skip with warning). Use `tabulate` for the table.

---

## 4. README.md

Write a README with these exact sections:

```markdown
# Efficient AI Inference for Edge Devices

> ReLU Technologies — AI Systems Internship Challenge

## Results at a Glance

[Insert the benchmark comparison table from benchmark.py output here]

[Insert the Pareto plot image here: ![Pareto](benchmarks/plots/pareto_accuracy_latency.png)]

## Key Findings

- **ONNX Runtime + INT8 quantization** reduces latency by Xx while losing <0.5% accuracy
- **Knowledge distillation to LSTM** achieves Xx size reduction (255MB → 3MB) with ~5% accuracy drop
- The LSTM student runs at X samples/sec on CPU — viable for real-time edge inference

## Quick Start

```bash
git clone https://github.com/[username]/relu-ai-intern-challenge.git
cd relu-ai-intern-challenge
make setup
make all        # Train, optimize, benchmark — everything
make demo       # Quick inference demo
```

## Project Structure

[Insert the tree from Section 1]

## Approach

### Dataset
IMDB sentiment classification (25K train / 25K test). Binary classification.
Chose this over CIFAR-10 because transformer optimization for NLP is more
challenging and demonstrates deeper understanding of the inference pipeline.

### Baseline
DistilBERT (66M parameters) fine-tuned for 2 epochs. Sequence length 256.

### Optimization Techniques

1. **Dynamic INT8 Quantization** — PyTorch `quantize_dynamic` on all Linear layers
2. **ONNX Runtime Optimization** — Graph-level operator fusion (attention + LayerNorm),
   constant folding, plus ONNX-level INT8 quantization
3. **Knowledge Distillation** — Distilled DistilBERT into a 2-layer bidirectional LSTM
   (128-dim hidden, ~1M parameters) using KL-divergence soft targets (T=4, α=0.7)

### Edge Simulation
All benchmarks run on CPU with `torch.set_num_threads(1)` to simulate
single-core edge deployment. Batch size 1 for latency, batch size 32 for throughput.

## Benchmark Details

[Link to full results in benchmarks/results.json]

## Lessons Learned

[Write 3-4 bullet points after seeing actual results — e.g.:
- Dynamic quantization is nearly free for transformers — always do it
- ONNX graph optimization matters more than I expected for attention-heavy models
- Knowledge distillation quality is highly sensitive to temperature — T=4 worked much better than T=2
- The Pareto frontier clearly shows there's no single "best" model — it depends on deployment constraints]

## Author
[Your name] — UVCE B.Tech
```

---

## 5. TECHNICAL REPORT (`report/report.pdf`)

3 pages. Use LaTeX or Markdown→PDF. Structure:

**Page 1: Introduction + Approach**
- Problem statement (2 sentences)
- Why DistilBERT + IMDB (3 sentences)
- Optimization techniques overview (brief paragraph each for quantization, ONNX, distillation)

**Page 2: Results**
- The full benchmark table
- The Pareto plot
- 2-3 sentences interpreting each

**Page 3: Edge Analysis + Lessons**
- Edge simulation findings
- Accuracy/efficiency tradeoffs discussion
- What would change on real hardware (ARM, Jetson)
- 3-4 lessons learned
- Future work (1 paragraph): QAT, pruning, TFLite conversion, actual Raspberry Pi testing

---

## 6. CRITICAL IMPLEMENTATION NOTES

### Things Codex must get right:

1. **All inference benchmarking must be on CPU**: Even if training uses GPU, ALL timing/memory measurements happen on CPU with `model.cpu()` and `torch.set_num_threads(1)` for single-core simulation.

2. **Warmup runs**: Always do 100 warmup iterations before timing. First runs include JIT compilation and cache warming — don't count them.

3. **Use `time.perf_counter()`** not `time.time()` — perf_counter has nanosecond resolution.

4. **ONNX Runtime session options**:
   ```python
   import onnxruntime as ort
   sess_options = ort.SessionOptions()
   sess_options.intra_op_num_threads = 1  # Match edge simulation
   sess_options.inter_op_num_threads = 1
   session = ort.InferenceSession("model.onnx", sess_options)
   ```

5. **Quantized model loading**: PyTorch dynamic quantized models must be loaded on CPU. Do NOT try `.cuda()` on them.

6. **ONNX export — use Optimum, not raw torch.onnx.export**: The `optimum` library handles dynamic axes, attention masks, and tokenizer integration properly. Raw export is error-prone for transformers.

7. **Distillation — teacher must be in eval mode with no_grad**:
   ```python
   teacher.eval()
   with torch.no_grad():
       teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits
   ```

8. **Seeds for reproducibility**:
   ```python
   import torch, random, numpy as np
   SEED = 42
   random.seed(SEED)
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.backends.cudnn.deterministic = True
   ```
   Put this at the top of every script.

9. **Graceful GPU/CPU handling**: Each script should auto-detect:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```
   Use GPU for training, but ALWAYS switch to CPU for benchmarking.

10. **The LSTM student uses the DistilBERT tokenizer**: Don't create a separate vocabulary. The student model's embedding layer has `vocab_size=30522` (DistilBERT's vocab size). This means you can use the exact same tokenized inputs for both teacher and student.

---

## 7. COMMIT STRATEGY

Make meaningful commits, not one big dump:

```
1. "Initial project structure, requirements, Makefile"
2. "Add baseline DistilBERT training script"
3. "Add dynamic INT8 quantization"
4. "Add ONNX export and optimization pipeline"
5. "Add knowledge distillation: DistilBERT → LSTM"
6. "Add comprehensive benchmarking with plots"
7. "Add edge simulation and readiness report"
8. "Add demo inference script"
9. "Add README with results"
10. "Add technical report"
```