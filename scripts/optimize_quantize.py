import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

SEED = 42
MAX_LENGTH = 256
BASELINE_DIR = Path("./models/baseline")
OUTPUT_DIR = Path("./models/quantized")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def evaluate(model, tokenizer):
    dataset = load_dataset("imdb", split="test")

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    dataloader = DataLoader(tokenized, batch_size=64)
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating quantized model"):
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            logits = model(**model_inputs).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained(str(BASELINE_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(BASELINE_DIR))
    model.eval()
    model.cpu()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    state_dict_path = OUTPUT_DIR / "distilbert_imdb_int8.pt"
    full_model_path = OUTPUT_DIR / "distilbert_imdb_int8_full.pt"
    torch.save(quantized_model.state_dict(), state_dict_path)
    torch.save(quantized_model, full_model_path)
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    metrics = evaluate(quantized_model, tokenizer)
    metrics["size_mb"] = round(file_size_mb(full_model_path), 3)

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
