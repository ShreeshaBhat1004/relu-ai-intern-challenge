import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
MAX_LENGTH = 256
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("./models/baseline")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def model_size_mb(model_path: Path) -> float:
    size_bytes = sum(
        file_path.stat().st_size
        for file_path in model_path.iterdir()
        if file_path.suffix in {".bin", ".safetensors"}
    )
    return size_bytes / (1024 * 1024)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("imdb")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

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

    train_dataset = tokenized["train"]
    test_dataset = tokenized["test"]

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "training_output"),
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
        fp16=use_fp16,
        dataloader_num_workers=2,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    eval_metrics = trainer.evaluate(test_dataset)
    metrics = {
        "accuracy": float(eval_metrics["eval_accuracy"]),
        "f1": float(eval_metrics["eval_f1"]),
        "size_mb": round(model_size_mb(OUTPUT_DIR), 3),
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
