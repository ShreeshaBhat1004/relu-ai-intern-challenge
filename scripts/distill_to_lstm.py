import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
TEMPERATURE = 4.0
ALPHA = 0.7
BASELINE_DIR = Path("./models/baseline")
OUTPUT_DIR = Path("./models/distilled")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LSTMStudent(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embeds)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(hidden_cat)


def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def collate_dataset(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def tokenize_dataset(tokenizer, split):
    dataset = load_dataset("imdb", split=split)

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
    return tokenized


def evaluate(student, dataloader, device):
    student.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)
            logits = student(input_ids, attention_mask=attention_mask)
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained(str(BASELINE_DIR))
    teacher = DistilBertForSequenceClassification.from_pretrained(str(BASELINE_DIR)).to(device)
    teacher.eval()

    student = LSTMStudent().to(device)
    optimizer = Adam(student.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5)

    train_dataset = tokenize_dataset(tokenizer, "train")
    test_dataset = tokenize_dataset(tokenizer, "test")
    train_loader = collate_dataset(train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(1, EPOCHS + 1):
        student.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids,
                    attention_mask=attention_mask,
                ).logits
            student_logits = student(input_ids, attention_mask=attention_mask)
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=TEMPERATURE,
                alpha=ALPHA,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(student, test_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        scheduler.step(1 - metrics["accuracy"])
        print(
            f"Epoch {epoch}: train_loss={avg_loss:.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}, val_f1={metrics['f1']:.4f}"
        )

    model_path = OUTPUT_DIR / "lstm_student.pt"
    torch.save(student.state_dict(), model_path)

    final_metrics = evaluate(student, test_loader, device)
    final_metrics["size_mb"] = round(model_size_mb(model_path), 3)

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
