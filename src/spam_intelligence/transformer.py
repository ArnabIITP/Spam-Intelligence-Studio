from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .evaluation import write_json

LABEL2ID = {"ham": 0, "spam": 1}
ID2LABEL = {0: "ham", 1: "spam"}


def _compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    probabilities = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    predictions = np.argmax(logits, axis=1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "spam_precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "spam_recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "pr_auc": average_precision_score(labels, probabilities),
    }


class EncodedTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 160):
        self.labels = torch.tensor(frame["label"].map(LABEL2ID).tolist(), dtype=torch.long)
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _sample_per_label(frame: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    label_count = frame["label"].nunique()
    per_label = max(1, max_samples // max(label_count, 1))
    sampled = pd.concat(
        [
            chunk.sample(n=min(len(chunk), per_label), random_state=42)
            for _, chunk in frame.groupby("label")
        ],
        ignore_index=True,
    )
    return sampled


def train_transformer_benchmark(
    corpus: pd.DataFrame,
    output_dir: Path,
    model_name: str = "distilbert-base-uncased",
    epochs: float = 1.0,
    batch_size: int = 8,
    max_train_samples: int | None = 2000,
    max_eval_samples: int | None = 800,
) -> dict[str, float]:
    train_frame = corpus[corpus["split"] == "train"][["text", "label"]].copy()
    validation_frame = corpus[corpus["split"] == "validation"][["text", "label"]].copy()

    if max_train_samples is not None and len(train_frame) > max_train_samples:
        train_frame = _sample_per_label(train_frame, max_train_samples)
    if max_eval_samples is not None and len(validation_frame) > max_eval_samples:
        validation_frame = _sample_per_label(validation_frame, max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = EncodedTextDataset(train_frame, tokenizer)
    validation_dataset = EncodedTextDataset(validation_frame, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        report_to=[],
        use_cpu=True,
        seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    write_json(output_dir / "transformer_metrics.json", metrics)
    return metrics
