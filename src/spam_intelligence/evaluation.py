from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def probability_scores(model, features: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw))
    return None


def evaluate_predictions(y_true, y_pred, y_score=None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "spam_precision": precision_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "spam_recall": recall_score(y_true, y_pred, pos_label="spam", zero_division=0),
    }
    if y_score is not None:
        binary_true = pd.Series(y_true).map({"ham": 0, "spam": 1}).to_numpy()
        metrics["pr_auc"] = average_precision_score(binary_true, y_score)
    else:
        metrics["pr_auc"] = 0.0
    return metrics


def evaluate_model(model, frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = frame["label"]
    y_pred = model.predict(frame)
    y_score = probability_scores(model, frame)
    metrics = evaluate_predictions(y_true, y_pred, y_score)
    report_frame = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T
    return metrics, report_frame


def metrics_table(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["composite_score", "pr_auc", "spam_recall"], ascending=False)


def collect_error_examples(model, frame: pd.DataFrame, limit: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.Series(model.predict(frame), index=frame.index, name="predicted_label")
    scores = probability_scores(model, frame)
    result = frame[["text", "label", "channel", "source"]].copy()
    result["predicted_label"] = predictions
    if scores is not None:
        result["spam_score"] = scores
    false_positives = result[(result["label"] == "ham") & (result["predicted_label"] == "spam")].head(limit)
    false_negatives = result[(result["label"] == "spam") & (result["predicted_label"] == "ham")].head(limit)
    return false_positives, false_negatives


def save_confusion_matrix(model, frame: pd.DataFrame, output_path: Path) -> None:
    predictions = model.predict(frame)
    matrix = confusion_matrix(frame["label"], predictions, labels=["ham", "spam"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_precision_recall_curve(model, frame: pd.DataFrame, output_path: Path) -> None:
    scores = probability_scores(model, frame)
    if scores is None:
        return
    binary_true = frame["label"].map({"ham": 0, "spam": 1}).to_numpy()
    precision, recall, _ = precision_recall_curve(binary_true, scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, color="#1f77b4")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
