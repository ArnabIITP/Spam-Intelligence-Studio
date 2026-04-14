from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_engineered_features, normalize_text
from .models import load_model


def _heuristic_signals(text: str) -> list[str]:
    normalized = normalize_text(text)
    signals: list[str] = []
    if "http" in normalized or "www." in normalized:
        signals.append("contains a link")
    if any(token in normalized for token in ["free", "winner", "claim", "urgent", "prize", "offer"]):
        signals.append("contains high-risk promotional language")
    if sum(char.isdigit() for char in text) >= 6:
        signals.append("contains many digits or contact-like patterns")
    if text.count("!") >= 2:
        signals.append("uses strong punctuation emphasis")
    if not signals:
        signals.append("message language looks conversational and low urgency")
    return signals[:3]


def _probabilities(model, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw))
    return None


def predict_messages(model_path: Path, messages: list[str]) -> list[dict[str, object]]:
    model = load_model(model_path)
    frame = pd.DataFrame(
        {
            "text": messages,
            "label": ["ham"] * len(messages),
            "channel": ["inference"] * len(messages),
            "source": ["cli"] * len(messages),
            "split": ["inference"] * len(messages),
        }
    )
    features = add_engineered_features(frame)
    predictions = model.predict(features)
    scores = _probabilities(model, features)
    results: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        result = {
            "text": message,
            "prediction": str(predictions[index]),
            "confidence": float(scores[index]) if scores is not None else None,
            "signals": _heuristic_signals(message),
        }
        results.append(result)
    return results


def predict_to_json(model_path: Path, messages: list[str], output_path: Path | None = None) -> str:
    payload = predict_messages(model_path, messages)
    text = json.dumps(payload, indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text
