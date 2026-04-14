from __future__ import annotations

import re
import string

import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
CURRENCY_RE = re.compile(r"([$£€₹]\s?\d+|\d+\s?(usd|eur|gbp|inr|rs))", re.IGNORECASE)
TOKEN_RE = re.compile(r"\b\w+\b")

SPAM_KEYWORDS = {
    "free",
    "winner",
    "urgent",
    "offer",
    "cash",
    "claim",
    "prize",
    "click",
    "credit",
    "bonus",
    "limited",
    "guaranteed",
    "win",
    "call",
    "subscribe",
}


def normalize_text(text: str) -> str:
    lowered = str(text or "").lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _keyword_count(text: str) -> int:
    tokens = TOKEN_RE.findall(text.lower())
    return sum(token in SPAM_KEYWORDS for token in tokens)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["normalized_text"] = enriched["text"].map(normalize_text)
    enriched["char_length"] = enriched["text"].str.len().astype(float)
    enriched["token_count"] = enriched["normalized_text"].str.split().str.len().fillna(0).astype(float)
    enriched["digit_ratio"] = enriched["text"].map(
        lambda value: _safe_ratio(sum(char.isdigit() for char in str(value)), max(len(str(value)), 1))
    )
    enriched["uppercase_ratio"] = enriched["text"].map(
        lambda value: _safe_ratio(sum(char.isupper() for char in str(value)), max(len(str(value)), 1))
    )
    enriched["punctuation_ratio"] = enriched["text"].map(
        lambda value: _safe_ratio(sum(char in string.punctuation for char in str(value)), max(len(str(value)), 1))
    )
    enriched["url_count"] = enriched["text"].str.count(URL_RE).astype(float)
    enriched["phone_count"] = enriched["text"].str.count(PHONE_RE).astype(float)
    enriched["currency_count"] = enriched["text"].str.count(CURRENCY_RE).astype(float)
    enriched["exclamation_count"] = enriched["text"].str.count("!").astype(float)
    enriched["keyword_count"] = enriched["normalized_text"].map(_keyword_count).astype(float)
    enriched["avg_token_length"] = enriched.apply(
        lambda row: _safe_ratio(row["char_length"], row["token_count"]) if row["token_count"] else 0.0,
        axis=1,
    )
    return enriched


def feature_columns() -> list[str]:
    return [
        "char_length",
        "token_count",
        "digit_ratio",
        "uppercase_ratio",
        "punctuation_ratio",
        "url_count",
        "phone_count",
        "currency_count",
        "exclamation_count",
        "keyword_count",
        "avg_token_length",
    ]
