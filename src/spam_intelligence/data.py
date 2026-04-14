from __future__ import annotations

import html
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS = ["text", "label", "channel", "source", "split"]
LABEL_MAP = {"ham": "ham", "spam": "spam", "0": "ham", "1": "spam"}
EMAIL_FOLDER_LABELS = {
    "20021010_easy_ham": "ham",
    "20030228_easy_ham_2": "ham",
    "20021010_hard_ham": "ham",
    "20021010_spam": "spam",
    "20050311_spam_2": "spam",
}
WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_label(label: str) -> str:
    key = str(label).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_MAP[key]


def _decode_bytes(payload: bytes, charset: str | None) -> str:
    candidate_charsets = [charset, "utf-8", "latin-1", "cp1252"]
    for candidate in candidate_charsets:
        if not candidate:
            continue
        try:
            return payload.decode(candidate, errors="ignore")
        except LookupError:
            continue
    return payload.decode("utf-8", errors="ignore")


def _stratify_keys(frame: pd.DataFrame) -> pd.Series:
    return frame["label"] + "__" + frame["channel"]


def _assign_splits(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    frame = frame.copy()
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_state,
        stratify=_stratify_keys(frame),
    )
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=random_state,
        stratify=_stratify_keys(temp_frame),
    )
    frame["split"] = "train"
    frame.loc[validation_frame.index, "split"] = "validation"
    frame.loc[test_frame.index, "split"] = "test"
    return frame


def load_sms_dataset(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, encoding="latin-1")
    renamed = frame.rename(columns={"Message": "text", "Class": "label"})
    renamed = renamed[["text", "label"]].copy()
    renamed["label"] = renamed["label"].map(normalize_label)
    renamed["text"] = renamed["text"].map(clean_text)
    renamed["channel"] = "sms"
    renamed["source"] = "uci_sms_spam_collection"
    return renamed


def _extract_email_text(raw_bytes: bytes) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    segments: list[str] = []
    subject = clean_text(message.get("subject", ""))
    if subject:
        segments.append(subject)

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type not in {"text/plain", "text/html"}:
                continue
            try:
                content = part.get_content()
            except Exception:
                payload = part.get_payload(decode=True) or b""
                content = _decode_bytes(payload, part.get_content_charset())
            if content_type == "text/html":
                content = HTML_TAG_RE.sub(" ", content)
            cleaned = clean_text(content)
            if cleaned:
                segments.append(cleaned)
    else:
        try:
            body = message.get_content()
        except Exception:
            payload = message.get_payload(decode=True) or b""
            body = _decode_bytes(payload, message.get_content_charset())
        cleaned = clean_text(HTML_TAG_RE.sub(" ", body))
        if cleaned:
            segments.append(cleaned)

    if not segments:
        return clean_text(raw_bytes.decode("utf-8", errors="ignore"))
    return clean_text(" ".join(segments))


def load_spamassassin_dataset(root: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for folder_name, label in EMAIL_FOLDER_LABELS.items():
        folder = root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing email corpus folder: {folder}")
        for email_file in sorted(path for path in folder.rglob("*") if path.is_file()):
            text = _extract_email_text(email_file.read_bytes())
            if not text:
                continue
            records.append(
                {
                    "text": text,
                    "label": label,
                    "channel": "email",
                    "source": folder_name,
                }
            )
    return pd.DataFrame(records)


def validate_corpus(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        issues.append(f"Missing columns: {', '.join(missing)}")
    if frame["text"].isna().any():
        issues.append("Text column contains null values.")
    if frame["label"].isna().any():
        issues.append("Label column contains null values.")
    empty_rows = (frame["text"].astype(str).str.strip() == "").sum()
    if empty_rows:
        issues.append(f"Text column contains {empty_rows} empty rows.")
    invalid_labels = sorted(set(frame["label"]) - {"ham", "spam"})
    if invalid_labels:
        issues.append(f"Unexpected labels: {', '.join(invalid_labels)}")
    duplicate_rows = frame.duplicated(subset=["text", "label", "channel"]).sum()
    if duplicate_rows:
        issues.append(f"Found {duplicate_rows} duplicate records.")
    return issues


def build_message_corpus(
    sms_path: Path,
    email_root: Path,
    output_path: Path | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    sms = load_sms_dataset(sms_path)
    email = load_spamassassin_dataset(email_root)
    frame = pd.concat([sms, email], ignore_index=True)
    frame["text"] = frame["text"].map(clean_text)
    frame["text_key"] = (
        frame["text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    frame = frame.drop_duplicates(subset=["text_key", "label", "channel"]).drop(columns=["text_key"])
    frame = _assign_splits(frame, random_state=random_state)
    issues = validate_corpus(frame)
    if issues:
        raise ValueError("Corpus validation failed: " + " | ".join(issues))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame


def load_corpus(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    issues = validate_corpus(frame)
    if issues:
        raise ValueError("Corpus validation failed: " + " | ".join(issues))
    return frame


def dataset_audit(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.assign(char_length=frame["text"].str.len())
        .groupby(["source", "channel", "label"])
        .agg(records=("text", "size"), average_length=("char_length", "mean"))
        .reset_index()
        .sort_values(["channel", "source", "label"])
    )
    return summary
