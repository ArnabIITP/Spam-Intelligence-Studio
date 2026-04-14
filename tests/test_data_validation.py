import pandas as pd

from spam_intelligence.data import validate_corpus


def test_validate_corpus_flags_missing_columns_and_duplicates():
    frame = pd.DataFrame(
        {
            "text": ["hello", "hello"],
            "label": ["ham", "ham"],
            "channel": ["sms", "sms"],
            "source": ["sample", "sample"],
            "split": ["train", "train"],
        }
    )
    issues = validate_corpus(frame)
    assert any("duplicate" in issue.lower() for issue in issues)


def test_validate_corpus_catches_bad_labels():
    frame = pd.DataFrame(
        {
            "text": ["hello"],
            "label": ["maybe"],
            "channel": ["sms"],
            "source": ["sample"],
            "split": ["train"],
        }
    )
    issues = validate_corpus(frame)
    assert any("unexpected labels" in issue.lower() for issue in issues)
