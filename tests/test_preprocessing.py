import pandas as pd

from spam_intelligence.data import normalize_label
from spam_intelligence.features import add_engineered_features, normalize_text


def test_normalize_label_maps_binary_labels():
    assert normalize_label("spam") == "spam"
    assert normalize_label("ham") == "ham"
    assert normalize_label("1") == "spam"
    assert normalize_label("0") == "ham"


def test_engineered_features_capture_risky_patterns():
    frame = pd.DataFrame(
        {
            "text": ["FREE cash!!! Visit http://example.com now 999999"],
            "label": ["spam"],
            "channel": ["sms"],
            "source": ["test"],
            "split": ["train"],
        }
    )
    enriched = add_engineered_features(frame)
    row = enriched.iloc[0]
    assert normalize_text(frame.iloc[0]["text"]).startswith("free cash")
    assert row["url_count"] == 1
    assert row["exclamation_count"] == 3
    assert row["digit_ratio"] > 0
    assert row["keyword_count"] >= 2
