from pathlib import Path
from uuid import uuid4

import pandas as pd

from spam_intelligence.models import evaluate_saved_models, fit_benchmark_models
from spam_intelligence.predict import predict_messages


def _sample_corpus() -> pd.DataFrame:
    records = [
        ("Free cash prize claim now", "spam", "sms", "sms_source", "train"),
        ("Win a free ticket today", "spam", "sms", "sms_source", "train"),
        ("Urgent account update click now", "spam", "email", "email_source", "train"),
        ("Cheap meds and bonus offer", "spam", "email", "email_source", "train"),
        ("Are we still meeting for lunch", "ham", "sms", "sms_source", "train"),
        ("Please send the project notes", "ham", "email", "email_source", "train"),
        ("Can you call me tonight", "ham", "sms", "sms_source", "validation"),
        ("Team sync has moved to noon", "ham", "email", "email_source", "validation"),
        ("Limited offer claim reward now", "spam", "sms", "sms_source", "validation"),
        ("Congratulations you won cash", "spam", "email", "email_source", "validation"),
        ("Dinner is ready at home", "ham", "sms", "sms_source", "test"),
        ("Quarterly report attached for review", "ham", "email", "email_source", "test"),
        ("Winner! call now for prize", "spam", "sms", "sms_source", "test"),
        ("Reset password to avoid suspension", "spam", "email", "email_source", "test"),
    ]
    return pd.DataFrame(records, columns=["text", "label", "channel", "source", "split"])


def test_training_pipeline_runs_end_to_end():
    corpus = _sample_corpus()
    train_frame = corpus[corpus["split"] == "train"]
    validation_frame = corpus[corpus["split"] == "validation"]
    test_frame = corpus[corpus["split"] == "test"]
    output_dir = Path("artifacts") / f"test_smoke_{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)

    models, validation_summary = fit_benchmark_models(train_frame, validation_frame, output_dir)
    assert not validation_summary.empty

    test_summary, best_model_name = evaluate_saved_models(models, test_frame, output_dir)
    assert best_model_name in set(test_summary["model"])
    best_model_path = output_dir / f"{best_model_name}.joblib"
    assert best_model_path.exists()

    predictions = predict_messages(best_model_path, ["Free reward waiting for you", "Let's catch up tomorrow"])
    assert len(predictions) == 2
    assert {entry["prediction"] for entry in predictions}.issubset({"ham", "spam"})
