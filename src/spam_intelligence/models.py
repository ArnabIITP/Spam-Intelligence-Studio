from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .evaluation import (
    collect_error_examples,
    evaluate_model,
    metrics_table,
    save_confusion_matrix,
    save_precision_recall_curve,
    write_json,
)
from .features import add_engineered_features, feature_columns


def build_preprocessor() -> ColumnTransformer:
    numeric_columns = feature_columns()
    return ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    max_features=15000,
                ),
                "normalized_text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=8000,
                ),
                "normalized_text",
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        sparse_threshold=0.2,
    )


def model_registry(random_state: int = 42, calibration_folds: int = 3) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "linear_svc": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            ),
            cv=calibration_folds,
        ),
        "complement_nb": ComplementNB(alpha=0.6),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
    }


def _composite_score(metrics: dict[str, float], model_name: str) -> float:
    bias = 0.002 if model_name == "logistic_regression" else 0.0
    return (
        0.45 * metrics["pr_auc"]
        + 0.35 * metrics["spam_recall"]
        + 0.20 * metrics["macro_f1"]
        + bias
    )


def fit_benchmark_models(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    output_dir: Path,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    trained_models: dict[str, Pipeline] = {}
    rows: list[dict[str, float | str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    train_features = add_engineered_features(train_frame)
    validation_features = add_engineered_features(validation_frame)
    min_class_count = int(train_frame["label"].value_counts().min())
    calibration_folds = max(2, min(3, min_class_count))

    for model_name, estimator in model_registry(
        random_state=random_state,
        calibration_folds=calibration_folds,
    ).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(train_features, train_features["label"])
        metrics, _ = evaluate_model(pipeline, validation_features)
        metrics["composite_score"] = _composite_score(metrics, model_name)
        rows.append({"model": model_name, **metrics})
        trained_models[model_name] = pipeline
        joblib.dump(pipeline, output_dir / f"{model_name}.joblib")

    summary = metrics_table(rows)
    summary.to_csv(output_dir / "validation_metrics.csv", index=False)
    return trained_models, summary


def load_model(model_path: Path):
    return joblib.load(model_path)


def top_linear_features(model: Pipeline, top_n: int = 20) -> dict[str, list[dict[str, float | str]]]:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return {"positive": [], "negative": []}
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = classifier.coef_[0]
    sorted_indices = np.argsort(coefficients)
    negative = [
        {"feature": str(feature_names[index]), "weight": float(coefficients[index])}
        for index in sorted_indices[:top_n]
    ]
    positive = [
        {"feature": str(feature_names[index]), "weight": float(coefficients[index])}
        for index in sorted_indices[-top_n:][::-1]
    ]
    return {"positive": positive, "negative": negative}


def evaluate_saved_models(
    models: dict[str, Pipeline],
    test_frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, str]:
    test_features = add_engineered_features(test_frame)
    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        metrics, _ = evaluate_model(model, test_features)
        metrics["composite_score"] = _composite_score(metrics, model_name)
        rows.append({"model": model_name, **metrics})

    summary = metrics_table(rows)
    summary.to_csv(output_dir / "test_metrics.csv", index=False)
    best_model_name = str(summary.iloc[0]["model"])
    best_model = models[best_model_name]
    save_confusion_matrix(best_model, test_features, output_dir / "confusion_matrix.png")
    save_precision_recall_curve(best_model, test_features, output_dir / "precision_recall_curve.png")
    false_positives, false_negatives = collect_error_examples(best_model, test_features, limit=10)
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)
    if "logistic_regression" in models:
        write_json(output_dir / "top_linear_features.json", top_linear_features(models["logistic_regression"]))
    return summary, best_model_name


def run_source_holdout_experiment(
    frame: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for holdout_channel in sorted(frame["channel"].unique()):
        train_frame = frame[frame["channel"] != holdout_channel]
        test_frame = frame[frame["channel"] == holdout_channel]
        if train_frame["label"].nunique() < 2 or test_frame["label"].nunique() < 2:
            continue
        min_class_count = int(train_frame["label"].value_counts().min())
        calibration_folds = max(2, min(3, min_class_count))
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    model_registry(
                        random_state=random_state,
                        calibration_folds=calibration_folds,
                    )[model_name],
                ),
            ]
        )
        pipeline.fit(add_engineered_features(train_frame), train_frame["label"])
        metrics, _ = evaluate_model(pipeline, add_engineered_features(test_frame))
        rows.append(
            {
                "holdout_channel": holdout_channel,
                "model": model_name,
                **metrics,
                "composite_score": _composite_score(metrics, model_name),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "cross_channel_holdout.csv", index=False)
    return summary


def save_training_manifest(
    output_dir: Path,
    corpus_path: Path,
    best_model_name: str,
    model_summary: pd.DataFrame,
) -> None:
    payload = {
        "corpus_path": str(corpus_path),
        "best_model": best_model_name,
        "models": model_summary.to_dict(orient="records"),
    }
    write_json(output_dir / "training_manifest.json", payload)
