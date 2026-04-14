from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import build_message_corpus, dataset_audit, load_corpus
from .models import (
    evaluate_saved_models,
    fit_benchmark_models,
    run_source_holdout_experiment,
    save_training_manifest,
)
from .predict import predict_to_json


DEFAULT_CORPUS_PATH = Path("data/processed/message_corpus.csv")
DEFAULT_SMS_PATH = Path("spam.csv")
DEFAULT_EMAIL_PATH = Path("data/raw/spamassassin_clean")
DEFAULT_CLASSICAL_DIR = Path("artifacts/classical")
DEFAULT_TRANSFORMER_DIR = Path("artifacts/transformer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the spam intelligence project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus_parser = subparsers.add_parser("build-corpus", help="Build the normalized multi-dataset corpus.")
    corpus_parser.add_argument("--sms-path", type=Path, default=DEFAULT_SMS_PATH)
    corpus_parser.add_argument("--email-root", type=Path, default=DEFAULT_EMAIL_PATH)
    corpus_parser.add_argument("--output", type=Path, default=DEFAULT_CORPUS_PATH)

    train_parser = subparsers.add_parser("train-classical", help="Train benchmark classical models.")
    train_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_CLASSICAL_DIR)

    transformer_parser = subparsers.add_parser("train-transformer", help="Train the transformer benchmark.")
    transformer_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    transformer_parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRANSFORMER_DIR)
    transformer_parser.add_argument("--epochs", type=float, default=1.0)
    transformer_parser.add_argument("--max-train-samples", type=int, default=2000)
    transformer_parser.add_argument("--max-eval-samples", type=int, default=800)

    predict_parser = subparsers.add_parser("predict", help="Classify raw messages with a saved model.")
    predict_parser.add_argument("--model-path", type=Path, default=DEFAULT_CLASSICAL_DIR / "best_model.joblib")
    predict_parser.add_argument("--message", action="append", required=True)
    predict_parser.add_argument("--output", type=Path, default=None)

    audit_parser = subparsers.add_parser("audit", help="Print dataset audit information.")
    audit_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    return parser


def command_build_corpus(args) -> int:
    corpus = build_message_corpus(args.sms_path, args.email_root, output_path=args.output)
    print(corpus.groupby(["channel", "label"]).size())
    return 0


def command_train_classical(args) -> int:
    corpus = load_corpus(args.corpus)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = corpus[corpus["split"] == "train"].copy()
    validation_frame = corpus[corpus["split"] == "validation"].copy()
    test_frame = corpus[corpus["split"] == "test"].copy()

    trained_models, validation_summary = fit_benchmark_models(train_frame, validation_frame, output_dir)
    test_summary, best_model_name = evaluate_saved_models(trained_models, test_frame, output_dir)
    run_source_holdout_experiment(corpus, best_model_name, output_dir)
    save_training_manifest(output_dir, args.corpus, best_model_name, test_summary)

    best_model_path = output_dir / f"{best_model_name}.joblib"
    (output_dir / "best_model.joblib").write_bytes(best_model_path.read_bytes())
    print("Validation metrics:")
    print(validation_summary.to_string(index=False))
    print("\nTest metrics:")
    print(test_summary.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    return 0


def command_train_transformer(args) -> int:
    from .transformer import train_transformer_benchmark

    corpus = load_corpus(args.corpus)
    metrics = train_transformer_benchmark(
        corpus=corpus,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    print(metrics)
    return 0


def command_predict(args) -> int:
    print(predict_to_json(args.model_path, args.message, output_path=args.output))
    return 0


def command_audit(args) -> int:
    corpus = load_corpus(args.corpus)
    print(dataset_audit(corpus).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-corpus":
        return command_build_corpus(args)
    if args.command == "train-classical":
        return command_train_classical(args)
    if args.command == "train-transformer":
        return command_train_transformer(args)
    if args.command == "predict":
        return command_predict(args)
    if args.command == "audit":
        return command_audit(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
