# Spam Intelligence Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#quick-start)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)
[![Task](https://img.shields.io/badge/Task-Spam%20Detection-orange)](#what-this-project-does)
[![Channels](https://img.shields.io/badge/Channels-SMS%20%2B%20Email-informational)](#datasets)

Spam Intelligence Studio is a multi-dataset spam-filtering project for classifying suspicious messages across channels. Instead of treating spam detection as a single CSV exercise, this repository builds a unified workflow that merges SMS spam and email spam into one normalized corpus, benchmarks multiple classical NLP models, includes a transformer comparison, and exposes a small inference interface for real message classification.


## What This Project Does

The repository is designed to read like a complete machine learning project:

- reusable package code instead of notebook-only logic
- explicit data ingestion and validation
- benchmark training and evaluation flows
- saved artifacts for analysis and reporting
- a final notebook that consumes package outputs instead of reimplementing the pipeline

### Project Goals

1. Create a realistic and well-structured spam-filtering pipeline for multi-channel message classification.
2. Combine multiple public sources so the problem is broader than SMS-only classification.
3. Compare strong classical baselines with a transformer benchmark.
4. Keep the project easy to run on a CPU-first environment while still feeling advanced and production-aware.

## Quick Start

The file `messageSpamFiltering.py` acts as a thin launcher for the package CLI.

```bash
pip install -e .
python messageSpamFiltering.py build-corpus
python messageSpamFiltering.py train-classical
python messageSpamFiltering.py train-transformer --epochs 1 --max-train-samples 2000 --max-eval-samples 800
python messageSpamFiltering.py predict --message "Free reward waiting for you" --message "Let's meet at 6"
```

<details>
<summary>See full command workflow</summary>

### 1. Install

```bash
pip install -e .
```

### 2. Build the merged corpus

```bash
python messageSpamFiltering.py build-corpus
```

This step:

- loads the SMS dataset
- loads the SpamAssassin email corpus
- cleans and normalizes records
- removes duplicates
- creates `train`, `validation`, and `test` splits
- writes `data/processed/message_corpus.csv`

### 3. Train and evaluate the classical benchmark

```bash
python messageSpamFiltering.py train-classical
```

This step writes:

- trained model files
- validation metrics
- test metrics
- confusion matrix
- precision-recall curve
- false positive examples
- false negative examples
- top linear feature weights
- training manifest

### 4. Run the transformer benchmark

```bash
python messageSpamFiltering.py train-transformer --epochs 1 --max-train-samples 2000 --max-eval-samples 800
```

For faster smoke runs, use smaller values such as:

```bash
python messageSpamFiltering.py train-transformer --epochs 0.2 --max-train-samples 400 --max-eval-samples 200
```

### 5. Inspect the dataset audit

```bash
python messageSpamFiltering.py audit
```

### 6. Predict on new messages

```bash
python messageSpamFiltering.py predict --message "Free reward waiting for you" --message "Let's meet at 6"
```

The prediction output includes:

- predicted label
- confidence score
- short heuristic explanation signals

</details>

## Datasets

The project currently uses two public data sources:

- `spam.csv`
  - local SMS spam dataset included in the repository
  - normalized into `text`, `label`, `channel`, `source`, `split`
  - labeled as `channel = sms`
- Apache SpamAssassin public corpus
  - downloaded into `data/raw/archives/`
  - extracted into a readable corpus under `data/raw/spamassassin_clean/`
  - labeled as `channel = email`

The merged processed dataset is saved to:

```text
data/processed/message_corpus.csv
```

The current built corpus contains both ham and spam across SMS and email channels, which lets the project measure both within-source performance and cross-channel generalization.

## Normalized Schema

Every record in the processed corpus follows the same schema:

- `text` - cleaned message body used for modeling
- `label` - `ham` or `spam`
- `channel` - `sms` or `email`
- `source` - source dataset or source folder name
- `split` - `train`, `validation`, or `test`

This schema keeps the training code independent from the raw source format.

## Modeling Approach

### Classical Benchmark

The main benchmark pipeline combines text representations with engineered metadata:

- word-level TF-IDF using uni-grams and bi-grams
- character TF-IDF using character windows
- handcrafted metadata features such as:
  - message length
  - token count
  - digit ratio
  - uppercase ratio
  - punctuation ratio
  - URL count
  - phone-like pattern count
  - currency-token count
  - exclamation count
  - promotional keyword count
  - average token length

The benchmark models are:

- Logistic Regression
- Linear SVC with probability calibration
- Complement Naive Bayes
- Random Forest

Model selection is not based on accuracy alone. The project ranks models using a weighted composite centered on:

- PR-AUC
- spam recall
- macro F1

### Transformer Benchmark

The advanced comparison track fine-tunes:

- `distilbert-base-uncased`

The transformer path is intentionally lightweight by default so the benchmark can still be executed on CPU with capped sample sizes.

## Current Results

The saved classical benchmark currently selects `logistic_regression` as the best overall model. Based on the latest generated artifacts:

- Test accuracy: `0.9782`
- Test macro F1: `0.9683`
- Spam precision: `0.9558`
- Spam recall: `0.9454`
- PR-AUC: `0.9830`

The project also saves a cross-channel holdout analysis. That experiment shows that moving between SMS and email causes a noticeable generalization drop, which is useful because it proves the repository is studying domain shift rather than reporting a single easy score.

The latest lightweight transformer run completed successfully and produced a saved metrics file under:

```text
artifacts/transformer/transformer_metrics.json
```

Because that run was intentionally short, it should be treated as a benchmark scaffold rather than a fully tuned final model.

## Repository Structure

```text
.
|-- artifacts/
|   |-- classical/
|   `-- transformer/
|-- data/
|   |-- processed/
|   `-- raw/
|-- notebooks/
|   `-- spam_intelligence_report.ipynb
|-- src/
|   `-- spam_intelligence/
|-- tests/
|-- messageSpamFiltering.py
|-- pyproject.toml
|-- README.md
|-- requirements.txt
`-- spam.csv
```

<details>
<summary>Important directories and what they contain</summary>

- `src/spam_intelligence/`
  - package code for loading data, building features, training models, evaluation, prediction, and transformer benchmarking
- `data/raw/archives/`
  - downloaded SpamAssassin archive files
- `data/raw/spamassassin_clean/`
  - cleaned extracted email corpus used by the loader
- `data/processed/`
  - processed merged dataset
- `artifacts/classical/`
  - saved classical models, metrics, plots, and error-analysis outputs
- `artifacts/transformer/`
  - saved transformer model outputs and transformer metrics
- `notebooks/spam_intelligence_report.ipynb`
  - final report notebook for presenting results and analysis
- `tests/`
  - validation and smoke-test coverage for the workflow

</details>

## Generated Artifacts

After running the classical benchmark, the main outputs are:

- `data/processed/message_corpus.csv`
- `artifacts/classical/best_model.joblib`
- `artifacts/classical/logistic_regression.joblib`
- `artifacts/classical/linear_svc.joblib`
- `artifacts/classical/complement_nb.joblib`
- `artifacts/classical/random_forest.joblib`
- `artifacts/classical/validation_metrics.csv`
- `artifacts/classical/test_metrics.csv`
- `artifacts/classical/cross_channel_holdout.csv`
- `artifacts/classical/confusion_matrix.png`
- `artifacts/classical/precision_recall_curve.png`
- `artifacts/classical/false_positives.csv`
- `artifacts/classical/false_negatives.csv`
- `artifacts/classical/top_linear_features.json`
- `artifacts/classical/training_manifest.json`

The transformer benchmark writes:

- `artifacts/transformer/model/`
- `artifacts/transformer/transformer_metrics.json`

## Report Notebook

The main analysis notebook is:

```text
notebooks/spam_intelligence_report.ipynb
```

The notebook imports the package and presents the project in a report format instead of embedding the full pipeline inside notebook cells. It covers:

1. project framing
2. dataset audit and imbalance discussion
3. channel and source analysis
4. feature engineering overview
5. classical benchmark review
6. transformer benchmark review
7. qualitative error analysis
8. deployment-style inference examples

## Testing

The test suite currently covers:

- label normalization
- engineered feature behavior
- corpus validation
- benchmark training smoke coverage
- notebook contract validation

Run all tests with:

```bash
pytest
```

## Practical Notes

- `spam.csv` provides the local SMS source dataset used by the pipeline.
- The transformer benchmark may need network access the first time it downloads model files.
- On restrictive Windows environments, notebook execution or temporary cache creation may require elevated permissions.
- The classical benchmark is the most reliable path for quick local results.

## License

The repository code is released under the MIT License. See `LICENSE`.

Dataset and model-source note:

- the repository code is covered by the MIT license
- third-party datasets are not relicensed by this repository
- `spam.csv` and the Apache SpamAssassin corpus remain subject to their original dataset terms, attribution requirements, and distribution conditions
- pretrained transformer weights and tokenizer assets also remain subject to their original upstream licenses and usage terms
