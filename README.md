# Industrial Predictive Maintenance — Task 9

A reproducible ML project using the **UCI AI4I 2020 Predictive Maintenance** dataset. The model predicts the binary `Machine failure` target from machine type and sensor readings, with explicit attention to costly false alarms.

## Scope

- Download data directly from UCI; no dataset files are committed.
- Explore class imbalance and numeric sensor correlations.
- Train a preprocessing + classifier pipeline.
- Select the decision threshold on validation data using a recall floor and false discovery rate (FDR).
- Evaluate once on an untouched test split.
- Export one Joblib artifact containing preprocessing, classifier, threshold, schema and metrics.
- Reuse the artifact in Task 10's FastAPI service.

**Target clarification:** AI4I has a binary `Machine failure` target and separate failure-mode indicator columns. Those indicators are not a mutually exclusive five-class target. This project starts with binary failure prediction. The dataset does not provide genuine remaining-useful-life labels, so RUL regression is not claimed from it.

## Run in Google Colab

Open [the notebook](notebooks/task_09_predictive_maintenance.ipynb) in Colab using **File → Open notebook → GitHub**, or open its GitHub URL directly. Run cells from top to bottom. The notebook installs dependencies, clones this branch's source, retrieves the dataset from UCI, trains/evaluates the model and optionally saves outputs to Google Drive.

## Optional local run

```bash
python -m pip install -r requirements.txt
python -m src.train --output-dir artifacts --model random_forest --min-recall 0.50
pytest -q
```

The artifact is written to `artifacts/predictive_maintenance_pipeline.joblib`. Generated artifacts and datasets are excluded from Git.

## Evaluation terminology

- **FDR:** FP / (TP + FP), the fraction of positive alerts that are false.
- **False positive rate:** FP / (FP + TN), the fraction of actual non-failure cases incorrectly alerted.

Threshold selection uses validation predictions only. The test set is held out from fitting and threshold selection. A lower FDR can come at the cost of missed failures, so interpret the threshold alongside recall and the confusion matrix.

## Layout

- `notebooks/` — Colab workflow
- `src/` — data loading, training, evaluation and inference
- `tests/` — unit tests
- `artifacts/` — generated outputs; not committed

## Task 10 handoff

Task 10 will load the exported Joblib artifact in FastAPI, validate sensor payloads with Pydantic, expose prediction and health endpoints, add API tests, containerize with Docker, and deploy the API/frontend on free-tier services where available. The model will be transferred as a versioned deployment asset; the deployed service will not depend on private Google Drive authentication.

## Dataset

UCI Machine Learning Repository — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
