# Industrial Predictive Maintenance — Task 9

A reproducible machine-learning project using the **UCI AI4I 2020 Predictive Maintenance** dataset. It predicts overall machine failure and independently predicts the five annotated failure modes, with explicit attention to costly false alarms.

## Project scope

- Download the dataset directly from UCI; no dataset files are committed.
- Explore class imbalance and numeric sensor relationships.
- Train an overall binary machine-failure classifier.
- Train five independent binary classifiers for the AI4I failure-mode indicators: TWF, HDF, PWF, OSF and RNF.
- Select each decision threshold on validation data to minimize False Discovery Rate (FDR), subject to a configurable recall floor.
- Evaluate on a held-out test split that is not used for fitting or threshold selection.
- Export reusable Joblib inference artifacts and JSON/CSV evaluation reports.
- Reuse both prediction interfaces in Task 10's FastAPI service.

## Targets and failure modes

The overall target is the binary `Machine failure` label. The dataset also supplies five binary failure-mode indicators:

| Indicator | Failure mode |
|---|---|
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |

These indicators are **multi-label**, not five mutually exclusive classes. A row can have more than one indicator, so the project trains one binary classifier per mode rather than forcing every observation into exactly one class. The overall failure model remains available as a separate baseline and alert.

## Industry constraint: minimize false discoveries

FDR is defined as `FP / (TP + FP)`: the proportion of positive alerts that are false. Threshold selection uses validation predictions only and minimizes FDR while respecting a recall floor (0.50 for the overall model; 0.30 by default for each rarer failure mode). The untouched test split is used only for final evaluation.

The recall floor is an explicit operating trade-off, not a claim that missed failures are harmless. Review FDR, precision, recall, confusion matrices and per-mode support together before using predictions operationally. For the five failure-mode models, alerting eligibility is determined exclusively from validation data: a mode is enabled only when its validation threshold satisfies both the recall floor and a maximum FDR of 0.50. If validation has no suitable threshold, the mode is marked unavailable for alerting (its probability score remains available) rather than silently falling back to an all-positive or F1-only threshold. The fitted estimator used to generate validation probabilities is retained unchanged for both held-out test evaluation and deployment, so the reported test metrics correspond to the exported model. The untouched test split is used only once for final performance reporting; test results do not change thresholds, model selection, or alerting eligibility. The overall binary model retains its existing threshold-selection behavior.

## Run in Google Colab

Open [the notebook](notebooks/task_09_predictive_maintenance.ipynb) in Colab using **File → Open notebook → GitHub**, or open its GitHub URL directly. Run cells from top to bottom. The notebook installs dependencies, clones the project branch, downloads AI4I from UCI, trains both model components, produces correlation analysis, and optionally backs up artifacts to Google Drive.

## Optional local run

```bash
python -m pip install -r requirements.txt
python -m src.train --output-dir artifacts --model random_forest --min-recall 0.50
python -m src.failure_modes --output-dir artifacts --min-recall 0.30
pytest -q
```

Generated outputs include:

- `artifacts/predictive_maintenance_pipeline.joblib` — overall binary model.
- `artifacts/failure_modes_pipeline.joblib` — five failure-mode models and thresholds.
- `artifacts/metrics.json` and `artifacts/failure_modes_metrics.json` — test metrics.
- `artifacts/sensor_failure_correlations.csv` — ranked Pearson and Spearman sensor associations.
- Feature-schema JSON files for reproducible inference.

Generated artifacts and dataset files are excluded from Git.

## Sensor correlation analysis

The analysis ranks each numeric sensor's Pearson and Spearman association with overall machine failure and each failure-mode indicator. It is descriptive: correlation does not establish causation, and this static dataset does not establish that a sensor is a temporal lead indicator. The categorical machine Type is handled by the predictive models but is not included in numeric correlation ranking.

## Inference interface

The existing `PredictiveMaintenanceModel` binary `predict()` interface remains backward-compatible. To return overall failure plus mode-specific probabilities and threshold decisions, initialize it with both artifacts:

```python
from src.predict import PredictiveMaintenanceModel

model = PredictiveMaintenanceModel(
    artifact_path="artifacts/predictive_maintenance_pipeline.joblib",
    failure_modes_artifact_path="artifacts/failure_modes_pipeline.joblib",
)

predictions = model.predict_with_failure_modes([
    {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }
])
```

The output includes the overall failure probability and decision, each mode's probability and decision, and the list of predicted modes. Multiple modes can be predicted for the same observation.

## Bonus: Remaining Useful Life

A Time-to-Failure / Remaining Useful Life regression model is not claimed. The AI4I 2020 dataset does not provide genuine per-machine operating histories and remaining-life labels needed to train and validate that regression task without fabricating targets. A suitable run-to-failure dataset would be required.

## Layout

- `notebooks/` — Colab workflow
- `src/` — loading, training, evaluation, correlation analysis and inference
- `tests/` — unit tests
- `artifacts/` — generated model artifacts and reports (not committed)

## Dataset

UCI Machine Learning Repository — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).


---

# Task 10 — Containerized inference API

Task 10 packages the Task 9 models behind a FastAPI service. The service downloads both Joblib artifacts from the public Hugging Face model repository at startup; model binaries and datasets remain outside Git.

## API capabilities

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Process liveness |
| GET | /ready | Model-artifact readiness |
| POST | /predict | JSON batch inference |
| POST | /predict/csv | CSV upload and batch inference |
| GET | /docs | Interactive OpenAPI documentation |

Both prediction endpoints return the overall failure probability/decision and the five independent failure-mode probabilities/decisions from Task 9. A failure mode whose validation-selected threshold was unavailable remains score-only and does not generate an alert.

### JSON request

Example body:

    {
      "readings": [{
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0
      }]
    }

The JSON body contains a readings array with 1–5,000 observations. Each observation must contain exactly the six sensor fields shown above. Type must be L, M or H. Sensor values must be JSON numbers (not numeric strings); extra or missing fields are rejected. Invalid request bodies and Pydantic validation failures return HTTP 400. If model artifacts cannot be loaded, prediction endpoints return HTTP 503 while /health remains available and /ready reports the initialization problem.

### CSV request

Upload a CSV file to POST /predict/csv using multipart form-data with the field name file. The header must contain exactly these six columns:

    Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]

CSV numeric columns are parsed as numbers and validated before inference. Uploads are limited to 5 MiB and 5,000 rows. Invalid CSV files return HTTP 400.

## Run locally

Python 3.11 is recommended.

    python -m venv .venv
    pip install -r requirements.txt

Set the Hugging Face model repository and an immutable commit SHA from the model repository's revision history. The verified Task 9 artifacts are pinned to `a0735888cd3bbd8c47abc012521d54c89ed5f53b`; do not replace it with `main` for a reproducible deployment.

Linux/macOS:

    export HF_REPO_ID=Desire-dev/predictive-maintenance
    export HF_MODEL_REVISION=<immutable-hugging-face-commit-sha>
    uvicorn api.main:app --host 0.0.0.0 --port 8000

PowerShell:

    $env:HF_REPO_ID = "Desire-dev/predictive-maintenance"
    $env:HF_MODEL_REVISION = "<immutable-hugging-face-commit-sha>"
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Open http://localhost:8000/docs for interactive API testing. The model repository is public, so runtime downloads do not require a Hugging Face write token. Never place a write token in source code or deployment environment variables.

## Docker

    docker build -t predictive-maintenance-api .
    docker run --rm -p 8000:8000 -e HF_REPO_ID=Desire-dev/predictive-maintenance -e HF_MODEL_REVISION=<immutable-hugging-face-commit-sha> predictive-maintenance-api

The image runs as a non-root user. Model artifacts are downloaded into the container's temporary/cache directory at startup and are not baked into the image.

## Tests and CI

    PYTHONPATH=. pytest -q

GitHub Actions runs the Python test suite and builds the Docker image on pushes to main, task-09/**, and task-10/**, and on pull requests targeting main. The Postman collection is at postman/Task-10-Predictive-Maintenance.postman_collection.json; set its baseUrl variable to the local or deployed API URL.

## Render deployment configuration

The repository includes render.yaml for a Docker-based Render web service. The Render Blueprint pins HF_MODEL_REVISION to the verified immutable model commit `a0735888cd3bbd8c47abc012521d54c89ed5f53b`. If configuring the service manually, use that exact value. The service's readiness check uses /ready, so it will not be considered ready until both model files have downloaded and loaded. No Hugging Face token is required for the public model repository.

The service is not deployed automatically by this repository change. Review the pull request and configure the model revision before deployment.
