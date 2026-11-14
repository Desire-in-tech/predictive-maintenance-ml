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

The recall floor is an explicit operating trade-off, not a claim that missed failures are harmless. Review FDR, precision, recall, confusion matrices and per-mode support together before using predictions operationally. For the five failure-mode models, alerting eligibility is determined exclusively from validation data: a mode is enabled only when its validation threshold satisfies both the recall floor and a maximum FDR of 0.50. If validation has no suitable threshold, the mode is marked unavailable for alerting (its probability score remains available) rather than silently falling back to an all-positive or F1-only threshold. The untouched test split is used only once for final performance reporting; test results do not change thresholds, model selection, or alerting eligibility. The overall binary model retains its existing threshold-selection behavior.

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
