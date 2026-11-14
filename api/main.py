"""HTTP API for the Task 9 predictive-maintenance models."""
import csv
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import PredictionRequest, PredictionResponse, SensorReading

logger = logging.getLogger(__name__)
ARTIFACT_NAMES = (
    "predictive_maintenance_pipeline.joblib",
    "failure_modes_pipeline.joblib",
)
FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_BATCH_SIZE = 5000


def load_model():
    """Download the two immutable model files and construct the Task 9 wrapper."""
    from huggingface_hub import hf_hub_download
    from src.predict import PredictiveMaintenanceModel

    repo_id = os.getenv("HF_REPO_ID", "Desire-dev/predictive-maintenance")
    revision = os.getenv("HF_MODEL_REVISION", "main")
    artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", "/tmp/predictive-maintenance"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    local_paths = {}
    for filename in ARTIFACT_NAMES:
        local_paths[filename] = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            cache_dir=str(artifact_dir / "hub-cache"),
        )

    return PredictiveMaintenanceModel(
        artifact_path=local_paths[ARTIFACT_NAMES[0]],
        failure_modes_artifact_path=local_paths[ARTIFACT_NAMES[1]],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = load_model()
        logger.info("Predictive-maintenance model artifacts loaded.")
    except Exception as exc:  # readiness reports the failure; app can still serve /health
        app.state.model_error = str(exc)
        logger.exception("Model initialization failed.")
    yield


app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "Serves the Task 9 overall machine-failure model and five independent "
        "failure-mode models trained on the UCI AI4I 2020 dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """The task specification requires HTTP 400 for invalid/malformed inputs."""
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request", "errors": exc.errors()},
    )


def _ensure_ready(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Check /ready for initialization details.",
        )
    return model


def _run_predictions(model, readings):
    records = [reading.model_dump(by_alias=True) for reading in readings]
    return model.predict_with_failure_modes(records)


@app.get("/health")
async def health():
    """Liveness endpoint; does not imply model readiness."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    """Readiness endpoint indicating whether both model artifacts loaded."""
    if getattr(request.app.state, "model", None) is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": getattr(request.app.state, "model_error", None)
                or "Model initialization has not completed.",
            },
        )
    return {"status": "ready", "model_repo": os.getenv(
        "HF_REPO_ID", "Desire-dev/predictive-maintenance"
    ), "model_revision": os.getenv("HF_MODEL_REVISION", "main")}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, request: Request):
    """Predict overall machine failure and each independently modelled failure mode."""
    model = _ensure_ready(request)
    predictions = _run_predictions(model, payload.readings)
    return {"predictions": predictions, "count": len(predictions)}


@app.post("/predict/csv", response_model=PredictionResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    """Accept a CSV with the exact six AI4I sensor column names."""
    model = _ensure_ready(request)
    if file.content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
        raise HTTPException(status_code=400, detail="Upload must be a CSV file.")

    raw = await file.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(status_code=400, detail="CSV exceeds the 5 MiB upload limit.")
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise ValueError("CSV is empty or has no header row.")
        if set(reader.fieldnames) != set(FEATURE_COLUMNS):
            raise ValueError(
                "CSV header must contain exactly these columns: "
                + ", ".join(FEATURE_COLUMNS)
            )
        rows = list(reader)
        if not rows:
            raise ValueError("CSV must contain at least one data row.")
        if len(rows) > MAX_BATCH_SIZE:
            raise ValueError(f"CSV cannot exceed {MAX_BATCH_SIZE} rows.")
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("CSV contains malformed or incomplete rows.")
        readings = [SensorReading.model_validate(row) for row in rows]
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    except Exception as exc:
        # Pydantic validation errors from CSV are converted to the same 400 contract.
        from pydantic import ValidationError
        if isinstance(exc, ValidationError):
            raise HTTPException(
                status_code=400,
                detail={"message": "CSV contains invalid sensor values.", "errors": exc.errors()},
            ) from exc
        raise

    predictions = _run_predictions(model, readings)
    return {"predictions": predictions, "count": len(predictions)}
