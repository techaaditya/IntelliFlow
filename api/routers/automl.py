"""FastAPI routes for IntelliFlow Engine 1."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from engines.automl.pipeline import detect_task_type, run_automl
from engines.automl.registry import AutoMLRegistry


router = APIRouter(prefix="/automl", tags=["AutoML"])

TaskType = Literal["classification", "regression", "auto"]


class TrainRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    metric: str | None = None
    n_trials: int = Field(default=5, ge=1, le=200)
    task_type: TaskType = "auto"


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    model_uri: str | None = None


class PredictResponse(BaseModel):
    model_uri: str
    predictions: list[Any]


class TrainResponse(BaseModel):
    run_id: str
    best_score: float
    best_model_family: str
    best_params: dict[str, Any]
    model_name: str
    model_version: str | None = None
    model_uri: str
    endpoint_url: str
    task_type: str
    metric: str
    target_column: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_uri: str
    latest_version: str | None = None
    alias: str


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return metadata for the latest registered full AutoML pipeline."""

    try:
        info = AutoMLRegistry().get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ModelInfoResponse(**info.__dict__)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict from raw rows using the registered full AutoML pipeline."""

    try:
        registry = AutoMLRegistry()
        predictions = registry.predict(request.rows, model_uri=request.model_uri)
        model_uri = request.model_uri or registry.get_model_uri()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(model_uri=model_uri, predictions=_json_safe_list(predictions))


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """Train Engine 1 from JSON rows."""

    dataset = pd.DataFrame(request.rows)
    return _train_dataframe(
        dataset=dataset,
        target_column=request.target_column,
        metric=request.metric,
        n_trials=request.n_trials,
        task_type=request.task_type,
    )


@router.post("/upload-train", response_model=TrainResponse)
async def upload_train(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    metric: str | None = Form(default=None),
    n_trials: int = Form(default=5),
    task_type: TaskType = Form(default="auto"),
) -> TrainResponse:
    """Train Engine 1 from an uploaded CSV or Excel file."""

    filename = file.filename or ""
    try:
        if filename.lower().endswith(".csv"):
            dataset = pd.read_csv(file.file)
        elif filename.lower().endswith((".xlsx", ".xls")):
            dataset = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Upload a .csv, .xlsx, or .xls file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {exc}") from exc

    return _train_dataframe(
        dataset=dataset,
        target_column=target_column,
        metric=metric,
        n_trials=n_trials,
        task_type=task_type,
    )


def _train_dataframe(
    *,
    dataset: pd.DataFrame,
    target_column: str,
    metric: str | None,
    n_trials: int,
    task_type: TaskType,
) -> TrainResponse:
    if dataset.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")
    if target_column not in dataset.columns:
        raise HTTPException(status_code=400, detail=f"Target column {target_column!r} was not found.")

    dataset = dataset.dropna(subset=[target_column])
    if dataset.empty:
        raise HTTPException(status_code=400, detail="No rows remain after dropping missing target values.")

    resolved_task_type = detect_task_type(dataset[target_column]) if task_type == "auto" else task_type
    resolved_metric = metric or ("accuracy" if resolved_task_type == "classification" else "r2")

    try:
        result = run_automl(
            dataset=dataset,
            target_column=target_column,
            metric=resolved_metric,
            n_trials=n_trials,
            task_type=resolved_task_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TrainResponse(
        run_id=result["run_id"],
        best_score=result["best_score"],
        best_model_family=result["best_model_family"],
        best_params=result["best_params"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        model_uri=result["model_uri"],
        endpoint_url=result["endpoint_url"],
        task_type=result["task_type"],
        metric=result["metric"],
        target_column=result["target_column"],
    )


def _json_safe_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [item.item() if hasattr(item, "item") else item for item in values]

