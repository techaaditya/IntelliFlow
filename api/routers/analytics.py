"""FastAPI routes for IntelliFlow Engine 2 (Analytics & EDA).

Mirrors the Engine 1 router conventions: JSON-row and file-upload entry points,
Pydantic request models, and clear ``HTTPException`` errors with recovery hints.
Accepts CSV / JSON / Parquet / Excel uploads as required by the spec.
"""

from __future__ import annotations

import io
import json
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engines.analytics.pipeline import CAPABILITIES, run_eda
from engines.analytics.profiling import DataProfiler

router = APIRouter(prefix="/analytics", tags=["Analytics"])

ReportFormat = Literal["json", "html"]


class AnalyzeRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, description="Dataset as a list of record dicts.")
    target_column: str | None = None
    timestamp_column: str | None = None
    user_id_column: str | None = None
    event_column: str | None = None
    segment_columns: list[str] | None = None
    funnel_steps: list[str] | None = None
    target_event: str | None = None
    freq: str = "day"
    retention_unit: Literal["day", "week"] = "day"
    capabilities: list[str] | None = Field(default=None, description=f"Subset of {list(CAPABILITIES)}.")


class ProfileRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """List the analytical capabilities this engine can run."""

    return {"capabilities": list(CAPABILITIES)}


@router.post("/profile")
def profile(request: ProfileRequest) -> dict[str, Any]:
    """Quick data-profiling report (Capability 1) from JSON rows."""

    dataset = _frame_from_rows(request.rows)
    try:
        return DataProfiler().profile(dataset).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Run the full EDA suite from JSON rows and return the structured report."""

    dataset = _frame_from_rows(request.rows)
    report = _run(dataset, request)
    return report.to_dict()


@router.post("/upload-analyze")
async def upload_analyze(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
    timestamp_column: str | None = Form(default=None),
    user_id_column: str | None = Form(default=None),
    event_column: str | None = Form(default=None),
    segment_columns: str | None = Form(default=None, description="Comma-separated column names."),
    funnel_steps: str | None = Form(default=None, description="Comma-separated step names."),
    target_event: str | None = Form(default=None),
    freq: str = Form(default="day"),
    retention_unit: str = Form(default="day"),
    report_format: ReportFormat = Query(default="json"),
) -> Any:
    """Run EDA on an uploaded CSV/JSON/Parquet/Excel file.

    Set ``report_format=html`` to receive the interactive dashboard instead of JSON.
    """

    dataset = await _read_upload(file)
    request = AnalyzeRequest(
        rows=[{}],  # placeholder; we pass the dataframe directly below
        target_column=target_column,
        timestamp_column=timestamp_column,
        user_id_column=user_id_column,
        event_column=event_column,
        segment_columns=_split(segment_columns),
        funnel_steps=_split(funnel_steps),
        target_event=target_event,
        freq=freq,
        retention_unit=retention_unit if retention_unit in ("day", "week") else "day",
    )
    report = _run(dataset, request)
    if report_format == "html":
        return HTMLResponse(content=report_to_html(report))
    return report.to_dict()


# --------------------------------------------------------------------- helpers
def report_to_html(report: Any) -> str:
    from engines.analytics.report import render_html

    return render_html(
        title="IntelliFlow EDA Report",
        summary=report.summary(),
        insights=[i.to_dict() for i in report.insights],
        charts=report.charts,
    )


def _run(dataset: pd.DataFrame, request: AnalyzeRequest) -> Any:
    try:
        return run_eda(
            dataset,
            target_column=request.target_column,
            timestamp_column=request.timestamp_column,
            user_id_column=request.user_id_column,
            event_column=request.event_column,
            segment_columns=request.segment_columns,
            funnel_steps=request.funnel_steps,
            target_event=request.target_event,
            freq=request.freq,
            retention_unit=request.retention_unit,
            capabilities=request.capabilities,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


def _frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    dataset = pd.DataFrame(rows)
    if dataset.empty or dataset.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No data: 'rows' produced an empty dataset.")
    return dataset


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        if filename.endswith(".csv") or filename.endswith(".tsv"):
            sep = "\t" if filename.endswith(".tsv") else ","
            return pd.read_csv(io.BytesIO(raw), sep=sep)
        if filename.endswith(".json"):
            try:
                return pd.read_json(io.BytesIO(raw))
            except ValueError:
                return pd.json_normalize(json.loads(raw.decode("utf-8")))
        if filename.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(raw))
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(raw))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse uploaded file: {exc}") from exc
    raise HTTPException(status_code=400, detail="Unsupported file type. Upload .csv, .json, .parquet, or .xlsx.")


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None
