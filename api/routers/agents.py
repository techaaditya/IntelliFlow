"""FastAPI routes for IntelliFlow Engine 3 (Agent Orchestration).

Mirrors the Engine 1/2 router conventions: JSON-row and file-upload entry
points, Pydantic request models, and ``HTTPException`` errors with recovery
hints. Engine 3 accepts a natural-language ``query`` plus the dataset and runs
the multi-agent crew, which can call Engine 1 (AutoML) and Engine 2 (Analytics)
as tools.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from engines.agents import AgentMemory, get_agent_config, run_agent_query
from engines.agents.llm import LLMError
from engines.agents.roles import AGENTS

router = APIRouter(prefix="/agents", tags=["Agents"])


class QueryRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, description="Dataset as a list of record dicts.")
    query: str = Field(..., min_length=1, description="Natural-language question for the agent crew.")
    session_id: str | None = Field(default=None, description="Groups queries for long-term memory.")
    max_steps: int | None = Field(default=None, ge=1, le=20, description="Cap on tool-calling iterations.")


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """List the crew's agents, their tools, and the LLM configuration.

    Runs without contacting the LLM, so it doubles as a health/inspection route.
    """

    config = get_agent_config()
    return {
        "agents": [agent.to_dict() for agent in AGENTS],
        "llm": config.to_dict(),
        "cross_engine": ["automl_tool -> Engine 1", "analytics_tool -> Engine 2", "profiler_tool -> Engine 2"],
    }


@router.get("/history")
def history(session_id: str = Query(..., min_length=1), limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    """Return previous agent queries and answers for a session."""

    try:
        records = AgentMemory().history(session_id, limit=limit)
    except Exception as exc:  # pragma: no cover - unexpected storage error
        raise HTTPException(status_code=500, detail=f"Could not read history: {exc}") from exc
    return {"session_id": session_id, "history": [r.to_dict() for r in records]}


@router.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Run the agent crew on JSON rows and return the structured result."""

    dataset = _frame_from_rows(request.rows)
    return _run(dataset, request.query, request.session_id, request.max_steps)


@router.post("/upload-query")
async def upload_query(
    file: UploadFile = File(...),
    query: str = Form(...),
    session_id: str | None = Form(default=None),
    max_steps: int | None = Form(default=None),
) -> dict[str, Any]:
    """Run the agent crew on an uploaded CSV/JSON/Parquet/Excel file."""

    dataset = await _read_upload(file)
    return _run(dataset, query, session_id, max_steps)


# --------------------------------------------------------------------- helpers
def _run(dataset: pd.DataFrame, query_text: str, session_id: str | None, max_steps: int | None) -> dict[str, Any]:
    try:
        result = run_agent_query(dataset=dataset, query=query_text, session_id=session_id, max_steps=max_steps)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM backbone error: {exc}") from exc
    except Exception as exc:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc
    return result.to_dict()


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
