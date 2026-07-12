"""Tools the agent crew can call during a query (Engine 3).

Tools are the crew's hands: they run real pandas operations, profile the data,
trigger Engine 1 (AutoML) and Engine 2 (Analytics) in-process, compose charts,
and (optionally) search the web. Each tool returns a compact **string
observation** for the LLM and may append structured artifacts (``ChartSpec`` /
``Insight`` objects, a model endpoint) onto the shared :class:`AgentContext`.

Design rules that match the rest of the platform:

- Reuse Engine 2's shared output contract (``ChartSpec``/``Insight``/``json_safe``)
  so the UI/report layers render agent charts with the existing ``to_plotly``.
- Never ``eval``/``exec`` model-provided text. ``pandas_tool`` exposes a fixed,
  whitelisted set of operations instead.
- Heavy/optional engines are imported lazily inside each tool, so importing this
  module never drags in mlflow/optuna/plotly or a web-search package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from engines.analytics.base import ChartKind, ChartSpec, Insight, json_safe, safe_float

from .config import AgentConfig, get_agent_config

# Keep AutoML runs snappy inside an interactive agent loop.
DEFAULT_AUTOML_TRIALS = 8
MAX_AUTOML_TRIALS = 40
MAX_TABLE_ROWS = 20


@dataclass
class AgentContext:
    """Mutable state shared across a single crew run."""

    dataset: pd.DataFrame
    config: AgentConfig = field(default_factory=get_agent_config)
    llm: Any = None  # OllamaChat, set by the crew (used by tools that need the model)
    charts: list[ChartSpec] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    model_endpoint: str | None = None
    model_uri: str | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.dataset.columns]

    def add_chart(self, chart: ChartSpec) -> None:
        self.charts.append(chart)

    def add_insight(self, insight: Insight) -> None:
        self.insights.append(insight)


@dataclass
class Tool:
    """An LLM-callable tool: a name, a description, a JSON-schema, and a fn."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------- helpers
def _obj_param(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


class ToolUsageHint(Exception):
    """A recoverable 'you called me wrong' signal — surfaced to the LLM, not logged as an error."""


def _require_column(context: AgentContext, column: str) -> str:
    if not column:
        raise ToolUsageHint(f"Provide a `column`. Available columns: {context.columns}")
    if column not in context.dataset.columns:
        raise ToolUsageHint(f"Column {column!r} not found. Available columns: {context.columns}")
    return column


def _truncate(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit] + " …(truncated)"


# ------------------------------------------------------------------ pandas_tool
_PANDAS_OPS = (
    "describe", "head", "shape", "columns", "value_counts", "unique_count",
    "missing_summary", "column_stats", "groupby_agg", "correlation", "filter_count",
)


def pandas_tool(
    context: AgentContext,
    operation: str,
    column: str | None = None,
    columns: list[str] | None = None,
    by: str | None = None,
    agg: str = "mean",
    query: str | None = None,
    top_n: int = 10,
) -> str:
    """Run one whitelisted pandas operation on the dataset and return the result."""

    df = context.dataset
    operation = (operation or "").strip().lower()
    if operation not in _PANDAS_OPS:
        return f"Unknown operation {operation!r}. Choose one of: {list(_PANDAS_OPS)}"

    if operation == "shape":
        return f"rows={df.shape[0]}, columns={df.shape[1]}"
    if operation == "columns":
        return json_str({c: str(df[c].dtype) for c in df.columns})
    if operation == "head":
        return json_str(json_safe(df.head(min(int(top_n or 5), MAX_TABLE_ROWS)).to_dict(orient="records")))
    if operation == "describe":
        target = df[columns] if columns else df
        return json_str(json_safe(target.describe(include="all").to_dict()))
    if operation == "missing_summary":
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        return json_str({str(k): int(v) for k, v in miss.items()}) if not miss.empty else "No missing values."
    if operation == "value_counts":
        _require_column(context, column or "")
        counts = df[column].value_counts(dropna=False).head(int(top_n or 10))
        return json_str({str(k): int(v) for k, v in counts.items()})
    if operation == "unique_count":
        _require_column(context, column or "")
        return f"{column}: {int(df[column].nunique(dropna=True))} unique values"
    if operation == "column_stats":
        _require_column(context, column or "")
        series = df[column]
        stats: dict[str, Any] = {
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            stats.update(
                mean=safe_float(series.mean()), std=safe_float(series.std()),
                min=safe_float(series.min()), max=safe_float(series.max()),
                median=safe_float(series.median()),
            )
        else:
            top = series.value_counts().head(5)
            stats["top_values"] = {str(k): int(v) for k, v in top.items()}
        return json_str(stats)
    if operation == "groupby_agg":
        _require_column(context, by or "")
        _require_column(context, column or "")
        agg = agg if agg in ("mean", "sum", "count", "min", "max", "median", "std") else "mean"
        grouped = df.groupby(by)[column].agg(agg).sort_values(ascending=False).head(int(top_n or 10))
        return json_str({str(k): safe_float(v) for k, v in grouped.items()})
    if operation == "correlation":
        numeric = df.select_dtypes("number")
        if numeric.shape[1] < 2:
            return "Not enough numeric columns for a correlation matrix."
        cols = [c for c in (columns or list(numeric.columns)) if c in numeric.columns][:8] or list(numeric.columns)[:8]
        corr = numeric[cols].corr().round(3)
        return json_str(json_safe(corr.to_dict()))
    if operation == "filter_count":
        if not query:
            return "Provide a pandas `query` expression, e.g. \"age > 30 and churn == 'Yes'\"."
        try:
            n = int(df.query(query).shape[0])
        except Exception as exc:  # invalid/unsafe expression → report, don't crash
            return f"Could not evaluate filter {query!r}: {exc}"
        return f"{n} of {len(df)} rows match `{query}`"
    return f"Operation {operation!r} not implemented."


# ----------------------------------------------------------------- profiler_tool
def profiler_tool(context: AgentContext) -> str:
    """Run Engine 2's data profiler and return a quality summary."""

    from engines.analytics.profiling import DataProfiler

    result = DataProfiler().profile(context.dataset)
    data = result.to_dict()
    summary = {
        "quality_score": data.get("quality_score"),
        "quality_grade": data.get("quality_grade"),
        "n_rows": data.get("n_rows"),
        "n_columns": data.get("n_columns"),
        "missing_pct": data.get("missing_pct"),
        "duplicate_pct": data.get("duplicate_pct"),
        "semantic_types": data.get("semantic_types"),
    }
    return _truncate(json_str(json_safe(summary)))


# ---------------------------------------------------------------- analytics_tool
def analytics_tool(
    context: AgentContext,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    capabilities: list[str] | None = None,
) -> str:
    """Run Engine 2's EDA suite; harvest its insights and charts into the context."""

    from engines.analytics import run_eda

    report = run_eda(
        context.dataset,
        target_column=target_column,
        timestamp_column=timestamp_column,
        capabilities=capabilities,
    )
    for insight in report.insights[:8]:
        context.add_insight(insight)
        if insight.visualization is not None:
            context.add_chart(insight.visualization)
    for chart in report.charts[:6]:
        context.add_chart(chart)
    summary = {
        "data_quality_score": report.data_quality_score,
        "data_quality_grade": report.data_quality_grade,
        "n_insights": len(report.insights),
        "top_insights": [
            {"title": i.title, "insight": i.insight, "severity": i.severity} for i in report.insights[:5]
        ],
        "charts_added": len(report.charts),
    }
    return _truncate(json_str(json_safe(summary)))


# ------------------------------------------------------------------ plotly_tool
def plotly_tool(
    context: AgentContext,
    kind: str,
    title: str,
    x: str | None = None,
    y: str | None = None,
    top_n: int = 15,
) -> str:
    """Compose a chart from dataset columns and add it to the result."""

    df = context.dataset
    kind = (kind or "bar").strip().lower()
    valid = {k.value for k in ChartKind}
    if kind not in valid:
        kind = "bar"

    data: dict[str, Any] = {}
    try:
        if kind == "histogram" and x:
            _require_column(context, x)
            data = {"values": json_safe(df[x].dropna().tolist()[:2000]), "x_title": x}
        elif kind in ("bar", "line") and x and y:
            _require_column(context, x)
            _require_column(context, y)
            grouped = df.groupby(x)[y].mean().sort_values(ascending=False).head(int(top_n or 15))
            data = {"x": [str(i) for i in grouped.index], "y": [safe_float(v) for v in grouped.values],
                    "x_title": x, "y_title": y}
        elif kind == "bar" and x:
            _require_column(context, x)
            counts = df[x].value_counts().head(int(top_n or 15))
            data = {"x": [str(i) for i in counts.index], "y": [int(v) for v in counts.values],
                    "x_title": x, "y_title": "count"}
        elif kind == "scatter" and x and y:
            _require_column(context, x)
            _require_column(context, y)
            sample = df[[x, y]].dropna().head(2000)
            data = {"x": json_safe(sample[x].tolist()), "y": json_safe(sample[y].tolist()),
                    "x_title": x, "y_title": y}
        else:
            return "Provide valid `kind` plus the columns it needs (e.g. bar/line need x[,y]; histogram needs x)."
    except (ValueError, ToolUsageHint) as exc:
        return str(exc)

    spec = ChartSpec(kind=kind, title=title or f"{kind} chart", data=data)
    context.add_chart(spec)
    return f"Added a {kind} chart titled {spec.title!r}."


# ------------------------------------------------------------------ automl_tool
def automl_tool(
    context: AgentContext,
    target_column: str,
    metric: str | None = None,
    task_type: str = "auto",
    n_trials: int = DEFAULT_AUTOML_TRIALS,
) -> str:
    """Cross-engine flagship: run Engine 1 (AutoML) on the dataset and register a model."""

    from engines.automl.pipeline import detect_task_type, run_automl

    _require_column(context, target_column)
    dataset = context.dataset.dropna(subset=[target_column]).copy()
    if dataset.empty:
        return f"No rows remain after dropping missing values in target {target_column!r}."

    resolved_task = detect_task_type(dataset[target_column]) if task_type in ("auto", None, "") else task_type
    resolved_metric = metric or ("accuracy" if resolved_task == "classification" else "r2")
    trials = max(1, min(int(n_trials or DEFAULT_AUTOML_TRIALS), MAX_AUTOML_TRIALS))

    # XGBoost multiclass requires integer-encoded labels; encode string
    # classification targets so every model family (RF/XGBoost/LightGBM) trains.
    class_mapping: dict[int, str] | None = None
    if resolved_task == "classification" and not pd.api.types.is_numeric_dtype(dataset[target_column]):
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        dataset[target_column] = encoder.fit_transform(dataset[target_column].astype(str))
        class_mapping = {int(i): str(cls) for i, cls in enumerate(encoder.classes_)}

    result = run_automl(
        dataset=dataset,
        target_column=target_column,
        metric=resolved_metric,
        n_trials=trials,
        task_type=resolved_task,
    )
    context.model_endpoint = result.get("endpoint_url")
    context.model_uri = result.get("model_uri")
    summary = {
        "task_type": result.get("task_type"),
        "metric": result.get("metric"),
        "best_score": safe_float(result.get("best_score")),
        "best_model_family": result.get("best_model_family"),
        "model_uri": result.get("model_uri"),
        "endpoint_url": result.get("endpoint_url"),
        "n_rows": result.get("n_rows"),
        "test_metrics": json_safe(result.get("test_metrics", {})),
    }
    if class_mapping is not None:
        summary["class_mapping"] = class_mapping  # endpoint predicts these integer codes
    context.add_insight(
        Insight(
            title=f"Model trained for {target_column}",
            insight=(
                f"AutoML selected {result.get('best_model_family')} with "
                f"{resolved_metric}={safe_float(result.get('best_score'))}."
            ),
            action=f"Serve predictions at {result.get('endpoint_url')}.",
            metric=summary,
            severity="info",
            category="automl",
            confidence=0.9,
        )
    )
    return _truncate(json_str(summary))


# ------------------------------------------------------------------ mlflow_tool
def mlflow_tool(context: AgentContext) -> str:
    """Report the latest registered AutoML model from the MLflow registry."""

    from engines.automl.registry import AutoMLRegistry

    try:
        info = AutoMLRegistry().get_model_info()
    except Exception as exc:  # no model registered yet / mlflow issue
        return f"No registered model available yet: {exc}. Use automl_tool to train one."
    return json_str({
        "model_name": info.model_name,
        "model_uri": info.model_uri,
        "latest_version": info.latest_version,
        "alias": info.alias,
    })


# --------------------------------------------------------------- web_search_tool
def web_search_tool(context: AgentContext, query: str, max_results: int = 4) -> str:
    """Search the web for external context; degrades gracefully when unavailable."""

    query = (query or "").strip()
    if not query:
        return "Provide a non-empty search query."
    try:
        from ddgs import DDGS  # optional dependency
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return (
                "Web search is unavailable (install `ddgs` to enable it). "
                "Answer from the model's own knowledge and say so."
            )
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max(1, min(int(max_results or 4), 8))))
    except Exception as exc:  # network/rate-limit → graceful note
        return f"Web search failed ({exc}). Answer from the model's own knowledge and say so."
    if not hits:
        return "No web results found."
    lines = [f"- {h.get('title', '')}: {h.get('body', '')[:200]} ({h.get('href', '')})" for h in hits]
    return _truncate("\n".join(lines))


# --------------------------------------------------------------------- registry
def json_str(obj: Any) -> str:
    import json

    return json.dumps(json_safe(obj), ensure_ascii=False, default=str)


def build_tool_registry(context: AgentContext) -> dict[str, Tool]:
    """Build the callable tool registry bound to a specific run context."""

    columns_hint = f"Dataset columns: {context.columns}."

    def bind(fn: Callable[..., str]) -> Callable[..., str]:
        def wrapped(**kwargs: Any) -> str:
            return fn(context, **kwargs)

        return wrapped

    tools: list[Tool] = [
        Tool(
            name="pandas_tool",
            description=(
                "Run one whitelisted pandas operation on the loaded dataset and return the result. "
                f"{columns_hint} Operations: {list(_PANDAS_OPS)}."
            ),
            parameters=_obj_param(
                {
                    "operation": {"type": "string", "enum": list(_PANDAS_OPS)},
                    "column": {"type": "string", "description": "Target column for column-scoped ops."},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "by": {"type": "string", "description": "Group-by column for groupby_agg."},
                    "agg": {"type": "string", "enum": ["mean", "sum", "count", "min", "max", "median", "std"]},
                    "query": {"type": "string", "description": "pandas query() expression for filter_count."},
                    "top_n": {"type": "integer"},
                },
                required=["operation"],
            ),
            fn=bind(pandas_tool),
        ),
        Tool(
            name="profiler_tool",
            description="Profile the dataset (Engine 2) and return a 0-100 quality score and column overview.",
            parameters=_obj_param({}),
            fn=bind(profiler_tool),
        ),
        Tool(
            name="analytics_tool",
            description=(
                "Run Engine 2's full EDA suite (profiling, correlation, anomaly, recommendations, and "
                "product-analytics when applicable). Adds its insights and charts to the final answer."
            ),
            parameters=_obj_param(
                {
                    "target_column": {"type": "string"},
                    "timestamp_column": {"type": "string"},
                    "capabilities": {"type": "array", "items": {"type": "string"}},
                }
            ),
            fn=bind(analytics_tool),
        ),
        Tool(
            name="plotly_tool",
            description="Compose a chart from dataset columns and attach it to the answer.",
            parameters=_obj_param(
                {
                    "kind": {"type": "string", "enum": sorted(k.value for k in ChartKind)},
                    "title": {"type": "string"},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "top_n": {"type": "integer"},
                },
                required=["kind", "title"],
            ),
            fn=bind(plotly_tool),
        ),
        Tool(
            name="automl_tool",
            description=(
                "Train Engine 1 (AutoML) on the dataset for a target column: preprocessing, Optuna HPO across "
                "RandomForest/XGBoost/LightGBM, MLflow tracking, and model registration. Returns the best score "
                "and a live prediction endpoint. Use this whenever the user wants a prediction/model."
            ),
            parameters=_obj_param(
                {
                    "target_column": {"type": "string"},
                    "metric": {"type": "string"},
                    "task_type": {"type": "string", "enum": ["auto", "classification", "regression"]},
                    "n_trials": {"type": "integer"},
                },
                required=["target_column"],
            ),
            fn=bind(automl_tool),
        ),
        Tool(
            name="mlflow_tool",
            description="Look up the latest AutoML model registered in the MLflow registry.",
            parameters=_obj_param({}),
            fn=bind(mlflow_tool),
        ),
        Tool(
            name="web_search_tool",
            description="Search the web for external context, domain knowledge, or benchmarks.",
            parameters=_obj_param(
                {"query": {"type": "string"}, "max_results": {"type": "integer"}}, required=["query"]
            ),
            fn=bind(web_search_tool),
        ),
    ]
    return {tool.name: tool for tool in tools}


def tool_specs(registry: dict[str, Tool], names: list[str] | None = None) -> list[dict[str, Any]]:
    """OpenAI-style tool specs for the given tool names (all by default)."""

    selected = registry.values() if names is None else (registry[n] for n in names if n in registry)
    return [tool.openai_spec() for tool in selected]
