"""Orchestrator for IntelliFlow Engine 2 -- the single entry point for EDA.

:func:`run_eda` inspects which columns the caller supplied and runs exactly the
capabilities those inputs unlock (profiling and recommendations always; funnel
only with steps; retention only with a user id + timestamp; and so on). It
returns an :class:`EDAReport` that bundles every section, a ranked list of
insights, the rolled-up data-quality score, and one-call exporters to JSON, an
interactive HTML dashboard, CSV summaries, and static images.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anomaly import AnomalyDetector, AnomalyResult
from .base import ChartSpec, Insight, grade_from_score, json_safe, sort_insights
from .correlation import CorrelationAnalyzer, CorrelationResult
from .events import EventStreamAnalyzer, EventStreamResult
from .funnel import FunnelAnalyzer, FunnelResult
from .profiling import DataProfiler, ProfileResult
from .recommendations import FeatureAdvisor, RecommendationResult
from .retention import RetentionAnalyzer, RetentionResult
from .utils import infer_semantic_types

CAPABILITIES = ("profiling", "correlation", "recommendations", "anomaly", "funnel", "retention", "events")


@dataclass
class EDAReport:
    """Aggregated result of an EDA run with one-call exporters."""

    n_rows: int
    n_columns: int
    generated_at: str
    inputs: dict[str, Any]
    data_quality_score: float | None
    data_quality_grade: str | None
    results: dict[str, Any] = field(default_factory=dict)  # raw result objects
    insights: list[Insight] = field(default_factory=list)
    charts: list[ChartSpec] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- accessors
    @property
    def sections(self) -> dict[str, Any]:
        return {name: result.to_dict() for name, result in self.results.items()}

    def summary(self) -> dict[str, Any]:
        critical = sum(1 for i in self.insights if i.severity == "critical")
        warnings = sum(1 for i in self.insights if i.severity == "warning")
        highlights = {
            "Insights": len(self.insights),
            "Critical": critical,
            "Warnings": warnings,
            "Capabilities run": len(self.results),
        }
        return {
            "data_quality_score": self.data_quality_score,
            "data_quality_grade": self.data_quality_grade,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "generated_at": self.generated_at,
            "highlights": highlights,
        }

    def to_dict(self) -> dict[str, Any]:
        # json_safe is the final guarantee: regardless of dtypes flowing through
        # any capability, the aggregated report is always JSON-serialisable.
        return json_safe(
            {
                "metadata": {
                    "n_rows": self.n_rows,
                    "n_columns": self.n_columns,
                    "generated_at": self.generated_at,
                    "inputs": self.inputs,
                    "capabilities_run": list(self.results.keys()),
                    "errors": self.errors,
                },
                "data_quality_score": self.data_quality_score,
                "data_quality_grade": self.data_quality_grade,
                "insights": [i.to_dict() for i in self.insights],
                "sections": self.sections,
            }
        )

    # ------------------------------------------------------------- exporters
    def to_json(self, path: str | Path, indent: int = 2) -> str:
        from .report import write_json

        return write_json(self.to_dict(), path, indent=indent)

    def to_html(self, path: str | Path, title: str = "IntelliFlow EDA Report") -> str:
        from .report import write_html

        return write_html(
            path,
            title=title,
            summary=self.summary(),
            insights=[i.to_dict() for i in self.insights],
            charts=self.charts,
        )

    def save_csv_summaries(self, out_dir: str | Path) -> list[str]:
        from .report import write_csv_summaries

        return write_csv_summaries(self.sections, out_dir)

    def save_charts(self, out_dir: str | Path, fmt: str = "png") -> list[str]:
        from .report import export_charts

        return export_charts(self.charts, out_dir, fmt=fmt)


def run_eda(
    dataset: pd.DataFrame,
    *,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    user_id_column: str | None = None,
    event_column: str | None = None,
    segment_columns: list[str] | None = None,
    funnel_steps: list[str] | None = None,
    target_event: str | None = None,
    freq: str = "day",
    retention_unit: str = "day",
    capabilities: list[str] | None = None,
) -> EDAReport:
    """Run the FAANG-style EDA suite over ``dataset``.

    Parameters
    ----------
    dataset:
        The data to analyse. Must be a non-empty :class:`pandas.DataFrame`.
    target_column:
        Optional supervised target -- unlocks correlation ranking, leakage
        detection and time-series anomaly metric selection.
    timestamp_column, user_id_column, event_column:
        Enable the event-centric capabilities (funnel, retention, event stream).
    segment_columns:
        Optional grouping columns; the first is used for funnel/retention segments.
    funnel_steps:
        Ordered step names (event values, or indicator columns) for funnel analysis.
    target_event:
        Event value used for Kaplan-Meier time-to-event analysis.
    capabilities:
        Explicit subset of :data:`CAPABILITIES` to run; defaults to every
        capability the provided columns support.

    Returns
    -------
    EDAReport
        Aggregated insights, per-capability sections, quality score and exporters.
    """

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("run_eda expects a pandas DataFrame. Load CSV/JSON/Parquet with the loaders in api or pandas.")
    if dataset.empty or dataset.shape[1] == 0:
        raise ValueError("Dataset is empty. Provide at least one row and one column.")
    _validate_columns(dataset, target_column, timestamp_column, user_id_column, event_column, segment_columns)

    requested = set(capabilities or CAPABILITIES)
    segment_col = segment_columns[0] if segment_columns else None

    # Identifier-style columns (user id, event name) are not predictive features:
    # exclude them from correlation / recommendations / anomaly so they never get
    # treated as measures, while the event-centric capabilities still use them.
    id_like = [c for c in (user_id_column, event_column) if c and c in dataset.columns]
    feature_df = dataset.drop(columns=id_like) if id_like else dataset
    feature_types = infer_semantic_types(feature_df)

    report = EDAReport(
        n_rows=int(dataset.shape[0]),
        n_columns=int(dataset.shape[1]),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        inputs={
            "target_column": target_column,
            "timestamp_column": timestamp_column,
            "user_id_column": user_id_column,
            "event_column": event_column,
            "segment_columns": segment_columns,
            "funnel_steps": funnel_steps,
        },
        data_quality_score=None,
        data_quality_grade=None,
    )

    # 1. Profiling (always) -- also yields the headline quality score.
    if "profiling" in requested:
        profile = _safe_run(report, "profiling", lambda: DataProfiler().profile(dataset))
        if isinstance(profile, ProfileResult):
            report.data_quality_score = profile.quality_score
            report.data_quality_grade = profile.quality_grade

    # 2. Correlation & feature intelligence (always).
    if "correlation" in requested:
        _safe_run(report, "correlation", lambda: CorrelationAnalyzer().analyze(feature_df, target_column=target_column))

    # 3. Feature warnings & recommendations (always).
    if "recommendations" in requested:
        _safe_run(report, "recommendations", lambda: FeatureAdvisor().analyze(feature_df, target_column=target_column))

    # 4. Anomaly detection (always univariate/multivariate; time-series if possible).
    if "anomaly" in requested:
        value_col = _pick_metric_column(feature_df, feature_types, target_column)
        anomaly_numeric = [c for c in feature_types.numeric if c in dataset.columns]
        _safe_run(
            report,
            "anomaly",
            lambda: AnomalyDetector().analyze(
                dataset,
                timestamp_col=timestamp_column if value_col else None,
                value_col=value_col,
                columns=anomaly_numeric or None,
                freq=freq,
            ),
        )

    # 5. Funnel (needs explicit steps).
    if "funnel" in requested and funnel_steps:
        _safe_run(
            report,
            "funnel",
            lambda: FunnelAnalyzer().analyze(
                dataset,
                funnel_steps,
                user_col=user_id_column,
                event_col=event_column,
                timestamp_col=timestamp_column,
                segment_col=segment_col,
            ),
        )

    # 6. Retention (needs user id + timestamp).
    if "retention" in requested and user_id_column and timestamp_column:
        _safe_run(
            report,
            "retention",
            lambda: RetentionAnalyzer().analyze(
                dataset,
                user_col=user_id_column,
                timestamp_col=timestamp_column,
                retention_unit=retention_unit,
                segment_col=segment_col,
            ),
        )

    # 7. Event stream (needs timestamp).
    if "events" in requested and timestamp_column:
        _safe_run(
            report,
            "events",
            lambda: EventStreamAnalyzer().analyze(
                dataset,
                timestamp_col=timestamp_column,
                user_col=user_id_column,
                event_col=event_column,
                freq=freq,
                target_event=target_event,
            ),
        )

    _aggregate(report)
    return report


# --------------------------------------------------------------------- helpers
def _safe_run(report: EDAReport, name: str, fn: Any) -> Any:
    """Run one capability, capturing failures so one bad section never aborts EDA."""

    try:
        result = fn()
        report.results[name] = result
        return result
    except Exception as exc:  # capability isolation
        report.errors[name] = f"{type(exc).__name__}: {exc}"
        return None


def _aggregate(report: EDAReport) -> None:
    insights: list[Insight] = []
    charts: list[ChartSpec] = []
    seen_titles: set[str] = set()

    for result in report.results.values():
        insights.extend(getattr(result, "insights", []) or [])
        for attr in ("visualization", "heatmap", "dendrogram", "curve_chart", "timeseries_chart", "survival_chart"):
            spec = getattr(result, attr, None)
            if isinstance(spec, ChartSpec) and spec.title not in seen_titles:
                charts.append(spec)
                seen_titles.add(spec.title)

    report.insights = sort_insights(insights)
    report.charts = charts


def _pick_metric_column(dataset: pd.DataFrame, types: Any, target_column: str | None) -> str | None:
    """Choose a numeric column to track for time-series anomalies."""

    if target_column and target_column in dataset.columns and pd.api.types.is_numeric_dtype(dataset[target_column]):
        return target_column
    candidates = [c for c in types.numeric if dataset[c].nunique(dropna=True) > 1]
    if not candidates:
        return None
    variances = {c: float(dataset[c].std(ddof=0) or 0.0) for c in candidates}
    return max(variances, key=variances.get)


def _validate_columns(
    dataset: pd.DataFrame,
    target_column: str | None,
    timestamp_column: str | None,
    user_id_column: str | None,
    event_column: str | None,
    segment_columns: list[str] | None,
) -> None:
    named = {
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "user_id_column": user_id_column,
        "event_column": event_column,
    }
    for label, col in named.items():
        if col is not None and col not in dataset.columns:
            raise ValueError(f"{label}={col!r} is not a column in the dataset. Available: {list(dataset.columns)[:20]}")
    for col in segment_columns or []:
        if col not in dataset.columns:
            raise ValueError(f"segment column {col!r} is not in the dataset.")
