"""IntelliFlow Engine 2 -- Analytics & FAANG-style EDA.

The "understanding engine": it turns a raw dataset into ranked, actionable
insights with the rigour of product-analytics systems. The single high-level
entry point is :func:`run_eda`, which returns an :class:`EDAReport` exportable to
JSON, an interactive HTML dashboard, CSV summaries and static images.

Every capability is also usable standalone:

>>> from engines.analytics import profile_dataset, analyze_funnel, run_eda
>>> report = run_eda(df, target_column="revenue", timestamp_column="date",
...                  user_id_column="user_id", event_column="event")
>>> report.to_html("eda.html")
"""

from __future__ import annotations

from .anomaly import AnomalyDetector, AnomalyResult, detect_anomalies
from .base import ChartKind, ChartSpec, Insight, Severity, json_safe
from .correlation import (
    CorrelationAnalyzer,
    CorrelationResult,
    analyze_correlations,
    correlation_ratio,
    cramers_v,
)
from .events import EventStreamAnalyzer, EventStreamResult, analyze_events
from .funnel import FunnelAnalyzer, FunnelResult, analyze_funnel
from .pipeline import CAPABILITIES, EDAReport, run_eda
from .profiling import DataProfiler, ProfileResult, profile_dataset
from .recommendations import FeatureAdvisor, RecommendationResult, recommend_features
from .retention import RetentionAnalyzer, RetentionResult, analyze_retention
from .utils import SemanticTypes, infer_semantic_types, wilson_interval

__all__ = [
    # high-level
    "run_eda",
    "EDAReport",
    "CAPABILITIES",
    # core types
    "Insight",
    "ChartSpec",
    "ChartKind",
    "Severity",
    "json_safe",
    # capability 1
    "DataProfiler",
    "ProfileResult",
    "profile_dataset",
    # capability 2
    "CorrelationAnalyzer",
    "CorrelationResult",
    "analyze_correlations",
    "cramers_v",
    "correlation_ratio",
    # capability 3
    "FunnelAnalyzer",
    "FunnelResult",
    "analyze_funnel",
    # capability 4
    "RetentionAnalyzer",
    "RetentionResult",
    "analyze_retention",
    # capability 5
    "EventStreamAnalyzer",
    "EventStreamResult",
    "analyze_events",
    # capability 6
    "AnomalyDetector",
    "AnomalyResult",
    "detect_anomalies",
    # capability 7
    "FeatureAdvisor",
    "RecommendationResult",
    "recommend_features",
    # utilities
    "SemanticTypes",
    "infer_semantic_types",
    "wilson_interval",
]
