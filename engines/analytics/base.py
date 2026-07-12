"""Core types and helpers shared across IntelliFlow Engine 2 (Analytics & EDA).

Every analytical capability in this engine speaks the same structured language so
that downstream consumers -- the JSON report, the HTML dashboard, the API gateway,
or a notebook -- can treat all discoveries uniformly. The shape mirrors how
analytics teams at large product companies communicate a finding:

- ``insight``        a plain-English discovery
- ``metric``         the quantified evidence behind the discovery
- ``visualization``  a serialisable chart spec, rendered lazily by ``visualization.py``
- ``action``         the recommended next step
- ``confidence``     a statistical-significance / data-quality score in ``[0, 1]``

Keeping the analysis layer free of any plotting library (charts are described as
plain data via :class:`ChartSpec`) means a report is always JSON-serialisable and
the heavy ``plotly`` import only happens when a human actually asks for a picture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

# Ordered from least to most urgent. Reports surface ``critical`` first.
Severity = Literal["ok", "info", "warning", "critical"]

_SEVERITY_RANK: dict[str, int] = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


class ChartKind(str, Enum):
    """Supported, renderer-agnostic chart kinds.

    A capability emits one of these together with a small ``data`` payload;
    :mod:`engines.analytics.visualization` knows how to turn each kind into a
    Plotly figure or a Matplotlib PNG.
    """

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    FUNNEL = "funnel"
    TABLE = "table"
    DENDROGRAM = "dendrogram"


@dataclass
class ChartSpec:
    """A renderer-agnostic description of a single visualization.

    Parameters
    ----------
    kind:
        One of :class:`ChartKind` (or its string value).
    title:
        Human-readable chart title.
    data:
        Kind-specific, JSON-safe payload (e.g. ``{"x": [...], "y": [...]}``).
    layout:
        Optional layout hints such as axis titles or orientation.
    """

    kind: ChartKind | str
    title: str
    data: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = ChartKind(self.kind) if not isinstance(self.kind, ChartKind) else self.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "data": json_safe(self.data),
            "layout": json_safe(self.layout),
        }


@dataclass
class Insight:
    """A single structured finding in the platform's house style.

    This is the atomic unit of every analysis. Aggregating insights from all
    capabilities yields the narrative of a report.
    """

    title: str
    insight: str
    action: str
    metric: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    severity: Severity = "info"
    category: str = "general"
    visualization: ChartSpec | None = None

    def __post_init__(self) -> None:
        self.confidence = _clamp_unit(self.confidence)
        if self.severity not in _SEVERITY_RANK:
            self.severity = "info"

    @property
    def severity_rank(self) -> int:
        return _SEVERITY_RANK[self.severity]

    @property
    def confidence_label(self) -> str:
        return confidence_label(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "insight": self.insight,
            "metric": json_safe(self.metric),
            "action": self.action,
            "confidence": round(float(self.confidence), 4),
            "confidence_label": self.confidence_label,
            "severity": self.severity,
            "category": self.category,
            "visualization": self.visualization.to_dict() if self.visualization else None,
        }


def confidence_label(score: float) -> str:
    """Map a ``[0, 1]`` confidence score to a coarse human label."""

    score = _clamp_unit(score)
    if score >= 0.85:
        return "very high"
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "moderate"
    if score >= 0.25:
        return "low"
    return "very low"


def grade_from_score(score: float) -> str:
    """Translate a 0-100 health score into a familiar letter grade."""

    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def sort_insights(insights: list[Insight]) -> list[Insight]:
    """Order insights most-urgent first, then by descending confidence.

    A stable sort keeps the original emission order for ties, which keeps the
    narrative readable (e.g. profiling findings stay grouped together).
    """

    return sorted(insights, key=lambda i: (-i.severity_rank, -i.confidence))


def _clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return max(0.0, min(1.0, number))


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce to a finite float, mapping ``NaN``/``inf``/errors to ``default``.

    JSON has no representation for ``NaN`` or ``Infinity``; routing every numeric
    metric through this keeps reports valid and comparisons well-defined.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas/dataclass objects into JSON-native types.

    Handles the awkward cases that ``json.dumps`` chokes on: numpy scalars and
    arrays, pandas ``Series``/``DataFrame``/``Timestamp``/``Timedelta``, ``NaN``
    and ``Infinity`` (mapped to ``None``), enums, dataclasses, and arbitrary
    objects exposing ``to_dict``.
    """

    # Order matters: bool is a subclass of int, NaN handling before generic float.
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return safe_float(obj)
    if isinstance(obj, (str, bytes)):
        return obj.decode() if isinstance(obj, bytes) else obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.timedelta64):
        return pd.Timedelta(obj).total_seconds()
    if isinstance(obj, np.ndarray):
        return [json_safe(item) for item in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {json_safe(k): json_safe(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return [json_safe(record) for record in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Index):
        return [json_safe(item) for item in obj.tolist()]
    if isinstance(obj, dict):
        return {json_safe(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(item) for item in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return json_safe(obj.to_dict())
    if is_dataclass(obj) and not isinstance(obj, type):
        from dataclasses import asdict

        return json_safe(asdict(obj))
    return str(obj)
