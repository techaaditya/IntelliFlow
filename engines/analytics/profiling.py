"""Capability 1 -- Data Profiling Intelligence.

Turns a raw DataFrame into an actionable health report: where data is missing and
whether it matters, how each numeric feature is distributed (skew, kurtosis,
modality, outliers), which categoricals are dangerously high-cardinality, whether
any column's stored dtype disagrees with its actual contents, and a single 0-100
quality score that rolls all of that up for a human or a downstream gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .base import ChartSpec, Insight, grade_from_score, json_safe, safe_float, sort_insights
from .utils import SemanticTypes, infer_semantic_types

# Tunable thresholds, named so the prose and the logic never drift apart.
MISSING_WARN = 0.20
MISSING_CRITICAL = 0.50
SKEW_THRESHOLD = 1.0
HIGH_KURTOSIS = 3.0
HIGH_CARDINALITY_RATIO = 0.5
HIGH_CARDINALITY_ABSOLUTE = 50
QUASI_CONSTANT_DOMINANCE = 0.99
OUTLIER_IQR_MULTIPLIER = 1.5
MAX_DETAILED_INSIGHTS = 8  # cap per-column noise; the rest is summarised


@dataclass
class ColumnProfile:
    """Per-column profile covering structure, completeness and distribution."""

    name: str
    dtype: str
    semantic_type: str
    count: int
    missing: int
    missing_pct: float
    n_unique: int
    unique_pct: float
    # Numeric-only fields (None for non-numeric columns).
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    median: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zeros: int | None = None
    negatives: int | None = None
    n_outliers: int | None = None
    outlier_pct: float | None = None
    modality: str | None = None
    # Categorical-only fields.
    top_value: Any | None = None
    top_freq: int | None = None
    top_pct: float | None = None
    flags: list[str] = field(default_factory=list)
    histogram: ChartSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type,
            "count": self.count,
            "missing": self.missing,
            "missing_pct": round(self.missing_pct, 4),
            "n_unique": self.n_unique,
            "unique_pct": round(self.unique_pct, 4),
            "flags": list(self.flags),
        }
        for key in (
            "mean", "std", "minimum", "maximum", "median", "skewness", "kurtosis",
            "zeros", "negatives", "n_outliers", "outlier_pct", "modality",
            "top_value", "top_freq", "top_pct",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.histogram is not None:
            data["histogram"] = self.histogram.to_dict()
        # top_value / bounds may be a Timestamp or numpy scalar -> keep JSON-safe.
        return json_safe(data)


@dataclass
class ProfileResult:
    """Dataset-level profile with column detail, quality score and insights."""

    n_rows: int
    n_columns: int
    total_cells: int
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    duplicate_pct: float
    memory_bytes: int
    semantic_types: dict[str, list[str]]
    columns: list[ColumnProfile]
    quality_score: float
    quality_grade: str
    quality_breakdown: dict[str, float]
    insights: list[Insight]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "total_cells": self.total_cells,
            "missing_cells": self.missing_cells,
            "missing_pct": round(self.missing_pct, 4),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": round(self.duplicate_pct, 4),
            "memory_bytes": self.memory_bytes,
            "semantic_types": self.semantic_types,
            "quality_score": round(self.quality_score, 2),
            "quality_grade": self.quality_grade,
            "quality_breakdown": {k: round(v, 2) for k, v in self.quality_breakdown.items()},
            "columns": [c.to_dict() for c in self.columns],
            "insights": [i.to_dict() for i in self.insights],
        }


class DataProfiler:
    """Compute a full profiling report for a tabular dataset.

    Parameters
    ----------
    skew_threshold:
        ``|skew|`` above this marks a numeric column as skewed (transform candidate).
    high_cardinality_ratio / high_cardinality_absolute:
        A categorical column is "high cardinality" when its distinct count exceeds
        ``high_cardinality_absolute`` *and* ``unique/non-null`` exceeds the ratio.
    """

    def __init__(
        self,
        skew_threshold: float = SKEW_THRESHOLD,
        high_cardinality_ratio: float = HIGH_CARDINALITY_RATIO,
        high_cardinality_absolute: int = HIGH_CARDINALITY_ABSOLUTE,
        quasi_constant_dominance: float = QUASI_CONSTANT_DOMINANCE,
    ) -> None:
        self.skew_threshold = skew_threshold
        self.high_cardinality_ratio = high_cardinality_ratio
        self.high_cardinality_absolute = high_cardinality_absolute
        self.quasi_constant_dominance = quasi_constant_dominance

    # ------------------------------------------------------------------ public
    def profile(self, df: pd.DataFrame) -> ProfileResult:
        if df is None or df.shape[1] == 0:
            raise ValueError("Cannot profile a dataset with no columns.")

        n_rows, n_cols = df.shape
        types = infer_semantic_types(df)
        type_lookup = self._semantic_lookup(types)

        columns = [self._profile_column(df[col], type_lookup.get(col, "categorical")) for col in df.columns]

        total_cells = int(n_rows * n_cols)
        missing_cells = int(df.isna().sum().sum())
        missing_pct = missing_cells / total_cells if total_cells else 0.0
        duplicate_rows = int(df.duplicated().sum()) if n_rows else 0
        duplicate_pct = duplicate_rows / n_rows if n_rows else 0.0

        score, breakdown = self._quality_score(
            missing_pct=missing_pct,
            duplicate_pct=duplicate_pct,
            columns=columns,
            n_rows=n_rows,
        )

        insights = self._build_insights(
            columns=columns,
            missing_pct=missing_pct,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            n_rows=n_rows,
            score=score,
        )

        return ProfileResult(
            n_rows=int(n_rows),
            n_columns=int(n_cols),
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_pct=missing_pct,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            memory_bytes=int(df.memory_usage(deep=True).sum()),
            semantic_types=types.as_dict(),
            columns=columns,
            quality_score=score,
            quality_grade=grade_from_score(score),
            quality_breakdown=breakdown,
            insights=sort_insights(insights),
        )

    # --------------------------------------------------------------- per column
    def _profile_column(self, series: pd.Series, semantic_type: str) -> ColumnProfile:
        n = len(series)
        non_null = series.dropna()
        missing = n - len(non_null)
        n_unique = int(non_null.nunique())

        profile = ColumnProfile(
            name=str(series.name),
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            count=int(len(non_null)),
            missing=int(missing),
            missing_pct=missing / n if n else 0.0,
            n_unique=n_unique,
            unique_pct=n_unique / len(non_null) if len(non_null) else 0.0,
        )

        if semantic_type in {"numeric", "boolean"} and pd.api.types.is_numeric_dtype(series):
            self._fill_numeric(profile, non_null)
        else:
            self._fill_categorical(profile, non_null)

        self._flag_column(profile, series, non_null, semantic_type)
        return profile

    def _fill_numeric(self, profile: ColumnProfile, values: pd.Series) -> None:
        arr = values.to_numpy(dtype="float64")
        if arr.size == 0:
            return

        profile.mean = safe_float(np.mean(arr))
        profile.std = safe_float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        profile.minimum = safe_float(np.min(arr))
        profile.maximum = safe_float(np.max(arr))
        profile.median = safe_float(np.median(arr))
        profile.zeros = int(np.sum(arr == 0))
        profile.negatives = int(np.sum(arr < 0))

        if arr.size >= 3 and np.std(arr) > 0:
            profile.skewness = safe_float(stats.skew(arr, bias=False))
            profile.kurtosis = safe_float(stats.kurtosis(arr, fisher=True, bias=False))
        else:
            profile.skewness = 0.0
            profile.kurtosis = 0.0

        n_out, pct_out = self._iqr_outliers(arr)
        profile.n_outliers = n_out
        profile.outlier_pct = pct_out
        profile.modality = self._estimate_modality(arr)
        profile.histogram = self._histogram_spec(profile.name, arr)

    def _fill_categorical(self, profile: ColumnProfile, values: pd.Series) -> None:
        if values.empty:
            return
        counts = values.value_counts()
        profile.top_value = counts.index[0]
        profile.top_freq = int(counts.iloc[0])
        profile.top_pct = float(counts.iloc[0] / len(values))

    # ------------------------------------------------------------------- flags
    def _flag_column(
        self,
        profile: ColumnProfile,
        series: pd.Series,
        non_null: pd.Series,
        semantic_type: str,
    ) -> None:
        if profile.count == 0:
            profile.flags.append("all_missing")
            return
        if profile.missing_pct >= MISSING_CRITICAL:
            profile.flags.append("high_missing")

        if profile.n_unique <= 1:
            profile.flags.append("constant")
        elif semantic_type in {"categorical", "boolean"} and profile.top_pct and profile.top_pct >= self.quasi_constant_dominance:
            profile.flags.append("quasi_constant")

        if semantic_type in {"categorical", "identifier"}:
            ratio = profile.unique_pct
            if profile.n_unique > self.high_cardinality_absolute and ratio >= self.high_cardinality_ratio:
                profile.flags.append("high_cardinality")

        if profile.skewness is not None and abs(profile.skewness) >= self.skew_threshold:
            profile.flags.append("skewed")
        if profile.kurtosis is not None and profile.kurtosis >= HIGH_KURTOSIS:
            profile.flags.append("heavy_tailed")
        if profile.outlier_pct is not None and profile.outlier_pct > 0.01:
            profile.flags.append("has_outliers")

        # Type validity: a text column whose values are actually numeric/datetime.
        mismatch = self._type_mismatch(series, non_null, semantic_type)
        if mismatch:
            profile.flags.append(mismatch)

    def _type_mismatch(self, series: pd.Series, non_null: pd.Series, semantic_type: str) -> str | None:
        is_text = pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)
        if not is_text or non_null.empty:
            return None
        if semantic_type == "datetime":
            return "stored_as_text_is_datetime"
        sample = non_null.astype(str).head(200)
        numeric_share = pd.to_numeric(sample, errors="coerce").notna().mean()
        if numeric_share >= 0.95:
            return "stored_as_text_is_numeric"
        return None

    # -------------------------------------------------------- numeric helpers
    @staticmethod
    def _iqr_outliers(arr: np.ndarray) -> tuple[int, float]:
        if arr.size < 4:
            return 0, 0.0
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return 0, 0.0
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        mask = (arr < lower) | (arr > upper)
        n_out = int(mask.sum())
        return n_out, n_out / arr.size

    @staticmethod
    def _estimate_modality(arr: np.ndarray) -> str:
        """Estimate the number of modes from a smoothed histogram.

        We avoid a full KDE for speed; a moving-average-smoothed histogram with a
        prominence floor is robust enough to separate "single bell" from "two
        humps" for EDA narration.
        """

        clean = arr[np.isfinite(arr)]
        if clean.size < 20 or np.unique(clean).size < 5:
            return "undetermined"

        bins = int(min(50, max(10, np.sqrt(clean.size))))
        counts, _ = np.histogram(clean, bins=bins)
        if counts.max() == 0:
            return "undetermined"

        kernel = np.ones(3) / 3.0
        smoothed = np.convolve(counts.astype(float), kernel, mode="same")
        floor = 0.05 * smoothed.max()

        peaks = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1] and smoothed[i] > floor:
                peaks += 1
        if smoothed[0] > smoothed[1] and smoothed[0] > floor:
            peaks += 1
        if smoothed[-1] > smoothed[-2] and smoothed[-1] > floor:
            peaks += 1

        if peaks <= 1:
            return "unimodal"
        if peaks == 2:
            return "bimodal"
        return "multimodal"

    @staticmethod
    def _histogram_spec(name: str, arr: np.ndarray) -> ChartSpec | None:
        clean = arr[np.isfinite(arr)]
        if clean.size == 0 or np.unique(clean).size < 2:
            return None
        bins = int(min(50, max(10, np.sqrt(clean.size))))
        counts, edges = np.histogram(clean, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return ChartSpec(
            kind="histogram",
            title=f"Distribution of {name}",
            data={"bin_centers": centers.tolist(), "counts": counts.tolist()},
            layout={"xaxis_title": name, "yaxis_title": "count"},
        )

    # ----------------------------------------------------------- quality score
    def _quality_score(
        self,
        *,
        missing_pct: float,
        duplicate_pct: float,
        columns: list[ColumnProfile],
        n_rows: int,
    ) -> tuple[float, dict[str, float]]:
        """Transparent weighted 0-100 health score.

        Each component is a 0-1 "healthiness" fraction scaled by its weight, so a
        perfect dataset scores exactly 100 and every deduction is explainable.
        """

        n_cols = max(len(columns), 1)
        weights = {
            "completeness": 40.0,
            "row_uniqueness": 15.0,
            "type_validity": 15.0,
            "variability": 10.0,
            "outlier_health": 10.0,
            "cardinality_health": 10.0,
        }

        type_issue_cols = sum(1 for c in columns if any(f.startswith("stored_as_text") for f in c.flags))
        constant_cols = sum(1 for c in columns if "constant" in c.flags or "quasi_constant" in c.flags)
        high_card_cols = sum(1 for c in columns if "high_cardinality" in c.flags)
        mean_outlier_pct = float(np.mean([c.outlier_pct for c in columns if c.outlier_pct is not None])) if any(
            c.outlier_pct is not None for c in columns
        ) else 0.0

        fractions = {
            "completeness": 1.0 - missing_pct,
            "row_uniqueness": 1.0 - min(duplicate_pct, 1.0),
            "type_validity": 1.0 - type_issue_cols / n_cols,
            "variability": 1.0 - constant_cols / n_cols,
            "outlier_health": 1.0 - min(mean_outlier_pct * 5.0, 1.0),
            "cardinality_health": 1.0 - high_card_cols / n_cols,
        }

        breakdown = {key: weights[key] * max(0.0, min(1.0, frac)) for key, frac in fractions.items()}
        score = float(sum(breakdown.values()))
        return score, breakdown

    # ---------------------------------------------------------------- insights
    def _build_insights(
        self,
        *,
        columns: list[ColumnProfile],
        missing_pct: float,
        duplicate_rows: int,
        duplicate_pct: float,
        n_rows: int,
        score: float,
    ) -> list[Insight]:
        insights: list[Insight] = []

        # Headline quality score.
        insights.append(
            Insight(
                title="Dataset health score",
                insight=(
                    f"Overall data quality scores {score:.0f}/100 (grade {grade_from_score(score)}). "
                    f"{missing_pct * 100:.1f}% of cells are missing across {len(columns)} columns."
                ),
                action=(
                    "Address the flagged columns before modeling."
                    if score < 80
                    else "Dataset is in good shape; proceed to feature analysis."
                ),
                metric={"quality_score": round(score, 2), "grade": grade_from_score(score), "missing_pct": round(missing_pct, 4)},
                confidence=0.9,
                severity="ok" if score >= 80 else ("warning" if score >= 60 else "critical"),
                category="profiling",
                visualization=self._missingness_chart(columns),
            )
        )

        # Missingness, worst columns first.
        missing_cols = sorted(
            (c for c in columns if c.missing_pct > 0),
            key=lambda c: c.missing_pct,
            reverse=True,
        )
        for col in missing_cols[:MAX_DETAILED_INSIGHTS]:
            if col.missing_pct < MISSING_WARN and "all_missing" not in col.flags:
                continue
            severity = "critical" if col.missing_pct >= MISSING_CRITICAL else "warning"
            insights.append(
                Insight(
                    title=f"Missing data in '{col.name}'",
                    insight=f"Column '{col.name}' is {col.missing_pct * 100:.1f}% missing ({col.missing} of {n_rows} rows).",
                    action=(
                        "Drop this column -- it is empty."
                        if "all_missing" in col.flags
                        else "Impute (median/mode), add a missingness indicator, or drop if not informative."
                    ),
                    metric={"missing_pct": round(col.missing_pct, 4), "missing": col.missing},
                    confidence=0.95,
                    severity=severity,
                    category="profiling",
                )
            )

        # Skew / distribution.
        skewed = [c for c in columns if "skewed" in c.flags]
        if skewed:
            worst = max(skewed, key=lambda c: abs(c.skewness or 0))
            names = ", ".join(c.name for c in skewed[:6])
            insights.append(
                Insight(
                    title="Skewed numeric distributions",
                    insight=(
                        f"{len(skewed)} numeric column(s) are strongly skewed (e.g. '{worst.name}' "
                        f"skew={worst.skewness:.2f}, {worst.modality})."
                    ),
                    action="Apply a log/Box-Cox/Yeo-Johnson transform before linear or distance-based models.",
                    metric={"skewed_columns": names, "max_abs_skew": round(abs(worst.skewness or 0), 3)},
                    confidence=0.8,
                    severity="info",
                    category="profiling",
                    visualization=worst.histogram,
                )
            )

        # Outliers.
        outlier_cols = [c for c in columns if (c.outlier_pct or 0) > 0.01]
        if outlier_cols:
            worst = max(outlier_cols, key=lambda c: c.outlier_pct or 0)
            insights.append(
                Insight(
                    title="Outliers detected",
                    insight=(
                        f"{len(outlier_cols)} numeric column(s) carry IQR outliers; '{worst.name}' has "
                        f"{worst.outlier_pct * 100:.1f}% outlying values."
                    ),
                    action="Decide per column: cap/winsorize, transform, or keep if outliers are legitimate signal.",
                    metric={"columns": [c.name for c in outlier_cols][:10], "worst_pct": round(worst.outlier_pct or 0, 4)},
                    confidence=0.7,
                    severity="info",
                    category="profiling",
                )
            )

        # Cardinality and constants.
        high_card = [c for c in columns if "high_cardinality" in c.flags]
        if high_card:
            insights.append(
                Insight(
                    title="High-cardinality categoricals",
                    insight=(
                        f"{len(high_card)} categorical column(s) have many distinct values "
                        f"(e.g. '{high_card[0].name}': {high_card[0].n_unique} categories)."
                    ),
                    action="Avoid one-hot encoding; use target/frequency encoding or learned embeddings.",
                    metric={c.name: c.n_unique for c in high_card[:10]},
                    confidence=0.85,
                    severity="warning",
                    category="profiling",
                )
            )

        constants = [c for c in columns if "constant" in c.flags or "quasi_constant" in c.flags]
        if constants:
            insights.append(
                Insight(
                    title="Constant / near-constant columns",
                    insight=f"{len(constants)} column(s) carry (almost) no variance: {', '.join(c.name for c in constants[:8])}.",
                    action="Drop these -- they add no signal and can destabilise some models.",
                    metric={"columns": [c.name for c in constants]},
                    confidence=0.9,
                    severity="warning",
                    category="profiling",
                )
            )

        # Type mismatches.
        mismatches = [c for c in columns if any(f.startswith("stored_as_text") for f in c.flags)]
        if mismatches:
            insights.append(
                Insight(
                    title="Type mismatches",
                    insight=(
                        f"{len(mismatches)} column(s) are stored as text but hold numeric/date values "
                        f"(e.g. '{mismatches[0].name}')."
                    ),
                    action="Cast to the correct dtype so summaries, scaling and models behave correctly.",
                    metric={c.name: [f for f in c.flags if f.startswith("stored_as_text")][0] for c in mismatches[:10]},
                    confidence=0.85,
                    severity="warning",
                    category="profiling",
                )
            )

        # Duplicate rows.
        if duplicate_rows > 0:
            insights.append(
                Insight(
                    title="Duplicate rows",
                    insight=f"{duplicate_rows} duplicate row(s) found ({duplicate_pct * 100:.1f}% of the dataset).",
                    action="Confirm whether duplicates are legitimate; otherwise de-duplicate to avoid leakage and bias.",
                    metric={"duplicate_rows": duplicate_rows, "duplicate_pct": round(duplicate_pct, 4)},
                    confidence=0.9,
                    severity="warning" if duplicate_pct > 0.01 else "info",
                    category="profiling",
                )
            )

        return insights

    @staticmethod
    def _missingness_chart(columns: list[ColumnProfile]) -> ChartSpec | None:
        offenders = [c for c in columns if c.missing_pct > 0]
        if not offenders:
            return None
        offenders = sorted(offenders, key=lambda c: c.missing_pct, reverse=True)[:25]
        return ChartSpec(
            kind="bar",
            title="Missing data by column",
            data={
                "x": [round(c.missing_pct * 100, 2) for c in offenders],
                "y": [c.name for c in offenders],
            },
            layout={"orientation": "h", "xaxis_title": "% missing", "yaxis_title": "column"},
        )

    @staticmethod
    def _semantic_lookup(types: SemanticTypes) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for kind, names in types.as_dict().items():
            for name in names:
                lookup[name] = kind
        return lookup


def profile_dataset(df: pd.DataFrame, **kwargs: Any) -> ProfileResult:
    """Convenience wrapper -- profile ``df`` with default settings."""

    return DataProfiler(**kwargs).profile(df)
