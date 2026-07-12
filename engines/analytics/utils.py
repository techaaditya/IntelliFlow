"""Shared dataset-introspection and statistics helpers for Engine 2.

These are the small, well-tested primitives that several capabilities lean on:
semantic type inference (is this column a measure, a category, a date, or an
identifier?), safe datetime coercion, Wilson confidence intervals for conversion
rates, and a deterministic sampler so analyses stay responsive on large frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Heuristic thresholds. Centralised so behaviour is consistent across capabilities.
HIGH_CARDINALITY_RATIO = 0.5  # unique/non-null above this -> likely an identifier
HIGH_CARDINALITY_ABSOLUTE = 100  # distinct categories above this -> encoding concern
DATETIME_DETECTION_SAMPLE = 50
DATETIME_MATCH_THRESHOLD = 0.8


@dataclass
class SemanticTypes:
    """Columns grouped by their analytical role rather than raw dtype."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    # High-cardinality near-unique columns: free text, UUIDs, primary keys.
    identifier: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "datetime": list(self.datetime),
            "boolean": list(self.boolean),
            "identifier": list(self.identifier),
        }


def looks_like_datetime(series: pd.Series) -> bool:
    """Return True when an object/string column parses as dates for most rows.

    Cheap pre-filter (regex on a sample) before the comparatively expensive
    ``pd.to_datetime`` so we do not pay parsing cost on obviously-textual columns.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)):
        return False

    sample = series.dropna().astype(str).head(DATETIME_DETECTION_SAMPLE)
    if sample.empty:
        return False

    date_like = sample.str.contains(
        r"\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}:\d{2}",
        regex=True,
    )
    if date_like.mean() < DATETIME_MATCH_THRESHOLD:
        return False

    with pd.option_context("mode.chained_assignment", None):
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= DATETIME_MATCH_THRESHOLD


def coerce_datetime(series: pd.Series) -> pd.Series:
    """Best-effort conversion of a column to ``datetime64`` (NaT on failure)."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")


def infer_semantic_types(df: pd.DataFrame) -> SemanticTypes:
    """Classify each column into an analytical role.

    The distinction the raw dtype misses and that EDA cares about most is
    *identifier* vs *categorical*: a string column where almost every value is
    unique (user ids, emails, UUIDs) should never be one-hot encoded or treated
    as a low-cardinality dimension.
    """

    types = SemanticTypes()
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        if pd.api.types.is_bool_dtype(series):
            types.boolean.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            # Numeric columns that are really binary flags read better as boolean.
            uniques = set(non_null.unique().tolist())
            if uniques and uniques.issubset({0, 1, 0.0, 1.0}):
                types.boolean.append(col)
            elif _looks_like_numeric_id(non_null, n_rows):
                # Integer-valued and almost entirely unique (user ids, row keys):
                # an identifier, not a measure. Floats are never treated this way,
                # so genuinely continuous features are safe.
                types.identifier.append(col)
            else:
                types.numeric.append(col)
            continue
        if looks_like_datetime(series):
            types.datetime.append(col)
            continue

        # Remaining object/category columns: split identifiers from dimensions.
        n_unique = int(non_null.nunique())
        ratio = n_unique / max(len(non_null), 1)
        if n_rows >= 20 and ratio >= HIGH_CARDINALITY_RATIO and n_unique > HIGH_CARDINALITY_ABSOLUTE:
            types.identifier.append(col)
        else:
            types.categorical.append(col)

    return types


def _looks_like_numeric_id(non_null: pd.Series, n_rows: int) -> bool:
    """Heuristic: integer-valued column that is almost entirely unique.

    Restricted to integer-like values so continuous measures (which can also be
    near-unique, e.g. precise revenue floats) are never misread as identifiers.
    """

    if n_rows < 50 or non_null.empty:
        return False
    n_unique = int(non_null.nunique())
    if n_unique <= HIGH_CARDINALITY_ABSOLUTE or n_unique / len(non_null) < 0.99:
        return False
    if pd.api.types.is_integer_dtype(non_null):
        return True
    arr = non_null.to_numpy(dtype="float64", na_value=np.nan)
    return bool(np.all(np.isfinite(arr)) and np.allclose(arr, np.round(arr)))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returned as ``(point_estimate, lower, upper)``. The Wilson interval is the
    standard choice for conversion/retention rates because, unlike the naive
    normal approximation, it stays inside ``[0, 1]`` and behaves sensibly for
    small samples or proportions near 0/1.
    """

    if total <= 0:
        return 0.0, 0.0, 0.0

    phat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (phat + z2 / (2 * total)) / denom
    margin = (z * np.sqrt((phat * (1 - phat) + z2 / (4 * total)) / total)) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return float(phat), float(lower), float(upper)


def maybe_sample(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> tuple[pd.DataFrame, bool]:
    """Deterministically down-sample for expensive (super-linear) analyses.

    Returns ``(frame, was_sampled)`` so callers can annotate that a result is an
    estimate. Keeps the engine responsive on the 1M-row datasets in the spec
    without changing behaviour on the small frames used in tests.
    """

    if max_rows <= 0 or len(df) <= max_rows:
        return df, False
    return df.sample(n=max_rows, random_state=random_state), True


def numeric_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return only the usable numeric columns (drops all-NaN / constant-NaN)."""

    frame = df if columns is None else df[columns]
    numeric = frame.select_dtypes(include=[np.number])
    return numeric.loc[:, numeric.notna().any(axis=0)]


def format_count(value: int) -> str:
    """Compact human formatting for counts used in insight prose (1.2K, 3.4M)."""

    value = int(value)
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return str(value)
