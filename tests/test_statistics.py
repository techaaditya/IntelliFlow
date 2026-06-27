"""Tests for the standalone statistical functions and core helpers."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from engines.analytics.base import Insight, ChartSpec, grade_from_score, json_safe, safe_float
from engines.analytics.correlation import CorrelationAnalyzer, correlation_ratio, cramers_v
from engines.analytics.events import EventStreamAnalyzer
from engines.analytics.utils import infer_semantic_types, looks_like_datetime, wilson_interval


# ------------------------------------------------------------------ base utils
def test_json_safe_handles_numpy_pandas_and_nan():
    payload = {
        "i": np.int64(3),
        "f": np.float32(1.5),
        "nan": float("nan"),
        "inf": np.float64(np.inf),
        "arr": np.array([1, 2]),
        "ts": pd.Timestamp("2024-01-01"),
        "td": pd.Timedelta("1D"),
        "b": np.bool_(True),
        "series": pd.Series([1, 2], index=["x", "y"]),
    }
    safe = json_safe(payload)
    json.dumps(safe)  # must not raise
    assert safe["i"] == 3 and isinstance(safe["i"], int)
    assert safe["nan"] is None and safe["inf"] is None
    assert safe["arr"] == [1, 2]
    assert safe["ts"].startswith("2024-01-01")
    assert safe["td"] == 86400.0
    assert safe["b"] is True
    assert safe["series"] == {"x": 1, "y": 2}


def test_safe_float_and_grade():
    assert safe_float("nope", default=-1) == -1
    assert safe_float(float("nan")) is None
    assert grade_from_score(95) == "A" and grade_from_score(50) == "F"


def test_insight_clamps_confidence_and_serialises():
    ins = Insight(title="t", insight="i", action="a", confidence=5.0, severity="nonsense")
    assert ins.confidence == 1.0
    assert ins.severity == "info"  # invalid severity falls back
    assert ins.to_dict()["confidence_label"] == "very high"


# ---------------------------------------------------------------- wilson interval
def test_wilson_interval_bounds_and_degenerate():
    p, lo, hi = wilson_interval(30, 100)
    assert lo < p < hi
    assert 0.0 <= lo and hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0, 0.0)
    # Certain success stays within [0, 1].
    _, lo2, hi2 = wilson_interval(10, 10)
    assert hi2 <= 1.0 and lo2 > 0.5


# --------------------------------------------------------------- cramers / eta
def test_cramers_v_perfect_and_independent():
    a = pd.Series(["x", "y"] * 100)
    perfect = a.map({"x": "1", "y": "2"})
    assert cramers_v(a, perfect) > 0.99
    gen = np.random.default_rng(0)
    indep = pd.Series(gen.choice(["p", "q"], 400))
    rand = pd.Series(gen.choice(["m", "n"], 400))
    assert cramers_v(indep, rand) < 0.2


def test_cramers_v_handles_single_category():
    assert cramers_v(pd.Series(["a"] * 10), pd.Series(["b"] * 10)) == 0.0


def test_correlation_ratio_separates_groups():
    cats = pd.Series(["g1"] * 50 + ["g2"] * 50)
    vals = pd.Series(np.r_[np.zeros(50), np.ones(50) * 10])
    assert correlation_ratio(cats, vals) > 0.95
    noise = pd.Series(np.random.default_rng(1).normal(0, 1, 100))
    assert correlation_ratio(cats, noise) < 0.4


# ------------------------------------------------------------------------ VIF
def test_vif_flags_collinearity():
    gen = np.random.default_rng(2)
    x = gen.normal(0, 1, 300)
    df = pd.DataFrame({"x": x, "x_dup": x + gen.normal(0, 1e-3, 300), "y": gen.normal(0, 1, 300)})
    result = CorrelationAnalyzer().analyze(df)
    vif = {v.feature: v.vif for v in result.vif}
    assert vif["x"] > 5 and vif["x_dup"] > 5
    assert vif["y"] < 5


def test_high_correlation_detection():
    gen = np.random.default_rng(3)
    x = gen.normal(0, 1, 200)
    df = pd.DataFrame({"a": x, "b": x * 2 + 1, "c": gen.normal(0, 1, 200)})
    pairs = CorrelationAnalyzer().analyze(df).high_correlations
    assert any({p.feature_a, p.feature_b} == {"a", "b"} for p in pairs)


def test_target_ranking_and_nonlinear():
    gen = np.random.default_rng(4)
    n = 1000
    u = gen.uniform(-3, 3, n)
    df = pd.DataFrame({"u_shaped": u, "noise": gen.normal(0, 1, n)})
    df["target"] = u ** 2 * 3 + gen.normal(0, 0.4, n)
    result = CorrelationAnalyzer().analyze(df, target_column="target")
    top = result.target_ranking[0]
    assert top.feature == "u_shaped"
    assert top.nonlinear is True  # high MI but near-zero linear correlation


# ----------------------------------------------------------------- datetime / types
def test_looks_like_datetime():
    assert looks_like_datetime(pd.Series(["2024-01-01", "2024-02-15", "2024-03-30"]))
    assert not looks_like_datetime(pd.Series(["apple", "banana", "cherry"]))


def test_infer_semantic_types_separates_id_from_category():
    df = pd.DataFrame(
        {
            "amount": np.random.default_rng(0).exponential(50, 200),
            "flag": [0, 1] * 100,
            "category": (["a", "b", "c", "d"] * 50),
            "uuid": [f"id-{i}" for i in range(200)],
            "when": pd.date_range("2024-01-01", periods=200, freq="h").astype(str),
        }
    )
    types = infer_semantic_types(df)
    assert "amount" in types.numeric
    assert "flag" in types.boolean
    assert "category" in types.categorical
    assert "uuid" in types.identifier
    assert "when" in types.datetime


# --------------------------------------------------------------- kaplan-meier
def test_kaplan_meier_monotonic_and_censoring():
    gen = np.random.default_rng(5)
    durations = np.abs(gen.exponential(5, 200))
    observed = gen.integers(0, 2, 200)
    times, surv, at_risk, estimator = EventStreamAnalyzer._kaplan_meier(durations, observed)
    assert estimator in {"statsmodels", "manual"}
    # Survival probability is non-increasing and within [0, 1].
    assert np.all(np.diff(surv) <= 1e-9)
    assert surv.max() <= 1.0 and surv.min() >= 0.0
    assert len(times) == len(surv) == len(at_risk)
