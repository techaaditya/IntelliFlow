"""Behavioural and edge-case tests for each analytics capability."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engines.analytics.anomaly import detect_anomalies
from engines.analytics.events import analyze_events
from engines.analytics.funnel import analyze_funnel
from engines.analytics.profiling import profile_dataset
from engines.analytics.recommendations import recommend_features
from engines.analytics.retention import analyze_retention


def _assert_json_safe(result) -> None:
    json.dumps(result.to_dict())


# ----------------------------------------------------------------- profiling
def test_profiling_detects_issues_and_scores(feature_frame):
    result = profile_dataset(feature_frame)
    assert 0 <= result.quality_score <= 100
    flags = {c.name: c.flags for c in result.columns}
    assert "constant" in flags["const"]
    assert "skewed" in flags["skewed"]
    _assert_json_safe(result)


@pytest.mark.parametrize("case", ["single_row", "all_nulls", "no_variance", "single_column", "two_rows"])
def test_profiling_edge_cases(edge_cases, case):
    result = profile_dataset(edge_cases[case])
    assert 0 <= result.quality_score <= 100
    _assert_json_safe(result)


def test_profiling_all_nulls_flagged(edge_cases):
    result = profile_dataset(edge_cases["all_nulls"])
    assert all("all_missing" in c.flags for c in result.columns)


def test_profiling_rejects_empty():
    with pytest.raises(ValueError):
        profile_dataset(pd.DataFrame())


# -------------------------------------------------------------------- funnel
def test_funnel_event_log_conversion_and_dropoff(sales_event_log):
    result = analyze_funnel(
        sales_event_log,
        ["launch", "purchase", "repeat"],
        user_col="user_id",
        event_col="product",
        timestamp_col="date",
        segment_col="region",
    )
    users = [s.users for s in result.steps]
    assert users[0] >= users[1] >= users[2]  # monotone funnel
    assert 0 <= result.overall_conversion <= 1
    assert result.biggest_dropoff is not None
    assert len(result.segments) >= 1
    _assert_json_safe(result)


def test_funnel_wide_format():
    wide = pd.DataFrame(
        {"a": [1] * 100, "b": [1] * 60 + [0] * 40, "c": [1] * 30 + [0] * 70}
    )
    result = analyze_funnel(wide, ["a", "b", "c"])
    assert result.steps[1].users == 60 and result.steps[2].users == 30
    assert result.steps[1].conversion_from_prev == pytest.approx(0.6)


def test_funnel_requires_two_steps():
    with pytest.raises(ValueError):
        analyze_funnel(pd.DataFrame({"a": [1]}), ["a"])


def test_funnel_confidence_interval_present(sales_event_log):
    result = analyze_funnel(
        sales_event_log, ["launch", "purchase"], user_col="user_id", event_col="product", timestamp_col="date"
    )
    step = result.steps[1]
    assert step.ci_low <= step.conversion_from_prev <= step.ci_high


# ----------------------------------------------------------------- retention
def test_retention_curve_maturity_and_trend():
    gen = np.random.default_rng(11)
    start = pd.Timestamp("2024-01-01")
    rows = []
    for week in range(10):
        c0 = start + pd.Timedelta(weeks=week)
        for u in range(200):
            uid = f"w{week}_u{u}"
            rows.append((uid, c0))
            for d in (1, 7, 30):
                if gen.random() < (0.5 - week * 0.03):
                    rows.append((uid, c0 + pd.Timedelta(days=d)))
    log = pd.DataFrame(rows, columns=["user", "ts"])
    result = analyze_retention(log, user_col="user", timestamp_col="ts", retention_unit="day", horizons=[0, 1, 7, 30])
    by_off = {p.offset: p for p in result.curve}
    assert by_off[0].rate == pytest.approx(1.0)
    # D30 eligibility excludes immature cohorts -> fewer eligible than D1.
    assert by_off[30].eligible <= by_off[1].eligible
    assert result.new_cohort_trend.get("direction") == "declining"
    _assert_json_safe(result)


def test_retention_churn_features(sales_event_log):
    result = analyze_retention(sales_event_log, user_col="user_id", timestamp_col="date")
    assert "recency_days" in result.churn.feature_columns
    assert result.churn.total_users > 0


def test_retention_single_user():
    tiny = pd.DataFrame({"u": ["a"], "ts": [pd.Timestamp("2024-01-01")]})
    result = analyze_retention(tiny, user_col="u", timestamp_col="ts")
    assert result.n_users == 1
    assert result.curve[0].rate == pytest.approx(1.0)


# -------------------------------------------------------------------- events
def test_events_sessions_sequences_survival(sales_event_log):
    result = analyze_events(
        sales_event_log,
        timestamp_col="date",
        user_col="user_id",
        event_col="product",
        freq="day",
        target_event="purchase",
    )
    assert result.n_events == len(sales_event_log)
    assert result.sessions is not None and result.sessions.n_sessions > 0
    assert result.velocity["available"] is True
    assert result.survival is not None and result.survival.n_events > 0
    _assert_json_safe(result)


def test_events_minimal_timestamp_only():
    df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=50, freq="h")})
    result = analyze_events(df, timestamp_col="t", freq="hour")
    assert result.sessions is None
    assert result.n_events == 50


# ------------------------------------------------------------------- anomaly
def test_anomaly_timeseries_spike_and_root_cause():
    gen = np.random.default_rng(2)
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    revenue = 100 + 15 * np.sin(np.arange(120) * 2 * np.pi / 7) + gen.normal(0, 4, 120)
    marketing = revenue * 0.5 + gen.normal(0, 2, 120)
    revenue[60] += 160
    marketing[60] += 80
    df = pd.DataFrame({"date": days, "revenue": revenue, "marketing": marketing})
    result = detect_anomalies(df, timestamp_col="date", value_col="revenue", freq="day", agg="sum")
    assert len(result.timeseries_anomalies) >= 1
    assert any(c["feature"] == "marketing" for c in result.root_causes)
    _assert_json_safe(result)


def test_anomaly_multivariate_isolation_forest():
    gen = np.random.default_rng(7)
    X = gen.normal(0, 1, (300, 4))
    X[5] = [12, 12, 12, 12]
    df = pd.DataFrame(X, columns=list("abcd"))
    result = detect_anomalies(df, contamination=0.02)
    assert result.multivariate is not None
    assert result.multivariate.n_anomalies >= 1
    assert 5 in [a["index"] for a in result.multivariate.top_anomalies]


def test_anomaly_no_variance_is_safe(edge_cases):
    result = detect_anomalies(edge_cases["no_variance"])
    assert result.univariate == []  # constant column produces no outliers
    _assert_json_safe(result)


# --------------------------------------------------------------- recommendations
def test_recommendations_full(feature_frame):
    result = recommend_features(feature_frame, target_column="target")
    assert any(f.issue == "constant" for f in result.constant_features)
    assert any("leaky" == w.feature for w in result.leakage_warnings)
    assert any("region" in g.features and "region_relabel" in g.features for g in result.duplicate_groups)
    assert result.scaling.needed is True
    _assert_json_safe(result)


def test_encoding_recommendations_by_cardinality():
    gen = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "low": gen.choice(list("abc"), 500),
            "high": gen.choice([f"c{i}" for i in range(200)], 500),
            "target": gen.normal(0, 1, 500),
        }
    )
    recs = {e.feature: e.strategy for e in recommend_features(df, target_column="target").encoding}
    assert recs["low"] == "one-hot"
    assert recs["high"] in {"target-encoding", "frequency-encoding", "embedding"}
