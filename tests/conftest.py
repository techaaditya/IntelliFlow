"""Shared pytest fixtures for IntelliFlow Engine 2 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def sales_event_log() -> pd.DataFrame:
    """A small product-style event log: launch -> purchase -> repeat funnel.

    Includes a user id, timestamp, revenue measure, region segment, and a
    deliberately injected revenue spike so anomaly detection has something to find.
    """

    gen = np.random.default_rng(7)
    base = pd.Timestamp("2024-06-01")
    rows: list[tuple] = []
    for u in range(600):
        region = gen.choice(["NA", "EU", "APAC"], p=[0.5, 0.3, 0.2])
        t0 = base + pd.Timedelta(days=float(gen.integers(0, 30)))
        rev = {"NA": 120.0, "EU": 90.0, "APAC": 60.0}[region]
        rows.append((t0, u, float(gen.exponential(rev)), region, "launch"))
        if gen.random() < 0.55:
            rows.append((t0 + pd.Timedelta(days=float(gen.exponential(2))), u, float(gen.exponential(rev)), region, "purchase"))
            if gen.random() < 0.4:
                rows.append((t0 + pd.Timedelta(days=float(gen.exponential(8))), u, float(gen.exponential(rev)), region, "repeat"))
    df = pd.DataFrame(rows, columns=["date", "user_id", "revenue", "region", "product"])
    spike = base + pd.Timedelta(days=10)
    df.loc[df["date"].between(spike, spike + pd.Timedelta(days=1)), "revenue"] *= 6
    return df


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """A modelling-style frame with collinearity, skew, constants and a target."""

    gen = np.random.default_rng(3)
    n = 400
    x1 = gen.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x1_copy": x1 + gen.normal(0, 0.001, n),  # near-duplicate -> high VIF
            "x2": gen.normal(5, 2, n),
            "skewed": gen.exponential(3, n),
            "const": np.ones(n),
            "region": gen.choice(["A", "B", "C"], n),
            "region_relabel": None,
        }
    )
    df["region_relabel"] = df["region"].map({"A": "a", "B": "b", "C": "c"})  # functional dup
    df["target"] = 2 * df["x2"] + 0.5 * x1 + gen.normal(0, 0.5, n)
    df["leaky"] = df["target"] + gen.normal(0, 1e-3, n)  # leakage
    return df


@pytest.fixture
def edge_cases() -> dict[str, pd.DataFrame]:
    """The pathological frames every statistical function must survive."""

    return {
        "single_row": pd.DataFrame({"a": [1.0], "b": ["x"]}),
        "all_nulls": pd.DataFrame({"a": [np.nan] * 8, "b": [None] * 8}),
        "no_variance": pd.DataFrame({"a": [5.0] * 50, "b": ["k"] * 50}),
        "single_column": pd.DataFrame({"a": np.arange(20.0)}),
        "two_rows": pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}),
    }
