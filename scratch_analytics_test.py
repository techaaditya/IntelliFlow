"""Demo / verification script for IntelliFlow Engine 2 (Analytics & EDA).

Builds a synthetic product-analytics dataset (the spec's sales_data example:
date, user_id, revenue, region, product) with a funnel, cohorts, a revenue
anomaly, and a leaky feature, then runs the full EDA suite and writes a JSON
report, an interactive HTML dashboard, CSV summaries, and PNG charts.

    python scratch_analytics_test.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from engines.analytics import run_eda

warnings.filterwarnings("ignore")


def build_sales_data(seed: int = 42) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    base = pd.Timestamp("2024-06-01")
    rows: list[tuple] = []
    for user in range(2000):
        region = gen.choice(["NA", "EU", "APAC", "LATAM"], p=[0.45, 0.3, 0.15, 0.10])
        signup = base + pd.Timedelta(days=float(gen.integers(0, 45)))
        base_rev = {"NA": 130.0, "EU": 95.0, "APAC": 70.0, "LATAM": 55.0}[region]

        rows.append((signup, user, float(gen.exponential(base_rev)), region, "launch"))
        if gen.random() < 0.6:  # purchase
            rows.append((signup + pd.Timedelta(days=float(gen.exponential(2))), user, float(gen.exponential(base_rev)), region, "purchase"))
            if gen.random() < 0.35:  # repeat
                rows.append((signup + pd.Timedelta(days=float(gen.exponential(12))), user, float(gen.exponential(base_rev)), region, "repeat"))

    df = pd.DataFrame(rows, columns=["date", "user_id", "revenue", "region", "product"])
    # Inject a viral-event revenue spike.
    spike = base + pd.Timedelta(days=15)
    df.loc[df["date"].between(spike, spike + pd.Timedelta(days=1)), "revenue"] *= 6
    return df


def main() -> None:
    print("Building synthetic sales_data ...")
    df = build_sales_data()
    print(f"  {df.shape[0]:,} event rows x {df.shape[1]} columns\n")

    print("Running full EDA suite ...")
    report = run_eda(
        df,
        target_column="revenue",
        timestamp_column="date",
        user_id_column="user_id",
        event_column="product",
        segment_columns=["region"],
        funnel_steps=["launch", "purchase", "repeat"],
        target_event="purchase",
        freq="day",
        retention_unit="day",
    )

    print(f"\n=== EDA REPORT ===")
    print(f"Data quality: {report.data_quality_score:.0f}/100 (grade {report.data_quality_grade})")
    print(f"Capabilities run: {', '.join(report.results)}")
    if report.errors:
        print(f"Errors: {report.errors}")
    print(f"\nTop insights ({len(report.insights)} total):")
    for insight in report.insights[:12]:
        print(f"  [{insight.severity:>8}] {insight.title}")
        print(f"             {insight.insight}")
        print(f"             -> {insight.action}  (confidence: {insight.confidence_label})")

    out = Path("eda_output")
    out.mkdir(exist_ok=True)
    report.to_json(out / "report.json")
    report.to_html(out / "dashboard.html")
    report.save_csv_summaries(out)
    report.save_charts(out, "png")
    print(f"\nArtifacts written to ./{out}/ : report.json, dashboard.html, *.csv, *.png")


if __name__ == "__main__":
    main()
