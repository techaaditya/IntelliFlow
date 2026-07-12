"""End-to-end tests for the EDA orchestrator and its exporters."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engines.analytics import run_eda
from engines.analytics.pipeline import CAPABILITIES
from engines.analytics.report import render_html
from engines.analytics.visualization import render_image, to_plotly


def test_run_eda_full(sales_event_log):
    report = run_eda(
        sales_event_log,
        target_column="revenue",
        timestamp_column="date",
        user_id_column="user_id",
        event_column="product",
        segment_columns=["region"],
        funnel_steps=["launch", "purchase", "repeat"],
        target_event="purchase",
    )
    assert set(report.results.keys()) == set(CAPABILITIES)
    assert report.errors == {}
    assert report.data_quality_score is not None
    assert len(report.insights) > 0
    json.dumps(report.to_dict())  # JSON-safe


def test_run_eda_minimal_tabular(feature_frame):
    # No event columns: only the always-on capabilities should run.
    report = run_eda(feature_frame, target_column="target")
    assert "profiling" in report.results
    assert "correlation" in report.results
    assert "recommendations" in report.results
    assert "funnel" not in report.results  # needs steps
    assert "retention" not in report.results  # needs user/timestamp


def test_run_eda_capability_isolation(sales_event_log):
    # A bad funnel step must not crash the whole run; it is captured in errors.
    report = run_eda(
        sales_event_log,
        timestamp_column="date",
        user_id_column="user_id",
        event_column="product",
        funnel_steps=["does_not_exist"],  # only one step -> ValueError inside funnel
    )
    assert "profiling" in report.results
    assert "funnel" in report.errors


def test_run_eda_validates_columns(feature_frame):
    with pytest.raises(ValueError):
        run_eda(feature_frame, target_column="nope")


def test_run_eda_rejects_non_dataframe():
    with pytest.raises(TypeError):
        run_eda([{"a": 1}])


def test_run_eda_rejects_empty():
    with pytest.raises(ValueError):
        run_eda(pd.DataFrame())


def test_report_exporters(sales_event_log, tmp_path: Path):
    report = run_eda(
        sales_event_log,
        target_column="revenue",
        timestamp_column="date",
        user_id_column="user_id",
        event_column="product",
        funnel_steps=["launch", "purchase", "repeat"],
    )
    json_path = report.to_json(tmp_path / "report.json")
    html_path = report.to_html(tmp_path / "dash.html")
    csvs = report.save_csv_summaries(tmp_path)
    pngs = report.save_charts(tmp_path, "png")

    assert Path(json_path).exists() and Path(json_path).stat().st_size > 0
    assert Path(html_path).exists() and "plotly" in Path(html_path).read_text().lower()
    assert len(csvs) > 0
    assert len(pngs) > 0
    json.loads(Path(json_path).read_text())  # valid JSON on disk


def test_charts_render_without_error(sales_event_log, tmp_path: Path):
    report = run_eda(sales_event_log, timestamp_column="date", user_id_column="user_id", event_column="product")
    for spec in report.charts:
        assert to_plotly(spec) is not None
        out = render_image(spec, str(tmp_path / "c.png"))
        assert Path(out).stat().st_size > 0
