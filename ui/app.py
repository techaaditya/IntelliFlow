"""Streamlit dashboard for IntelliFlow Engines 1 and 2."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from engines.analytics import run_eda
from engines.analytics.visualization import to_plotly
from engines.automl.pipeline import detect_task_type, run_automl
from engines.automl.registry import AutoMLRegistry


st.set_page_config(page_title="IntelliFlow", page_icon="IF", layout="wide")


CSS = """
<style>
:root {
  --if-ink: #18212f;
  --if-muted: #647084;
  --if-border: #d8dde6;
  --if-panel: #ffffff;
  --if-soft: #f6f8fb;
  --if-teal: #0f766e;
  --if-blue: #2457a7;
  --if-amber: #b7791f;
  --if-red: #b42318;
}
.stApp {
  background: #ffffff;
  color: var(--if-ink);
}
.block-container {
  padding-top: 1.25rem;
  max-width: 1500px;
}
[data-testid="stSidebar"] {
  background: #f8fafc;
  border-right: 1px solid var(--if-border);
}
[data-testid="stSidebar"] * {
  color: var(--if-ink);
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  background: #ffffff;
  border-color: var(--if-border);
}
h1, h2, h3 {
  letter-spacing: 0;
  color: var(--if-ink);
}
p, label, span, div {
  letter-spacing: 0;
}
.stMarkdown, .stText, [data-testid="stMarkdownContainer"] {
  color: var(--if-ink);
}
.if-title {
  display: flex;
  align-items: baseline;
  gap: .75rem;
  padding: .25rem 0 .75rem;
  border-bottom: 1px solid var(--if-border);
  margin-bottom: 1rem;
}
.if-title h1 {
  margin: 0;
  font-size: 1.75rem;
}
.if-title span {
  color: var(--if-muted);
  font-size: .95rem;
}
.if-band {
  background: var(--if-soft);
  border: 1px solid var(--if-border);
  border-radius: 8px;
  color: var(--if-ink);
  padding: .9rem 1rem;
  margin: .5rem 0 1rem;
}
.metric-row [data-testid="stMetric"] {
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: 8px;
  padding: .7rem .85rem;
}
.if-status {
  display: inline-block;
  border-radius: 999px;
  border: 1px solid var(--if-border);
  padding: .18rem .55rem;
  font-size: .78rem;
  color: var(--if-muted);
  background: #fff;
}
.if-critical { color: var(--if-red); font-weight: 700; }
.if-warning { color: var(--if-amber); font-weight: 700; }
.if-info { color: var(--if-blue); font-weight: 700; }
.if-ok { color: var(--if-teal); font-weight: 700; }
div[data-testid="stTabs"] button {
  min-height: 2.6rem;
  color: var(--if-ink);
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  border-bottom-color: var(--if-teal);
}
[data-testid="stAlert"] {
  border-radius: 8px;
  border: 1px solid #c7d2fe;
  background: #eef4ff;
  color: var(--if-ink);
}
[data-testid="stButton"] button[kind="primary"] {
  background: #0f766e;
  border-color: #0f766e;
  color: #ffffff;
}
[data-testid="stButton"] button[kind="primary"] * {
  color: #ffffff;
}
</style>
"""


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='if-title'><h1>IntelliFlow</h1><span>Unified AutoML and Analytics workbench</span></div>",
        unsafe_allow_html=True,
    )

    dataset = sidebar_dataset()
    if dataset is None:
        st.info("Upload a dataset or load a sample from the sidebar to begin.")
        return

    st.session_state["dataset"] = dataset
    overview(dataset)

    dataset_tab, analytics_tab, automl_tab, predict_tab, api_tab = st.tabs(
        ["Dataset", "Analytics Engine", "AutoML Engine", "Prediction", "API"]
    )
    with dataset_tab:
        dataset_view(dataset)
    with analytics_tab:
        analytics_view(dataset)
    with automl_tab:
        automl_view(dataset)
    with predict_tab:
        prediction_view(dataset)
    with api_tab:
        api_view()


def sidebar_dataset() -> pd.DataFrame | None:
    with st.sidebar:
        st.subheader("Dataset")
        source = st.radio("Source", ["Upload file", "Iris sample", "Product analytics sample"], label_visibility="collapsed")
        if source == "Upload file":
            uploaded = st.file_uploader("CSV, Excel, JSON, or Parquet", type=["csv", "tsv", "xlsx", "xls", "json", "parquet"])
            if uploaded is None:
                return st.session_state.get("dataset")
            try:
                return read_uploaded(uploaded)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                return None
        if source == "Iris sample":
            return iris_sample()
        return product_analytics_sample()


def overview(dataset: pd.DataFrame) -> None:
    numeric = len(dataset.select_dtypes(include=np.number).columns)
    missing = int(dataset.isna().sum().sum())
    duplicate_rows = int(dataset.duplicated().sum())
    st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dataset):,}")
    c2.metric("Columns", f"{dataset.shape[1]:,}")
    c3.metric("Numeric", f"{numeric:,}")
    c4.metric("Missing cells", f"{missing:,}", delta=f"{duplicate_rows} duplicate rows")
    st.markdown("</div>", unsafe_allow_html=True)


def dataset_view(dataset: pd.DataFrame) -> None:
    st.subheader("Dataset Preview")
    st.dataframe(dataset.head(100), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Column Types")
        types = pd.DataFrame({"column": dataset.columns, "dtype": [str(dataset[c].dtype) for c in dataset.columns]})
        st.dataframe(types, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Missing Values")
        missing = dataset.isna().sum().reset_index()
        missing.columns = ["column", "missing"]
        st.dataframe(missing.sort_values("missing", ascending=False), use_container_width=True, hide_index=True)


def analytics_view(dataset: pd.DataFrame) -> None:
    st.subheader("Engine 2: Analytics and EDA")
    columns = list(dataset.columns)
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target column", ["None"] + columns, key="eda_target")
    timestamp = c2.selectbox("Timestamp column", ["None"] + columns, key="eda_time")
    user_id = c3.selectbox("User ID column", ["None"] + columns, key="eda_user")
    c4, c5, c6 = st.columns(3)
    event = c4.selectbox("Event column", ["None"] + columns, key="eda_event")
    segment = c5.multiselect("Segment columns", columns, key="eda_segment")
    funnel_steps = c6.text_input("Funnel steps", placeholder="launch,purchase,repeat")

    if st.button("Run analytics", type="primary", use_container_width=True):
        with st.spinner("Running profiling, correlations, anomalies, and available product analytics..."):
            report = run_eda(
                dataset,
                target_column=none_to_value(target),
                timestamp_column=none_to_value(timestamp),
                user_id_column=none_to_value(user_id),
                event_column=none_to_value(event),
                segment_columns=segment or None,
                funnel_steps=csv_list(funnel_steps),
                target_event=None,
            )
            st.session_state["eda_report"] = report

    report = st.session_state.get("eda_report")
    if not report:
        st.markdown("<div class='if-band'>Run analytics to generate quality scores, insights, and charts.</div>", unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality score", f"{report.data_quality_score:.0f}" if report.data_quality_score is not None else "N/A")
    c2.metric("Grade", report.data_quality_grade or "N/A")
    c3.metric("Insights", len(report.insights))
    c4.metric("Capabilities", len(report.results))

    if report.errors:
        with st.expander("Capability errors"):
            st.json(report.errors)

    st.subheader("Top Insights")
    for insight in report.insights[:12]:
        severity_class = f"if-{insight.severity}"
        st.markdown(
            f"<div class='if-band'><span class='{severity_class}'>{insight.severity.upper()}</span> "
            f"<strong>{insight.title}</strong><br>{insight.insight}<br><span class='if-status'>{insight.confidence_label} confidence</span> "
            f"{insight.action}</div>",
            unsafe_allow_html=True,
        )

    if report.charts:
        st.subheader("Visualizations")
        for chart in report.charts[:8]:
            try:
                st.plotly_chart(to_plotly(chart), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render {chart.title}: {exc}")


def automl_view(dataset: pd.DataFrame) -> None:
    st.subheader("Engine 1: AutoML")
    columns = list(dataset.columns)
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    target = c1.selectbox("Target column", columns, key="ml_target")
    task_choice = c2.selectbox("Task type", ["auto", "classification", "regression"], key="ml_task")
    metric_default = "accuracy" if detect_task_type(dataset[target]) == "classification" else "r2"
    metric = c3.text_input("Metric", value=metric_default, key="ml_metric")
    n_trials = c4.number_input("Trials", min_value=1, max_value=100, value=5, step=1)

    st.markdown(
        "<div class='if-band'>AutoML trains Random Forest, XGBoost, and LightGBM when available, tracks trials in MLflow, and registers the complete prediction pipeline.</div>",
        unsafe_allow_html=True,
    )
    if st.button("Train and register model", type="primary", use_container_width=True):
        with st.spinner("Training AutoML pipeline. This can take a few minutes."):
            result = run_automl(
                dataset=dataset,
                target_column=target,
                metric=metric,
                n_trials=int(n_trials),
                task_type=task_choice,
            )
            st.session_state["automl_result"] = result

    result = st.session_state.get("automl_result")
    if not result:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best score", f"{result['best_score']:.4f}")
    c2.metric("Model family", result["best_model_family"])
    c3.metric("Task", result["task_type"])
    c4.metric("Version", result.get("model_version") or "N/A")

    st.subheader("Registered Model")
    st.code(result["model_uri"])
    st.subheader("Test Metrics")
    st.json(result.get("test_metrics", {}))
    st.subheader("Best Parameters")
    st.json(result["best_params"])

    trials = pd.DataFrame(result.get("trials", []))
    if not trials.empty:
        st.subheader("Trial Leaderboard")
        st.dataframe(trials.sort_values("cv_score", ascending=False), use_container_width=True, hide_index=True)


def prediction_view(dataset: pd.DataFrame) -> None:
    st.subheader("Prediction")
    st.markdown("<div class='if-band'>Predictions use the registered full pipeline, so raw rows are transformed exactly like training data.</div>", unsafe_allow_html=True)
    sample_size = st.slider("Rows to predict from current dataset", min_value=1, max_value=min(20, len(dataset)), value=min(5, len(dataset)))
    preview = dataset.drop(columns=[st.session_state.get("ml_target")], errors="ignore").head(sample_size)
    edited = st.data_editor(preview, use_container_width=True, num_rows="dynamic")

    if st.button("Predict with registered model", type="primary", use_container_width=True):
        try:
            registry = AutoMLRegistry()
            predictions = registry.predict(edited)
            st.success("Prediction complete.")
            output = edited.copy()
            output["prediction"] = predictions
            st.dataframe(output, use_container_width=True, hide_index=True)
            st.code(registry.get_model_uri())
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def api_view() -> None:
    st.subheader("API Gateway")
    st.markdown("Run the API server from the project root:")
    st.code("uvicorn api.main:app --reload", language="bash")
    st.markdown("Then open:")
    st.code("http://127.0.0.1:8000/docs")
    st.markdown("Available routes:")
    routes = pd.DataFrame(
        [
            ["GET", "/health", "Service health"],
            ["POST", "/automl/train", "Train AutoML from JSON rows"],
            ["POST", "/automl/upload-train", "Train AutoML from CSV/Excel"],
            ["POST", "/automl/predict", "Predict with registered AutoML pipeline"],
            ["GET", "/automl/model-info", "Registered model metadata"],
            ["POST", "/analytics/analyze", "Run EDA from JSON rows"],
            ["POST", "/analytics/upload-analyze", "Run EDA from uploaded dataset"],
            ["POST", "/analytics/profile", "Quick data profile"],
            ["GET", "/analytics/capabilities", "List analytics capabilities"],
        ],
        columns=["Method", "Route", "Purpose"],
    )
    st.dataframe(routes, use_container_width=True, hide_index=True)


def read_uploaded(uploaded: Any) -> pd.DataFrame:
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith(".tsv"):
        return pd.read_csv(io.BytesIO(raw), sep="\t")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".json"):
        return pd.read_json(io.BytesIO(raw))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    raise ValueError("Unsupported file type.")


def iris_sample() -> pd.DataFrame:
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    frame["target"] = iris.target
    frame["species"] = [iris.target_names[i] for i in iris.target]
    return frame


def product_analytics_sample(seed: int = 42) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    base = pd.Timestamp("2024-06-01")
    rows: list[tuple[Any, ...]] = []
    for user in range(1000):
        region = gen.choice(["NA", "EU", "APAC", "LATAM"], p=[0.45, 0.3, 0.15, 0.10])
        signup = base + pd.Timedelta(days=float(gen.integers(0, 45)))
        base_rev = {"NA": 130.0, "EU": 95.0, "APAC": 70.0, "LATAM": 55.0}[region]
        rows.append((signup, user, float(gen.exponential(base_rev)), region, "launch"))
        if gen.random() < 0.6:
            rows.append((signup + pd.Timedelta(days=float(gen.exponential(2))), user, float(gen.exponential(base_rev)), region, "purchase"))
            if gen.random() < 0.35:
                rows.append((signup + pd.Timedelta(days=float(gen.exponential(12))), user, float(gen.exponential(base_rev)), region, "repeat"))
    frame = pd.DataFrame(rows, columns=["date", "user_id", "revenue", "region", "product"])
    spike = base + pd.Timedelta(days=15)
    frame.loc[frame["date"].between(spike, spike + pd.Timedelta(days=1)), "revenue"] *= 6
    return frame


def none_to_value(value: str) -> str | None:
    return None if value == "None" else value


def csv_list(value: str) -> list[str] | None:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


if __name__ == "__main__":
    main()
