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
  --if-ink: #111827;
  --if-muted: #667085;
  --if-border: #d6deea;
  --if-panel: #ffffff;
  --if-soft: #f7f9fd;
  --if-midnight: #0b1020;
  --if-navy: #111a31;
  --if-teal: #00a991;
  --if-blue: #3167ff;
  --if-violet: #7c3aed;
  --if-amber: #f59e0b;
  --if-red: #b42318;
  --if-shadow: 0 18px 50px rgba(20, 28, 45, .12);
}
.stApp {
  background:
    linear-gradient(180deg, #f7fbff 0%, #eef3f8 46%, #f9fbfd 100%);
  color: var(--if-ink);
}
.block-container {
  padding-top: 1rem;
  padding-bottom: 3rem;
  max-width: 1500px;
}
[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, #0b1020 0%, #111a31 58%, #172033 100%);
  border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * {
  color: #e5edf7;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  background: rgba(255,255,255,.08);
  border-color: rgba(255,255,255,.18);
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
  background: #ffffff;
  color: #111827;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 8px;
  padding: .32rem .5rem;
  margin-bottom: .25rem;
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
.if-shell {
  background: rgba(255,255,255,.78);
  border: 1px solid rgba(214,222,234,.85);
  border-radius: 8px;
  box-shadow: var(--if-shadow);
  overflow: hidden;
}
.if-hero {
  position: relative;
  background:
    linear-gradient(135deg, #0b1020 0%, #13213d 48%, #123b47 100%);
  border-radius: 8px;
  color: #ffffff;
  padding: 1.35rem 1.5rem;
  margin-bottom: 1rem;
  overflow: hidden;
}
.if-hero:before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,.07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px);
  background-size: 34px 34px;
  opacity: .32;
}
.if-hero > * {
  position: relative;
  z-index: 1;
}
.if-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: .95rem;
}
.if-brand {
  display: flex;
  align-items: center;
  gap: .75rem;
}
.if-mark {
  width: 42px;
  height: 42px;
  border-radius: 8px;
  background: linear-gradient(135deg, #00a991 0%, #3167ff 100%);
  display: grid;
  place-items: center;
  font-weight: 900;
  letter-spacing: 0;
  color: #ffffff;
  box-shadow: 0 12px 32px rgba(0,169,145,.28);
}
.if-title-row h1 {
  margin: 0;
  font-size: 2.05rem;
  color: #ffffff;
}
.if-title-row span, .if-hero p {
  color: #c7d7ea;
  font-size: .95rem;
}
.if-pill {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  border-radius: 999px;
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.18);
  color: #e8f1ff;
  padding: .42rem .72rem;
  font-size: .82rem;
  white-space: nowrap;
}
.if-hero-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .75rem;
}
.if-hero-card {
  background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 8px;
  padding: .85rem;
  min-height: 92px;
}
.if-hero-card strong {
  display: block;
  color: #ffffff;
  margin-bottom: .35rem;
}
.if-hero-card span {
  color: #c7d7ea;
  font-size: .84rem;
}
.if-sidebar-brand {
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.08);
  border-radius: 8px;
  padding: .85rem;
  margin: .35rem 0 1rem;
}
.if-sidebar-brand strong {
  display: block;
  font-size: 1.08rem;
  color: #ffffff;
}
.if-sidebar-brand span {
  color: #b7c5d7;
  font-size: .82rem;
}
.if-band {
  background: var(--if-soft);
  border: 1px solid var(--if-border);
  border-radius: 8px;
  color: var(--if-ink);
  padding: 1rem;
  margin: .65rem 0 1rem;
  box-shadow: 0 10px 28px rgba(20, 28, 45, .06);
}
.if-section-title {
  margin: 1.35rem 0 .8rem;
}
.if-section-title span {
  color: var(--if-blue);
  font-size: .76rem;
  font-weight: 800;
  text-transform: uppercase;
}
.if-section-title h2 {
  margin: .15rem 0 .2rem;
  font-size: 1.55rem;
}
.if-section-title p {
  margin: 0;
  color: var(--if-muted);
}
.if-stat-card {
  position: relative;
  background: #ffffff;
  border: 1px solid var(--if-border);
  border-radius: 8px;
  padding: 1rem;
  min-height: 118px;
  overflow: hidden;
  box-shadow: 0 14px 35px rgba(20, 28, 45, .08);
}
.if-stat-card:before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--if-blue), var(--if-teal));
}
.if-stat-label {
  color: var(--if-muted);
  font-size: .78rem;
  text-transform: uppercase;
  font-weight: 800;
}
.if-stat-value {
  font-size: 2rem;
  line-height: 1.15;
  font-weight: 850;
  color: var(--if-ink);
  margin-top: .45rem;
}
.if-stat-note {
  color: var(--if-muted);
  font-size: .84rem;
  margin-top: .38rem;
}
.if-engine-strip {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .85rem;
  margin: .9rem 0 1rem;
}
.if-engine-card {
  border-radius: 8px;
  padding: 1rem;
  background: #ffffff;
  border: 1px solid var(--if-border);
  box-shadow: 0 14px 35px rgba(20, 28, 45, .08);
}
.if-engine-card strong {
  color: var(--if-ink);
  font-size: 1rem;
}
.if-engine-card p {
  margin: .35rem 0 0;
  color: var(--if-muted);
  font-size: .88rem;
}
.if-empty {
  background:
    linear-gradient(135deg, #ffffff 0%, #f5f8ff 100%);
  border: 1px solid var(--if-border);
  border-radius: 8px;
  padding: 1.25rem;
  box-shadow: var(--if-shadow);
}
.if-empty h2 {
  margin: 0 0 .35rem;
}
.if-empty-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .8rem;
  margin-top: 1rem;
}
.if-step {
  border-radius: 8px;
  border: 1px solid var(--if-border);
  background: #ffffff;
  padding: .85rem;
}
.if-step b {
  color: var(--if-blue);
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
  min-height: 2.8rem;
  color: var(--if-ink);
  border-radius: 8px 8px 0 0;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  border-bottom-color: var(--if-teal);
  color: var(--if-teal);
}
[data-testid="stAlert"] {
  border-radius: 8px;
  border: 1px solid #c7d2fe;
  background: #eef4ff;
  color: var(--if-ink);
}
[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(90deg, #0f766e 0%, #3167ff 100%);
  border: 0;
  color: #ffffff;
  min-height: 3rem;
  border-radius: 8px;
  box-shadow: 0 16px 30px rgba(49,103,255,.18);
}
[data-testid="stButton"] button[kind="primary"] * {
  color: #ffffff;
}
@media (max-width: 900px) {
  .if-hero-grid, .if-engine-strip, .if-empty-grid {
    grid-template-columns: 1fr;
  }
  .if-title-row {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
"""


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    dataset = sidebar_dataset()
    if dataset is None:
        empty_state()
        return

    st.session_state["dataset"] = dataset
    overview(dataset)
    engine_strip()

    dataset_tab, analytics_tab, automl_tab, predict_tab, api_tab = st.tabs(
        ["01 Dataset", "02 Analytics", "03 AutoML", "04 Prediction", "05 API"]
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
        st.markdown(
            "<div class='if-sidebar-brand'><strong>IntelliFlow</strong><span>Dataset in. Intelligence out.</span></div>",
            unsafe_allow_html=True,
        )
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


def render_header() -> None:
    st.markdown(
        """
        <section class="if-hero">
          <div class="if-title-row">
            <div class="if-brand">
              <div class="if-mark">IF</div>
              <div>
                <h1>IntelliFlow Command Center</h1>
                <span>Unified AutoML, EDA, model registry, and prediction workspace</span>
              </div>
            </div>
            <div class="if-pill">Engine 1 + Engine 2 integrated</div>
          </div>
          <div class="if-hero-grid">
            <div class="if-hero-card"><strong>Analyze</strong><span>Profile data quality, correlations, anomalies, trends, and product behavior.</span></div>
            <div class="if-hero-card"><strong>Train</strong><span>Run preprocessing, feature engineering, HPO, tracking, and registry in one flow.</span></div>
            <div class="if-hero-card"><strong>Deploy</strong><span>Use the registered pipeline for consistent raw-row predictions.</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def empty_state() -> None:
    st.markdown(
        """
        <div class="if-empty">
          <h2>Start with a dataset</h2>
          <p>Upload your own file or choose a sample from the left panel. The same dataset powers Engine 2 analytics and Engine 1 AutoML, so you can inspect the data before training.</p>
          <div class="if-empty-grid">
            <div class="if-step"><b>01</b><br><strong>Load</strong><br><span>CSV, Excel, JSON, Parquet, or built-in samples.</span></div>
            <div class="if-step"><b>02</b><br><strong>Understand</strong><br><span>Run EDA, quality scoring, insights, and visualizations.</span></div>
            <div class="if-step"><b>03</b><br><strong>Model</strong><br><span>Train and register a complete AutoML prediction pipeline.</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="if-stat-card">
          <div class="if-stat-label">{label}</div>
          <div class="if-stat-value">{value}</div>
          <div class="if-stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="if-section-title">
          <span>{kicker}</span>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview(dataset: pd.DataFrame) -> None:
    numeric = len(dataset.select_dtypes(include=np.number).columns)
    missing = int(dataset.isna().sum().sum())
    duplicate_rows = int(dataset.duplicated().sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Rows", f"{len(dataset):,}", "Records available for analysis")
    with c2:
        stat_card("Columns", f"{dataset.shape[1]:,}", "Fields detected in the workspace")
    with c3:
        stat_card("Numeric", f"{numeric:,}", "Columns ready for scoring and charts")
    with c4:
        stat_card("Missing cells", f"{missing:,}", f"{duplicate_rows} duplicate rows")


def engine_strip() -> None:
    st.markdown(
        """
        <div class="if-engine-strip">
          <div class="if-engine-card"><strong>Engine 2: Analytics Intelligence</strong><p>Profiles the dataset, surfaces quality issues, finds correlations and anomalies, and builds visual insight cards before modeling.</p></div>
          <div class="if-engine-card"><strong>Engine 1: AutoML Factory</strong><p>Uses the same dataset to preprocess, engineer features, optimize models, track MLflow runs, and register a reusable prediction pipeline.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_view(dataset: pd.DataFrame) -> None:
    section_title("Workspace", "Dataset Preview", "Inspect the exact table that flows into analytics, training, and prediction.")
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
    section_title("Engine 2", "Analytics and EDA", "Generate a fast intelligence layer before you decide what to train.")
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
    section_title("Engine 1", "AutoML Factory", "Train, track, compare, and register a production-ready prediction pipeline.")
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
    section_title("Serving", "Prediction", "Score raw rows through the registered full pipeline.")
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
    section_title("Integration", "API Gateway", "Use the FastAPI routes when the UI needs to connect with another system.")
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
