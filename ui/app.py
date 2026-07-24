"""Streamlit dashboard for IntelliFlow Engines 1, 2, and 3.

This module is orchestration only: it loads a dataset, calls the engines, and
hands the results to :mod:`ui.components` for rendering. All styling lives in
:mod:`ui.theme`, and the engines themselves are untouched -- chart theming is
applied by post-processing the figure returned by ``to_plotly``.
"""

from __future__ import annotations

import io
import sys
import uuid
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from engines.agents import get_agent_config, run_agent_query
from engines.agents.llm import LLMError
from engines.analytics import run_eda
from engines.analytics.visualization import to_plotly
from engines.automl.pipeline import detect_task_type, run_automl
from engines.automl.registry import AutoMLRegistry
from ui import components as ui
from ui import theme

theme.page_config()

MAX_INSIGHTS = 12
MAX_CHARTS = 8


def main() -> None:
    active_theme = theme.current_theme()
    theme.inject(active_theme)

    dataset = sidebar()
    render_header(dataset)

    if dataset is None:
        ui.empty_state()
        return

    st.session_state["dataset"] = dataset
    forget_stale_results(dataset)
    overview(dataset)
    ui.engine_cards()

    dataset_tab, analytics_tab, automl_tab, predict_tab, agents_tab, api_tab = st.tabs(
        ["Dataset", "Analytics", "AutoML", "Prediction", "Agents", "API"]
    )
    with dataset_tab:
        dataset_view(dataset)
    with analytics_tab:
        analytics_view(dataset, active_theme)
    with automl_tab:
        automl_view(dataset)
    with predict_tab:
        prediction_view(dataset)
    with agents_tab:
        agents_view(dataset, active_theme)
    with api_tab:
        api_view()


# ------------------------------------------------------------------- sidebar
def sidebar() -> pd.DataFrame | None:
    """Brand block, appearance toggle, and the dataset source picker."""

    with st.sidebar:
        ui.render(
            '<div class="if-brand" style="margin:.35rem 0 1.1rem">',
            '<div class="if-mark">IF</div>',
            '<div><strong style="display:block;font-size:1rem;color:var(--if-ink)">IntelliFlow</strong>',
            '<span style="color:var(--if-muted);font-size:.8rem">Dataset in. Intelligence out.</span>',
            "</div></div>",
        )

        st.radio(
            "Appearance",
            options=list(theme.THEMES),
            format_func=str.capitalize,
            horizontal=True,
            key="theme",
        )

        st.markdown("### Dataset")
        source = st.radio(
            "Source",
            ["Upload file", "Iris sample", "Product analytics sample"],
            key="data_source",
            label_visibility="collapsed",
        )

        dataset: pd.DataFrame | None
        if source == "Upload file":
            uploaded = st.file_uploader(
                "CSV, Excel, JSON, or Parquet",
                type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
            )
            if uploaded is None:
                dataset = st.session_state.get("dataset")
            else:
                try:
                    dataset = read_uploaded(uploaded)
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")
                    dataset = None
        elif source == "Iris sample":
            dataset = iris_sample()
        else:
            dataset = product_analytics_sample()

        if dataset is not None:
            ui.render(
                '<div class="if-callout" style="margin-top:1rem">',
                f"<div><strong>{dataset.shape[0]:,}</strong> rows &middot; "
                f"<strong>{dataset.shape[1]:,}</strong> columns loaded</div></div>",
            )
        return dataset


def forget_stale_results(dataset: pd.DataFrame) -> None:
    """Drop results that describe a *different* dataset.

    Every result panel is captioned as if it belongs to the table on screen, so
    a report computed from the previous dataset must not survive a switch.
    """

    signature = (tuple(str(column) for column in dataset.columns), dataset.shape)
    if st.session_state.get("dataset_signature") == signature:
        return
    st.session_state["dataset_signature"] = signature
    for key in ("eda_report", "automl_result", "agent_result", "agent_history"):
        st.session_state.pop(key, None)


def render_header(dataset: pd.DataFrame | None) -> None:
    meta: list[tuple[str, str | None]] = [
        ("Analytics", "accent"),
        ("AutoML", "accent"),
        ("Agents", "accent"),
    ]
    if dataset is not None:
        meta.append((f"{len(dataset):,} rows loaded", None))
    ui.hero(
        "IntelliFlow Command Center",
        "One workspace for exploratory analytics, automated modelling, and an agent crew that uses both.",
        meta,
    )


def overview(dataset: pd.DataFrame) -> None:
    numeric = len(dataset.select_dtypes(include=np.number).columns)
    missing = int(dataset.isna().sum().sum())
    duplicates = int(dataset.duplicated().sum())
    cells = int(dataset.size) or 1
    ui.stat_row(
        [
            ui.stat_card("Rows", f"{len(dataset):,}", "Records available for analysis"),
            ui.stat_card("Columns", f"{dataset.shape[1]:,}", f"{numeric:,} numeric, {dataset.shape[1] - numeric:,} other"),
            ui.stat_card("Missing cells", f"{missing:,}", f"{missing / cells:.1%} of the table"),
            ui.stat_card("Duplicate rows", f"{duplicates:,}", "Exact repeats across all columns"),
        ]
    )


# ------------------------------------------------------------------- dataset
def dataset_view(dataset: pd.DataFrame) -> None:
    ui.section_header(
        "Workspace",
        "Dataset preview",
        "The exact table that flows into analytics, training, prediction, and the agent crew.",
    )
    st.dataframe(dataset.head(100), width="stretch", height=380)
    ui.note(f"Showing the first {min(100, len(dataset)):,} of {len(dataset):,} rows.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Column types")
        ui.render(
            '<div class="if-kv"><div class="if-kv-title">Detected dtypes</div>',
            *[
                '<div class="if-kv-row">'
                f'<span class="if-kv-key">{ui.esc(column)}</span>'
                f'<span class="if-kv-val">{ui.chip(str(dataset[column].dtype), tone=_dtype_tone(dataset[column]))}</span>'
                "</div>"
                for column in dataset.columns
            ],
            "</div>",
        )
    with right:
        st.markdown("#### Missing values")
        missing = dataset.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            ui.callout(
                "<strong>No missing values.</strong> Every cell in the table is populated.",
                tone="success",
                icon="✓",
            )
        else:
            ui.kv_panel(
                "Cells missing per column",
                {
                    column: f"{int(count):,}  ({count / len(dataset):.1%})"
                    for column, count in missing.items()
                },
            )
        st.markdown("#### Unique values")
        ui.kv_panel(
            "Distinct values per column",
            {column: int(dataset[column].nunique(dropna=True)) for column in dataset.columns},
        )


def _dtype_tone(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "warning"
    if pd.api.types.is_numeric_dtype(series):
        return "accent"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "success"
    return "default"


# ----------------------------------------------------------------- analytics
def analytics_view(dataset: pd.DataFrame, active_theme: str) -> None:
    ui.section_header(
        "Engine 2",
        "Analytics and EDA",
        "Profiling, correlations, anomalies, and product behaviour — a fast intelligence "
        "layer before you decide what to train.",
    )

    columns = list(dataset.columns)
    with st.container(border=True):
        st.markdown("**Configure the run** — every column you map unlocks another capability.")
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Target column", ["None"] + columns, key="eda_target")
        timestamp = c2.selectbox("Timestamp column", ["None"] + columns, key="eda_time")
        user_id = c3.selectbox("User ID column", ["None"] + columns, key="eda_user")
        c4, c5, c6 = st.columns(3)
        event = c4.selectbox("Event column", ["None"] + columns, key="eda_event")
        segment = c5.multiselect("Segment columns", columns, key="eda_segment")
        funnel_steps = c6.text_input("Funnel steps", placeholder="launch,purchase,repeat")
        run = st.button("Run analytics", type="primary", width="stretch")

    if run:
        with st.spinner("Profiling, correlating, and hunting anomalies…"):
            st.session_state["eda_report"] = run_eda(
                dataset,
                target_column=none_to_value(target),
                timestamp_column=none_to_value(timestamp),
                user_id_column=none_to_value(user_id),
                event_column=none_to_value(event),
                segment_columns=segment or None,
                funnel_steps=csv_list(funnel_steps),
                target_event=None,
            )

    report = st.session_state.get("eda_report")
    if not report:
        ui.callout(
            "Run analytics to generate a quality score, ranked insights, and themed charts.",
            icon="→",
        )
        return

    critical = sum(1 for insight in report.insights if insight.severity == "critical")
    warnings = sum(1 for insight in report.insights if insight.severity == "warning")

    score_col, stats_col = st.columns([1.05, 1.95])
    with score_col:
        ui.score_display(
            report.data_quality_score,
            report.data_quality_grade,
            f"Rolled up across {len(report.results)} capabilities.",
        )
    with stats_col:
        ui.stat_row(
            [
                ui.stat_card("Insights", f"{len(report.insights):,}", "Ranked most urgent first"),
                ui.stat_card("Critical", f"{critical:,}", "Need attention before modelling"),
                ui.stat_card("Warnings", f"{warnings:,}", "Worth reviewing"),
            ],
            columns=3,
        )

    ui.section_header("Findings", "Top insights", "Ordered by severity, then confidence.")
    ui.insight_grid(report.insights[:MAX_INSIGHTS])
    if len(report.insights) > MAX_INSIGHTS:
        ui.note(f"Showing {MAX_INSIGHTS} of {len(report.insights):,} insights.")

    if report.charts:
        ui.section_header("Evidence", "Visualizations", "Charts emitted by the capabilities that ran.")
        render_charts(report.charts[:MAX_CHARTS], active_theme)

    if report.errors:
        with st.expander(f"Capability notes ({len(report.errors)})"):
            ui.kv_panel("Capabilities that could not complete", report.errors)


def render_charts(charts: list[Any], active_theme: str, columns: int = 2) -> None:
    """Two-up chart grid; each figure is re-skinned to the active theme."""

    for row_start in range(0, len(charts), columns):
        row = charts[row_start : row_start + columns]
        for slot, spec in zip(st.columns(len(row)), row):
            with slot:
                try:
                    ui.chart(theme.style_figure(to_plotly(spec), active_theme))
                except Exception as exc:  # one bad spec must not sink the grid
                    ui.callout(
                        f"Could not render <strong>{ui.esc(spec.title)}</strong>: {ui.esc(exc)}",
                        tone="warning",
                        icon="!",
                    )


# -------------------------------------------------------------------- automl
def automl_view(dataset: pd.DataFrame) -> None:
    ui.section_header(
        "Engine 1",
        "AutoML factory",
        "Preprocess, engineer features, optimise, track in MLflow, and register a servable pipeline.",
    )

    columns = list(dataset.columns)
    with st.container(border=True):
        st.markdown("**Training run** — the registered pipeline is what the Prediction tab serves.")
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        target = c1.selectbox("Target column", columns, key="ml_target")
        task_choice = c2.selectbox("Task type", ["auto", "classification", "regression"], key="ml_task")
        metric_default = "accuracy" if detect_task_type(dataset[target]) == "classification" else "r2"
        # Keying the metric on the target makes the default follow the detected
        # task instead of stranding "r2" on a classification run.
        metric = c3.text_input("Metric", value=metric_default, key=f"ml_metric::{target}")
        n_trials = c4.number_input("Trials", min_value=1, max_value=100, value=5, step=1)
        run = st.button("Train and register model", type="primary", width="stretch")

    ui.callout(
        "Random Forest, XGBoost, and LightGBM compete when installed. Every trial is tracked "
        "in MLflow and the winning <strong>full pipeline</strong> is registered, so raw rows can "
        "be scored later without repeating any preprocessing.",
        icon="ℹ",
    )

    if run:
        with st.spinner("Training the AutoML pipeline — this can take a few minutes…"):
            st.session_state["automl_result"] = run_automl(
                dataset=dataset,
                target_column=target,
                metric=metric,
                n_trials=int(n_trials),
                task_type=task_choice,
            )

    result = st.session_state.get("automl_result")
    if not result:
        return

    headline, side = st.columns([1, 2])
    with headline:
        ui.render(
            '<div class="if-card" style="border-top:2px solid var(--if-accent)">',
            f'<div class="if-stat-label">Best {ui.esc(result.get("metric", "score"))}</div>',
            f'<div class="if-stat-value" style="font-size:2.6rem">{result["best_score"]:.4f}</div>',
            f'<div class="if-stat-note">Cross-validated on the {ui.esc(result["task_type"])} task</div>',
            "</div>",
        )
    with side:
        ui.stat_row(
            [
                ui.stat_card("Model family", str(result["best_model_family"]).replace("_", " ").title(), "Winner of the search"),
                ui.stat_card("Target", str(result.get("target_column", "—")), "Column the pipeline predicts"),
                ui.stat_card("Version", str(result.get("model_version") or "—"), f"Registered as {result.get('model_name', 'model')}"),
            ],
            columns=3,
        )

    ui.callout(
        "<strong>Registered model URI</strong> — load it anywhere with <code>mlflow.pyfunc.load_model</code>.",
        tone="success",
        icon="✓",
        code=str(result["model_uri"]),
    )

    left, right = st.columns(2)
    with left:
        ui.kv_panel("Test metrics", result.get("test_metrics") or {}, empty="No hold-out metrics recorded.")
    with right:
        ui.kv_panel("Best hyper-parameters", result.get("best_params") or {}, empty="The winner used its defaults.")

    trials = pd.DataFrame(result.get("trials", []))
    if not trials.empty:
        ui.section_header("Search", "Trial leaderboard", "Every configuration the optimiser evaluated.")
        table = trials.sort_values("cv_score", ascending=False).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
        display = table[["rank", "model_family", "cv_score", "params"]].copy()
        display["params"] = display["params"].map(_compact_params)
        low, high = _score_bounds(display["cv_score"])
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "model_family": st.column_config.TextColumn("Model family"),
                "cv_score": st.column_config.ProgressColumn(
                    "CV score", format="%.4f", min_value=low, max_value=high
                ),
                "params": st.column_config.TextColumn("Parameters", width="large"),
            },
        )


def _score_bounds(scores: pd.Series) -> tuple[float, float]:
    """Progress-bar bounds that survive negative, NaN, or all-equal scores."""

    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    if numeric.empty:
        return 0.0, 1.0
    low = min(0.0, float(numeric.min()))
    high = float(numeric.max())
    return (low, high) if high > low else (low, low + 1.0)


def _compact_params(params: Any) -> str:
    if not isinstance(params, dict):
        return str(params)
    return ", ".join(f"{key}={value}" for key, value in params.items())


# ---------------------------------------------------------------- prediction
def prediction_view(dataset: pd.DataFrame) -> None:
    ui.section_header(
        "Serving",
        "Prediction",
        "Score raw rows through the registered full pipeline — the same transforms as training.",
    )

    with st.container(border=True):
        st.markdown("**Rows to score** — edit any cell, or add rows with the ✚ control.")
        sample_size = st.slider(
            "Rows from the current dataset",
            min_value=1,
            max_value=min(20, len(dataset)),
            value=min(5, len(dataset)),
        )
        preview = dataset.drop(columns=[st.session_state.get("ml_target")], errors="ignore").head(sample_size)
        edited = st.data_editor(preview, width="stretch", num_rows="dynamic")
        predict = st.button("Predict with registered model", type="primary", width="stretch")

    if not predict:
        return

    try:
        registry = AutoMLRegistry()
        predictions = registry.predict(edited)
    except Exception as exc:
        ui.callout(
            f"<strong>Prediction failed.</strong> {ui.esc(exc)}",
            tone="danger",
            icon="✕",
        )
        return

    # A dataset can legitimately carry its own "prediction" column; the model's
    # output replaces it rather than colliding with it.
    output = edited.drop(columns=["prediction"], errors="ignore").copy()
    output.insert(0, "prediction", predictions)
    ui.callout(
        f"<strong>{len(output):,} row(s) scored</strong> through the registered pipeline.",
        tone="success",
        icon="✓",
    )
    st.dataframe(
        output,
        width="stretch",
        hide_index=True,
        column_config={"prediction": st.column_config.TextColumn("▸ Prediction", width="medium")},
    )
    try:
        ui.callout("<strong>Served by</strong>", tone="accent", icon="◆", code=registry.get_model_uri())
    except Exception:  # the URI is a nicety, never a failure mode
        pass


# -------------------------------------------------------------------- agents
def agents_view(dataset: pd.DataFrame, active_theme: str) -> None:
    ui.section_header(
        "Engine 3",
        "Agent orchestration",
        "Ask in plain English. The crew plans, analyses the loaded dataset, and can trigger "
        "AutoML and Analytics for you.",
    )

    config = get_agent_config()
    if not config.has_credentials:
        ui.callout(
            "<strong>No LLM key detected.</strong> Add <code>OLLAMA_API_KEY</code> to a "
            f"<code>.env</code> file in the project root — model <strong>{ui.esc(config.model)}</strong> "
            f"via <strong>{ui.esc(config.host)}</strong> — then rerun. Every other tab works without it.",
            tone="warning",
            icon="!",
        )

    st.session_state.setdefault("agent_session", uuid.uuid4().hex[:12])
    st.session_state.setdefault("agent_history", [])

    ui.callout(
        "The crew — <strong>Planner, Data Analyst, ML Engineer, Visualizer, Researcher, "
        "Synthesizer</strong> — shares this exact dataset. Try: <em>“Which features best "
        "predict the target, and how accurate is a model?”</em>",
        icon="◆",
    )

    for message in st.session_state["agent_history"]:
        ui.chat_message(message["role"], message["text"], message.get("agents", ()))

    with st.container(border=True):
        query = st.text_area(
            "Your question",
            key="agent_query",
            placeholder="e.g. Find anomalies in revenue and tell me what drives them.",
        )
        c1, c2 = st.columns([2, 1])
        ask = c1.button("Ask the crew", type="primary", width="stretch")
        max_steps = int(
            c2.number_input("Max tool steps", min_value=1, max_value=20, value=int(config.max_steps), step=1)
        )

    if ask and query.strip():
        with st.spinner("The crew is planning, analysing, and (if needed) training a model…"):
            try:
                result = run_agent_query(
                    query.strip(),
                    dataset,
                    session_id=st.session_state["agent_session"],
                    max_steps=max_steps,
                )
            except (ValueError, TypeError) as exc:
                ui.callout(f"<strong>Invalid request.</strong> {ui.esc(exc)}", tone="danger", icon="✕")
                return
            except LLMError as exc:
                ui.callout(f"<strong>LLM backbone error.</strong> {ui.esc(exc)}", tone="danger", icon="✕")
                return
            except Exception as exc:  # pragma: no cover - surface anything else
                ui.callout(f"<strong>Agent run failed.</strong> {ui.esc(exc)}", tone="danger", icon="✕")
                return

        st.session_state["agent_result"] = result
        st.session_state["agent_history"].extend(
            [
                {"role": "user", "text": result.query},
                {"role": "assistant", "text": result.answer or "_No answer produced._",
                 "agents": list(result.agents_used)},
            ]
        )
        st.rerun()

    result = st.session_state.get("agent_result")
    if not result:
        return

    if result.model_endpoint:
        ui.callout(
            "<strong>A model was trained during this run.</strong> Prediction endpoint:",
            tone="success",
            icon="✓",
            code=str(result.model_endpoint),
        )

    if result.insights:
        ui.section_header("Findings", "Insights the crew gathered", "Same severity system as Engine 2.")
        ui.insight_grid(result.insights[:6])

    if result.charts:
        ui.section_header("Evidence", "Charts the crew produced", "")
        render_charts(list(result.charts)[:6], active_theme)

    with st.expander("Agent trace — what each step did", expanded=False):
        ui.trace_timeline(result.trace)
        if result.tools_used:
            ui.render(
                '<div style="margin-top:.6rem">',
                ui.chips(result.tools_used, tone="accent"),
                "</div>",
            )

    if result.errors:
        with st.expander(f"Run notes / recovered errors ({len(result.errors)})"):
            ui.kv_panel("Steps that reported a problem", result.errors)


# ----------------------------------------------------------------------- api
API_ROUTES = (
    ("Platform", (("GET", "/health", "Service health"),)),
    (
        "Engine 1 — AutoML",
        (
            ("POST", "/automl/train", "Train from JSON rows"),
            ("POST", "/automl/upload-train", "Train from CSV/Excel"),
            ("POST", "/automl/predict", "Predict with the registered pipeline"),
            ("GET", "/automl/model-info", "Registered model metadata"),
        ),
    ),
    (
        "Engine 2 — Analytics",
        (
            ("POST", "/analytics/analyze", "Run EDA from JSON rows"),
            ("POST", "/analytics/upload-analyze", "Run EDA from an uploaded dataset"),
            ("POST", "/analytics/profile", "Quick data profile"),
            ("GET", "/analytics/capabilities", "List analytics capabilities"),
        ),
    ),
    (
        "Engine 3 — Agents",
        (
            ("POST", "/agents/query", "Ask the crew from JSON rows"),
            ("POST", "/agents/upload-query", "Ask the crew from an uploaded dataset"),
            ("GET", "/agents/history", "Past queries for a session"),
            ("GET", "/agents/capabilities", "List agents, tools, and LLM config"),
        ),
    ),
)


def api_view() -> None:
    ui.section_header(
        "Integration",
        "API gateway",
        "Every engine is reachable over HTTP when another system needs the same results.",
    )
    left, right = st.columns(2)
    with left:
        ui.callout("<strong>Start the server</strong> from the project root", icon="1", code="uvicorn api.main:app --reload")
    with right:
        ui.callout("<strong>Then open the interactive docs</strong>", icon="2", code="http://127.0.0.1:8000/docs")

    for group, routes in API_ROUTES:
        ui.route_list(group, routes)


# ------------------------------------------------------------------- loaders
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
