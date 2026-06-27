"""Report serialisation: JSON, an interactive HTML dashboard, CSV summaries, images.

This module is deliberately decoupled from the analysis code -- it consumes plain
dictionaries (the ``to_dict()`` output of any result) plus a list of
:class:`ChartSpec` objects, so it can render a report without importing every
capability. :class:`engines.analytics.pipeline.EDAReport` is the main caller.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .base import ChartSpec, json_safe
from .visualization import export_chart_safe, figure_to_html_div

_SEVERITY_COLOR = {
    "critical": "#d62728",
    "warning": "#ff7f0e",
    "info": "#1f77b4",
    "ok": "#2ca02c",
}
_CATEGORY_TITLES = {
    "profiling": "Data Profiling",
    "correlation": "Correlation & Feature Intelligence",
    "funnel": "Funnel Analysis",
    "retention": "Cohort Retention",
    "events": "Event Stream Analytics",
    "anomaly": "Anomaly Detection",
    "recommendations": "Feature Warnings & Recommendations",
    "general": "General",
}


def write_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> str:
    """Serialise a report dictionary to a JSON file (NaN/inf safe)."""

    path = Path(path)
    path.write_text(json.dumps(json_safe(data), indent=indent), encoding="utf-8")
    return str(path)


def export_charts(charts: Iterable[ChartSpec], out_dir: str | Path, fmt: str = "png") -> list[str]:
    """Render each chart spec to ``out_dir`` as a static image. Returns paths."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for i, spec in enumerate(charts):
        slug = _slug(spec.title) or f"chart_{i}"
        target = out / f"{i:02d}_{slug}.{fmt}"
        if export_chart_safe(spec, str(target), fmt):
            written.append(str(target))
    return written


def write_csv_summaries(sections: dict[str, Any], out_dir: str | Path) -> list[str]:
    """Write per-capability tabular summaries as CSV files for downstream tools."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def dump(name: str, records: Any) -> None:
        if not records:
            return
        try:
            frame = pd.DataFrame(records)
        except Exception:
            return
        if frame.empty:
            return
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(str(path))

    profiling = sections.get("profiling") or {}
    dump("profiling_columns", profiling.get("columns"))

    corr = sections.get("correlation") or {}
    dump("target_ranking", corr.get("target_ranking"))
    dump("vif", corr.get("vif"))
    dump("high_correlations", corr.get("high_correlations"))

    dump("funnel_steps", (sections.get("funnel") or {}).get("steps"))

    retention = sections.get("retention") or {}
    dump("retention_curve", retention.get("curve"))
    dump("churn_at_risk", (retention.get("churn") or {}).get("top_at_risk"))

    events = sections.get("events") or {}
    dump("event_timeseries", events.get("timeseries"))
    dump("event_sequences", events.get("top_sequences"))

    anomaly = sections.get("anomaly") or {}
    dump("anomaly_univariate", anomaly.get("univariate"))
    dump("anomaly_timeseries", anomaly.get("timeseries_anomalies"))

    rec = sections.get("recommendations") or {}
    dump("encoding_recommendations", rec.get("encoding"))
    dump("leakage_warnings", rec.get("leakage_warnings"))
    dump("constant_features", rec.get("constant_features"))

    return written


def render_html(
    *,
    title: str,
    summary: dict[str, Any],
    insights: list[dict[str, Any]],
    charts: list[ChartSpec],
) -> str:
    """Build a self-contained interactive HTML dashboard string."""

    insight_charts = {id(i): i.get("visualization") for i in insights}
    chart_divs = "\n".join(_chart_card(spec, include_js=(idx == 0)) for idx, spec in enumerate(charts))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for ins in insights:
        grouped.setdefault(ins.get("category", "general"), []).append(ins)

    sections_html = []
    for category, items in grouped.items():
        cards = "\n".join(_insight_card(i) for i in items)
        sections_html.append(
            f"<section class='cat'><h2>{html.escape(_CATEGORY_TITLES.get(category, category.title()))}"
            f" <span class='count'>{len(items)}</span></h2>{cards}</section>"
        )

    score = summary.get("data_quality_score")
    grade = summary.get("data_quality_grade", "")
    score_html = (
        f"<div class='score'><div class='score-num'>{score:.0f}</div>"
        f"<div class='score-grade'>Grade {html.escape(str(grade))}</div>"
        f"<div class='score-label'>Data Quality</div></div>"
        if isinstance(score, (int, float))
        else ""
    )
    chips = "".join(
        f"<div class='chip'><span>{html.escape(str(v))}</span><label>{html.escape(k)}</label></div>"
        for k, v in (summary.get("highlights") or {}).items()
    )

    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        style=_CSS,
        score=score_html,
        chips=chips,
        generated=summary.get("generated_at", ""),
        rows=summary.get("n_rows", "?"),
        cols=summary.get("n_columns", "?"),
        insights=" ".join(sections_html),
        charts=chart_divs or "<p class='muted'>No charts available.</p>",
    )


def write_html(path: str | Path, **kwargs: Any) -> str:
    """Render and write the HTML dashboard. Accepts the same args as :func:`render_html`."""

    path = Path(path)
    path.write_text(render_html(**kwargs), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------- helpers
def _insight_card(insight: dict[str, Any]) -> str:
    severity = insight.get("severity", "info")
    color = _SEVERITY_COLOR.get(severity, "#1f77b4")
    metric = insight.get("metric") or {}
    metric_html = ""
    if isinstance(metric, dict) and metric:
        scalars = {k: v for k, v in metric.items() if isinstance(v, (str, int, float, bool))}
        if scalars:
            metric_html = "<div class='metric'>" + "".join(
                f"<code>{html.escape(str(k))} = {html.escape(str(v))}</code>" for k, v in list(scalars.items())[:6]
            ) + "</div>"
    viz = insight.get("visualization")
    viz_html = _chart_card(_spec_from_dict(viz)) if viz else ""
    return (
        f"<div class='card' style='border-left-color:{color}'>"
        f"<div class='card-head'><span class='sev' style='background:{color}'>{html.escape(severity)}</span>"
        f"<strong>{html.escape(insight.get('title', ''))}</strong>"
        f"<span class='conf'>conf: {html.escape(str(insight.get('confidence_label', '')))}</span></div>"
        f"<p class='ins'>{html.escape(insight.get('insight', ''))}</p>"
        f"{metric_html}"
        f"<p class='act'>&rarr; {html.escape(insight.get('action', ''))}</p>"
        f"{viz_html}"
        f"</div>"
    )


def _chart_card(spec: ChartSpec | None, include_js: bool = False) -> str:
    if spec is None:
        return ""
    return f"<div class='chart'>{figure_to_html_div(spec, include_js=include_js)}</div>"


def _spec_from_dict(viz: dict[str, Any]) -> ChartSpec | None:
    try:
        return ChartSpec(kind=viz["kind"], title=viz.get("title", ""), data=viz.get("data", {}), layout=viz.get("layout", {}))
    except Exception:
        return None


def _slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")[:40]


_CSS = """
:root{--bg:#0f1117;--card:#1a1d27;--text:#e6e8ee;--muted:#9aa0ad;--border:#2a2e3a;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:28px 32px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:28px;flex-wrap:wrap}
header h1{margin:0;font-size:22px}
.meta{color:var(--muted);font-size:12px}
.score{margin-left:auto;text-align:center;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px 22px}
.score-num{font-size:34px;font-weight:700;line-height:1}.score-grade{color:var(--muted);font-size:12px;margin-top:4px}.score-label{font-size:11px;color:var(--muted)}
.chips{display:flex;gap:14px;flex-wrap:wrap;padding:18px 32px}
.chip{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 16px;min-width:120px}
.chip span{font-size:20px;font-weight:600;display:block}.chip label{color:var(--muted);font-size:11px}
main{padding:8px 32px 60px}
.cat{margin-top:30px}.cat h2{font-size:16px;border-bottom:1px solid var(--border);padding-bottom:8px}
.count{color:var(--muted);font-size:12px;font-weight:400}
.card{background:var(--card);border:1px solid var(--border);border-left:4px solid #1f77b4;border-radius:10px;padding:14px 16px;margin:12px 0}
.card-head{display:flex;align-items:center;gap:10px}
.sev{color:#fff;font-size:10px;text-transform:uppercase;padding:2px 8px;border-radius:20px;letter-spacing:.04em}
.conf{margin-left:auto;color:var(--muted);font-size:11px}
.ins{margin:8px 0 6px}.act{color:#8fc7ff;margin:6px 0 0}
.metric{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0}
.metric code{background:#0c0e14;border:1px solid var(--border);border-radius:6px;padding:2px 8px;font-size:12px;color:#c8cdd8}
.chart{margin-top:12px;background:#fff;border-radius:8px;overflow:hidden}
.muted{color:var(--muted)}.chart-error{color:#d62728;padding:10px}
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{style}</style></head>
<body>
<header>
  <div><h1>{title}</h1><div class="meta">Generated {generated} &middot; {rows} rows &times; {cols} columns</div></div>
  {score}
</header>
<div class="chips">{chips}</div>
<main>
  {insights}
  <section class="cat"><h2>Visualizations</h2>{charts}</section>
</main>
</body></html>
"""
