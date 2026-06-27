"""Renderers that turn a :class:`~engines.analytics.base.ChartSpec` into pictures.

The analysis layer never imports a plotting library; it only emits ``ChartSpec``
data. This module is the single place those specs become something visual:

* :func:`to_plotly` -> an interactive ``plotly`` figure for the HTML dashboard.
* :func:`render_image` -> a static PNG/SVG via Matplotlib for reports and slides.

Both are best-effort and defensive: an unknown or malformed spec degrades to a
simple representation rather than raising, so one bad chart never sinks a report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import ChartKind, ChartSpec


def to_plotly(spec: ChartSpec) -> Any:
    """Build an interactive Plotly figure from a chart spec."""

    import plotly.graph_objects as go

    kind = spec.kind if isinstance(spec.kind, ChartKind) else ChartKind(spec.kind)
    data = spec.data or {}
    layout = spec.layout or {}
    fig = go.Figure()

    if kind == ChartKind.BAR:
        horizontal = layout.get("orientation") == "h"
        if horizontal:
            fig.add_bar(x=data.get("x", []), y=data.get("y", []), orientation="h")
        else:
            fig.add_bar(x=data.get("x", []), y=data.get("y", []))
    elif kind == ChartKind.LINE:
        fig.add_scatter(x=data.get("x", []), y=data.get("y", []), mode="lines+markers", name=spec.title)
        if "ci_low" in data and "ci_high" in data:
            x = list(data.get("x", []))
            fig.add_scatter(
                x=x + x[::-1],
                y=list(data["ci_high"]) + list(data["ci_low"])[::-1],
                fill="toself", fillcolor="rgba(0,100,200,0.15)",
                line={"color": "rgba(255,255,255,0)"}, hoverinfo="skip", showlegend=False,
            )
        if data.get("anomaly_x"):
            fig.add_scatter(x=data["anomaly_x"], y=data["anomaly_y"], mode="markers",
                            marker={"color": "red", "size": 10, "symbol": "x"}, name="anomaly")
    elif kind == ChartKind.SCATTER:
        fig.add_scatter(x=data.get("x", []), y=data.get("y", []), mode="markers")
    elif kind == ChartKind.HISTOGRAM:
        fig.add_bar(x=data.get("bin_centers", []), y=data.get("counts", []))
    elif kind == ChartKind.BOX:
        for name, values in (data.get("groups") or {"": data.get("values", [])}).items():
            fig.add_box(y=values, name=str(name))
    elif kind == ChartKind.HEATMAP:
        fig.add_heatmap(
            z=data.get("matrix", []),
            x=data.get("x", data.get("labels", [])),
            y=data.get("labels", []),
            zmin=data.get("zmin"), zmax=data.get("zmax"),
            colorscale=layout.get("colorscale", "RdBu"),
        )
    elif kind == ChartKind.FUNNEL:
        fig.add_trace(go.Funnel(y=data.get("stages", []), x=data.get("users", [])))
    elif kind == ChartKind.TABLE:
        header = data.get("header", [])
        cells = data.get("cells", [])
        fig.add_trace(go.Table(header={"values": header}, cells={"values": cells}))
    elif kind == ChartKind.DENDROGRAM:
        _dendrogram_to_plotly(fig, data)
    else:  # pragma: no cover - defensive fallback
        fig.add_annotation(text=f"Unsupported chart: {kind}", showarrow=False)

    fig.update_layout(
        title=spec.title,
        xaxis_title=layout.get("xaxis_title"),
        yaxis_title=layout.get("yaxis_title"),
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
    )
    return fig


def _dendrogram_to_plotly(fig: Any, data: dict[str, Any]) -> None:
    """Draw a dendrogram from a stored SciPy linkage matrix."""

    from scipy.cluster.hierarchy import dendrogram

    linkage = np.asarray(data.get("linkage", []), dtype="float64")
    labels = data.get("labels", [])
    if linkage.size == 0:
        fig.add_annotation(text="No linkage available", showarrow=False)
        return
    dgram = dendrogram(linkage, labels=labels, no_plot=True)
    for xs, ys in zip(dgram["icoord"], dgram["dcoord"]):
        fig.add_scatter(x=xs, y=ys, mode="lines", line={"color": "#4C78A8"}, hoverinfo="skip", showlegend=False)
    tick_positions = list(range(5, 10 * len(dgram["ivl"]) + 5, 10))
    fig.update_xaxes(tickmode="array", tickvals=tick_positions, ticktext=dgram["ivl"])
    fig.update_yaxes(title="distance")


def render_image(spec: ChartSpec, path: str, fmt: str = "png", dpi: int = 150) -> str:
    """Render a chart spec to a static image file via Matplotlib. Returns the path."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = spec.kind if isinstance(spec.kind, ChartKind) else ChartKind(spec.kind)
    data = spec.data or {}
    layout = spec.layout or {}
    fig, ax = plt.subplots(figsize=(9, 5))

    try:
        if kind == ChartKind.BAR:
            if layout.get("orientation") == "h":
                ax.barh(data.get("y", []), data.get("x", []), color="#4C78A8")
            else:
                ax.bar(data.get("x", []), data.get("y", []), color="#4C78A8")
        elif kind in (ChartKind.LINE,):
            ax.plot(data.get("x", []), data.get("y", []), marker="o", ms=3)
            if data.get("anomaly_x"):
                ax.scatter(data["anomaly_x"], data["anomaly_y"], color="red", marker="x", zorder=5)
            _thin_xticks(ax)
        elif kind == ChartKind.SCATTER:
            ax.scatter(data.get("x", []), data.get("y", []), s=12, alpha=0.6)
        elif kind == ChartKind.HISTOGRAM:
            ax.bar(data.get("bin_centers", []), data.get("counts", []), width=_bar_width(data.get("bin_centers", [])), color="#72B7B2")
        elif kind == ChartKind.HEATMAP:
            matrix = np.array(data.get("matrix", []), dtype="float64")
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=data.get("zmin"), vmax=data.get("zmax"), aspect="auto")
            labels = data.get("labels", [])
            xlabels = data.get("x", labels)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            fig.colorbar(im, ax=ax)
        elif kind == ChartKind.FUNNEL:
            stages = data.get("stages", [])
            users = data.get("users", [])
            ax.barh(range(len(stages)), users, color="#4C78A8")
            ax.set_yticks(range(len(stages)))
            ax.set_yticklabels(stages)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, f"{kind.value}", ha="center", va="center")
        ax.set_title(spec.title)
        if layout.get("xaxis_title"):
            ax.set_xlabel(layout["xaxis_title"])
        if layout.get("yaxis_title"):
            ax.set_ylabel(layout["yaxis_title"])
        fig.tight_layout()
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _bar_width(centers: list[float]) -> float:
    if len(centers) < 2:
        return 1.0
    return float(np.median(np.diff(centers))) * 0.9


def _thin_xticks(ax: Any, max_ticks: int = 12) -> None:
    ticks = ax.get_xticks()
    if len(ticks) > max_ticks:
        step = int(np.ceil(len(ticks) / max_ticks))
        ax.set_xticks(ticks[::step])
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(7)


def export_chart_safe(spec: ChartSpec, path: str, fmt: str = "png") -> bool:
    """Render a chart to ``path``; return False instead of raising on failure."""

    try:
        render_image(spec, path, fmt=fmt)
        return True
    except Exception:  # pragma: no cover - defensive
        return False


def figure_to_html_div(spec: ChartSpec, include_js: bool = False) -> str:
    """Render a chart spec to an embeddable Plotly ``<div>`` (best effort)."""

    try:
        fig = to_plotly(spec)
        return fig.to_html(full_html=False, include_plotlyjs=("cdn" if include_js else False), default_height=420)
    except Exception as exc:  # pragma: no cover - defensive
        return f"<div class='chart-error'>Could not render '{spec.title}': {exc}</div>"
