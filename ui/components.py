"""Reusable render helpers for the IntelliFlow dashboard.

Every helper here builds markup against the CSS variables defined in
:mod:`ui.theme`, so a component is written once and is correct in both light
and dark mode. Nothing in this module reaches into the engines; helpers take
plain values (or the engines' public dataclasses) and return/render markup.

Two conventions keep the HTML safe and predictable inside Streamlit:

* every caller-supplied string goes through :func:`esc`;
* markup is emitted as one whitespace-collapsed line, because Streamlit's
  Markdown parser turns indented HTML into a code block.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st

from . import theme

__all__ = [
    "esc", "render", "hero", "section_header", "stat_card", "stat_row", "callout",
    "chip", "chips", "insight_card", "insight_grid", "kv_panel", "score_display",
    "chat_message", "trace_timeline", "route_list", "empty_state", "engine_cards",
    "chart", "note", "markdown_to_html",
]

# Collapses only line breaks between tags: indented HTML would be parsed as a
# Markdown code block, while a single space between tags is meaningful text.
_COLLAPSE = re.compile(r">[^\S\n]*\n\s*<")

ENGINES = (
    ("Engine 2", "Analytics Intelligence", "accent-2",
     "Profiles the dataset, scores its quality, and surfaces correlations, "
     "anomalies, and product behaviour as ranked insights before you model."),
    ("Engine 1", "AutoML Factory", "accent",
     "Preprocesses, engineers features, optimises hyper-parameters with Optuna, "
     "tracks every trial in MLflow, and registers a servable pipeline."),
    ("Engine 3", "Agent Orchestration", "accent-3",
     "A crew of specialist agents answers questions in plain English and calls "
     "Engines 1 and 2 on your behalf, with a full trace of what it did."),
)


# ------------------------------------------------------------------- helpers
def esc(value: Any) -> str:
    """HTML-escape any value for safe interpolation."""

    return html.escape("" if value is None else str(value), quote=True)


def render(*parts: str) -> None:
    """Emit markup as a single collapsed line so Streamlit renders it as HTML."""

    st.markdown(_COLLAPSE.sub("><", "".join(parts).strip()), unsafe_allow_html=True)


def markdown_to_html(text: str) -> str:
    """Convert a small, safe subset of Markdown to HTML.

    Used where markdown has to live *inside* one of our own containers (the
    chat bubble), which ``st.markdown`` cannot do on its own. Input is escaped
    before any tag is introduced, so untrusted model output stays inert.
    """

    escaped = esc(text or "").replace("\r\n", "\n")
    escaped = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', escaped)

    blocks: list[str] = []
    items: list[str] = []
    tag = "ul"

    def flush() -> None:
        if items:
            blocks.append(f"<{tag}>" + "".join(f"<li>{item}</li>" for item in items) + f"</{tag}>")
            items.clear()

    for raw_line in escaped.split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            level = min(len(heading.group(1)) + 2, 6)
            blocks.append(f"<h{level}>{heading.group(2)}</h{level}>")
            continue
        bullet = re.match(r"^([-*+]|\d+[.)])\s+(.*)$", line)
        if bullet:
            marker = "ol" if bullet.group(1)[0].isdigit() else "ul"
            if marker != tag:
                flush()
                tag = marker
            items.append(bullet.group(2))
            continue
        flush()
        blocks.append(f"<p>{line}</p>")
    flush()
    return "".join(blocks) or "<p></p>"


def _tone_class(tone: str | None) -> str:
    return f" is-{tone}" if tone in ("accent", "success", "warning", "danger") else ""


def _severity(severity: str) -> tuple[str, str, str]:
    """(token prefix, display label, css colour expression) for a severity."""

    key = theme.SEVERITY_TOKENS.get(severity, "info")
    label = theme.SEVERITY_LABELS.get(severity, severity.title())
    return key, label, f"var(--if-{key})"


def _fmt(value: Any) -> str:
    """Render a metric-ish value the way a person would write it."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "—"
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_fmt(item) for item in value) or "—"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}: {_fmt(val)}" for key, val in value.items()) or "—"
    return str(value)


def _label(key: Any) -> str:
    return str(key).replace("_", " ").strip().capitalize()


# ---------------------------------------------------------------- components
def hero(title: str, subtitle: str, meta: Sequence[tuple[str, str | None]] = ()) -> None:
    """The page masthead: brand mark, title, and status chips."""

    meta_html = "".join(chip(text, tone=tone) for text, tone in meta)
    render(
        '<section class="if-hero">',
        '<div class="if-hero-top">',
        '<div class="if-brand">',
        '<div class="if-mark">IF</div>',
        f"<div><h1>{esc(title)}</h1>",
        f'<div class="if-hero-sub">{esc(subtitle)}</div></div>',
        "</div>",
        f'<div class="if-hero-meta">{meta_html}</div>',
        "</div></section>",
    )


def section_header(kicker: str, title: str, description: str = "") -> None:
    render(
        '<div class="if-section">',
        f'<div class="if-kicker">{esc(kicker)}</div>',
        f"<h2>{esc(title)}</h2>",
        f"<p>{esc(description)}</p>" if description else "",
        "</div>",
    )


def stat_card(label: str, value: str, note: str = "") -> str:
    """Markup for a single KPI tile (use :func:`stat_row` to place them)."""

    return (
        '<div class="if-card">'
        f'<div class="if-stat-label">{esc(label)}</div>'
        f'<div class="if-stat-value">{esc(value)}</div>'
        f'<div class="if-stat-note">{esc(note)}</div>'
        "</div>"
    )


def stat_row(cards: Sequence[str], columns: int = 4) -> None:
    render(f'<div class="if-grid if-grid-{columns}">', *cards, "</div>")


def chip(text: str, tone: str | None = None, dot: str | None = None) -> str:
    dot_html = f'<span class="if-dot" style="background:{dot}"></span>' if dot else ""
    return f'<span class="if-chip{_tone_class(tone)}">{dot_html}{esc(text)}</span>'


def chips(items: Iterable[str], tone: str | None = None) -> str:
    return "".join(chip(item, tone=tone) for item in items)


def callout(text: str, tone: str | None = None, icon: str = "", code: str = "") -> None:
    """A single, consistent way to say something in prose next to the UI.

    ``text`` may contain ``<strong>`` / ``<code>`` produced by the caller via
    :func:`esc`-safe helpers; pass ``code`` for a monospaced tail such as a
    model URI.
    """

    code_html = f" <code>{esc(code)}</code>" if code else ""
    icon_html = f'<span class="if-callout-icon">{esc(icon)}</span>' if icon else ""
    render(
        f'<div class="if-callout{_tone_class(tone)}">',
        icon_html,
        f"<div>{text}{code_html}</div>",
        "</div>",
    )


def note(text: str) -> None:
    render(f'<p class="if-note">{esc(text)}</p>')


def insight_card(insight: Any) -> str:
    """One severity system for every insight, wherever it is shown."""

    key, label, color = _severity(getattr(insight, "severity", "info"))
    confidence = getattr(insight, "confidence_label", "")
    category = getattr(insight, "category", "") or ""
    action = getattr(insight, "action", "") or ""
    return (
        '<div class="if-insight">'
        '<div class="if-insight-head">'
        f'<span class="if-dot" style="background:{color}"></span>'
        f'<span class="if-insight-sev" style="color:{color}">{esc(label)}</span>'
        + (f'<span class="if-insight-cat">{esc(category)}</span>' if category else "")
        + "</div>"
        f"<h4>{esc(getattr(insight, 'title', ''))}</h4>"
        f'<div class="if-insight-body">{esc(getattr(insight, "insight", ""))}</div>'
        '<div class="if-insight-foot">'
        + (chip(f"{confidence} confidence", tone="accent") if confidence else "")
        + (f'<span class="if-insight-action">{esc(action)}</span>' if action else "")
        + "</div></div>"
    )


def insight_grid(insights: Sequence[Any], columns: int = 2) -> None:
    if not insights:
        callout("No insights were produced for this configuration.", icon="○")
        return
    render(
        f'<div class="if-grid if-grid-{columns}">',
        *[insight_card(item) for item in insights],
        "</div>",
    )


def kv_panel(title: str, mapping: Mapping[str, Any] | None, empty: str = "Nothing to show.") -> None:
    """A formatted key/value grid -- the replacement for every raw ``st.json``."""

    rows = list((mapping or {}).items())
    if not rows:
        render(
            '<div class="if-kv">',
            f'<div class="if-kv-title">{esc(title)}</div>',
            f'<div class="if-kv-empty">{esc(empty)}</div>',
            "</div>",
        )
        return
    render(
        '<div class="if-kv">',
        f'<div class="if-kv-title">{esc(title)}</div>',
        *[
            f'<div class="if-kv-row"><span class="if-kv-key">{esc(_label(key))}</span>'
            f'<span class="if-kv-val">{esc(_fmt(value))}</span></div>'
            for key, value in rows
        ],
        "</div>",
    )


def score_display(score: float | None, grade: str | None, caption: str = "") -> None:
    """The headline data-quality figure, with the grade as the second channel."""

    if score is None:
        value, pct, key = "N/A", 0.0, "info"
    else:
        value, pct = f"{score:.0f}", max(0.0, min(100.0, float(score)))
        key = "success" if pct >= 80 else "warning" if pct >= 60 else "danger"
    color = f"var(--if-{key})"
    render(
        f'<div class="if-score" style="--if-score:{pct:.1f};--if-score-color:{color}">',
        '<div class="if-score-ring">',
        f'<div class="if-score-ring-inner">{esc(value)}</div>',
        "</div>",
        '<div class="if-score-meta">',
        "<h3>Data quality score",
        f' <span class="if-score-grade">Grade {esc(grade or "N/A")}</span>' if grade or score is not None else "",
        "</h3>",
        f"<p>{esc(caption)}</p>" if caption else "",
        "</div></div>",
    )


def chat_message(role: str, text: str, agents: Sequence[str] = ()) -> None:
    """A chat bubble; assistant text keeps its Markdown formatting."""

    is_user = role == "user"
    body = esc(text).replace("\n", "<br>") if is_user else markdown_to_html(text)
    agent_html = (
        f'<div class="if-msg-agents">{chips(agents, tone="accent")}</div>' if agents else ""
    )
    render(
        f'<div class="if-msg {"is-user" if is_user else "is-agent"}">',
        f'<div class="if-msg-avatar">{"YOU" if is_user else "IF"}</div>',
        f'<div class="if-msg-bubble">{body}{agent_html}</div>',
        "</div>",
    )


def trace_timeline(trace: Sequence[Mapping[str, Any]]) -> None:
    """The crew's run as a stepped timeline instead of a JSON blob."""

    if not trace:
        callout("No trace was recorded for this run.", icon="○")
        return

    steps: list[str] = []
    for entry in trace:
        step = str(entry.get("step", "step"))
        if step == "plan":
            name, detail, color = "Planner", entry.get("plan", ""), "var(--if-accent)"
        elif step == "recall":
            name = "Memory recall"
            detail = f"{entry.get('recalled', 0)} prior exchange(s) retrieved"
            color = "var(--if-muted)"
        elif step == "tool":
            name = f"{entry.get('agent') or 'Agent'} → {entry.get('tool', 'tool')}"
            arguments = entry.get("arguments") or {}
            observation = str(entry.get("observation", "")).strip()
            args_text = ", ".join(f"{k}={_fmt(v)}" for k, v in arguments.items())
            detail = " · ".join(part for part in (args_text, observation[:320]) if part)
            color = "var(--if-accent-2)"
        elif step == "synthesize":
            name, detail, color = "Synthesizer", "Composed the final answer", "var(--if-success)"
        else:
            name, detail, color = step.title(), _fmt(dict(entry)), "var(--if-muted)"

        steps.append(
            f'<div class="if-step" style="--if-step-color:{color}">'
            '<div class="if-step-head">'
            f'<span class="if-step-name">{esc(name)}</span>'
            "</div>"
            + (f'<div class="if-step-detail">{esc(detail)}</div>' if detail else "")
            + "</div>"
        )
    render('<div class="if-timeline">', *steps, "</div>")


def route_list(title: str, routes: Sequence[tuple[str, str, str]]) -> None:
    render(
        '<div class="if-routes">',
        f'<div class="if-routes-head">{esc(title)}</div>',
        *[
            '<div class="if-route">'
            f'<span class="if-method {esc(method.lower())}">{esc(method)}</span>'
            f'<span class="if-route-path">{esc(path)}</span>'
            f'<span class="if-route-desc">{esc(description)}</span>'
            "</div>"
            for method, path, description in routes
        ],
        "</div>",
    )


def engine_cards() -> None:
    """All three engines, described in the language the app actually uses."""

    cards = [
        f'<div class="if-engine" style="--if-engine-accent:var(--if-{token})">'
        f'<div class="if-engine-tag">{esc(tag)}</div>'
        f"<strong>{esc(name)}</strong><p>{esc(description)}</p></div>"
        for tag, name, token, description in ENGINES
    ]
    render('<div class="if-grid if-grid-3">', *cards, "</div>")


def empty_state() -> None:
    steps = (
        ("01", "Load", "A CSV, Excel, JSON or Parquet file — or one of the two built-in samples."),
        ("02", "Understand", "Engine 2 scores data quality and ranks the insights worth acting on."),
        ("03", "Model & ask", "Engine 1 trains and registers a pipeline; Engine 3 answers questions about it."),
    )
    render(
        '<div class="if-empty">',
        "<h2>Start with a dataset</h2>",
        "<p>Choose a source in the left panel. The same table feeds analytics, training, "
        "prediction, and the agent crew, so everything you see stays consistent.</p>",
        '<div class="if-grid if-grid-3">',
        *[
            f'<div class="if-stepcard"><b>{esc(num)}</b><strong>{esc(title)}</strong>'
            f"<span>{esc(body)}</span></div>"
            for num, title, body in steps
        ],
        "</div></div>",
    )


def chart(figure: Any) -> None:
    """Render an already-themed Plotly figure with the app's toolbar config."""

    st.plotly_chart(figure, width="stretch", config=theme.plotly_config())
