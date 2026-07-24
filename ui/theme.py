"""Design system for the IntelliFlow dashboard.

This module is the single source of truth for how the UI looks. It owns:

* :data:`TOKENS`   -- the light and dark colour/elevation scales.
* :func:`build_css`/:func:`inject` -- one ``<style>`` block that themes both our
  own components and every CSS-controllable native Streamlit widget.
* :func:`style_figure` -- the same palette applied to Plotly figures, so charts
  belong to the page instead of arriving from a stock template.

The palette comes from the validated ``dataviz`` reference instance: the eight
categorical slots pass the lightness-band, chroma-floor and colour-vision
(CVD ΔE 24.2 light / 10.3 dark) checks against these exact surfaces, and every
ink/accent pairing clears WCAG AA on the surface it renders on.

Nothing here imports from ``engines`` -- theming is strictly a UI concern, and
chart styling is applied by post-processing the figure that
:func:`engines.analytics.visualization.to_plotly` returns.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

THEMES = ("light", "dark")
DEFAULT_THEME = "light"

FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, '
    '"Helvetica Neue", Arial, "Noto Sans", sans-serif'
)
MONO_STACK = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)

# The eight categorical slots, stepped per surface. Order is the CVD-safety
# mechanism (max-min adjacent ΔE), so it must not be reshuffled.
_COLORWAY_LIGHT = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
_COLORWAY_DARK = [
    "#3987e5",
    "#199e70",
    "#c98500",
    "#008300",
    "#9085e9",
    "#e66767",
    "#d55181",
    "#d95926",
]

# Diverging (blue <-> red, neutral gray midpoint) for correlation-style matrices.
_DIVERGING_LIGHT = [
    [0.0, "#0d366b"], [0.25, "#5598e7"], [0.5, "#f0efec"],
    [0.75, "#e34948"], [1.0, "#8f1f1f"],
]
_DIVERGING_DARK = [
    [0.0, "#86b6ef"], [0.25, "#2a78d6"], [0.5, "#383835"],
    [0.75, "#d03b3b"], [1.0, "#f3a3a3"],
]

# Single-hue sequential ramp for magnitude-only matrices.
_SEQUENTIAL_LIGHT = [
    [0.0, "#eaf2fd"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"],
    [0.75, "#2a78d6"], [1.0, "#104281"],
]
_SEQUENTIAL_DARK = [
    [0.0, "#132436"], [0.25, "#184f95"], [0.5, "#2a78d6"],
    [0.75, "#6da7ec"], [1.0, "#cde2fb"],
]


TOKENS: dict[str, dict[str, Any]] = {
    "light": {
        # surfaces
        "canvas": "#f9f9f7",
        "panel": "#ffffff",
        "soft": "#f3f3f0",
        "sunken": "#efeeea",
        # ink
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#6b6a66",
        # lines
        "border": "#e4e3dd",
        "border_strong": "#d3d2ca",
        # accent
        "accent": "#1c5cab",
        "accent_solid": "#1c5cab",
        "accent_hover": "#184f95",
        "accent_soft": "#e8f1fd",
        "accent_2": "#1baf7a",
        "accent_3": "#4a3aa7",
        "on_accent": "#ffffff",
        # status: *_text is legible ink, plain key is the vivid dot/bar
        "success": "#0ca30c",
        "success_text": "#006300",
        "success_soft": "#e6f5e6",
        "warning": "#fab219",
        "warning_text": "#8a5a00",
        "warning_soft": "#fdf3dd",
        "danger": "#d03b3b",
        "danger_text": "#b0201f",
        "danger_soft": "#fbeaea",
        "info": "#2a78d6",
        "info_text": "#1c5cab",
        "info_soft": "#e8f1fd",
        # elevation
        "shadow": "0 1px 2px rgba(11,11,11,.04), 0 6px 20px rgba(11,11,11,.05)",
        "shadow_lg": "0 1px 3px rgba(11,11,11,.06), 0 18px 44px rgba(11,11,11,.08)",
        # charts
        "colorway": _COLORWAY_LIGHT,
        "grid": "#e9e8e2",
        "axis": "#c3c2b7",
        "diverging": _DIVERGING_LIGHT,
        "sequential": _SEQUENTIAL_LIGHT,
    },
    "dark": {
        "canvas": "#0d0d0d",
        "panel": "#1a1a19",
        "soft": "#232321",
        "sunken": "#151514",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#9c9a93",
        "border": "#2f2f2c",
        "border_strong": "#3d3d39",
        "accent": "#6da7ec",
        "accent_solid": "#256abf",
        "accent_hover": "#2a78d6",
        "accent_soft": "#15263a",
        "accent_2": "#199e70",
        "accent_3": "#9085e9",
        "on_accent": "#ffffff",
        "success": "#3ec93e",
        "success_text": "#3ec93e",
        "success_soft": "#132513",
        "warning": "#fab219",
        "warning_text": "#fab219",
        "warning_soft": "#2b2210",
        "danger": "#e66767",
        "danger_text": "#e66767",
        "danger_soft": "#2c1717",
        "info": "#6da7ec",
        "info_text": "#6da7ec",
        "info_soft": "#15263a",
        "shadow": "0 1px 2px rgba(0,0,0,.5), 0 6px 20px rgba(0,0,0,.4)",
        "shadow_lg": "0 1px 3px rgba(0,0,0,.55), 0 18px 44px rgba(0,0,0,.5)",
        "colorway": _COLORWAY_DARK,
        "grid": "#2c2c2a",
        "axis": "#4a4a46",
        "diverging": _DIVERGING_DARK,
        "sequential": _SEQUENTIAL_DARK,
    },
}

# Insight severities -> the token prefix that carries their colour.
SEVERITY_TOKENS = {
    "critical": "danger",
    "warning": "warning",
    "info": "info",
    "ok": "success",
}
SEVERITY_LABELS = {
    "critical": "Critical",
    "warning": "Warning",
    "info": "Insight",
    "ok": "Healthy",
}


# --------------------------------------------------------------------- state
def normalize(theme: str | None) -> str:
    """Coerce anything to a supported theme name."""

    value = (theme or "").strip().lower()
    return value if value in THEMES else DEFAULT_THEME


def current_theme() -> str:
    """The theme selected in the sidebar for this rerun."""

    return normalize(st.session_state.get("theme"))


def tokens(theme: str | None = None) -> dict[str, Any]:
    """Token table for ``theme`` (defaults to the active one)."""

    return TOKENS[normalize(theme) if theme else current_theme()]


def page_config() -> None:
    """Streamlit page setup. Must run before any other Streamlit call."""

    st.set_page_config(
        page_title="IntelliFlow",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ----------------------------------------------------------------------- css
# Streamlit paints the dataframe/data-editor grid onto a canvas with a theme it
# builds from `.streamlit/config.toml` at page load, so the `--gdg-*` custom
# properties above cannot repaint it at runtime. Inverting the canvas (and
# rotating hue back) is the only way to keep the grid readable in dark mode; it
# is scoped to the grid surface alone, never the surrounding chrome.
_GRID_DARK_FIX = """
[data-testid="stDataFrameResizable"] {
  filter: invert(1) hue-rotate(180deg);
  background: #ffffff;
}
[data-testid="stDataFrameResizable"] ::-webkit-scrollbar-thumb { border-color: #ffffff; }
"""


def build_css(theme: str) -> str:
    """The complete ``<style>`` block for ``theme``.

    Covers our own ``.if-*`` components *and* the native Streamlit chrome, so a
    theme flip changes every surface rather than leaving default widgets behind.
    """

    t = tokens(theme)
    dark = normalize(theme) == "dark"
    # The dataframe/data-editor grid is a canvas: it is only reachable through
    # glide-data-grid's own custom properties, which we set alongside ours.
    grid_scheme = "dark" if dark else "light"

    return f"""
<style>
:root, .stApp {{
  --if-canvas: {t["canvas"]};
  --if-panel: {t["panel"]};
  --if-soft: {t["soft"]};
  --if-sunken: {t["sunken"]};
  --if-ink: {t["ink"]};
  --if-secondary: {t["secondary"]};
  --if-muted: {t["muted"]};
  --if-border: {t["border"]};
  --if-border-strong: {t["border_strong"]};
  --if-accent: {t["accent"]};
  --if-accent-solid: {t["accent_solid"]};
  --if-accent-hover: {t["accent_hover"]};
  --if-accent-soft: {t["accent_soft"]};
  --if-accent-2: {t["accent_2"]};
  --if-accent-3: {t["accent_3"]};
  --if-on-accent: {t["on_accent"]};
  --if-success: {t["success"]};
  --if-success-text: {t["success_text"]};
  --if-success-soft: {t["success_soft"]};
  --if-warning: {t["warning"]};
  --if-warning-text: {t["warning_text"]};
  --if-warning-soft: {t["warning_soft"]};
  --if-danger: {t["danger"]};
  --if-danger-text: {t["danger_text"]};
  --if-danger-soft: {t["danger_soft"]};
  --if-info: {t["info"]};
  --if-info-text: {t["info_text"]};
  --if-info-soft: {t["info_soft"]};
  --if-shadow: {t["shadow"]};
  --if-shadow-lg: {t["shadow_lg"]};
  --if-radius: 12px;
  --if-radius-sm: 8px;
  --if-font: {FONT_STACK};
  --if-mono: {MONO_STACK};

  /* Streamlit's own theme variables, kept in sync for widgets that read them. */
  --primary-color: {t["accent_solid"]};
  --background-color: {t["canvas"]};
  --secondary-background-color: {t["soft"]};
  --text-color: {t["ink"]};
  --font: {FONT_STACK};

  color-scheme: {grid_scheme};
}}

/* ------------------------------------------------------------ foundations */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--if-canvas);
  color: var(--if-ink);
  font-family: var(--if-font);
  -webkit-font-smoothing: antialiased;
}}
[data-testid="stHeader"], [data-testid="stToolbar"] {{
  background: transparent;
}}
[data-testid="stDecoration"] {{
  background: linear-gradient(90deg, var(--if-accent-solid), var(--if-accent-2));
  height: 2px;
}}
[data-testid="stToolbar"] button, [data-testid="stHeader"] svg {{
  color: var(--if-muted);
  fill: var(--if-muted);
}}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1460px;
  padding: 2.5rem 2.5rem 5rem !important;
}}
/* Bordered containers (st.container(border=True)) are plain vertical blocks;
   only they have a visible border, so recolouring every block is safe. */
[data-testid="stVerticalBlock"] {{
  border-color: var(--if-border) !important;
  border-radius: var(--if-radius) !important;
}}

h1, h2, h3, h4, h5, h6 {{
  color: var(--if-ink);
  font-family: var(--if-font);
  letter-spacing: -.014em;
  font-weight: 640;
}}
h1 {{ font-size: 2rem; line-height: 1.2; }}
h2 {{ font-size: 1.35rem; line-height: 1.28; }}
h3 {{ font-size: 1.1rem; line-height: 1.35; }}
h4 {{ font-size: .96rem; line-height: 1.4; }}
h5, h6 {{ font-size: .88rem; line-height: 1.4; }}
[data-testid="stHeadingWithActionElements"] {{ scroll-margin-top: 1rem; }}
p, li, label, .stMarkdown, [data-testid="stMarkdownContainer"] {{
  color: var(--if-ink);
  font-family: var(--if-font);
  font-size: .94rem;
  line-height: 1.62;
}}
[data-testid="stMarkdownContainer"] a {{ color: var(--if-accent); }}
hr, [data-testid="stMarkdownContainer"] hr {{
  border: 0;
  border-top: 1px solid var(--if-border);
  margin: 1.1rem 0;
}}
::selection {{ background: var(--if-accent-soft); color: var(--if-ink); }}
* {{ scrollbar-color: var(--if-border-strong) transparent; scrollbar-width: thin; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
  background: var(--if-border-strong);
  border: 3px solid var(--if-canvas);
  border-radius: 999px;
}}

/* ---------------------------------------------------------------- sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
  background: var(--if-panel);
}}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--if-border); }}
[data-testid="stSidebarHeader"] {{ padding-bottom: .2rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top: .5rem; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: var(--if-ink); }}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {{ color: var(--if-muted); }}

/* ------------------------------------------------------------------- tabs */
[data-testid="stTabs"] [role="tablist"] {{
  gap: .2rem;
  background: var(--if-soft);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: .3rem;
  margin-bottom: 1.1rem;
  display: flex;
  flex-wrap: wrap;
}}
[data-testid="stTab"], [data-baseweb="tab-list"] button[data-baseweb="tab"] {{
  height: auto;
  min-height: 2.35rem;
  display: flex;
  align-items: center;
  padding: .4rem 1rem;
  border-radius: var(--if-radius-sm);
  color: var(--if-muted);
  background: transparent;
  border: 1px solid transparent;
  cursor: pointer;
  transition: background .15s ease, color .15s ease;
}}
[data-testid="stTab"]:hover {{ color: var(--if-ink); background: var(--if-panel); }}
[data-testid="stTab"][data-selected="true"], [data-baseweb="tab-list"] button[aria-selected="true"] {{
  background: var(--if-panel);
  color: var(--if-ink);
  border-color: var(--if-border);
  box-shadow: var(--if-shadow);
}}
[data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
  font-size: .88rem;
  font-weight: 560;
  color: inherit;
  margin: 0;
}}
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
  background: transparent !important;
  display: none;
}}
[data-testid="stTabPanel"] {{ padding-top: .1rem; }}

/* ---------------------------------------------------------------- buttons */
.stButton button, .stDownloadButton button, .stFormSubmitButton button,
[data-testid^="stBaseButton-"] {{
  border-radius: var(--if-radius-sm);
  font-family: var(--if-font);
  font-weight: 560;
  font-size: .9rem;
  min-height: 2.7rem;
  transition: background .15s ease, border-color .15s ease, color .15s ease;
}}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"],
button[kind="primary"] {{
  background: var(--if-accent-solid) !important;
  border: 1px solid var(--if-accent-solid) !important;
  color: var(--if-on-accent) !important;
  box-shadow: var(--if-shadow);
}}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] div,
button[kind="primary"] p {{ color: var(--if-on-accent) !important; }}
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {{
  background: var(--if-accent-hover) !important;
  border-color: var(--if-accent-hover) !important;
}}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"],
button[kind="secondary"] {{
  background: var(--if-panel) !important;
  border: 1px solid var(--if-border-strong) !important;
  color: var(--if-ink) !important;
}}
[data-testid="stBaseButton-secondary"] p, button[kind="secondary"] p {{ color: var(--if-ink); }}
[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {{
  border-color: var(--if-accent) !important;
  color: var(--if-accent) !important;
}}
[data-testid="stBaseButton-secondary"]:hover p {{ color: var(--if-accent) !important; }}
[data-testid^="stBaseButton-header"], [data-testid="stBaseButton-elementToolbar"] {{
  background: transparent !important;
  color: var(--if-muted) !important;
  border: 0 !important;
  min-height: 0;
  box-shadow: none;
}}
[data-testid="stElementToolbar"] {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius-sm);
  box-shadow: var(--if-shadow);
}}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {{
  outline: 2px solid var(--if-accent);
  outline-offset: 2px;
  box-shadow: none;
}}

/* ------------------------------------------------------- inputs & selects */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
  color: var(--if-secondary);
  font-size: .82rem;
  font-weight: 560;
  letter-spacing: .005em;
}}
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
[data-testid="stNumberInputContainer"], .react-aria-ComboBox [role="group"],
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div {{
  background: var(--if-panel) !important;
  border-color: var(--if-border-strong) !important;
  border-radius: var(--if-radius-sm) !important;
  color: var(--if-ink) !important;
}}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within,
.react-aria-ComboBox [role="group"]:focus-within,
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
  border-color: var(--if-accent) !important;
  box-shadow: 0 0 0 3px var(--if-accent-soft) !important;
}}
input, textarea, select, [data-testid="stNumberInputField"],
.react-aria-ComboBox input, [data-testid="stMultiSelect"] input {{
  color: var(--if-ink) !important;
  font-family: var(--if-font) !important;
  background: transparent !important;
  -webkit-text-fill-color: var(--if-ink);
}}
input::placeholder, textarea::placeholder {{
  color: var(--if-muted) !important;
  -webkit-text-fill-color: var(--if-muted);
  opacity: 1;
}}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{
  background: var(--if-soft) !important;
  color: var(--if-secondary) !important;
  border-color: var(--if-border) !important;
}}
.react-aria-Popover, [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"],
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectVirtualDropdown"] {{
  background: var(--if-panel) !important;
  border: 1px solid var(--if-border) !important;
  border-radius: var(--if-radius-sm) !important;
  box-shadow: var(--if-shadow-lg) !important;
  color: var(--if-ink) !important;
}}
[role="option"], [data-baseweb="menu"] li {{ color: var(--if-ink) !important; }}
[role="option"][data-focused="true"], [role="option"]:hover, li[aria-selected="true"] {{
  background: var(--if-soft) !important;
}}
[role="option"][data-selected="true"], [role="option"][aria-selected="true"] {{
  background: var(--if-accent-soft) !important;
  color: var(--if-accent) !important;
}}
[data-baseweb="tag"], [data-testid="stMultiSelectDeleteIcon"] ~ span, .react-aria-Tag {{
  background: var(--if-accent-soft) !important;
  color: var(--if-accent) !important;
  border-radius: 999px !important;
}}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg, .react-aria-Tag svg {{
  color: var(--if-accent) !important;
  fill: var(--if-accent) !important;
}}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg,
.react-aria-ComboBox button svg {{ fill: var(--if-muted); color: var(--if-muted); }}

/* radio & checkbox */
[data-testid="stRadioGroup"] {{ gap: .3rem; }}
[data-testid="stRadioOption"] {{ color: var(--if-ink); }}
[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{
  color: var(--if-ink);
  font-size: .9rem;
}}
[data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] p {{
  color: var(--if-ink);
  font-weight: 560;
}}
/* The radio marker is a nested pair of divs whose colour Streamlit bakes from
   config.toml; owning it here keeps the control on-palette even when the app is
   launched from a directory where that config is not picked up. */
[data-testid="stRadioOption"] > div > div > div:first-child {{
  background: var(--if-soft) !important;
  box-shadow: inset 0 0 0 1px var(--if-border-strong);
}}
[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {{
  background: var(--if-accent-solid) !important;
  box-shadow: none;
}}
[data-testid="stRadioOption"] > div > div > div:first-child > div {{
  background: var(--if-panel) !important;
}}
[data-testid="stCheckbox"] span[data-rac], [data-testid="stCheckbox"] [data-baseweb="checkbox"] span {{
  border-color: var(--if-border-strong) !important;
}}
input[type="radio"], input[type="checkbox"], input[type="range"] {{
  accent-color: var(--if-accent-solid);
}}

/* slider */
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"], [data-testid="stThumbValue"] {{
  color: var(--if-muted) !important;
  font-size: .76rem;
}}
[data-testid="stSlider"] [role="slider"] {{ box-shadow: var(--if-shadow); }}

/* file uploader */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {{
  background: var(--if-soft);
  border: 1px dashed var(--if-border-strong);
  border-radius: var(--if-radius-sm);
  color: var(--if-secondary);
}}
[data-testid="stFileUploaderDropzone"] button {{
  background: var(--if-panel);
  border: 1px solid var(--if-border-strong);
  color: var(--if-ink);
}}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {{ color: var(--if-muted); }}
[data-testid="stFileUploaderFile"] {{ color: var(--if-ink); }}

/* --------------------------------------------------------------- metrics */
[data-testid="stMetric"] {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: .85rem 1rem;
  box-shadow: var(--if-shadow);
}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
  color: var(--if-muted);
  font-size: .74rem;
  font-weight: 620;
  letter-spacing: .06em;
  text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{
  color: var(--if-ink);
  font-size: 1.65rem;
  font-weight: 640;
  letter-spacing: -.02em;
}}
[data-testid="stMetricDelta"] {{ font-size: .82rem; }}

/* ------------------------------------------------------- alerts & expander */
[data-testid="stAlert"], [data-testid="stAlertContainer"] {{
  background: var(--if-soft);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius-sm);
  color: var(--if-ink);
  box-shadow: none;
}}
[data-testid="stAlert"] p, [data-testid="stAlertContainer"] p {{ color: var(--if-ink); }}
[data-testid="stAlertContentSuccess"] {{
  background: var(--if-success-soft);
  border-color: var(--if-success);
}}
[data-testid="stAlertContentWarning"] {{
  background: var(--if-warning-soft);
  border-color: var(--if-warning);
}}
[data-testid="stAlertContentError"] {{
  background: var(--if-danger-soft);
  border-color: var(--if-danger);
}}
[data-testid="stAlertContentInfo"] {{
  background: var(--if-info-soft);
  border-color: var(--if-info);
}}
[data-testid="stExpander"], details[data-testid="stExpander"] {{
  background: var(--if-panel);
  border: 1px solid var(--if-border) !important;
  border-radius: var(--if-radius) !important;
  box-shadow: none;
}}
[data-testid="stExpander"] summary {{
  color: var(--if-secondary);
  font-weight: 560;
  font-size: .88rem;
  border-radius: var(--if-radius);
}}
[data-testid="stExpander"] summary:hover {{ color: var(--if-accent); }}
[data-testid="stExpander"] svg {{ fill: var(--if-muted); }}

/* ------------------------------------------------------------------- code */
code, kbd, [data-testid="stCode"] code, .stCode code {{
  font-family: var(--if-mono) !important;
  font-size: .84rem !important;
}}
[data-testid="stMarkdownContainer"] code {{
  background: var(--if-soft);
  color: var(--if-accent);
  padding: .1rem .35rem;
  border-radius: 5px;
  border: 1px solid var(--if-border);
}}
[data-testid="stCode"], .stCode, pre {{
  background: var(--if-sunken) !important;
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius-sm);
}}
[data-testid="stCode"] pre, .stCode pre {{ background: transparent !important; }}
[data-testid="stCode"] code span, .stCode code span {{ color: var(--if-ink) !important; }}
[data-testid="stCodeCopyButton"] button, [data-testid="stCode"] button {{ color: var(--if-muted); }}

/* -------------------------------------------------------------- dataframe */
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"],
[data-testid="stDataEditor"] {{
  border: 1px solid var(--if-border) !important;
  border-radius: var(--if-radius) !important;
  overflow: hidden;
  background: var(--if-panel);
}}
[data-testid="stDataFrame"] *, [data-testid="stDataEditor"] * {{
  font-family: var(--if-font);
}}
/* glide-data-grid renders to a canvas; these custom properties are the only
   supported way to recolour it, so the grid follows the theme too. */
.stApp, [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
  --gdg-bg-cell: {t["panel"]};
  --gdg-bg-cell-medium: {t["soft"]};
  --gdg-bg-header: {t["soft"]};
  --gdg-bg-header-has-focus: {t["sunken"]};
  --gdg-bg-header-hovered: {t["sunken"]};
  --gdg-text-dark: {t["ink"]};
  --gdg-text-medium: {t["secondary"]};
  --gdg-text-light: {t["muted"]};
  --gdg-text-header: {t["secondary"]};
  --gdg-text-header-selected: {t["ink"]};
  --gdg-text-group-header: {t["secondary"]};
  --gdg-border-color: {t["border"]};
  --gdg-horizontal-border-color: {t["border"]};
  --gdg-drilldown-border: {t["border"]};
  --gdg-accent-color: {t["accent_solid"]};
  --gdg-accent-fg: {t["on_accent"]};
  --gdg-accent-light: {t["accent_soft"]};
  --gdg-bg-bubble: {t["soft"]};
  --gdg-bg-bubble-selected: {t["accent_soft"]};
  --gdg-bg-search-result: {t["warning_soft"]};
  --gdg-bg-icon-header: {t["muted"]};
  --gdg-fg-icon-header: {t["panel"]};
  --gdg-font-family: {FONT_STACK};
}}
{_GRID_DARK_FIX if dark else ""}

/* ------------------------------------------------------------ misc chrome */
[data-testid="stSpinner"] i {{ border-top-color: var(--if-accent) !important; }}
[data-testid="stSpinner"] p, [data-testid="stStatusWidget"] {{ color: var(--if-secondary); }}
[data-testid="stProgress"] div[role="progressbar"] > div {{ background: var(--if-accent-solid); }}
[data-testid="stToast"] {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  color: var(--if-ink);
}}
.js-plotly-plot .plotly .modebar {{ background: transparent !important; }}
.js-plotly-plot .plotly .modebar-btn path {{ fill: var(--if-muted) !important; }}
.js-plotly-plot .plotly .modebar-btn:hover path {{ fill: var(--if-accent) !important; }}
[data-testid="stPlotlyChart"] {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  box-shadow: var(--if-shadow);
}}
/* Streamlit fixes the chart element's height to the figure height, so the card
   gets its breathing room from the figure's own margins, not from padding. */
[data-testid="stPlotlyChart"] .js-plotly-plot, [data-testid="stPlotlyChart"] .main-svg {{
  border-radius: var(--if-radius);
}}

/* ===================================================================== */
/* IntelliFlow components                                                 */
/* ===================================================================== */
.if-hero {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: 1.5rem 1.6rem;
  box-shadow: var(--if-shadow);
  margin-bottom: 1rem;
}}
.if-hero-top {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.25rem;
  flex-wrap: wrap;
}}
.if-brand {{ display: flex; align-items: center; gap: .85rem; }}
.if-mark {{
  width: 42px; height: 42px;
  flex: 0 0 42px;
  border-radius: 11px;
  background: var(--if-accent-solid);
  color: var(--if-on-accent);
  display: grid; place-items: center;
  font-weight: 680; font-size: .95rem; letter-spacing: -.02em;
}}
.if-hero h1 {{ margin: 0; font-size: 1.75rem; letter-spacing: -.02em; }}
.if-hero-sub {{ color: var(--if-muted); font-size: .9rem; margin-top: .18rem; }}
.if-hero-meta {{ display: flex; gap: .4rem; flex-wrap: wrap; align-items: center; }}

.if-chip {{
  display: inline-flex; align-items: center; gap: .38rem;
  border-radius: 999px;
  border: 1px solid var(--if-border);
  background: var(--if-soft);
  color: var(--if-secondary);
  padding: .26rem .62rem;
  font-size: .76rem;
  font-weight: 550;
  white-space: nowrap;
  line-height: 1.35;
}}
.if-chip.is-accent {{
  background: var(--if-accent-soft);
  border-color: var(--if-accent-soft);
  color: var(--if-accent);
}}
.if-chip.is-success {{ background: var(--if-success-soft); border-color: var(--if-success-soft); color: var(--if-success-text); }}
.if-chip.is-warning {{ background: var(--if-warning-soft); border-color: var(--if-warning-soft); color: var(--if-warning-text); }}
.if-chip.is-danger  {{ background: var(--if-danger-soft);  border-color: var(--if-danger-soft);  color: var(--if-danger-text); }}
.if-dot {{ width: 7px; height: 7px; border-radius: 50%; flex: 0 0 7px; }}

.if-section {{ margin: 1.3rem 0 .85rem; }}
.if-kicker {{
  color: var(--if-accent);
  font-size: .72rem;
  font-weight: 680;
  letter-spacing: .1em;
  text-transform: uppercase;
}}
.if-section h2 {{ margin: .28rem 0 .2rem; font-size: 1.32rem; }}
.if-section p {{ margin: 0; color: var(--if-muted); font-size: .9rem; max-width: 74ch; }}

.if-grid {{ display: grid; gap: .75rem; margin: .1rem 0 .6rem; }}
.if-grid-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.if-grid-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.if-grid-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}

.if-card {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: 1rem 1.05rem;
  box-shadow: var(--if-shadow);
}}
.if-stat-label {{
  color: var(--if-muted);
  font-size: .73rem;
  font-weight: 640;
  letter-spacing: .07em;
  text-transform: uppercase;
}}
.if-stat-value {{
  color: var(--if-ink);
  font-size: 1.75rem;
  font-weight: 640;
  letter-spacing: -.025em;
  line-height: 1.15;
  margin-top: .4rem;
}}
.if-stat-note {{ color: var(--if-muted); font-size: .82rem; margin-top: .3rem; }}

.if-engine {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: 1rem 1.05rem;
  box-shadow: var(--if-shadow);
  border-top: 2px solid var(--if-engine-accent, var(--if-accent));
}}
.if-engine-tag {{
  color: var(--if-engine-accent, var(--if-accent));
  font-size: .71rem; font-weight: 680; letter-spacing: .1em; text-transform: uppercase;
}}
.if-engine strong {{ display: block; margin: .3rem 0 .25rem; font-size: .99rem; color: var(--if-ink); font-weight: 620; }}
.if-engine p {{ margin: 0; color: var(--if-muted); font-size: .85rem; line-height: 1.55; }}

.if-callout {{
  display: flex; gap: .7rem; align-items: flex-start;
  background: var(--if-soft);
  border: 1px solid var(--if-border);
  border-left: 3px solid var(--if-muted);
  border-radius: var(--if-radius-sm);
  padding: .8rem .95rem;
  margin: .55rem 0;
  color: var(--if-secondary);
  font-size: .88rem;
  line-height: 1.6;
}}
.if-callout strong {{ color: var(--if-ink); font-weight: 620; }}
.if-callout code {{
  font-family: var(--if-mono); font-size: .82rem;
  background: var(--if-panel); border: 1px solid var(--if-border);
  border-radius: 5px; padding: .08rem .32rem; color: var(--if-accent);
  overflow-wrap: anywhere;
}}
.if-callout.is-accent  {{ background: var(--if-accent-soft);  border-color: var(--if-accent-soft);  border-left-color: var(--if-accent); }}
.if-callout.is-success {{ background: var(--if-success-soft); border-color: var(--if-success-soft); border-left-color: var(--if-success); }}
.if-callout.is-warning {{ background: var(--if-warning-soft); border-color: var(--if-warning-soft); border-left-color: var(--if-warning); }}
.if-callout.is-danger  {{ background: var(--if-danger-soft);  border-color: var(--if-danger-soft);  border-left-color: var(--if-danger); }}
.if-callout-icon {{ font-size: .95rem; line-height: 1.5; flex: 0 0 auto; }}

.if-insight {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: .95rem 1.05rem;
  box-shadow: var(--if-shadow);
  height: 100%;
  display: flex; flex-direction: column; gap: .45rem;
}}
.if-insight-head {{ display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; }}
.if-insight-sev {{
  font-size: .71rem; font-weight: 680; letter-spacing: .08em; text-transform: uppercase;
}}
.if-insight-cat {{ color: var(--if-muted); font-size: .74rem; margin-left: auto; }}
.if-insight h4 {{ margin: 0; font-size: .97rem; font-weight: 620; color: var(--if-ink); letter-spacing: -.01em; }}
.if-insight-body {{ color: var(--if-secondary); font-size: .875rem; line-height: 1.6; }}
.if-insight-foot {{
  display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
  margin-top: auto; padding-top: .55rem; border-top: 1px solid var(--if-border);
}}
.if-insight-action {{ color: var(--if-muted); font-size: .82rem; }}

.if-kv {{
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  overflow: hidden;
  background: var(--if-panel);
  box-shadow: var(--if-shadow);
}}
.if-kv-title {{
  padding: .6rem .95rem;
  background: var(--if-soft);
  border-bottom: 1px solid var(--if-border);
  color: var(--if-muted);
  font-size: .73rem; font-weight: 660; letter-spacing: .07em; text-transform: uppercase;
}}
.if-kv-row {{
  display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
  padding: .52rem .95rem;
  border-bottom: 1px solid var(--if-border);
  font-size: .87rem;
}}
.if-kv-row:last-child {{ border-bottom: 0; }}
.if-kv-key {{ color: var(--if-muted); overflow-wrap: anywhere; }}
.if-kv-val {{
  color: var(--if-ink); font-weight: 560; text-align: right;
  font-variant-numeric: tabular-nums; overflow-wrap: anywhere;
}}
.if-kv-empty {{ padding: .8rem .95rem; color: var(--if-muted); font-size: .86rem; }}

.if-score {{
  display: flex; align-items: center; gap: 1.1rem;
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: 1.05rem 1.2rem;
  box-shadow: var(--if-shadow);
}}
.if-score-ring {{
  width: 84px; height: 84px; flex: 0 0 84px; border-radius: 50%;
  display: grid; place-items: center;
  background: conic-gradient(var(--if-score-color) calc(var(--if-score) * 1%), var(--if-soft) 0);
}}
.if-score-ring-inner {{
  width: 66px; height: 66px; border-radius: 50%;
  background: var(--if-panel);
  display: grid; place-items: center;
  font-size: 1.35rem; font-weight: 660; color: var(--if-ink); letter-spacing: -.02em;
}}
.if-score-meta h3 {{ margin: 0 0 .18rem; font-size: 1.02rem; font-weight: 620; }}
.if-score-meta p {{ margin: 0; color: var(--if-muted); font-size: .86rem; }}
.if-score-grade {{
  font-size: .73rem; font-weight: 680; letter-spacing: .08em; text-transform: uppercase;
  color: var(--if-score-color);
}}

.if-msg {{ display: flex; gap: .7rem; margin: .55rem 0; }}
.if-msg-avatar {{
  width: 30px; height: 30px; flex: 0 0 30px; border-radius: 9px;
  display: grid; place-items: center;
  font-size: .72rem; font-weight: 680;
  background: var(--if-soft); color: var(--if-secondary);
  border: 1px solid var(--if-border);
}}
.if-msg.is-user {{ flex-direction: row-reverse; }}
.if-msg.is-user .if-msg-avatar {{ background: var(--if-accent-solid); color: var(--if-on-accent); border-color: var(--if-accent-solid); }}
.if-msg-bubble {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: .7rem .95rem;
  max-width: 78%;
  color: var(--if-ink);
  font-size: .9rem; line-height: 1.62;
  box-shadow: var(--if-shadow);
}}
.if-msg.is-user .if-msg-bubble {{
  background: var(--if-accent-soft);
  border-color: var(--if-accent-soft);
}}
.if-msg-agents {{ display: flex; gap: .3rem; flex-wrap: wrap; margin-top: .55rem; }}
.if-msg-bubble p {{ margin: 0 0 .5rem; font-size: .9rem; }}
.if-msg-bubble p:last-of-type {{ margin-bottom: 0; }}
.if-msg-bubble h3, .if-msg-bubble h4, .if-msg-bubble h5, .if-msg-bubble h6 {{
  font-size: .95rem;
  font-weight: 640;
  margin: .2rem 0 .45rem;
}}
.if-msg-bubble ul, .if-msg-bubble ol {{ margin: .1rem 0 .55rem; padding-left: 1.15rem; }}
.if-msg-bubble li {{ font-size: .9rem; margin-bottom: .18rem; }}
.if-msg-bubble code {{
  font-family: var(--if-mono); font-size: .82rem;
  background: var(--if-soft); border: 1px solid var(--if-border);
  border-radius: 5px; padding: .06rem .3rem; color: var(--if-accent);
}}
.if-msg-bubble a {{ color: var(--if-accent); }}

.if-timeline {{ position: relative; padding-left: 1.35rem; margin: .35rem 0; }}
.if-timeline:before {{
  content: ""; position: absolute; left: 6px; top: .45rem; bottom: .45rem;
  width: 1px; background: var(--if-border);
}}
.if-step {{ position: relative; padding: 0 0 .95rem; }}
.if-step:last-child {{ padding-bottom: 0; }}
.if-step:before {{
  content: ""; position: absolute; left: -1.35rem; top: .34rem;
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--if-panel);
  border: 2px solid var(--if-step-color, var(--if-accent));
}}
.if-step-head {{ display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; }}
.if-step-name {{ color: var(--if-ink); font-size: .88rem; font-weight: 600; }}
.if-step-detail {{
  color: var(--if-muted); font-size: .82rem; margin-top: .25rem;
  font-family: var(--if-mono); line-height: 1.55; overflow-wrap: anywhere;
}}

.if-routes {{
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  overflow: hidden;
  background: var(--if-panel);
  box-shadow: var(--if-shadow);
  margin-bottom: .8rem;
}}
.if-routes-head {{
  padding: .6rem .95rem;
  background: var(--if-soft);
  border-bottom: 1px solid var(--if-border);
  color: var(--if-ink);
  font-size: .84rem; font-weight: 620;
}}
.if-route {{
  display: flex; align-items: center; gap: .7rem;
  padding: .52rem .95rem;
  border-bottom: 1px solid var(--if-border);
}}
.if-route:last-child {{ border-bottom: 0; }}
.if-method {{
  flex: 0 0 46px; text-align: center;
  font-size: .68rem; font-weight: 700; letter-spacing: .05em;
  border-radius: 5px; padding: .16rem 0;
}}
.if-method.get {{ background: var(--if-info-soft); color: var(--if-info-text); }}
.if-method.post {{ background: var(--if-success-soft); color: var(--if-success-text); }}
.if-route-path {{
  font-family: var(--if-mono); font-size: .82rem; color: var(--if-ink);
  flex: 0 0 auto; overflow-wrap: anywhere;
}}
.if-route-desc {{ color: var(--if-muted); font-size: .82rem; margin-left: auto; text-align: right; }}

.if-empty {{
  background: var(--if-panel);
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius);
  padding: 1.6rem 1.7rem;
  box-shadow: var(--if-shadow-lg);
}}
.if-empty h2 {{ margin: 0 0 .3rem; }}
.if-empty > p {{ color: var(--if-muted); margin: 0 0 1.15rem; max-width: 70ch; }}
.if-stepcard {{
  border: 1px solid var(--if-border);
  border-radius: var(--if-radius-sm);
  background: var(--if-soft);
  padding: .9rem 1rem;
}}
.if-stepcard b {{ color: var(--if-accent); font-size: .74rem; letter-spacing: .1em; }}
.if-stepcard strong {{ display: block; margin: .28rem 0 .2rem; color: var(--if-ink); font-size: .95rem; }}
.if-stepcard span {{ color: var(--if-muted); font-size: .84rem; line-height: 1.55; }}

.if-note {{ color: var(--if-muted); font-size: .8rem; margin: .35rem 0 0; }}

@media (max-width: 1100px) {{
  .if-grid-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 820px) {{
  .if-grid-2, .if-grid-3, .if-grid-4 {{ grid-template-columns: 1fr; }}
  .if-hero-top {{ flex-direction: column; align-items: flex-start; }}
  .if-msg-bubble {{ max-width: 100%; }}
  .if-route {{ flex-wrap: wrap; }}
  .if-route-desc {{ margin-left: 0; text-align: left; width: 100%; }}
  .stAppViewMain .block-container, .main .block-container {{ padding: 1rem 1rem 3rem; }}
}}
</style>
"""


def inject(theme: str | None = None) -> None:
    """Write the theme's stylesheet into the page."""

    st.markdown(build_css(normalize(theme) if theme else current_theme()), unsafe_allow_html=True)


# -------------------------------------------------------------------- plotly
def plotly_layout(theme: str | None = None) -> dict[str, Any]:
    """Layout overrides that make a figure look native to the dashboard."""

    t = tokens(theme)
    axis = {
        "gridcolor": t["grid"],
        "linecolor": t["axis"],
        "zerolinecolor": t["grid"],
        "tickfont": {"color": t["muted"], "size": 11},
        "title": {"font": {"color": t["secondary"], "size": 12}},
        "automargin": True,
    }
    return {
        "paper_bgcolor": t["panel"],
        "plot_bgcolor": t["panel"],
        "colorway": list(t["colorway"]),
        "font": {"family": FONT_STACK, "color": t["secondary"], "size": 12},
        "title": {
            "font": {"family": FONT_STACK, "color": t["ink"], "size": 15},
            "x": 0.012,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
            "pad": {"l": 8, "t": 6},
        },
        "margin": {"l": 64, "r": 28, "t": 56, "b": 62},
        "hoverlabel": {
            "bgcolor": t["panel"],
            "bordercolor": t["border_strong"],
            "font": {"family": FONT_STACK, "color": t["ink"], "size": 12},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": t["secondary"], "size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        "xaxis": axis,
        "yaxis": dict(axis),
        "separators": ".,",
    }


def style_figure(fig: Any, theme: str | None = None) -> Any:
    """Re-skin a figure produced by the engine renderer, in place.

    The analytics engine owns *what* is drawn; this owns *how it looks*. Every
    step is best-effort so an unusual trace type can never break a chart.
    """

    t = tokens(theme)
    try:
        fig.update_layout(**plotly_layout(theme))
    except Exception:  # pragma: no cover - defensive
        return fig

    for trace in fig.data:
        kind = getattr(trace, "type", "")
        try:
            if kind == "bar":
                # A hairline of surface between bars keeps adjacent fills readable.
                trace.marker.line.color = t["panel"]
                trace.marker.line.width = 1
                if trace.marker.color is None:
                    trace.marker.color = t["colorway"][0]
                try:
                    trace.marker.cornerradius = 4
                except Exception:
                    pass
            elif kind in ("scatter", "scattergl"):
                if getattr(trace, "name", "") == "anomaly":
                    trace.marker.color = t["danger"]
                    trace.marker.size = 9
                elif "lines" in (getattr(trace, "mode", "") or ""):
                    if trace.line.color in (None, "#4C78A8"):
                        trace.line.color = t["colorway"][0]
                    trace.line.width = 2
                    if trace.marker.size is None:
                        trace.marker.size = 6
                elif trace.marker.color is None:
                    trace.marker.color = t["colorway"][0]
                    trace.marker.size = 8
                    trace.marker.opacity = 0.75
                if getattr(trace, "fill", None) == "toself":
                    trace.fillcolor = _rgba(t["colorway"][0], 0.14)
            elif kind == "heatmap":
                diverging = (trace.zmin is not None and trace.zmin < 0) or trace.colorscale in (
                    "RdBu", "rdbu",
                )
                trace.colorscale = t["diverging"] if diverging else t["sequential"]
                trace.colorbar.outlinewidth = 0
                trace.colorbar.tickfont = {"color": t["muted"], "size": 10}
            elif kind == "box":
                trace.marker.outliercolor = t["danger"]
                trace.line.width = 1.5
            elif kind == "funnel":
                trace.marker.color = t["colorway"][0]
                trace.connector.line.color = t["border_strong"]
            elif kind == "table":
                trace.header.fill.color = t["soft"]
                trace.header.font = {"color": t["ink"], "family": FONT_STACK, "size": 12}
                trace.header.line.color = t["border"]
                trace.cells.fill.color = t["panel"]
                trace.cells.font = {"color": t["secondary"], "family": FONT_STACK, "size": 11}
                trace.cells.line.color = t["border"]
        except Exception:  # pragma: no cover - one odd trace must not break the page
            continue

    # A lone series is already named by the chart title; a legend box would just
    # repeat it (the same rule the dataviz method applies).
    legend_entries = sum(1 for trace in fig.data if getattr(trace, "showlegend", None) is not False)
    if legend_entries < 2:
        fig.update_layout(showlegend=False)

    for annotation in fig.layout.annotations or ():
        try:
            annotation.font.color = t["muted"]
        except Exception:  # pragma: no cover
            continue
    return fig


def plotly_config() -> dict[str, Any]:
    """Chart toolbar config: keep the useful tools, drop the noisy ones."""

    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        "displayModeBar": "hover",
    }


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
