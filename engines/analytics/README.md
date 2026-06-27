# IntelliFlow Engine 2 — Analytics & FAANG-style EDA

The "understanding engine." It transforms a raw dataset into ranked, actionable
insights with the rigour of the analytics systems used at large product
companies — profiling, correlation/feature intelligence, funnels, cohort
retention, event-stream analytics, anomaly detection, and modelling
recommendations — all behind one call: `run_eda(...)`.

## Quick start

```python
import pandas as pd
from engines.analytics import run_eda

df = pd.read_csv("sales_data.csv")  # date, user_id, revenue, region, product

report = run_eda(
    df,
    target_column="revenue",
    timestamp_column="date",
    user_id_column="user_id",
    event_column="product",
    segment_columns=["region"],
    funnel_steps=["launch", "purchase", "repeat"],
    target_event="purchase",
)

print(report.data_quality_score, report.data_quality_grade)
for insight in report.insights[:10]:
    print(insight.severity, insight.title, "->", insight.action)

report.to_json("eda_output/report.json")          # full metrics + findings
report.to_html("eda_output/dashboard.html")        # interactive Plotly dashboard
report.save_csv_summaries("eda_output")            # CSVs for downstream tools
report.save_charts("eda_output", fmt="png")        # static PNG/SVG charts
```

`run_eda` runs exactly the capabilities your columns unlock: profiling,
correlation and recommendations always run; funnel needs `funnel_steps`;
retention needs `user_id_column` + `timestamp_column`; event-stream needs a
`timestamp_column`. Each capability is **isolated** — if one fails it is recorded
in `report.errors` and the rest still run.

## The structured output contract

Every finding from every capability is the same `Insight` object, so reports,
dashboards and the API all speak one language:

| Field | Meaning |
| --- | --- |
| `insight` | plain-English discovery |
| `metric` | the quantified evidence (JSON-safe) |
| `visualization` | a `ChartSpec` rendered lazily to Plotly/Matplotlib |
| `action` | the recommended next step |
| `confidence` | statistical-significance / data-quality score in `[0, 1]` |
| `severity` | `ok` / `info` / `warning` / `critical` |

Charts are described as **data only** (`ChartSpec`); no plotting library is
imported during analysis. `visualization.py` turns a spec into an interactive
Plotly figure (for HTML) or a static Matplotlib image (for PNG/SVG) on demand.

## Capabilities & key modules

| # | Module | What it produces |
| --- | --- | --- |
| 1 | `profiling.py` | nulls, distribution (skew/kurtosis/modality/outliers), cardinality, type-mismatch flags, **0–100 quality score** |
| 2 | `correlation.py` | Pearson/Spearman/**Cramér's V**, **VIF** multicollinearity, mutual-information target ranking, non-linear detection, heatmap + dendrogram |
| 3 | `funnel.py` | step conversion, drop-off, segment funnels, time-to-convert, **Wilson confidence intervals** |
| 4 | `retention.py` | cohorts, N-day/week curves (**maturity-corrected**), cohort×offset heatmap, churn-risk features, declining-cohort trend |
| 5 | `events.py` | time-series aggregation, sessions, journey n-grams/transitions, **Kaplan-Meier** time-to-event, velocity |
| 6 | `anomaly.py` | Z-score/IQR, **Isolation Forest** (with feature contributions), **STL** residual anomalies, level shifts, co-movement root causes |
| 7 | `recommendations.py` | constant/quasi-constant, duplicate features, **leakage** warnings, encoding (one-hot/target/embedding), scaling |

Supporting modules: `base.py` (core types + `json_safe`), `utils.py` (semantic
type inference, Wilson interval, sampling), `visualization.py` (renderers),
`report.py` (JSON/HTML/CSV/PNG), `pipeline.py` (`run_eda` orchestrator).

## Design notes worth knowing

- **Maturity-correct retention.** A cohort that signed up yesterday cannot have a
  Day-30 number, so it is excluded from the Day-30 denominator rather than
  counted as zero. Naive implementations let new cohorts drag the curve down.
- **Robust anomaly stats.** Univariate detection uses a median/MAD modified
  Z-score so the outliers it hunts don't distort the threshold; the Isolation
  Forest reports per-feature contributions so an alert explains *why*.
- **Identifiers ≠ measures.** Integer/near-unique id columns are detected and
  kept out of correlation, VIF and anomaly metrics; `run_eda` also excludes the
  declared `user_id`/`event` columns from feature-level analysis.
- **Always serialisable.** Every `to_dict()` is JSON-safe (NaN/Inf → null, numpy
  and pandas scalars coerced), and the report carries a final `json_safe` net.
- **Scales to ~1M rows.** Super-linear analyses (mutual information, Isolation
  Forest) deterministically sample large frames and annotate that they did.

## API

The engine is exposed through the gateway under `/analytics`
(`api/routers/analytics.py`):

- `POST /analytics/analyze` — full EDA from JSON rows.
- `POST /analytics/upload-analyze` — CSV/JSON/Parquet/Excel upload
  (`?report_format=html` returns the dashboard).
- `POST /analytics/profile` — quick profiling only.
- `GET /analytics/capabilities` — list available capabilities.

## Tests

```bash
python -m pytest tests/ -q
```

Covers every statistical function and the required edge cases (single row, all
nulls, no variance, single column) plus end-to-end orchestration and exporters.
Run the demo end-to-end with `python scratch_analytics_test.py`.
