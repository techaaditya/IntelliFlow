# IntelliFlow - Unified Intelligent Data Platform

IntelliFlow is a unified data platform that consolidates independent data science workflows into a single, cohesive system. This repository will eventually feature a shared data ingestion layer feeding three pluggable service engines, all accessible through a single API gateway and dashboard UI.

Currently, **Engine 1 (AutoML Pipeline)** and **Engine 2 (Analytics & EDA)** are fully implemented.

## Features

* **Engine 1 (AutoML):** Automatically preprocesses data, searches the hyperparameter space using Optuna, tracks experiments with MLflow, and deploys the best model.
* **Engine 2 (Analytics & EDA):** FAANG-style exploratory data analysis — data profiling with a 0–100 quality score, correlation/feature intelligence (Pearson/Spearman/Cramér's V, VIF, mutual-information target ranking), funnel & cohort-retention analysis, event-stream analytics (sessions, journeys, Kaplan-Meier), anomaly detection (Isolation Forest + STL), and modelling recommendations. Exports JSON, an interactive HTML dashboard, CSV summaries and PNG/SVG charts. See [`engines/analytics/README.md`](engines/analytics/README.md).
* **Engine 3 (Agent Orchestration):** *(Coming Soon)* Multi-agent CrewAI orchestration using Claude as the LLM backbone.

## Prerequisites

* **Python 3.10+** (developed and tested on 3.11)
* `pip` and `venv`

## Installation and Setup

### 1. Create and activate a virtual environment

It is recommended to run IntelliFlow in a Python virtual environment to avoid dependency conflicts.

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs everything for all engines: pandas, numpy, scikit-learn, scipy,
statsmodels, plotly, seaborn, matplotlib (data/EDA), optuna, mlflow, xgboost,
lightgbm (AutoML), and fastapi, uvicorn, pydantic (API gateway).

---

## Running the Engines

Each engine ships a self-contained verification/demo script you can run directly,
and is also exposed through the shared **API gateway** (see below).

### Engine 1 — AutoML Pipeline

`scratch_test.py` runs the complete AutoML pipeline on the Iris dataset:
preprocessing, Optuna hyperparameter optimization across model families
(Random Forest, XGBoost, LightGBM), and experiment tracking + model registration
in MLflow.

```bash
python scratch_test.py
```

View the tracking dashboard (runs, metrics, artifacts, confusion matrices,
feature importances):

```bash
mlflow ui          # then open http://127.0.0.1:5000
```

Programmatic use:

```python
import pandas as pd
from engines.automl.pipeline import run_automl

result = run_automl(dataset=df, target_column="target", metric="accuracy", n_trials=20)
print(result["best_score"], result["model_uri"])
```

### Engine 2 — Analytics & EDA

`scratch_analytics_test.py` builds a synthetic product-analytics dataset and runs
the full EDA suite, writing a JSON report, an interactive HTML dashboard, CSV
summaries, and PNG charts to `./eda_output/`.

```bash
python scratch_analytics_test.py
# then open eda_output/dashboard.html
```

Programmatic use:

```python
import pandas as pd
from engines.analytics import run_eda

report = run_eda(
    pd.read_csv("sales_data.csv"),
    target_column="revenue", timestamp_column="date",
    user_id_column="user_id", event_column="product",
    funnel_steps=["launch", "purchase", "repeat"],
)
print(report.data_quality_score, report.data_quality_grade)
report.to_html("dashboard.html")
report.to_json("report.json")
report.save_csv_summaries("eda_output")
```

See [`engines/analytics/README.md`](engines/analytics/README.md) for the full
capability reference and design notes.

### Engine 3 — Agent Orchestration

*(Coming soon)* Multi-agent CrewAI orchestration using Claude as the LLM backbone.

---

## API Gateway

A single FastAPI gateway exposes all implemented engines. Start it with:

```bash
uvicorn api.main:app --reload
```

Then open the interactive Swagger UI at **http://127.0.0.1:8000/docs**.

| Engine | Endpoints |
| --- | --- |
| Health | `GET /health` |
| Engine 1 (AutoML) | `POST /automl/train`, `POST /automl/upload-train`, `POST /automl/predict`, `GET /automl/model-info` |
| Engine 2 (Analytics) | `POST /analytics/analyze`, `POST /analytics/upload-analyze`, `POST /analytics/profile`, `GET /analytics/capabilities` |

Example — run EDA on an uploaded file and get the interactive dashboard back:

```bash
curl -X POST "http://127.0.0.1:8000/analytics/upload-analyze?report_format=html" \
  -F "file=@sales_data.csv" \
  -F "target_column=revenue" \
  -F "timestamp_column=date" \
  -F "user_id_column=user_id" \
  -F "funnel_steps=launch,purchase,repeat" \
  -o dashboard.html
```

---

## Running the Tests

```bash
python -m pytest tests/ -q
```

## Docker Deployment (Future)

Once all engines, APIs, and the Streamlit UI are built, you can run the entire platform using Docker Compose:

```bash
docker compose up --build
```
