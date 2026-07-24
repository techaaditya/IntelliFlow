# IntelliFlow - Unified Intelligent Data Platform

IntelliFlow is a unified data platform that consolidates independent data science workflows into a single, cohesive system. A shared dataset feeds three pluggable service engines, all accessible through a single API gateway and Streamlit dashboard.

All three engines are implemented: **Engine 1 (AutoML Pipeline)**, **Engine 2 (Analytics & EDA)**, and **Engine 3 (Agent Orchestration)**.

## Features

* **Engine 1 (AutoML):** Automatically preprocesses data, searches the hyperparameter space using Optuna, tracks experiments with MLflow, and deploys the best model.
* **Engine 2 (Analytics & EDA):** FAANG-style exploratory data analysis - data profiling with a 0–100 quality score, correlation/feature intelligence (Pearson/Spearman/Cramér's V, VIF, mutual-information target ranking), funnel & cohort-retention analysis, event-stream analytics (sessions, journeys, Kaplan-Meier), anomaly detection (Isolation Forest + STL), and modelling recommendations. Exports JSON, an interactive HTML dashboard, CSV summaries and PNG/SVG charts. See [`engines/analytics/README.md`](engines/analytics/README.md).
* **Engine 3 (Agent Orchestration):** A multi-agent crew (Planner → Data Analyst → ML Engineer → Visualizer → Researcher → Synthesizer) that answers natural-language questions about the loaded dataset. Agents call real tools - including **Engine 1 (AutoML)** and **Engine 2 (Analytics)** in-process - so a question like *"predict churn"* actually trains and registers a model. The LLM backbone is any OpenAI-compatible endpoint, defaulting to **Ollama Cloud's `gpt-oss:120b`**; long-term memory is a lightweight SQLite store. Built without the heavy CrewAI/LangChain/ChromaDB stack for a robust, conflict-free install.

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
lightgbm (AutoML), fastapi, uvicorn, pydantic (API gateway), and requests +
ddgs (Engine 3 agent crew). Engine 3 also needs an LLM key - see
[Engine 3 - Agent Orchestration](#engine-3--agent-orchestration) below.

---

## Running the Engines

Each engine ships a self-contained verification/demo script you can run directly,
and is also exposed through the shared **API gateway** (see below).

### Engine 1 - AutoML Pipeline

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

### Engine 2 - Analytics & EDA

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

### Engine 3 - Agent Orchestration

Engine 3 answers natural-language questions with a crew of agents that share the
same dataset and can trigger Engine 1 and Engine 2 as tools.

**1. Configure the LLM backbone.** Copy `.env.example` to `.env` and add your
[Ollama Cloud](https://ollama.com/settings/keys) API key:

```bash
cp .env.example .env
# then edit .env:
#   OLLAMA_API_KEY=your_key_here
#   OLLAMA_MODEL=gpt-oss:120b
```

The backbone is provider-swappable - point `OLLAMA_HOST`/`OLLAMA_MODEL` at a
local Ollama daemon, Groq, or any OpenAI-compatible endpoint. Non-secret
settings (model, temperature, max steps) also live under `agents:` in
[`config.yaml`](config.yaml).

**2. Run a demo** (builds a dataset, asks the crew, prints the answer, the
agents used, and any model endpoint it created):

```bash
python scratch_agent_test.py
```

Programmatic use:

```python
import pandas as pd
from engines.agents import run_agent_query

result = run_agent_query(
    "Which features best predict the target, and how accurate is a model?",
    pd.read_csv("data.csv"),
)
print(result.answer)
print("Agents used:", result.agents_used)
print("Model endpoint:", result.model_endpoint)  # set if the crew trained a model
result.to_dict()  # JSON-safe: answer, charts, insights, model_endpoint, trace
```

The crew's tools run offline-safe: without a key the engine imports fine and the
`/agents/capabilities` route still works; the web-search tool degrades
gracefully when `ddgs` or the network is unavailable.

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
| Engine 3 (Agents) | `POST /agents/query`, `POST /agents/upload-query`, `GET /agents/history`, `GET /agents/capabilities` |

Example - run EDA on an uploaded file and get the interactive dashboard back:

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

## Streamlit Dashboard

The integrated dashboard gives one shared dataset workspace for all three
engines:

```bash
streamlit run ui/app.py
```

Open the local URL Streamlit prints, usually **http://localhost:8501**. The UI
has six tabs: dataset upload (CSV/Excel/JSON/Parquet or sample datasets),
EDA reports (Engine 2), AutoML training (Engine 1), registered-model
prediction, an **Agents chat tab** (Engine 3), and a compact API route
reference.

---

## Running the Tests

```bash
python -m pytest tests/ -q
```

## Docker Deployment

A `docker-compose.yml` is included to run the API, Streamlit UI, and MLflow
together:

```bash
docker compose up --build
```

Set `OLLAMA_API_KEY` (and optionally `OLLAMA_HOST`/`OLLAMA_MODEL`) in your
shell or a `.env` file before starting - it's passed through to the `api`
service for Engine 3. This compose setup is not part of the automated test
suite; verify it in your own environment before relying on it.
