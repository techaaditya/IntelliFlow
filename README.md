# IntelliFlow - Unified Intelligent Data Platform

IntelliFlow is a unified data platform that consolidates independent data science workflows into a single, cohesive system. This repository will eventually feature a shared data ingestion layer feeding three pluggable service engines, all accessible through a single API gateway and dashboard UI.

Currently, **Engine 1 (AutoML Pipeline)** is fully implemented.

## Features

* **Engine 1 (AutoML):** Automatically preprocesses data, searches the hyperparameter space using Optuna, tracks experiments with MLflow, and deploys the best model.
* **Engine 2 (Analytics & EDA):** *(Coming Soon)* Automated exploratory data analysis and interactive dashboards.
* **Engine 3 (Agent Orchestration):** *(Coming Soon)* Multi-agent CrewAI orchestration using Claude as the LLM backbone.

## Installation and Setup

### 1. Create a Virtual Environment

It is recommended to run IntelliFlow in a Python virtual environment to avoid dependency conflicts.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install the required Python packages using `pip`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Verification Script

We have provided a verification script (`scratch_test.py`) that runs the complete AutoML pipeline on the Iris dataset. It preprocesses the data, runs Optuna hyperparameter optimization across different model families (Random Forest, XGBoost, LightGBM), and logs everything to MLflow.

```powershell
python scratch_test.py
```

### 4. View MLflow Dashboard

To view the tracking dashboard, artifacts, confusion matrices, and feature importance charts:

```powershell
mlflow ui
```
Then navigate your browser to `http://127.0.0.1:5000`.

## Docker Deployment (Future)

Once all engines, APIs, and the Streamlit UI are built, you can run the entire platform using Docker Compose:

```powershell
docker compose up --build
```
