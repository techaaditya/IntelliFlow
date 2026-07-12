"""Demo / verification script for IntelliFlow Engine 3 (Agent Orchestration).

Loads the Iris dataset and asks the agent crew a question that exercises the
flagship cross-engine feature: the crew plans, inspects the data, and triggers
Engine 1 (AutoML) to actually train and register a model — then synthesizes a
plain-English answer.

Requires an LLM key (see ``.env`` / ``.env.example``):

    OLLAMA_API_KEY=...     # Ollama Cloud key for gpt-oss:120b

Run:

    python scratch_agent_test.py
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd
from sklearn.datasets import load_iris

from engines.agents import get_agent_config, run_agent_query

warnings.filterwarnings("ignore")

# The LLM answer may contain Unicode; make console output UTF-8 safe on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - older stdio
    pass


def iris_frame() -> pd.DataFrame:
    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()
    df["species"] = pd.Categorical.from_codes(bunch.target, bunch.target_names)
    return df


def main() -> None:
    config = get_agent_config()
    print("=" * 70)
    print("IntelliFlow Engine 3 — Agent Orchestration demo")
    print(f"LLM backbone : {config.model} via {config.host}")
    print(f"API key set  : {config.has_credentials}")
    print("=" * 70)
    if not config.has_credentials:
        print("\nNo OLLAMA_API_KEY found. Copy .env.example to .env and add your key.")
        return

    df = iris_frame()
    question = (
        "Which measurements best separate the iris species, and how accurately "
        "can a model predict the species? Train a model and summarize the result."
    )
    print(f"\nDataset: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Question: {question}\n")
    print("Running the crew (this calls the LLM and may train a model)...\n")

    result = run_agent_query(question, df, session_id="demo-iris")

    print("-" * 70)
    print("ANSWER:\n")
    print(result.answer)
    print("-" * 70)
    print("Agents used   :", ", ".join(result.agents_used) or "(none)")
    print("Tools used    :", ", ".join(result.tools_used) or "(none)")
    print("Model endpoint:", result.model_endpoint or "(no model trained)")
    print("Charts        :", len(result.charts))
    print("Insights      :", len(result.insights))
    if result.errors:
        print("Recovered errors:", result.errors)
    print("\nAgent trace (steps):")
    for step in result.trace:
        label = step.get("tool") or step.get("agent") or step.get("step")
        print(f"  - {step.get('step'):10} {label}")


if __name__ == "__main__":
    main()
