"""Tests for IntelliFlow Engine 3 (Agent Orchestration).

The LLM backbone is mocked (``FakeLLM``) so every test runs offline and
deterministically — no network, no API key required. One test exercises the
real cross-engine ``automl_tool`` path (which trains a tiny model in-process).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from engines.agents import AgentMemory, run_agent_query
from engines.agents.crew import AgentQueryResult
from engines.agents.llm import ChatResponse, ToolCall
from engines.agents.memory import AgentMemory as Memory
from engines.agents.tools import AgentContext, build_tool_registry


# --------------------------------------------------------------------- fakes
def _assistant_raw(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


class FakeLLM:
    """A scripted LLM: ``chat`` pops from ``script``; ``complete`` is deterministic."""

    def __init__(self, script: list[ChatResponse] | None = None, embedding: list[float] | None = None):
        self.script = list(script or [])
        self.embedding = embedding
        self.chat_calls = 0
        self.complete_calls = 0

    def embed(self, text: str):
        return self.embedding

    def complete(self, prompt: str, system: str | None = None, temperature=None) -> str:
        self.complete_calls += 1
        if system and "Planner" in system:
            return "1. Inspect the data.\n2. Answer the question."
        return "Final synthesized answer with the key numbers."

    def chat(self, messages, tools=None, temperature=None, tool_choice=None) -> ChatResponse:
        self.chat_calls += 1
        if self.script:
            return self.script.pop(0)
        return ChatResponse(content="analysis complete", tool_calls=[], raw=_assistant_raw("analysis complete"))


def _tool_call_response(name: str, arguments: dict, call_id: str = "c1") -> ChatResponse:
    raw = _assistant_raw(
        "",
        [{"id": call_id, "type": "function",
          "function": {"name": name, "arguments": json.dumps(arguments)}}],
    )
    return ChatResponse(content="", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)], raw=raw)


def _stop_response() -> ChatResponse:
    return ChatResponse(content="done", tool_calls=[], raw=_assistant_raw("done"))


# --------------------------------------------------------------------- crew
def test_run_agent_query_basic(feature_frame, tmp_path):
    memory = Memory(path=str(tmp_path / "mem.db"))
    llm = FakeLLM(script=[
        _tool_call_response("pandas_tool", {"operation": "column_stats", "column": "target"}),
        _stop_response(),
    ])
    result = run_agent_query("Describe the target column.", feature_frame, llm=llm, memory=memory, session_id="s1")

    assert isinstance(result, AgentQueryResult)
    assert result.answer  # synthesizer produced text
    assert "Planner" in result.agents_used
    assert "Data Analyst" in result.agents_used  # pandas_tool owner
    assert "Synthesizer" in result.agents_used
    assert "pandas_tool" in result.tools_used
    assert result.errors == {}
    # Whole result must be JSON-serialisable (platform-wide contract).
    json.dumps(result.to_dict())


def test_run_agent_query_attaches_charts(feature_frame, tmp_path):
    memory = Memory(path=str(tmp_path / "mem.db"))
    llm = FakeLLM(script=[
        _tool_call_response("plotly_tool", {"kind": "histogram", "title": "Target dist", "x": "target"}),
        _stop_response(),
    ])
    result = run_agent_query("Plot the target.", feature_frame, llm=llm, memory=memory)
    assert len(result.charts) == 1
    assert result.charts[0].title == "Target dist"
    assert "Visualizer" in result.agents_used
    json.dumps(result.to_dict())


def test_bad_column_is_soft_hint(feature_frame, tmp_path):
    """Calling a tool with a bad column coaches the model — it is not a logged error."""

    memory = Memory(path=str(tmp_path / "mem.db"))
    llm = FakeLLM(script=[
        _tool_call_response("pandas_tool", {"operation": "value_counts", "column": "does_not_exist"}),
        _stop_response(),
    ])
    result = run_agent_query("Count a missing column.", feature_frame, llm=llm, memory=memory)
    assert result.answer  # still synthesized
    assert result.errors == {}  # usage hint, not an error
    assert any("not found" in str(s.get("observation", "")) for s in result.trace)
    json.dumps(result.to_dict())


def test_capability_isolation_tool_failure(monkeypatch, feature_frame, tmp_path):
    """A genuinely failing tool is captured in errors, not fatal — the run still completes."""

    import engines.agents.tools as tools_mod

    def boom(context, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(tools_mod, "profiler_tool", boom)
    memory = Memory(path=str(tmp_path / "mem.db"))
    llm = FakeLLM(script=[_tool_call_response("profiler_tool", {}), _stop_response()])
    result = run_agent_query("Profile the data.", feature_frame, llm=llm, memory=memory)
    assert result.answer  # still synthesized
    assert "profiler_tool" in result.errors  # real failure captured
    json.dumps(result.to_dict())


def test_cross_engine_automl_tool(feature_frame, tmp_path):
    """The ML Engineer's automl_tool really trains Engine 1 and sets an endpoint."""

    memory = Memory(path=str(tmp_path / "mem.db"))
    small = feature_frame.head(120)
    llm = FakeLLM(script=[
        _tool_call_response("automl_tool", {"target_column": "target", "task_type": "regression", "n_trials": 1}),
        _stop_response(),
    ])
    result = run_agent_query("Predict the target.", small, llm=llm, memory=memory, session_id="ml")
    assert result.model_endpoint  # endpoint URL produced
    assert "/automl/predict/" in result.model_endpoint
    assert "ML Engineer" in result.agents_used
    json.dumps(result.to_dict())


def test_input_validation():
    with pytest.raises(TypeError):
        run_agent_query("hi", ["not", "a", "frame"], llm=FakeLLM())
    with pytest.raises(ValueError):
        run_agent_query("hi", pd.DataFrame(), llm=FakeLLM())
    with pytest.raises(ValueError):
        run_agent_query("   ", pd.DataFrame({"a": [1]}), llm=FakeLLM())


# --------------------------------------------------------------------- tools
def test_pandas_tool_operations(feature_frame):
    ctx = AgentContext(dataset=feature_frame)
    reg = build_tool_registry(ctx)
    tool = reg["pandas_tool"].fn
    assert "rows=" in tool(operation="shape")
    assert "unique" in tool(operation="column_stats", column="region")
    assert "does_not_exist" not in tool(operation="value_counts", column="region")
    # filter_count on a valid expression
    assert "rows match" in tool(operation="filter_count", query="x2 > 5")


def test_plotly_tool_adds_chart(feature_frame):
    ctx = AgentContext(dataset=feature_frame)
    reg = build_tool_registry(ctx)
    msg = reg["plotly_tool"].fn(kind="bar", title="By region", x="region")
    assert "chart" in msg.lower()
    assert len(ctx.charts) == 1


def test_tool_specs_are_openai_shaped(feature_frame):
    ctx = AgentContext(dataset=feature_frame)
    reg = build_tool_registry(ctx)
    spec = reg["automl_tool"].openai_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "automl_tool"
    assert "target_column" in spec["function"]["parameters"]["properties"]


# --------------------------------------------------------------------- memory
def test_memory_roundtrip(tmp_path):
    mem = AgentMemory(path=str(tmp_path / "m.db"))
    mem.remember("sess", "what is churn?", "churn is ...", embedding=[0.1, 0.2, 0.3])
    mem.remember("sess", "top features?", "tenure and charges", embedding=[0.9, 0.1, 0.0])
    hist = mem.history("sess")
    assert len(hist) == 2
    # keyword recall (no query embedding) should find the churn entry first
    recalled = mem.recall("sess", "tell me about churn", n=1)
    assert recalled and "churn" in recalled[0].query
    # cosine recall with an embedding close to the second entry
    recalled_emb = mem.recall("sess", "features", n=1, query_embedding=[0.85, 0.15, 0.0])
    assert recalled_emb and recalled_emb[0].answer == "tenure and charges"


# --------------------------------------------------------------------- API
def test_agents_capabilities_endpoint():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    resp = client.get("/agents/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    names = [a["name"] for a in body["agents"]]
    assert names == ["Planner", "Data Analyst", "ML Engineer", "Visualizer", "Researcher", "Synthesizer"]
    assert body["llm"]["model"]


def test_agents_query_endpoint_mocked(monkeypatch, feature_frame):
    """/agents/query wiring: monkeypatch the crew entry point to avoid the network."""

    import api.routers.agents as agents_router

    def fake_run(dataset, query, session_id=None, max_steps=None):
        return AgentQueryResult(
            query=query, answer="mocked answer", session_id=session_id or "x",
            agents_used=["Planner", "Synthesizer"],
        )

    monkeypatch.setattr(agents_router, "run_agent_query", fake_run)

    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    rows = feature_frame.head(3).to_dict(orient="records")
    resp = client.post("/agents/query", json={"rows": rows, "query": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "mocked answer"
    assert body["agents_used"] == ["Planner", "Synthesizer"]
