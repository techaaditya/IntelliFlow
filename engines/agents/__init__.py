"""IntelliFlow Engine 3 — Multi-Agent Orchestration.

A lightweight multi-agent crew that answers natural-language questions about the
loaded dataset. A Planner decomposes the query; specialist agents (Data Analyst,
ML Engineer, Visualizer, Researcher) carry it out by calling real tools — including
**Engine 1 (AutoML)** and **Engine 2 (Analytics)** in-process — and a Synthesizer
composes the final answer. The LLM backbone is any OpenAI-compatible endpoint,
defaulting to Ollama Cloud's ``gpt-oss:120b``.

The single entry point mirrors the other engines::

    >>> from engines.agents import run_agent_query
    >>> result = run_agent_query("Which features best predict churn, and how accurate is a model?", df)
    >>> print(result.answer)
    >>> result.to_dict()  # JSON-safe: answer, charts, model_endpoint, agents_used, trace
"""

from __future__ import annotations

from .config import AgentConfig, get_agent_config, load_agent_config
from .crew import AgentCrew, AgentQueryResult, run_agent_query
from .llm import ChatResponse, LLMError, OllamaChat, ToolCall
from .memory import AgentMemory, MemoryRecord
from .roles import AGENTS, SPECIALIST_TOOL_NAMES, AgentRole, role_by_name
from .tools import AgentContext, Tool, build_tool_registry

__all__ = [
    "run_agent_query",
    "AgentCrew",
    "AgentQueryResult",
    "AgentConfig",
    "load_agent_config",
    "get_agent_config",
    "OllamaChat",
    "ChatResponse",
    "ToolCall",
    "LLMError",
    "AgentMemory",
    "MemoryRecord",
    "AgentRole",
    "AGENTS",
    "SPECIALIST_TOOL_NAMES",
    "role_by_name",
    "AgentContext",
    "Tool",
    "build_tool_registry",
]
