"""The agent crew orchestrator for IntelliFlow Engine 3.

``run_agent_query`` is the single public entry point, mirroring the shape of
Engine 1's ``run_automl`` and Engine 2's ``run_eda``: one call in, one
structured, JSON-safe result out.

The flow follows the documented crew without the heavyweight CrewAI stack:

1. **Recall** — pull relevant prior exchanges from long-term memory.
2. **Planner** — the LLM decomposes the query into an ordered plan (no tools).
3. **Executor** — a bounded tool-calling loop over ``gpt-oss:120b`` with all
   specialist tools available. It embodies the Data Analyst, ML Engineer,
   Visualizer, and Researcher; each tool that fires is attributed back to its
   owning agent so ``agents_used`` reflects the real work done.
4. **Synthesizer** — the LLM composes the final, decision-ready answer from
   every observation gathered (no tools).
5. **Remember** — persist the exchange (with an embedding when available).

Every step is wrapped so a single failure is recorded in ``errors`` rather than
crashing the run, matching Engine 2's capability-isolation convention.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from engines.analytics.base import ChartSpec, Insight, json_safe, sort_insights

from .config import AgentConfig, get_agent_config
from .llm import LLMError, OllamaChat, ToolCall
from .memory import AgentMemory
from .roles import AGENTS, PLANNER, SYNTHESIZER, TOOL_OWNER
from .tools import AgentContext, ToolUsageHint, build_tool_registry, tool_specs

_AGENT_ORDER = {agent.name: i for i, agent in enumerate(AGENTS)}


@dataclass
class AgentQueryResult:
    """Structured result of a crew run (matches the documented /agents/query schema)."""

    query: str
    answer: str
    session_id: str
    charts: list[ChartSpec] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    model_endpoint: str | None = None
    model_uri: str | None = None
    agents_used: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "query": self.query,
                "answer": self.answer,
                "session_id": self.session_id,
                "answer_charts": [c.to_dict() for c in self.charts],
                "charts": [c.to_dict() for c in self.charts],
                "insights": [i.to_dict() for i in self.insights],
                "model_endpoint": self.model_endpoint,
                "model_uri": self.model_uri,
                "agents_used": self.agents_used,
                "tools_used": self.tools_used,
                "trace": self.trace,
                "errors": self.errors,
            }
        )


# --------------------------------------------------------------------- prompts
def _dataset_overview(df: pd.DataFrame) -> str:
    lines = [f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."]
    parts = []
    for col in df.columns[:40]:
        parts.append(f"{col} ({df[col].dtype})")
    lines.append("Columns: " + ", ".join(parts))
    return "\n".join(lines)


_PLANNER_SYSTEM = (
    "You are the Planner in a data-science agent crew. Given a user's question and a dataset overview, "
    "produce a short, numbered plan (max 5 steps) describing exactly what to compute, whether a predictive "
    "model is needed, and which charts would help. Be concrete and reference real column names. "
    "Do not answer the question yourself — only plan."
)

_EXECUTOR_SYSTEM = (
    "You are the execution engine of a data-science agent crew (Data Analyst, ML Engineer, Visualizer, "
    "Researcher). Carry out the plan by CALLING TOOLS on the REAL dataset. Strict rules:\n"
    "- Every fact must come from a tool. Never invent numbers, models, code, or endpoints.\n"
    "- If the user asks to predict/classify/forecast, or asks how accurate a model is, you MUST call "
    "automl_tool with the correct target_column. Do NOT describe a model you did not train with automl_tool. "
    "The tool returns the real accuracy/score and a real prediction endpoint.\n"
    "- Use profiler_tool or analytics_tool for an overview; pandas_tool for specific statistics; plotly_tool "
    "or analytics_tool to add charts; web_search_tool only for external context.\n"
    "- Do NOT repeat a tool call with the same arguments. Each call should gather NEW information.\n"
    "- When you have gathered enough evidence (and trained a model if one was requested), stop calling tools "
    "and reply with a one-line note that analysis is complete."
)

_SYNTH_SYSTEM = (
    "You are the Synthesizer. Using ONLY the observations the crew actually gathered, write the final answer. "
    "Rules:\n"
    "- Cite only numbers that appear in the observations. Do NOT invent statistics, code, model files, or URLs.\n"
    "- If a trained-model endpoint is provided to you, report it verbatim. If NO endpoint is provided, state "
    "plainly that no model was trained — never fabricate a model, accuracy, or deployment code.\n"
    "- Be specific and quantitative, and end with a short 'Next steps' line. Use clear Markdown."
)

# Words that signal the user wants a predictive model built.
_MODELING_INTENT = (
    "predict", "prediction", "forecast", "classif", "regress", "model", "accuracy",
    "churn", "probability", "likelihood", "which features", "feature importance", "train",
)


def _wants_model(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in _MODELING_INTENT)


# --------------------------------------------------------------------- crew
class AgentCrew:
    """Runs a single natural-language query through the crew."""

    def __init__(self, config: AgentConfig | None = None, *, llm: OllamaChat | None = None,
                 memory: AgentMemory | None = None) -> None:
        self.config = config or get_agent_config()
        self.llm = llm or OllamaChat(self.config)
        self.memory = memory if memory is not None else AgentMemory(self.config)

    def run(self, query: str, dataset: pd.DataFrame, *, session_id: str | None = None,
            max_steps: int | None = None) -> AgentQueryResult:
        session_id = session_id or uuid.uuid4().hex[:12]
        max_steps = max_steps or self.config.max_steps
        overview = _dataset_overview(dataset)

        context = AgentContext(dataset=dataset, config=self.config, llm=self.llm)
        registry = build_tool_registry(context)
        result = AgentQueryResult(query=query, answer="", session_id=session_id)
        used_agents: set[str] = set()

        # 1. Recall prior context (best-effort).
        memory_context = self._recall(query, session_id, result)

        # 2. Planner.
        plan = self._plan(query, overview, memory_context, result, used_agents)

        # 3a. If the query needs a model, the ML Engineer trains one deterministically
        #     (the flagship cross-engine trigger — not left to the LLM's discretion).
        model_note = self._train_if_requested(query, registry, context, result, used_agents)

        # 3b. Executor tool-calling loop for analysis and charts.
        observations = self._execute(
            query, overview, plan, registry, context, result, used_agents, max_steps, model_note=model_note
        )
        if model_note:
            observations = [f"[automl_tool] {model_note}"] + observations

        # Collect artifacts produced by tools.
        result.charts = list(context.charts)
        result.insights = sort_insights(context.insights)
        result.model_endpoint = context.model_endpoint
        result.model_uri = context.model_uri
        for name, message in context.errors.items():
            result.errors.setdefault(name, message)

        # 4. Synthesizer.
        answer = self._synthesize(query, plan, observations, context, result, used_agents)
        result.answer = answer

        # Finalize attribution in canonical crew order.
        result.agents_used = sorted(used_agents, key=lambda n: _AGENT_ORDER.get(n, 99))

        # 5. Remember (best-effort).
        self._remember(session_id, query, answer, result)
        return result

    # ------------------------------------------------------------- steps
    def _recall(self, query: str, session_id: str, result: AgentQueryResult) -> str:
        try:
            embedding = self.llm.embed(query)
            records = self.memory.recall(session_id, query, query_embedding=embedding)
        except Exception as exc:  # memory must never break a run
            result.errors["memory_recall"] = str(exc)
            return ""
        if not records:
            return ""
        snippets = [f"- Q: {r.query}\n  A: {r.answer[:280]}" for r in records[:3]]
        result.trace.append({"step": "recall", "recalled": len(records)})
        return "Relevant prior exchanges:\n" + "\n".join(snippets)

    def _plan(self, query: str, overview: str, memory_context: str, result: AgentQueryResult,
              used_agents: set[str]) -> str:
        user = f"{overview}\n\n{memory_context}\n\nUser question: {query}\n\nProduce the plan."
        try:
            plan = self.llm.complete(user, system=_PLANNER_SYSTEM, temperature=0.1).strip()
        except LLMError as exc:
            result.errors["planner"] = str(exc)
            raise
        used_agents.add(PLANNER.name)
        result.trace.append({"step": "plan", "agent": PLANNER.name, "plan": plan})
        return plan

    def _train_if_requested(self, query: str, registry: dict[str, Any], context: AgentContext,
                            result: AgentQueryResult, used_agents: set[str]) -> str | None:
        """Deterministically run AutoML when the query asks for a model.

        The ML Engineer picks the target column (a single focused LLM call) and
        triggers Engine 1 directly, so the cross-engine feature never depends on
        the model electing to call the tool.
        """

        if not _wants_model(query):
            return None
        target = self._select_target(query, context)
        if not target:
            return None
        call = ToolCall(id="ml_engineer_automl", name="automl_tool", arguments={"target_column": target})
        observation = self._run_tool(registry, call, context, result, used_agents, step=-1)
        return observation

    def _select_target(self, query: str, context: AgentContext) -> str | None:
        columns = context.columns
        system = (
            "You choose the prediction target column for a machine-learning task. Reply with EXACTLY one "
            "column name copied verbatim from the provided list — the value the user wants to predict — or the "
            "single word NONE if no prediction is needed. Output only the column name, nothing else."
        )
        user = f"Columns: {columns}\nUser question: {query}\nTarget column:"
        try:
            answer = self.llm.complete(user, system=system, temperature=0).strip().strip("\"'`")
        except LLMError:
            return None
        if answer in columns:
            return answer
        # Tolerate minor formatting: exact case-insensitive match.
        lowered = {c.lower(): c for c in columns}
        if answer.lower() in lowered:
            return lowered[answer.lower()]
        return None

    def _execute(self, query: str, overview: str, plan: str, registry: dict[str, Any],
                 context: AgentContext, result: AgentQueryResult, used_agents: set[str],
                 max_steps: int, *, model_note: str | None = None) -> list[str]:
        specs = tool_specs(registry)
        if model_note:
            directive = (
                "\n\nA model has ALREADY been trained for you (see result below) — do NOT call automl_tool "
                f"again. Focus on statistics and charts.\nModel result: {model_note}"
            )
        else:
            directive = ""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _EXECUTOR_SYSTEM},
            {
                "role": "user",
                "content": f"{overview}\n\nPlan:\n{plan}\n\nUser question: {query}\n\n"
                           f"Execute the plan using the tools.{directive}",
            },
        ]
        observations: list[str] = []
        seen_calls: set[str] = set()
        for step in range(max_steps):
            try:
                response = self.llm.chat(messages, tools=specs, tool_choice="auto")
            except LLMError as exc:
                result.errors["executor"] = str(exc)
                break

            assistant_msg = self._assistant_message(response)
            messages.append(assistant_msg)

            if not response.has_tool_calls:
                if response.content.strip():
                    observations.append(response.content.strip())
                break

            for call in response.tool_calls:
                signature = f"{call.name}:{json_safe(call.arguments)}"
                if signature in seen_calls:
                    note = (
                        "You already ran this exact call. Call a DIFFERENT tool or different arguments, "
                        "or stop if analysis is complete."
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": note}
                    )
                    continue
                seen_calls.add(signature)
                observation = self._run_tool(registry, call, context, result, used_agents, step)
                observations.append(f"[{call.name}] {observation}")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": observation}
                )
        return observations

    def _run_tool(self, registry: dict[str, Any], call: Any, context: AgentContext,
                  result: AgentQueryResult, used_agents: set[str], step: int) -> str:
        tool = registry.get(call.name)
        if tool is None:
            return f"Unknown tool {call.name!r}."
        try:
            observation = tool.fn(**call.arguments)
        except ToolUsageHint as exc:  # "called me wrong" — coach the model, don't log an error
            observation = str(exc)
        except (ValueError, TypeError) as exc:
            observation = f"Tool input error: {exc}"
            context.errors[call.name] = str(exc)
        except Exception as exc:  # unexpected tool failure is isolated, not fatal
            observation = f"Tool failed: {exc}"
            context.errors[call.name] = str(exc)
        owner = TOOL_OWNER.get(call.name)
        if owner:
            used_agents.add(owner)
        if call.name not in result.tools_used:
            result.tools_used.append(call.name)
        result.trace.append(
            {"step": "tool", "index": step, "agent": owner, "tool": call.name,
             "arguments": call.arguments, "observation": observation[:600]}
        )
        return observation

    def _synthesize(self, query: str, plan: str, observations: list[str], context: AgentContext,
                    result: AgentQueryResult, used_agents: set[str]) -> str:
        evidence = "\n".join(observations) if observations else "No tool observations were produced."
        extras = []
        if context.model_endpoint:
            extras.append(f"A model WAS trained. Real prediction endpoint (use verbatim): {context.model_endpoint}")
        else:
            extras.append("NO model was trained (automl_tool was not run). Do not describe or invent any model.")
        if context.charts:
            extras.append(f"{len(context.charts)} chart(s) were attached to the answer.")
        extra_text = "\n" + "\n".join(extras)
        user = (
            f"User question: {query}\n\nPlan:\n{plan}\n\nObservations:\n{evidence}{extra_text}\n\n"
            "Write the final answer now."
        )
        try:
            answer = self.llm.complete(user, system=_SYNTH_SYSTEM, temperature=0.2).strip()
        except LLMError as exc:
            result.errors["synthesizer"] = str(exc)
            # Fall back to the raw observations so the user still gets something useful.
            answer = "The crew gathered the following findings:\n\n" + evidence
        used_agents.add(SYNTHESIZER.name)
        result.trace.append({"step": "synthesize", "agent": SYNTHESIZER.name})
        return answer

    def _remember(self, session_id: str, query: str, answer: str, result: AgentQueryResult) -> None:
        try:
            embedding = self.llm.embed(query)
            self.memory.remember(session_id, query, answer, embedding=embedding)
        except Exception as exc:  # persistence must never break the response
            result.errors.setdefault("memory_write", str(exc))

    @staticmethod
    def _assistant_message(response: Any) -> dict[str, Any]:
        """Reconstruct the assistant message (preserving tool_calls) for the next turn."""

        try:
            message = response.raw["choices"][0]["message"]
            if isinstance(message, dict):
                return message
        except (KeyError, IndexError, TypeError):
            pass
        return {"role": "assistant", "content": response.content or ""}


# --------------------------------------------------------------------- entry point
def run_agent_query(
    query: str,
    dataset: pd.DataFrame,
    *,
    session_id: str | None = None,
    config: AgentConfig | None = None,
    max_steps: int | None = None,
    llm: OllamaChat | None = None,
    memory: AgentMemory | None = None,
) -> AgentQueryResult:
    """Answer a natural-language ``query`` about ``dataset`` with the agent crew.

    Parameters
    ----------
    query:
        The user's natural-language question.
    dataset:
        The loaded dataframe every engine shares.
    session_id:
        Groups queries for long-term memory; generated when omitted.
    config / llm / memory:
        Optional overrides (dependency injection for tests / custom backends).
    max_steps:
        Cap on tool-calling iterations (defaults to the configured value).
    """

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            "run_agent_query expects a pandas DataFrame. Load CSV/JSON/Parquet with pandas or the API loaders."
        )
    if dataset.empty or dataset.shape[1] == 0:
        raise ValueError("The dataset is empty. Upload data with at least one column before querying the crew.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Provide a non-empty natural-language query for the agent crew.")

    crew = AgentCrew(config=config, llm=llm, memory=memory)
    return crew.run(query, dataset, session_id=session_id, max_steps=max_steps)
