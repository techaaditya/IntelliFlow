"""Agent role definitions for the IntelliFlow crew (Engine 3).

Each :class:`AgentRole` carries the persona and the set of tools it may use,
mirroring the documented CrewAI crew (Planner → Data Analyst → ML Engineer →
Visualizer → Researcher → Synthesizer). The backstories give the LLM the same
FAANG-team framing the design docs call for.

Reasoning-only agents (Planner, Synthesizer) hold no tools; the specialist
agents own the tools that call Engine 1 (AutoML), Engine 2 (Analytics), pandas,
and web search. :mod:`engines.agents.crew` runs a single tool-calling executor
that embodies the specialist agents and tracks which roles/tools actually fire,
so ``agents_used`` reflects the real work done.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentRole:
    """A named agent persona with a goal, backstory, and permitted tools."""

    name: str
    goal: str
    backstory: str
    tool_names: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": list(self.tool_names),
        }


PLANNER = AgentRole(
    name="Planner",
    goal="Interpret the user's natural-language query and decompose it into a concise, ordered plan of concrete data sub-tasks.",
    backstory="A lead data scientist who is expert at turning vague, ambiguous data-science requests into a crisp, actionable plan.",
    tool_names=(),
)

ANALYST = AgentRole(
    name="Data Analyst",
    goal="Answer quantitative questions about the loaded dataset by computing real statistics and profiling the data.",
    backstory="A senior analyst on a FAANG data team who never guesses a number they can compute directly from the data.",
    tool_names=("pandas_tool", "profiler_tool"),
)

ML_ENGINEER = AgentRole(
    name="ML Engineer",
    goal="Decide when a predictive model is warranted, trigger the AutoML engine on the dataset, and interpret the registered model's results.",
    backstory="An MLOps engineer with ten years of experience who ships production models and reads a leaderboard at a glance.",
    tool_names=("automl_tool", "mlflow_tool"),
)

VISUALIZER = AgentRole(
    name="Visualizer",
    goal="Turn findings into clear, correct charts by running the analytics engine and composing chart specifications.",
    backstory="A data-visualization specialist who believes the right chart makes an insight land instantly.",
    tool_names=("analytics_tool", "plotly_tool"),
)

RESEARCHER = AgentRole(
    name="Researcher",
    goal="Provide external context, domain knowledge, or benchmarks when the dataset alone cannot answer the question.",
    backstory="A resourceful research analyst who knows when a question needs outside context and how to find it succinctly.",
    tool_names=("web_search_tool",),
)

SYNTHESIZER = AgentRole(
    name="Synthesizer",
    goal="Combine every agent's findings into one coherent, well-structured, human-readable answer with clear next steps.",
    backstory="An executive communicator who distills a mass of analysis into a decision-ready narrative.",
    tool_names=(),
)


# Ordered roster (the documented six-agent crew).
AGENTS: tuple[AgentRole, ...] = (PLANNER, ANALYST, ML_ENGINEER, VISUALIZER, RESEARCHER, SYNTHESIZER)

# Which tool belongs to which agent — used to attribute tool usage back to a role.
TOOL_OWNER: dict[str, str] = {tool: agent.name for agent in AGENTS for tool in agent.tool_names}

# Tools available to the specialist executor (everything except reasoning-only roles).
SPECIALIST_TOOL_NAMES: tuple[str, ...] = tuple(TOOL_OWNER.keys())


def role_by_name(name: str) -> AgentRole | None:
    for agent in AGENTS:
        if agent.name == name:
            return agent
    return None
