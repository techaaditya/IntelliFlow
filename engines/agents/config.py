"""Configuration and secrets resolution for IntelliFlow Engine 3 (Agents).

This is the platform's first configuration/secrets plumbing. It deliberately
stays tiny and dependency-free: values are resolved in priority order

    1. real process environment variables (e.g. ``OLLAMA_API_KEY``)
    2. a ``.env`` file at the project root (parsed by a minimal built-in reader)
    3. the ``agents:`` block of ``config.yaml`` (non-secret settings only)
    4. hard-coded, sensible defaults

The LLM backbone is intentionally provider-swappable: point ``host``/``model``
at Ollama Cloud (the default), a local Ollama daemon, Groq, or any other
OpenAI-compatible endpoint without touching the crew code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ defaults
DEFAULT_HOST = "https://ollama.com"
DEFAULT_MODEL = "gpt-oss:120b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_MEMORY_RESULTS = 5
DEFAULT_REQUEST_TIMEOUT = 120

# Repo root = two levels up from this file (engines/agents/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AgentConfig:
    """Resolved settings for the agent crew and its LLM backbone."""

    api_key: str | None = None
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_steps: int = DEFAULT_MAX_STEPS
    max_memory_results: int = DEFAULT_MAX_MEMORY_RESULTS
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    memory_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "agent_memory.db"))

    @property
    def chat_url(self) -> str:
        """OpenAI-compatible chat-completions endpoint."""

        return f"{self.host.rstrip('/')}/v1/chat/completions"

    @property
    def embed_url(self) -> str:
        """Native Ollama embeddings endpoint (current ``/api/embed`` API)."""

        return f"{self.host.rstrip('/')}/api/embed"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key)

    def to_dict(self) -> dict[str, Any]:
        """Public, secret-free view of the configuration."""

        return {
            "host": self.host,
            "model": self.model,
            "embed_model": self.embed_model,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "max_memory_results": self.max_memory_results,
            "request_timeout": self.request_timeout,
            "api_key_configured": self.has_credentials,
        }


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``.env`` reader (no python-dotenv dependency).

    Supports ``KEY=value`` lines, ``#`` comments, blank lines, an optional
    ``export`` prefix, and surrounding single/double quotes. Unknown syntax is
    skipped rather than raising, so a malformed line never blocks startup.
    """

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _load_yaml_agents(path: Path) -> dict[str, Any]:
    """Read the optional ``agents:`` block from ``config.yaml`` if PyYAML is present."""

    if not path.is_file():
        return {}
    try:
        import yaml  # lazy: yaml is optional
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    block = data.get("agents") if isinstance(data, dict) else None
    return block if isinstance(block, dict) else {}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_agent_config(project_root: Path | None = None) -> AgentConfig:
    """Resolve an :class:`AgentConfig` from env, ``.env``, ``config.yaml``, defaults."""

    root = project_root or PROJECT_ROOT
    dotenv = _parse_env_file(root / ".env")
    yaml_block = _load_yaml_agents(root / "config.yaml")

    def pick(*keys: str, default: Any = None) -> Any:
        """First non-empty value across env → .env → yaml, by any of the given keys."""

        for key in keys:
            if os.environ.get(key):
                return os.environ[key]
        for key in keys:
            if dotenv.get(key):
                return dotenv[key]
        for key in keys:
            if key in yaml_block and yaml_block[key] not in (None, ""):
                return yaml_block[key]
        return default

    return AgentConfig(
        api_key=pick("OLLAMA_API_KEY", "OLLAMA_CLOUD_API_KEY", "api_key"),
        host=str(pick("OLLAMA_HOST", "host", default=DEFAULT_HOST)),
        model=str(pick("OLLAMA_MODEL", "model", default=DEFAULT_MODEL)),
        embed_model=str(pick("OLLAMA_EMBED_MODEL", "embed_model", default=DEFAULT_EMBED_MODEL)),
        temperature=_coerce_float(pick("OLLAMA_TEMPERATURE", "temperature"), DEFAULT_TEMPERATURE),
        max_steps=_coerce_int(pick("AGENT_MAX_STEPS", "max_steps"), DEFAULT_MAX_STEPS),
        max_memory_results=_coerce_int(
            pick("AGENT_MAX_MEMORY_RESULTS", "max_memory_results"), DEFAULT_MAX_MEMORY_RESULTS
        ),
        request_timeout=_coerce_int(pick("OLLAMA_TIMEOUT", "request_timeout"), DEFAULT_REQUEST_TIMEOUT),
    )


@lru_cache(maxsize=1)
def get_agent_config() -> AgentConfig:
    """Cached default configuration for convenience call-sites."""

    return load_agent_config()
