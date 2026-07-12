"""Ollama Cloud chat/embedding client for IntelliFlow Engine 3.

A deliberately thin wrapper over the OpenAI-compatible endpoint exposed by
Ollama Cloud (``https://ollama.com/v1/chat/completions``) and the native
embeddings endpoint. It uses :mod:`requests` (already a platform dependency)
rather than the heavy CrewAI/LiteLLM stack, so the agent engine adds no
conflict-prone packages.

The client is provider-agnostic: any OpenAI-compatible ``/v1/chat/completions``
endpoint (local Ollama, Groq, Anthropic-via-proxy, …) works by changing
``host``/``model`` in :class:`~engines.agents.config.AgentConfig`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig, get_agent_config


class LLMError(RuntimeError):
    """Raised when the LLM backbone cannot be reached or returns an error."""


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ChatResponse:
    """Normalized response from a chat completion."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class OllamaChat:
    """Chat + embedding client for an OpenAI-compatible LLM endpoint."""

    def __init__(self, config: AgentConfig | None = None, *, max_retries: int = 1) -> None:
        self.config = config or get_agent_config()
        self.max_retries = max(0, int(max_retries))

    # --------------------------------------------------------------- chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return a normalized response.

        Parameters
        ----------
        messages:
            OpenAI-style message dicts (``{"role": ..., "content": ...}``,
            optionally with ``tool_calls``/``tool_call_id``).
        tools:
            Optional OpenAI-style tool/function specifications.
        temperature:
            Overrides the configured default when provided.
        tool_choice:
            ``"auto"``, ``"none"``, or ``"required"`` (passed through when set).
        """

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        data = self._post_json(self.config.chat_url, payload)
        return self._parse_chat(data)

    def complete(self, prompt: str, *, system: str | None = None, temperature: float | None = None) -> str:
        """Convenience helper: single-turn prompt returning plain text."""

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature).content

    # ---------------------------------------------------------- embeddings
    def embed(self, text: str) -> list[float] | None:
        """Return an embedding vector, or ``None`` if embeddings are unavailable.

        Long-term memory degrades gracefully to recency/keyword recall when this
        returns ``None`` (missing embed model, network error, etc.).
        """

        text = (text or "").strip()
        if not text:
            return None
        payload = {"model": self.config.embed_model, "input": text}
        try:
            data = self._post_json(self.config.embed_url, payload, retries=0)
        except LLMError:
            return None
        # /api/embed returns {"embeddings": [[...]]}; older shape is {"embedding": [...]}.
        vector = data.get("embedding")
        if vector is None:
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                vector = embeddings[0]
        if isinstance(vector, list) and vector and all(isinstance(x, (int, float)) for x in vector):
            return [float(x) for x in vector]
        return None

    # -------------------------------------------------------------- internals
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post_json(self, url: str, payload: dict[str, Any], *, retries: int | None = None) -> dict[str, Any]:
        import requests  # lazy import keeps the module importable without the dep resolved

        attempts = self.max_retries if retries is None else retries
        last_exc: Exception | None = None
        for attempt in range(attempts + 1):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.config.request_timeout,
                )
            except requests.RequestException as exc:  # network/DNS/timeout
                last_exc = exc
                continue
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMError(f"LLM returned non-JSON response: {response.text[:200]}") from exc
            if response.status_code in (401, 403):
                raise LLMError(
                    "LLM authentication failed (HTTP "
                    f"{response.status_code}). Check OLLAMA_API_KEY in your .env / environment."
                )
            if response.status_code == 404:
                raise LLMError(
                    f"LLM endpoint or model not found (HTTP 404) at {url} for model "
                    f"{self.config.model!r}. Verify OLLAMA_HOST and OLLAMA_MODEL."
                )
            # 429/5xx: retry if attempts remain
            if response.status_code >= 500 or response.status_code == 429:
                last_exc = LLMError(f"LLM transient error HTTP {response.status_code}: {response.text[:200]}")
                if attempt < attempts:
                    continue
            raise LLMError(f"LLM request failed (HTTP {response.status_code}): {response.text[:300]}")
        raise LLMError(
            f"Could not reach the LLM endpoint at {url}: {last_exc}. "
            "Check your network connection and OLLAMA_HOST."
        )

    @staticmethod
    def _parse_chat(data: dict[str, Any]) -> ChatResponse:
        choices = data.get("choices") or []
        if not choices:
            # Some servers echo an error object with a 200; surface it clearly.
            if "error" in data:
                raise LLMError(f"LLM error: {data['error']}")
            raise LLMError("LLM returned no choices in the response.")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            fn = raw_call.get("function") or {}
            name = fn.get("name") or raw_call.get("name") or ""
            if not name:
                continue
            arguments = OllamaChat._parse_arguments(fn.get("arguments"))
            tool_calls.append(
                ToolCall(id=str(raw_call.get("id") or f"call_{len(tool_calls)}"), name=name, arguments=arguments)
            )
        return ChatResponse(content=content, tool_calls=tool_calls, raw=data, usage=data.get("usage") or {})

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments.strip():
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"_raw": arguments}
        return {}
