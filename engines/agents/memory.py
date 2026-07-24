"""Long-term memory for the agent crew (Engine 3), backed by SQLite.

Implements the two memory tiers the design docs call for:

- **Short-term** — the in-session message history, handled by the crew itself
  and passed to each LLM call.
- **Long-term** — a persistent store of past ``(query, answer)`` pairs per
  session/dataset. On a new query the crew recalls the most relevant prior
  answers as extra context.

Relevance uses cosine similarity over embeddings from the Ollama cloud embed
endpoint when available, and **degrades gracefully** to recency + keyword
overlap when embeddings cannot be produced (no embed model, offline, etc.).
SQLite is used (stdlib) to avoid a heavyweight vector-store dependency.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentConfig, get_agent_config


@dataclass
class MemoryRecord:
    """One remembered query/answer exchange."""

    id: int
    session_id: str
    query: str
    answer: str
    created_at: float
    score: float | None = None  # populated by recall()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "created_at": self.created_at,
            "score": self.score,
        }


class AgentMemory:
    """Persistent SQLite-backed store of prior agent exchanges."""

    def __init__(self, config: AgentConfig | None = None, *, path: str | Path | None = None) -> None:
        self.config = config or get_agent_config()
        self.path = str(path or self.config.memory_path)
        self._ensure_schema()

    # ------------------------------------------------------------------ schema
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_memory_session ON agent_memory(session_id, created_at)"
            )

    # ------------------------------------------------------------------ writes
    def remember(
        self,
        session_id: str,
        query: str,
        answer: str,
        *,
        embedding: list[float] | None = None,
    ) -> int:
        """Persist a query/answer exchange; returns the new row id."""

        session_id = session_id or "default"
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO agent_memory (session_id, query, answer, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    query or "",
                    answer or "",
                    json.dumps(embedding) if embedding else None,
                    time.time(),
                ),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------- reads
    def history(self, session_id: str, *, limit: int = 20) -> list[MemoryRecord]:
        """Return the most recent exchanges for a session, newest first."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, query, answer, created_at FROM agent_memory "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id or "default", int(limit)),
            ).fetchall()
        return [
            MemoryRecord(
                id=row["id"],
                session_id=row["session_id"],
                query=row["query"],
                answer=row["answer"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def recall(
        self,
        session_id: str,
        query: str,
        *,
        n: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryRecord]:
        """Return up to ``n`` prior exchanges most relevant to ``query``.

        Uses cosine similarity over stored embeddings when a ``query_embedding``
        is supplied and stored embeddings exist; otherwise falls back to keyword
        overlap, then recency.
        """

        n = n or self.config.max_memory_results
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, query, answer, embedding, created_at FROM agent_memory "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT 200",
                (session_id or "default", ),
            ).fetchall()
        if not rows:
            return []

        scored: list[MemoryRecord] = []
        query_terms = _tokenize(query)
        for row in rows:
            record = MemoryRecord(
                id=row["id"],
                session_id=row["session_id"],
                query=row["query"],
                answer=row["answer"],
                created_at=row["created_at"],
            )
            emb = _load_embedding(row["embedding"])
            if query_embedding is not None and emb is not None:
                record.score = _cosine(query_embedding, emb)
            else:
                record.score = _keyword_overlap(query_terms, _tokenize(record.query))
            scored.append(record)

        # If nothing scored above zero (e.g. no overlap and no embeddings), use recency.
        if all((r.score or 0.0) <= 0.0 for r in scored):
            scored.sort(key=lambda r: r.created_at, reverse=True)
        else:
            scored.sort(key=lambda r: (r.score or 0.0, r.created_at), reverse=True)
        return scored[: int(n)]

    def clear(self, session_id: str | None = None) -> int:
        """Delete memories (all, or for one session). Returns rows removed."""

        with self._connect() as conn:
            if session_id is None:
                cursor = conn.execute("DELETE FROM agent_memory")
            else:
                cursor = conn.execute("DELETE FROM agent_memory WHERE session_id = ?", (session_id, ))
            return int(cursor.rowcount)


# --------------------------------------------------------------------- helpers
def _load_embedding(value: Any) -> list[float] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, list) and data:
        return [float(x) for x in data]
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in (text or "")).split() if len(t) > 2}


def _keyword_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
