"""Small SQLite-backed semantic index for reusable task memories."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from memory import MEMORIES_DIR, ROOT


INDEX_PATH = ROOT / ".memory_index.sqlite3"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EmbeddingFunction = Callable[[list[str]], list[list[float]]]


def _memory_text(memory: dict[str, Any]) -> str:
    playbook = memory.get("playbook")
    if not isinstance(playbook, dict):
        playbook = {}
    return json.dumps(
        {
            "task": memory.get("task"),
            "task_signature": playbook.get("task_signature"),
            "applicability": playbook.get("applicability"),
            "learned_target_mappings": playbook.get("learned_target_mappings"),
            "preferred_plan": playbook.get("preferred_plan"),
            "fallbacks": playbook.get("fallbacks"),
            "environment_facts": playbook.get("environment_facts"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    return list(struct.unpack(f"<{dimensions}f", blob))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalized_words(text: Any) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", str(text).lower())
        if word not in {"a", "an", "the", "please", "now"}
    }


def _lexical_memory_score(query: str, memory: dict[str, Any]) -> float:
    query_words = _normalized_words(query)
    task_words = _normalized_words(memory.get("task", ""))
    if query_words and query_words == task_words:
        return 1.0

    playbook = memory.get("playbook")
    if not isinstance(playbook, dict):
        playbook = {}

    candidate_phrases = [
        playbook.get("task_signature", ""),
        playbook.get("applicability", ""),
    ]
    for mapping in playbook.get("learned_target_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        candidate_phrases.extend(
            [
                mapping.get("requested", ""),
                mapping.get("effective", ""),
            ]
        )

    best_score = 0.0
    for phrase in candidate_phrases:
        candidate_words = _normalized_words(phrase)
        if not query_words or not candidate_words:
            continue
        overlap = len(query_words & candidate_words)
        coverage = overlap / len(query_words)
        if coverage == 1.0:
            best_score = max(best_score, 0.9)
        elif coverage >= 0.5:
            best_score = max(best_score, 0.65 * coverage)
    return best_score


def _normalized_phrase(text: Any) -> str:
    return " ".join(sorted(_normalized_words(text)))


def _target_mapping_pairs(playbook: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for mapping in playbook.get("learned_target_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        requested = _normalized_phrase(mapping.get("requested", ""))
        effective = _normalized_phrase(mapping.get("effective", ""))
        if requested and effective:
            pairs.add((requested, effective))
    return pairs


def _mapping_is_covered(
    candidate_mapping: tuple[str, str],
    existing_mappings: set[tuple[str, str]],
) -> bool:
    candidate_requested = set(candidate_mapping[0].split())
    candidate_effective = set(candidate_mapping[1].split())
    for existing_requested_text, existing_effective_text in existing_mappings:
        existing_requested = set(existing_requested_text.split())
        existing_effective = set(existing_effective_text.split())
        same_effective = (
            candidate_effective == existing_effective
            or candidate_effective.issubset(existing_effective)
            or existing_effective.issubset(candidate_effective)
        )
        related_request = bool(candidate_requested & existing_requested)
        if same_effective and related_request:
            return True
    return False


def memory_save_decision(
    candidate: dict[str, Any],
    retrieved_matches: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Save only when an exact prior memory does not already cover the task."""

    candidate_task = _normalized_phrase(candidate.get("task", ""))
    exact_matches = [
        match
        for match in retrieved_matches
        if match.get("match_type") == "exact_task"
        or _normalized_phrase(match.get("task", "")) == candidate_task
    ]
    if not exact_matches:
        return True, "no exact existing memory"

    candidate_playbook = candidate.get("playbook")
    if not isinstance(candidate_playbook, dict):
        candidate_playbook = {}
    candidate_mappings = _target_mapping_pairs(candidate_playbook)

    existing_mappings: set[tuple[str, str]] = set()
    for match in exact_matches:
        playbook = match.get("playbook")
        if isinstance(playbook, dict):
            existing_mappings.update(_target_mapping_pairs(playbook))

    new_mappings = {
        mapping
        for mapping in candidate_mappings
        if not _mapping_is_covered(mapping, existing_mappings)
    }
    if new_mappings:
        return True, f"new target mapping: {sorted(new_mappings)}"

    return False, "exact task already has a reusable playbook"


def _response_vectors(response: Any) -> list[list[float]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not isinstance(data, list):
        raise TypeError("embedding response did not contain a data list.")

    vectors = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise TypeError("embedding response item did not contain a vector.")
        vectors.append([float(value) for value in embedding])
    return vectors


class MemoryVectorStore:
    """Persist memory embeddings in SQLite and search them by cosine similarity."""

    def __init__(
        self,
        *,
        memories_dir: Path = MEMORIES_DIR,
        index_path: Path = INDEX_PATH,
        api_key: str | None = None,
        embedding_model: str | None = None,
        embed_function: EmbeddingFunction | None = None,
    ) -> None:
        self.memories_dir = memories_dir
        self.index_path = index_path
        self.api_key = api_key
        self.embedding_model = embedding_model or os.getenv(
            "MMFCUA_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        )
        self._embed_function = embed_function
        self._query_cache: dict[str, list[float]] = {}
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    path TEXT PRIMARY KEY,
                    modified_ns INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    memory_json TEXT NOT NULL
                )
                """
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_function is not None:
            return self._embed_function(texts)

        from litellm import embedding

        response = embedding(
            model=self.embedding_model,
            input=texts,
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
        )
        return _response_vectors(response)

    def refresh(self) -> int:
        """Index new/changed memories and remove deleted entries."""

        self.memories_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(self.memories_dir.glob("*.json"))
        current_paths = {str(path.resolve()) for path in paths}

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT path, modified_ns, model FROM memory_vectors"
            ).fetchall()
            indexed = {
                row["path"]: (row["modified_ns"], row["model"])
                for row in rows
            }

        pending: list[tuple[Path, int, dict[str, Any], str]] = []
        for path in paths:
            resolved = str(path.resolve())
            modified_ns = path.stat().st_mtime_ns
            if indexed.get(resolved) == (modified_ns, self.embedding_model):
                continue
            memory = json.loads(path.read_text(encoding="utf-8"))
            pending.append((path, modified_ns, memory, _memory_text(memory)))

        if pending:
            vectors = self._embed([item[3] for item in pending])
            if len(vectors) != len(pending):
                raise RuntimeError("embedding response count did not match memories.")
            with self._connection() as connection:
                for (path, modified_ns, memory, _), vector in zip(pending, vectors):
                    connection.execute(
                        """
                        INSERT INTO memory_vectors (
                            path, modified_ns, model, dimensions, vector, memory_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            modified_ns=excluded.modified_ns,
                            model=excluded.model,
                            dimensions=excluded.dimensions,
                            vector=excluded.vector,
                            memory_json=excluded.memory_json
                        """,
                        (
                            str(path.resolve()),
                            modified_ns,
                            self.embedding_model,
                            len(vector),
                            _pack_vector(vector),
                            json.dumps(memory, ensure_ascii=False),
                        ),
                    )

        with self._connection() as connection:
            rows = connection.execute("SELECT path FROM memory_vectors").fetchall()
            deleted = [row["path"] for row in rows if row["path"] not in current_paths]
            connection.executemany(
                "DELETE FROM memory_vectors WHERE path = ?",
                [(path,) for path in deleted],
            )

        return len(pending)

    def search(
        self,
        query: str,
        *,
        limit: int = 2,
        minimum_score: float = 0.72,
    ) -> list[dict[str, Any]]:
        self.refresh()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT path, dimensions, vector, memory_json
                FROM memory_vectors
                WHERE model = ?
                """,
                (self.embedding_model,),
            ).fetchall()
        if not rows:
            return []

        parsed_rows = [
            (row, json.loads(row["memory_json"]))
            for row in rows
        ]
        lexical_scores = [
            _lexical_memory_score(query, memory)
            for _, memory in parsed_rows
        ]
        needs_vector_search = any(score < minimum_score for score in lexical_scores)
        query_vector: list[float] | None = None
        if needs_vector_search:
            query_vector = self._query_cache.get(query)
            if query_vector is None:
                query_vector = self._embed([query])[0]
                self._query_cache[query] = query_vector

        matches = []
        for (row, memory), lexical_score in zip(parsed_rows, lexical_scores):
            vector_score = 0.0
            if query_vector is not None:
                vector = _unpack_vector(row["vector"], row["dimensions"])
                vector_score = _cosine_similarity(query_vector, vector)
            score = max(lexical_score, vector_score)
            if score < minimum_score:
                continue
            matches.append(
                {
                    "score": round(score, 4),
                    "match_type": (
                        "exact_task"
                        if lexical_score == 1.0
                        else "lexical_alias"
                        if lexical_score >= vector_score
                        else "vector"
                    ),
                    "path": row["path"],
                    "run_id": memory.get("run_id"),
                    "task": memory.get("task"),
                    "verification": memory.get("verification"),
                    "playbook": memory.get("playbook"),
                }
            )

        matches.sort(
            key=lambda item: (
                item["score"],
                (item.get("verification") or {}).get("confidence", 0.0),
            ),
            reverse=True,
        )

        deduplicated = []
        seen_tasks = set()
        for match in matches:
            task_key = _normalized_phrase(match.get("task", ""))
            if task_key and task_key in seen_tasks:
                continue
            if task_key:
                seen_tasks.add(task_key)
            deduplicated.append(match)
        return deduplicated[: max(0, limit)]


def guidance_message(matches: list[dict[str, Any]]) -> str | None:
    if not matches:
        return None
    compact_matches = [
        {
            "score": match.get("score"),
            "match_type": match.get("match_type"),
            "task": match.get("task"),
            "verification": match.get("verification"),
            "playbook": match.get("playbook"),
        }
        for match in matches
    ]
    return json.dumps(
        {
            "retrieved_memory_guidance": compact_matches,
            "instruction": (
                "Use these as advisory prior experience only. The current screenshot "
                "is the source of truth. Revalidate aliases, installed applications, "
                "preconditions, and success checks. Never replay historical coordinates."
            ),
        },
        indent=2,
        ensure_ascii=False,
    )
