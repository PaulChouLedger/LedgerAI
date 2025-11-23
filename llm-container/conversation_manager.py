"""
Conversation Manager
====================

Provides passive transcript ingestion, keyword-triggered interaction windows,
and lightweight FAISS-backed conversation memory for the generic LLM container.

Key responsibilities:
    • Store every finalized transcript snippet in an embedding index so the RAG
      stack can recall prior conversations even when the device stayed silent.
    • Detect configurable activation keywords inside the transcript stream and
      open a short interaction window that produces an LLM response.
    • Supply conversation memory snippets alongside standard RAG context when
      generating responses.
"""

from __future__ import annotations

import os
import re
import time
import threading
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "faiss-cpu is required for conversation memory. "
        "Install it via the llm-container requirements."
    ) from exc

# Type aliases
EmbeddingFn = Callable[[List[str]], List[List[float]]]
ConversationHandler = Callable[[str, str, Optional[str]], str]
SearchResult = Dict[str, Any]


def _normalize_text(text: str) -> str:
    """Lowercase the text and collapse whitespace for reliable comparisons."""
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", clean).strip()


def _remove_keywords(text: str, keywords: List[str]) -> str:
    """Remove activation keywords from the provided text (case-insensitive)."""
    lowered = text.lower()
    for keyword in keywords:
        if not keyword:
            continue
        keyword_lower = keyword.lower()
        if keyword_lower in lowered:
            pattern = re.compile(rf"\b{re.escape(keyword_lower)}\b", re.IGNORECASE)
            text = pattern.sub("", text)
            lowered = text.lower()
    return re.sub(r"\s+", " ", text).strip()


class ConversationMemoryIndex:
    """
    Simple FAISS-backed memory store for transcript snippets.

    - Embeddings are normalised to unit length (L2) and stored in an IP index,
      which effectively yields cosine similarity scores.
    - The index persists to disk periodically so that past conversations remain
      available across container restarts.
    """

    def __init__(
        self,
        storage_dir: str,
        persist_every: int = 10,
        max_entries: int = 5000,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.persist_every = max(1, persist_every)
        self.max_entries = max_entries

        self._lock = threading.Lock()
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._embedding_dim: Optional[int] = None

        self._data_path = self.storage_dir / "conversation_memory.pkl"
        self._load()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _ensure_index(self, embedding_dim: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(embedding_dim)
            if self._vectors:
                stacked = np.vstack(self._vectors).astype("float32")
                faiss.normalize_L2(stacked)
                self._index.add(stacked)

    def _trim_if_needed(self) -> None:
        if self.max_entries and len(self._vectors) > self.max_entries:
            overflow = len(self._vectors) - self.max_entries
            if overflow <= 0:
                return
            self._vectors = self._vectors[overflow:]
            self._metadata = self._metadata[overflow:]
            if self._index is not None:
                self._index.reset()
                if self._vectors:
                    stacked = np.vstack(self._vectors).astype("float32")
                    faiss.normalize_L2(stacked)
                    self._index.add(stacked)

    def _persist(self) -> None:
        payload = {
            "embedding_dim": self._embedding_dim,
            "metadata": self._metadata,
            "vectors": [vec.tolist() for vec in self._vectors],
        }
        with open(self._data_path, "wb") as handle:
            pickle.dump(payload, handle)

    def _load(self) -> None:
        if not self._data_path.exists():
            return
        try:
            with open(self._data_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ConversationMemory] ⚠️ Failed to load stored memory: {exc}")
            return

        self._embedding_dim = payload.get("embedding_dim")
        vectors = payload.get("vectors") or []
        metadata = payload.get("metadata") or []

        if not vectors or not metadata or len(vectors) != len(metadata):
            return

        self._vectors = [np.asarray(vec, dtype="float32") for vec in vectors]
        self._metadata = list(metadata)
        if self._embedding_dim:
            self._ensure_index(self._embedding_dim)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def add_entry(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> None:
        if not text or not embedding:
            return

        vector = np.asarray(embedding, dtype="float32")
        if vector.ndim != 1:
            vector = vector.flatten()

        with self._lock:
            self._embedding_dim = self._embedding_dim or vector.shape[0]
            self._ensure_index(self._embedding_dim)

            faiss.normalize_L2(vector)
            if self._index is not None:
                self._index.add(vector.reshape(1, -1))

            self._vectors.append(vector)
            entry_metadata = metadata.copy() if metadata else {}
            entry_metadata.setdefault("text", text)
            entry_metadata.setdefault("stored_at", time.time())
            self._metadata.append(entry_metadata)

            self._trim_if_needed()

            if persist and len(self._vectors) % self.persist_every == 0:
                self._persist()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        min_score: float = 0.3,
    ) -> List[SearchResult]:
        if self._index is None or not self._vectors:
            return []
        vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        distances, indices = self._index.search(vector, top_k)
        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if score < min_score:
                continue
            metadata = self._metadata[idx]
            # Preserve stored text, but also return score for caller
            results.append(
                {
                    "score": float(score),
                    "text": metadata.get("text", ""),
                    "metadata": metadata,
                }
            )
        return results

    def flush(self) -> None:
        """Persist the current memory snapshot to disk."""
        with self._lock:
            if self._vectors:
                self._persist()


@dataclass
class SessionState:
    """Tracks per-session activation and transcript buffering state."""

    active: bool = False
    activation_keyword: Optional[str] = None
    activation_timestamp: float = 0.0
    last_update: float = 0.0
    cooldown_until: float = 0.0
    transcript_buffer: List[str] = field(default_factory=list)
    latest_metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationOrchestrator:
    """
    Coordinates passive listening, activation keyword detection, and response generation.
    """

    def __init__(
        self,
        memory_index: ConversationMemoryIndex,
        embed_fn: EmbeddingFn,
        conversation_handler: ConversationHandler,
        activation_keywords: Optional[List[str]] = None,
        activation_window: float = 15.0,
        activation_cooldown: float = 3.0,
        memory_top_k: int = 3,
        memory_min_score: float = 0.35,
    ) -> None:
        self.memory_index = memory_index
        self.embed_fn = embed_fn
        self.conversation_handler = conversation_handler
        self.activation_keywords = [
            kw.strip()
            for kw in (activation_keywords or ["hey aura"])
            if kw.strip()
        ]
        self.activation_window = activation_window
        self.activation_cooldown = activation_cooldown
        self.memory_top_k = memory_top_k
        self.memory_min_score = memory_min_score
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Core processing
    # ------------------------------------------------------------------ #

    def process_chunk(
        self,
        session_id: str,
        text: str,
        is_final: bool,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a transcript chunk, optionally triggering an LLM response.

        Returns a dictionary describing the state transition:
            {
                "ingested": bool,
                "activated": bool,
                "activation_keyword": Optional[str],
                "awaiting_response": bool,
                "response": Optional[str]
            }
        """
        timestamp = timestamp or time.time()
        text = text.strip()
        if not text:
            return {
                "ingested": False,
                "activated": False,
                "awaiting_response": False,
                "response": None,
            }

        metadata = metadata.copy() if metadata else {}
        metadata.setdefault("session_id", session_id)
        metadata.setdefault("timestamp", timestamp)
        metadata.setdefault("type", metadata.get("type", "user_transcript"))

        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.last_update = timestamp
            state.latest_metadata = metadata

            activation_update = self._handle_activation(state, text, timestamp)
            activation_triggered = activation_update["triggered"]

            if state.active:
                state.transcript_buffer.append(text)

        # Store transcript in memory when finalised
        ingested = False
        if is_final:
            self._store_in_memory(text, metadata)
            ingested = True

        response_payload: Dict[str, Any] = {
            "ingested": ingested,
            "activated": activation_triggered or activation_update["active"],
            "activation_keyword": activation_update["keyword"],
            "awaiting_response": False,
            "response": None,
        }

        # Only attempt to generate a response on finalised text during an active window
        if is_final:
            response_payload.update(self._maybe_generate_response(session_id))

        return response_payload

    # ------------------------------------------------------------------ #
    # Activation + response helpers
    # ------------------------------------------------------------------ #

    def _handle_activation(
        self,
        state: SessionState,
        text: str,
        timestamp: float,
    ) -> Dict[str, Any]:
        # Respect cooldown window after a response
        if state.cooldown_until and timestamp < state.cooldown_until:
            state.active = False
            state.transcript_buffer.clear()
            return {
                "triggered": False,
                "active": False,
                "keyword": state.activation_keyword,
                "state": state,
            }

        normalized = _normalize_text(text)
        triggered_keyword = None
        if not state.active:
            for keyword in self.activation_keywords:
                if keyword and keyword in normalized:
                    triggered_keyword = keyword
                    break

            if triggered_keyword:
                state.active = True
                state.activation_keyword = triggered_keyword
                state.activation_timestamp = timestamp
                state.transcript_buffer = []

        # Auto-expire active session if window elapsed without updates
        if state.active and (
            timestamp - state.activation_timestamp > self.activation_window
        ):
            state.active = False
            state.transcript_buffer.clear()

        return {
            "triggered": bool(triggered_keyword),
            "active": state.active,
            "keyword": state.activation_keyword,
            "state": state,
        }

    def _maybe_generate_response(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or not state.active or not state.transcript_buffer:
                if state is not None:
                    state.active = False
                    state.transcript_buffer.clear()
                return {
                    "awaiting_response": False,
                    "response": None,
                    "activated": False,
                }

            prompt_parts = list(state.transcript_buffer)
            state.transcript_buffer.clear()
            state.active = False
            state.cooldown_until = time.time() + self.activation_cooldown

        prompt = " ".join(prompt_parts).strip()
        prompt = _remove_keywords(prompt, self.activation_keywords)
        prompt = prompt.strip()

        if not prompt:
            return {
                "awaiting_response": False,
                "response": None,
                "activated": False,
            }

        memory_context = self._build_memory_context(prompt)
        response = self.conversation_handler(prompt, session_id, memory_context)

        # Store assistant response in memory for future recall
        self._store_in_memory(
            response,
            {
                "session_id": session_id,
                "timestamp": time.time(),
                "type": "assistant_response",
            },
        )

        return {
            "awaiting_response": False,
            "response": response,
            "activated": False,
        }

    def _build_memory_context(self, prompt: str) -> Optional[str]:
        embeddings = self.embed_fn([prompt])
        if not embeddings:
            return None
        results = self.memory_index.search(
            embeddings[0],
            top_k=self.memory_top_k,
            min_score=self.memory_min_score,
        )
        if not results:
            return None

        lines = []
        for item in results:
            text = item.get("text", "")
            score = item.get("score", 0.0)
            ts = item.get("metadata", {}).get("timestamp")
            timestamp_desc = ""
            if ts:
                timestamp_desc = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(ts)
                )
            lines.append(
                f"- (score {score:.2f}) {text}"
                + (f" — noted at {timestamp_desc}" if timestamp_desc else "")
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Memory storage
    # ------------------------------------------------------------------ #

    def _store_in_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = text.strip()
        if not text:
            return
        try:
            embeddings = self.embed_fn([text])
            # Check if embeddings is valid and not empty
            if not embeddings:
                return
            # Handle case where embeddings might be a tuple or other structure
            if isinstance(embeddings, tuple):
                embeddings = list(embeddings)
            if not isinstance(embeddings, list) or len(embeddings) == 0:
                return
            # Validate that embeddings[0] exists and is a list/tuple
            if not isinstance(embeddings[0], (list, tuple)):
                print(f"[ConversationMemory] ⚠️ Invalid embedding format: {type(embeddings[0])}")
                return
            # Convert tuple to list if needed
            embedding = list(embeddings[0]) if isinstance(embeddings[0], tuple) else embeddings[0]
            if not embedding or len(embedding) == 0:
                return
            self.memory_index.add_entry(text, embedding, metadata)
        except (IndexError, TypeError, AttributeError) as e:
            print(f"[ConversationMemory] ⚠️ Failed to store in memory: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"[ConversationMemory] ⚠️ Unexpected error storing in memory: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Maintenance helpers
    # ------------------------------------------------------------------ #

    def flush_memory(self) -> None:
        self.memory_index.flush()


