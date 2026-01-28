"""
Session-scoped state for the LLM container.

Today this is used for "pause / continue / repeat" voice UX:
- When we pause a long, sentence-tagged stream, we store the remaining iterator.
- The next user utterance can resume ("continue") or replay last part ("repeat").

This module centralizes state handling so container_rest.py stays readable and the
logic can later be swapped to a shared store (e.g., Redis / memory-container) if
you run multiple workers/replicas.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, Optional


@dataclass
class PendingContinuation:
    iter: Iterator[str]
    last_sentence: str = ""
    item_count: int = 0


class SessionState:
    """
    In-memory, process-local session state.

    NOTE: If the server is ever run with multiple processes/workers/replicas,
    this state must move to a shared store.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._pending: Dict[str, PendingContinuation] = {}

    def set_pending_continuation(
        self,
        session_id: str,
        pending_iter: Iterator[str],
        *,
        last_sentence: str,
        item_count: int,
    ) -> None:
        if not session_id:
            return
        with self._lock:
            self._pending[session_id] = PendingContinuation(
                iter=pending_iter,
                last_sentence=(last_sentence or "").strip(),
                item_count=int(item_count or 0),
            )

    def has_pending_continuation(self, session_id: str) -> bool:
        if not session_id:
            return False
        with self._lock:
            return session_id in self._pending

    def consume_pending_continuation(self, session_id: str) -> Optional[PendingContinuation]:
        """
        Atomically fetch + remove pending continuation for this session.
        """
        if not session_id:
            return None
        with self._lock:
            return self._pending.pop(session_id, None)

    def peek_pending_continuation(self, session_id: str) -> Optional[PendingContinuation]:
        if not session_id:
            return None
        with self._lock:
            return self._pending.get(session_id)

    def clear_pending_continuation(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._pending.pop(session_id, None)

    def pending_session_ids(self) -> Iterable[str]:
        with self._lock:
            return list(self._pending.keys())


# Singleton for this process
SESSION_STATE = SessionState()

