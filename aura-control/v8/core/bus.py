"""
core.bus -- Lightweight thread-safe event bus.

Every module communicates through named events. No module needs to import
another module directly (except core/ utilities).

Usage:
    from core.bus import bus

    # Subscribe
    bus.on("transcript.ready", lambda text, **kw: print(text))

    # Publish (any thread)
    bus.emit("transcript.ready", text="Hello world")

    # Unsubscribe
    bus.off("transcript.ready", my_callback)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List


class EventBus:
    """Minimal pub/sub.  All callbacks fire synchronously on the emitting thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: Dict[str, List[Callable]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on(self, event: str, callback: Callable) -> None:
        """Register *callback* for *event*."""
        with self._lock:
            if callback not in self._subs[event]:
                self._subs[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Remove *callback* from *event* (no-op if not registered)."""
        with self._lock:
            try:
                self._subs[event].remove(callback)
            except ValueError:
                pass

    def once(self, event: str, callback: Callable) -> None:
        """Register *callback* to fire only once for *event*."""
        def _wrapper(**kwargs: Any) -> None:
            self.off(event, _wrapper)
            callback(**kwargs)
        # Store a reference so off() can find it if the caller unsubscribes early
        _wrapper._inner = callback  # type: ignore[attr-defined]
        self.on(event, _wrapper)

    def emit(self, event: str, **kwargs: Any) -> None:
        """Fire all callbacks for *event*.  Exceptions are caught and printed."""
        with self._lock:
            listeners = list(self._subs.get(event, []))
        for cb in listeners:
            try:
                cb(**kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"[bus] ERROR in handler for '{event}': {exc}")

    def clear(self, event: str | None = None) -> None:
        """Remove all listeners for *event*, or all listeners if *event* is None."""
        with self._lock:
            if event is None:
                self._subs.clear()
            else:
                self._subs.pop(event, None)


# Module-level singleton — import this everywhere.
bus = EventBus()
