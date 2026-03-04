"""
core.state -- Persisted application settings + runtime flags.

All mutations fire bus events so the GUI / voice pipeline react without
coupling.  Settings are saved to data/app_settings.json automatically.

Usage:
    from core.state import state
    mode = state.llm_mode            # read
    state.llm_mode = "medical"       # write  (auto-saves + emits "state.llm_mode")
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.bus import bus
from core.config import SETTINGS_FILE, DEFAULT_DOCK, MAX_DOCK_SLOTS


class _State:
    """Thread-safe settings singleton.  Property setters auto-persist + emit."""

    _FIELDS = {
        # field_name: default_value
        "llm_mode":                        "generic",
        "llm_model":                       "",
        "wake_word_enabled":               True,
        "wake_word_sensitivity":           0.9,
        "wake_word_engine":                "openwakeword",
        "wake_word_model_path":            None,
        "tts_engine":                      "chatterbox",
        "chatterbox_voice_cloning_enabled": True,
        "whisper_model":                   None,       # None → read from container
        "dock":                            None,       # None → use DEFAULT_DOCK
        # Boot / voice enrollment
        "active_user_id":                  None,
        "active_user_name":                None,
        "boot_enrollment_done":            False,
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        # Runtime-only (not persisted)
        self._playing = False
        self._playing_since: Optional[float] = None
        self._shutdown = False
        self._restart_listener = False
        self._last_response_question = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load from disk (or create default file)."""
        path = Path(SETTINGS_FILE)
        if path.exists():
            try:
                with open(path) as f:
                    disk = json.load(f)
                for k, default in self._FIELDS.items():
                    self._data[k] = disk.get(k, default)
            except Exception as exc:
                print(f"[state] load error: {exc}")
                self._apply_defaults()
        else:
            self._apply_defaults()
            self._save()
            print(f"[state] created default settings: {path}")

    def _apply_defaults(self) -> None:
        for k, v in self._FIELDS.items():
            self._data.setdefault(k, v)

    def _save(self) -> None:
        path = Path(SETTINGS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Only persist non-None values
            out = {k: v for k, v in self._data.items() if v is not None}
            json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    # Generic getter / setter (drives the property helpers below)
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            old = self._data.get(key)
            self._data[key] = value
            self._save()
        if old != value:
            bus.emit(f"state.{key}", value=value, old=old)

    # ------------------------------------------------------------------
    # Typed properties  (IDE-friendly, validated)
    # ------------------------------------------------------------------

    @property
    def llm_mode(self) -> str:
        return self.get("llm_mode", "generic")

    @llm_mode.setter
    def llm_mode(self, v: str) -> None:
        if v in ("medical", "generic"):
            self.set("llm_mode", v)

    @property
    def llm_model(self) -> str:
        return self.get("llm_model", "")

    @llm_model.setter
    def llm_model(self, v: str) -> None:
        self.set("llm_model", v or "")

    @property
    def tts_engine(self) -> str:
        return self.get("tts_engine", "chatterbox")

    @tts_engine.setter
    def tts_engine(self, v: str) -> None:
        if v in ("chatterbox", "elevenlabs"):
            self.set("tts_engine", v)

    @property
    def wake_word_enabled(self) -> bool:
        return bool(self.get("wake_word_enabled", True))

    @wake_word_enabled.setter
    def wake_word_enabled(self, v: bool) -> None:
        self.set("wake_word_enabled", bool(v))

    @property
    def wake_word_sensitivity(self) -> float:
        return float(self.get("wake_word_sensitivity", 0.9))

    @wake_word_sensitivity.setter
    def wake_word_sensitivity(self, v: float) -> None:
        self.set("wake_word_sensitivity", max(0.0, min(1.0, float(v))))

    @property
    def chatterbox_voice_cloning_enabled(self) -> bool:
        return bool(self.get("chatterbox_voice_cloning_enabled", True))

    @chatterbox_voice_cloning_enabled.setter
    def chatterbox_voice_cloning_enabled(self, v: bool) -> None:
        self.set("chatterbox_voice_cloning_enabled", bool(v))

    @property
    def whisper_model(self) -> Optional[str]:
        return self.get("whisper_model")

    @whisper_model.setter
    def whisper_model(self, v: Optional[str]) -> None:
        self.set("whisper_model", v if v and v.strip() else None)

    # ------------------------------------------------------------------
    # Boot / voice enrollment
    # ------------------------------------------------------------------

    @property
    def active_user_id(self) -> Optional[str]:
        return self.get("active_user_id")

    @active_user_id.setter
    def active_user_id(self, v: Optional[str]) -> None:
        self.set("active_user_id", v if v and v.strip() else None)

    @property
    def active_user_name(self) -> Optional[str]:
        return self.get("active_user_name")

    @active_user_name.setter
    def active_user_name(self, v: Optional[str]) -> None:
        self.set("active_user_name", v if v and v.strip() else None)

    @property
    def boot_enrollment_done(self) -> bool:
        return bool(self.get("boot_enrollment_done", False))

    @boot_enrollment_done.setter
    def boot_enrollment_done(self, v: bool) -> None:
        self.set("boot_enrollment_done", bool(v))

    # ------------------------------------------------------------------
    # Dock  (the user's chosen perimeter complications)
    # ------------------------------------------------------------------

    @property
    def dock(self) -> List[str]:
        d = self.get("dock")
        if d is None:
            return list(DEFAULT_DOCK)
        return list(d)

    @dock.setter
    def dock(self, v: List[str]) -> None:
        self.set("dock", list(v)[:MAX_DOCK_SLOTS])

    def dock_add(self, name: str, slot: int | None = None) -> bool:
        """Add complication *name* to the dock.  Returns True if added."""
        d = self.dock
        if name in d:
            return False
        if len(d) >= MAX_DOCK_SLOTS:
            return False
        if slot is not None:
            d.insert(slot, name)
        else:
            d.append(name)
        self.dock = d
        return True

    def dock_remove(self, name: str) -> bool:
        """Remove *name* from dock.  Returns True if removed."""
        d = self.dock
        if name not in d:
            return False
        d.remove(name)
        self.dock = d
        return True

    def dock_swap(self, name: str, new_name: str) -> bool:
        """Replace *name* with *new_name* in-place."""
        d = self.dock
        if name not in d:
            return False
        idx = d.index(name)
        d[idx] = new_name
        self.dock = d
        return True

    # ------------------------------------------------------------------
    # Runtime-only flags (NOT persisted)
    # ------------------------------------------------------------------

    @property
    def playing(self) -> bool:
        # Watchdog: force-clear if stuck > 15 s
        if self._playing and self._playing_since:
            if (time.time() - self._playing_since) > 15.0:
                self._playing = False
                self._playing_since = None
                bus.emit("tts.finished")
        return self._playing

    @playing.setter
    def playing(self, v: bool) -> None:
        self._playing = v
        self._playing_since = time.time() if v else None
        bus.emit("tts.started" if v else "tts.finished")

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown

    def request_shutdown(self) -> None:
        self._shutdown = True
        bus.emit("shutdown")

    @property
    def restart_listener(self) -> bool:
        return self._restart_listener

    @restart_listener.setter
    def restart_listener(self, v: bool) -> None:
        self._restart_listener = v

    @property
    def last_response_question(self) -> bool:
        return self._last_response_question

    @last_response_question.setter
    def last_response_question(self, v: bool) -> None:
        self._last_response_question = v


# Module-level singleton
state = _State()
