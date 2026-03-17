"""
services.diaglog -- Simple diagnostic text log of what Aura hears and says.

Appends timestamped entries to /tmp/aura_diag.log.
Rotates daily (one file per day, keeps 7 days).
"""

from __future__ import annotations

import os
import time
from datetime import datetime

_LOG_DIR = "/tmp"
_MAX_DAYS = 7


def _log_path() -> str:
    return os.path.join(_LOG_DIR, f"aura_diag_{datetime.now():%Y-%m-%d}.log")


def _cleanup_old():
    """Remove diag logs older than _MAX_DAYS."""
    now = time.time()
    for f in os.listdir(_LOG_DIR):
        if f.startswith("aura_diag_") and f.endswith(".log"):
            path = os.path.join(_LOG_DIR, f)
            if now - os.path.getmtime(path) > _MAX_DAYS * 86400:
                try:
                    os.remove(path)
                except OSError:
                    pass


def heard(text: str) -> None:
    """Log what Whisper transcribed (accepted transcript)."""
    _write(f"HEARD: {text}")


def rejected(text: str, reason: str) -> None:
    """Log a rejected transcript or audio segment."""
    _write(f"REJECTED ({reason}): {text}")


def said(text: str, latency_ms: float = 0) -> None:
    """Log what Aura said back."""
    lat = f" [{latency_ms:.0f}ms]" if latency_ms else ""
    _write(f"SAID{lat}: {text}")


def _write(entry: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_log_path(), "a") as f:
            f.write(f"[{ts}] {entry}\n")
    except OSError:
        pass
