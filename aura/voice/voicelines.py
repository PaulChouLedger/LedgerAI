"""
voice.voicelines -- Random pre-baked voiceline lookup.

Pre-bake script (tools/prebake_voicelines.py) generates thousands of
unique LLM-written, Piper-synthesised opening lines per category and
writes them to data/voicelines/<category>/<hash>.wav with a manifest.

This module loads the manifest at import time (cheap -- it's just JSON
metadata, the actual WAVs are read on demand by aplay) and exposes
random_voiceline(category) for the welcome / idle paths.

The manifest is the source of truth: WAVs without a manifest entry are
ignored, and manifest entries whose WAV is missing are skipped at pick
time. Re-running the prebake script tops up the pool.
"""

from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import Optional

from core.config import DATA_DIR


_DIR = Path(DATA_DIR) / "voicelines"
_MANIFEST = _DIR / "manifest.json"

_lock = threading.Lock()
_loaded_at: float = 0.0          # mtime when last loaded
_pool: dict[str, list[dict]] = {}  # category -> list of entries


def _load_if_stale() -> None:
    """Reload the manifest if it's been updated on disk.

    Cheap to call -- only re-parses when mtime changes. This matters
    because the prebake script is resumable: it can be running in the
    background while the puck is live, and we want new entries to be
    pickable as soon as they're written without restarting Aura.
    """
    global _loaded_at, _pool
    if not _MANIFEST.exists():
        return
    try:
        mtime = _MANIFEST.stat().st_mtime
    except OSError:
        return
    if mtime <= _loaded_at:
        return
    try:
        data = json.loads(_MANIFEST.read_text())
    except Exception as e:
        print(f"[voicelines] manifest parse failed: {e}")
        return
    if not isinstance(data, dict):
        return
    with _lock:
        _pool = {k: v for k, v in data.items() if isinstance(v, list)}
        _loaded_at = mtime


def _resolve_wav(rel_or_abs: str) -> str:
    """Manifest stores paths relative to DATA_DIR's parent (the repo root).
    But on the puck the repo lives at ~/Aura4/, so relative resolution
    via DATA_DIR.parent is correct in both dev and prod.
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    return str(Path(DATA_DIR).parent / rel_or_abs)


def random_voiceline(category: str) -> Optional[tuple[str, str, str]]:
    """Return (wav_path, text, style) from the named category, or None
    if the category is empty or no WAV exists on disk yet.

    Caller is responsible for enqueuing via speaker.enqueue_wav(). The
    text is returned only for logging / live-caption emission -- the
    audio is fixed at bake time.
    """
    _load_if_stale()
    with _lock:
        entries = list(_pool.get(category, ()))
    if not entries:
        return None
    # Try a few times in case some WAVs are missing (e.g. partial sync).
    random.shuffle(entries)
    for entry in entries[:8]:
        try:
            wav = _resolve_wav(entry["wav"])
            text = entry.get("text", "")
            style = entry.get("style", "neutral")
        except Exception:
            continue
        if os.path.isfile(wav) and os.path.getsize(wav) > 1000:
            return wav, text, style
    return None


def category_size(category: str) -> int:
    """Number of entries in a category (for boot diagnostics)."""
    _load_if_stale()
    with _lock:
        return len(_pool.get(category, []))


def all_sizes() -> dict[str, int]:
    _load_if_stale()
    with _lock:
        return {k: len(v) for k, v in _pool.items()}
