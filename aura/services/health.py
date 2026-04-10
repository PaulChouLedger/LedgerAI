"""
services.health -- Lightweight service health checks.

Phase B: Whisper + LLM run in-process.  Only Memory stays as HTTP service.
This module provides health checks for both in-process engines and HTTP services.
"""

from __future__ import annotations

import urllib.request

from core.config import MEMORY_URL

# Only memory is still an HTTP service
_SERVICE_URLS = {
    "memory":  MEMORY_URL,
}


def check_service(name: str) -> bool:
    """Non-blocking health check for a single service by name.

    For 'whisper' and 'llm': checks in-process engine loaded state.
    For 'memory': HTTP ping to /health.
    """
    # In-process engines
    if name == "whisper":
        try:
            from voice.whisper_engine import whisper_engine
            return whisper_engine.loaded
        except Exception:
            return False
    if name == "llm":
        try:
            from voice.llm_engine import llm_engine
            return llm_engine.loaded
        except Exception:
            return False

    # HTTP services
    url = _SERVICE_URLS.get(name)
    if url is None:
        return False
    try:
        r = urllib.request.urlopen(f"{url}/health", timeout=2.0)
        return r.status == 200
    except Exception:
        return False


def ensure_containers() -> None:
    """No-op. Memory service started by start_aura.sh, Whisper+LLM in-process."""
    print("[health] Memory service managed by start_aura.sh; Whisper+LLM in-process")
