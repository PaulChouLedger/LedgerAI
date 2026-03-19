"""
services.health -- Lightweight service health checks.

start_aura.sh handles all container and LLM startup.
This module only provides a non-blocking HTTP ping for the boot orchestrator.
"""

from __future__ import annotations

import urllib.request

from core.config import WHISPER_URL, LLM_URL, MEMORY_URL

_SERVICE_URLS = {
    "whisper": WHISPER_URL,
    "llm":     LLM_URL,
    "memory":  MEMORY_URL,
}


def check_service(name: str) -> bool:
    """Non-blocking health check for a single service by name.

    Returns True if the service responds to /health, False otherwise.
    """
    url = _SERVICE_URLS.get(name)
    if url is None:
        return False
    try:
        r = urllib.request.urlopen(f"{url}/health", timeout=2.0)
        return r.status == 200
    except Exception:
        return False


def ensure_containers() -> None:
    """No-op. start_aura.sh handles all container and LLM startup."""
    print("[health] Containers managed by start_aura.sh — nothing to do")
