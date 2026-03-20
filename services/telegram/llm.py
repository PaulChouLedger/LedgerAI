"""
llm -- LLM client for Aura Telegram bot.

Tries Farsight first (perpetual/chat endpoint), falls back to local
Ollama (OpenAI-compatible API) if Farsight is unreachable.
"""

from __future__ import annotations

import logging
import requests

from config import LLM_ENDPOINT, LLM_MAX_TOKENS, LLM_TIMEOUT, FARSIGHT_URL

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:72b-instruct-q4_K_M"


def _try_farsight(prompt: str, system_prompt: str, max_tokens: int) -> str | None:
    """Farsight perpetual/chat endpoint."""
    try:
        resp = requests.post(
            LLM_ENDPOINT,
            json={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
            },
            timeout=LLM_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        text = resp.json().get("response", "").strip()
        return text or None
    except Exception:
        return None


def _try_ollama(prompt: str, system_prompt: str, max_tokens: int) -> str | None:
    """Ollama OpenAI-compatible chat endpoint."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            },
            timeout=LLM_TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning("Ollama HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
        choices = resp.json().get("choices", [])
        if not choices:
            return None
        text = choices[0].get("message", {}).get("content", "").strip()
        return text or None
    except requests.exceptions.Timeout:
        log.warning("Ollama request timed out (%ds)", LLM_TIMEOUT)
        return None
    except Exception as e:
        log.error("Ollama call failed: %s", e)
        return None


def llm_call(
    prompt: str,
    system_prompt: str,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str | None:
    """Send a prompt to the LLM. Tries Farsight, falls back to Ollama.

    Returns None if both fail (caller decides whether to stay silent).
    """
    # Try Farsight first
    result = _try_farsight(prompt, system_prompt, max_tokens)
    if result:
        log.debug("Response from Farsight (%d chars)", len(result))
        return result

    # Fallback to local Ollama
    log.info("Farsight unavailable, falling back to Ollama (%s)", OLLAMA_MODEL)
    result = _try_ollama(prompt, system_prompt, max_tokens)
    if result:
        log.debug("Response from Ollama (%d chars)", len(result))
        return result

    log.warning("Both Farsight and Ollama failed — no response")
    return None
