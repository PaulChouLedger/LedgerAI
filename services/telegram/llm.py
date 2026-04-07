"""
llm -- LLM client for Aura Telegram bot.

Primary: Ollama (72B Qwen on localhost:11434)
Fallback: Farsight perpetual/chat endpoint (if available)
"""

from __future__ import annotations

import logging
import requests

from config import LLM_ENDPOINT, LLM_MAX_TOKENS, LLM_TIMEOUT

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:70b-instruct-q5_K_M"


def _try_ollama(prompt: str, system_prompt: str, max_tokens: int) -> str | None:
    """Ollama native chat endpoint (primary)."""
    try:
        log.info("Ollama request: system=%d chars, prompt=%d chars", len(system_prompt), len(prompt))
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.85,
                    "num_ctx": 16384,
                },
                "stream": False,
            },
            timeout=LLM_TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning("Ollama HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()
        if not text:
            # Fallback to OpenAI format
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "").strip()
        return text or None
    except requests.exceptions.Timeout:
        log.warning("Ollama request timed out (%ds)", LLM_TIMEOUT)
        return None
    except Exception as e:
        log.error("Ollama call failed: %s", e)
        return None


def _try_farsight(prompt: str, system_prompt: str, max_tokens: int) -> str | None:
    """Farsight perpetual/chat endpoint (fallback)."""
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


def llm_call(
    prompt: str,
    system_prompt: str,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str | None:
    """Send a prompt to the LLM. Tries Ollama first, falls back to Farsight.

    Returns None if both fail (caller decides whether to stay silent).
    """
    result = _try_ollama(prompt, system_prompt, max_tokens)
    if result:
        return result

    # Fallback to Farsight
    log.info("Ollama unavailable, trying Farsight")
    result = _try_farsight(prompt, system_prompt, max_tokens)
    if result:
        return result

    log.warning("Both Ollama and Farsight failed — no response")
    return None
