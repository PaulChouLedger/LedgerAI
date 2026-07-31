"""
llm -- LLM client for Aura Telegram bot.

Primary: Ollama on localhost:11434 (model env-overridable)
Fallback: Farsight perpetual/chat endpoint (if available)

2026-07-31 rebuild notes:
- The model name was hardcoded to llama3.1:70b-instruct-q5_K_M, which was
  removed from the local Ollama store months ago. Every call 404'd, returned
  None, and None means "stay silent" -- so a dead model was indistinguishable
  from a well-behaved bot, for three months. Model and URL are env-overridable
  now, and the default is a model that actually exists on this box.
- num_predict was hardcoded to 512, silently truncating every caller that
  asked for more (the daily brief asks for 1500). It follows max_tokens now.
- Consecutive-failure logging: one warning per state CHANGE rather than one
  per call, plus a loud counter every 50 failures, so three months of silence
  can never again look like three months of discretion.
"""

from __future__ import annotations

import logging
import os
import requests

from config import LLM_ENDPOINT, LLM_MAX_TOKENS, LLM_TIMEOUT

log = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("AURA_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("AURA_OLLAMA_MODEL", "qwen2.5:72b-instruct-q8_0")

_consecutive_failures = 0


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
                    "num_predict": max(64, int(max_tokens)),
                    "temperature": 0.85,
                    "repeat_penalty": 1.1,
                    "num_ctx": 16384,
                },
                "keep_alive": "30m",
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
    global _consecutive_failures
    result = _try_ollama(prompt, system_prompt, max_tokens)
    if not result:
        log.info("Ollama unavailable, trying Farsight")
        result = _try_farsight(prompt, system_prompt, max_tokens)

    if result:
        if _consecutive_failures:
            log.warning("LLM recovered after %d consecutive failures",
                        _consecutive_failures)
        _consecutive_failures = 0
        return result

    _consecutive_failures += 1
    if _consecutive_failures == 1 or _consecutive_failures % 50 == 0:
        log.error("LLM DOWN: both Ollama and Farsight failing "
                  "(%d consecutive failures). The bot is mute, not polite.",
                  _consecutive_failures)
    else:
        log.warning("Both Ollama and Farsight failed — no response")
    return None
