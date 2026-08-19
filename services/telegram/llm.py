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

2026-08-19 -- IT HAPPENED AGAIN, AND THE FIX ABOVE IS WHY.
The 2026-07-31 note ends "the default is a model that actually exists on this
box." It did, on 2026-07-31. `qwen2.5:72b-instruct-q8_0` was pulled off this
box during a VRAM clean-up, and from 2026-08-06 every call 404'd again --
twelve days of a bot that decided to answer, scored the message, started the
typing indicator, and then said nothing. In Area31, to its owner, by name.

A pinned name is a claim about a machine's contents made by a file that
cannot see the machine. Ownership is upside down. So the pin is gone: the
model is RESOLVED against /api/tags, preferring one already resident in VRAM
(a cold 70B is a minute of silence) and then the largest chat model
installed. If the resolved model disappears mid-run the 404 re-resolves
rather than repeating forever.

The name is now a preference, never a requirement. The only thing that can
mute her is having no chat model at all, and that says so.

2026-08-19 (later) -- AND THE PREFERENCE IS NOT OURS TO STATE.
The fix above is right and stays. What it did not know is that a SECOND
repository answers with the same 70B: STERLING, the voice assistant in
~/Aura/sterling. That night this bot asked for num_ctx 16384 while STERLING
asked for 8192, and because num_ctx is part of an ollama runner's identity
the two of them spent half an hour evicting each other's 55 GB -- 23.20 s
and 15.42 s to first token against 0.6-0.96 s all day before. Nothing in
this repo could see the other half; nothing in that one could see this.

Quieter, and still live when it was found: `keep_alive` in the request BODY
overrides OLLAMA_KEEP_ALIVE, per model, last writer wins. STERLING pins the
model with -1. This file sent "30m". `ollama ps` therefore read "28 minutes
from now" on a model STERLING believed was pinned forever, and every message
she answered re-armed that timer -- so a quiet half-hour would have cost the
room a 71.83 s cold load with every service green.

Model name, num_ctx and keep_alive now come from ONE file that both repos
read at runtime (`ollama_policy.py` -> ~/Aura/config/ollama-policy.json),
and each request logs WHICH SOURCE it read them from. The resolution logic
below is untouched: the policy states a preference, this box's /api/tags
still has the last word, and a pinned name that stopped existing still
cannot mute her.
"""

from __future__ import annotations

import logging
import os
import requests

import ollama_policy as OP
from config import LLM_ENDPOINT, LLM_MAX_TOKENS, LLM_TIMEOUT

log = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("AURA_OLLAMA_URL", "http://localhost:11434")

#: What she'd LIKE to speak with. Not a requirement -- see _resolve_model.
#: AURA_OLLAMA_MODEL still wins if it is set AND installed. Kept accurate so
#: that MODEL SUBSTITUTED means real drift rather than a line people learn to
#: scroll past; being wrong here is now a logged inconvenience, not a mute.
#:
#: 2026-08-19: the NAME is no longer written here. STERLING, in another
#: repository, answers a room with the same 70B, and a model name / num_ctx
#: / quantisation is that runner's identity — two files with opinions about
#: it is two processes evicting 55 GB from each other. See ollama_policy.py.
#: The resolution logic below is unchanged and still the safety net: the
#: policy states a PREFERENCE, this box's /api/tags still has the last word.
OLLAMA_MODEL = os.environ.get("AURA_OLLAMA_MODEL", OP.MODEL)

#: substrings that mark a model as not-for-conversation. An embedder answers
#: every chat request with an error, which reads downstream as silence.
_NOT_CHAT = ("embed", "bge", "minilm", "nomic", "rerank", "clip", "whisper")

_resolved: str | None = None
_consecutive_failures = 0


def _param_b(details: dict) -> float:
    """Billions of parameters, from ollama's own '70.6B' string."""
    try:
        return float(str(details.get("parameter_size", "0")).rstrip("Bb"))
    except ValueError:
        return 0.0


def _resolve_model(force: bool = False) -> str | None:
    """The best chat model this box ACTUALLY has, asked rather than assumed.

    Preference order: the pinned name if it is installed, then whatever is
    already resident in VRAM (loading a cold 70B is ~60 s of silence and she
    is usually mid-sentence), then the largest installed chat model.

    Returns None only when ollama is unreachable or has nothing to talk
    with -- and says so at ERROR, because that is the one state that really
    does mute her.
    """
    global _resolved
    if _resolved and not force:
        return _resolved
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
        names = [m["name"] for m in tags.get("models", [])]
        sizes = {m["name"]: _param_b(m.get("details", {}))
                 for m in tags.get("models", [])}
    except Exception as e:                                    # noqa: BLE001
        log.error("Ollama unreachable at %s (%s) -- cannot pick a model",
                  OLLAMA_URL, e)
        return None

    chat = [n for n in names
            if not any(h in n.lower() for h in _NOT_CHAT)]
    if not chat:
        log.error("Ollama has NO chat model installed (%d tags, all "
                  "embedders). She is mute until one is pulled.", len(names))
        _resolved = None
        return None

    if OLLAMA_MODEL in chat:
        pick, why = OLLAMA_MODEL, "the pinned name, and it is installed"
    else:
        try:
            ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json()
            warm = {m["name"] for m in ps.get("models", [])} & set(chat)
        except Exception:                                     # noqa: BLE001
            warm = set()
        pool = warm or set(chat)
        pick = max(pool, key=lambda n: sizes.get(n, 0.0))
        why = ("already resident in VRAM" if warm
               else "largest installed chat model")
        #: LOUD. A substitution means she is not speaking with the voice she
        #: was configured for, and the last two outages were exactly this
        #: discrepancy going unnoticed.
        log.error("MODEL SUBSTITUTED: %r is not installed on this box. "
                  "Using %r (%s). Set AURA_OLLAMA_MODEL, or pull the pin.",
                  OLLAMA_MODEL, pick, why)

    if pick != _resolved:
        log.warning("LLM model resolved to %r (%s)", pick, why)
    _resolved = pick
    return pick


def _try_ollama(prompt: str, system_prompt: str, max_tokens: int,
                _retry: bool = True) -> str | None:
    """Ollama native chat endpoint (primary)."""
    model = _resolve_model()
    if not model:
        return None
    #: read per request, not at import: STERLING is a live room and this bot
    #: answers strangers, so a change to the shared policy has to reach both
    #: without either being restarted.
    pol = OP.policy()
    try:
        #: §15 corollary -- name the SOURCE, not just the value. A num_ctx
        #: that matches STERLING by luck and one that was read from the
        #: shared file are indistinguishable in `ollama ps`; they are not
        #: indistinguishable here.
        log.info("Ollama request (%s): system=%d chars, prompt=%d chars | %s",
                 model, len(system_prompt), len(prompt), OP.describe())
        if OP.from_fallback():
            log.error("SHARED OLLAMA POLICY NOT READ (%s). STERLING reads "
                      "that file; this process is on a local copy and the "
                      "two can now drift apart into a 55 GB eviction loop "
                      "with nothing in either log marked wrong.", OP.PATH)
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "num_predict": max(64, int(max_tokens)),
                    "temperature": 0.85,
                    "repeat_penalty": 1.1,
                    "num_ctx": pol["num_ctx"],
                },
                "keep_alive": pol["keep_alive"],
                "stream": False,
            },
            timeout=LLM_TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning("Ollama HTTP %d (%s): %s",
                        resp.status_code, model, resp.text[:200])
            #: the model went away underneath us -- deleted by a VRAM
            #: clean-up, which is precisely how the last two outages began.
            #: Ask again and answer this message, rather than 404ing until
            #: somebody reads a log.
            if resp.status_code == 404 and _retry:
                log.warning("%r vanished mid-run -- re-resolving", model)
                if _resolve_model(force=True):
                    return _try_ollama(prompt, system_prompt, max_tokens,
                                       _retry=False)
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
