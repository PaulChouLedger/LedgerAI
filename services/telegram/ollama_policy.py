"""Reader for the box's ONE ollama policy. No values live in this file.

── WHY (2026-08-19) ───────────────────────────────────────────────────────
This bot is not the only thing talking to this machine's ollama. STERLING —
a voice assistant in a room, in a DIFFERENT REPOSITORY (~/Aura/sterling) —
answers questions with the same llama3.1 70B. A warm 70B is 55 GB and takes
71.83 s to load off disk; it is not either process's resource, it is the
box's.

num_ctx, the model name and the quantisation are part of an ollama runner's
IDENTITY. Two callers asking for the same model with different values do not
share it, they EVICT each other. On 2026-08-19 this bot came back online at
00:32 asking for 16384 while STERLING asked for 8192, and for half an hour
they alternated 55 GB reloads — 23.20 s and 15.42 s to first token against
0.6-0.96 s all day before. From a Telegram user's side that is Aura going
quiet mid-conversation for no reason anybody could see from this repo.

Second, quieter half of the same bug, found the same night: `keep_alive` in
the request BODY overrides OLLAMA_KEEP_ALIVE, per model, last writer wins.
STERLING pins with -1; this file sent "30m"; `ollama ps` therefore read
"28 minutes from now" on a model STERLING believed was pinned forever, and
every message this bot answered re-armed that timer.

Neither repo can import from the other, so the values moved into one file
that both READ at runtime:

    ~/Aura/config/ollama-policy.json      ($AURA_OLLAMA_POLICY overrides)

── WHY A COPY OF THE READER RATHER THAN A SHARED IMPORT ───────────────────
The two repos share the POLICY, not the code that reads it. Importing a
module across checkouts would make a moved or renamed Aura directory an
ImportError inside a live Telegram bot, and this file's whole history is
about failure modes that mute her (see llm.py's header: a pinned model name
that stopped existing cost twelve days of silence). A duplicated 60-line
reader degrades to "she still answers, and the log says she used a local
fallback". That is the trade, made deliberately.

Nothing here raises.

── SAY WHERE THE VALUE CAME FROM ──────────────────────────────────────────
A value that agrees with a fallback is not evidence it came from the right
place. `describe()` names the source and llm.py logs it, so a policy file
that quietly went missing is one grep away instead of being invisible until
the next eviction storm.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

#: absolute, because this repo has no relationship to that one other than
#: sharing a GPU. Override with AURA_OLLAMA_POLICY if either moves.
_DEFAULT = "/home/paul/Aura/config/ollama-policy.json"
PATH = Path(os.environ.get("AURA_OLLAMA_POLICY", _DEFAULT))

#: ── THE FALLBACK, AND WHY IT IS NOT A SECOND OPINION ───────────────────
#: These exist so a missing policy file degrades to "she answers, loudly
#: annotated" rather than a crash. They are a SNAPSHOT of that file as of
#: 2026-08-19, not an independent decision. Editing them to change
#: behaviour re-creates the bug this module exists to end — edit the JSON.
FALLBACK = {
    "model": "llama3.1:70b-instruct-q5_K_M",
    "num_ctx": 16384,
    "keep_alive": -1,
}

_TTL_S = 5.0

_cache: dict | None = None
_cache_src = ""
_cache_at = 0.0
_cache_mtime = -1.0


def _load() -> tuple[dict, str]:
    """(values, source). Never raises."""
    try:
        mtime = PATH.stat().st_mtime
    except OSError as exc:
        return dict(FALLBACK), (
            f"FALLBACK in {Path(__file__).name} — {PATH} unreadable "
            f"({exc.__class__.__name__})")
    try:
        raw = json.loads(PATH.read_text())
    except Exception as exc:                                  # noqa: BLE001
        return dict(FALLBACK), (
            f"FALLBACK in {Path(__file__).name} — {PATH} is not valid JSON "
            f"({exc!r})")

    out, missing = {}, []
    for k in FALLBACK:
        if k in raw:
            out[k] = raw[k]
        else:
            out[k] = FALLBACK[k]
            missing.append(k)
    src = (f"{PATH} (mtime "
           f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(mtime))})")
    if missing:
        #: partial is worse than absent — it looks like it worked.
        src += f" — BUT {','.join(missing)} ABSENT, those came from FALLBACK"
    return out, src


def policy(force: bool = False) -> dict:
    """The current values. Cheap enough to call per request."""
    global _cache, _cache_src, _cache_at, _cache_mtime
    now = time.time()
    if _cache is not None and not force and (now - _cache_at) < _TTL_S:
        return _cache
    try:
        mtime = PATH.stat().st_mtime
    except OSError:
        mtime = -1.0
    if _cache is not None and not force and mtime == _cache_mtime:
        _cache_at = now
        return _cache
    _cache, _cache_src = _load()
    _cache_at, _cache_mtime = now, mtime
    return _cache


def describe() -> str:
    """One line naming WHERE the live values came from. Log this."""
    p = policy()
    return (f"ollama policy: model={p['model']} num_ctx={p['num_ctx']} "
            f"keep_alive={p['keep_alive']} <- {_cache_src}")


def from_fallback() -> bool:
    """True when any value came from this file rather than the policy."""
    policy()
    return "FALLBACK" in _cache_src


MODEL = policy()["model"]
NUM_CTX = policy()["num_ctx"]
KEEP_ALIVE = policy()["keep_alive"]

if __name__ == "__main__":
    print(describe())
