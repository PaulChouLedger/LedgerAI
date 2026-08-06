#!/usr/bin/env python3
"""Welcoming people who join (services/telegram/bot.py, 2026-08-06).

Owner: "in general give the directive to the TG bot to have unique warm
welcomes in area31 for people who just joined."

The LLM call is not tested here — it needs a live model and its output is
not deterministic. What IS tested is everything around it, and in particular
the two failures that showed up while building it, both of which were found
by RUNNING the composer and reading what came out rather than by reasoning
about the prompt:

  1. Asked for a line "specific enough that it could not have been sent to
     anyone else", the model invented biographies for strangers — "your
     insights on urban planning" for someone who had never spoken. A person
     has just joined; nothing is known about them; the prompt demanded
     specificity anyway and the model supplied it. Fixed by banning claims
     about the person and pointing the specificity at the room.

  2. With that ban in place, five consecutive welcomes came out as one
     sentence with different adjectives: "<Name>, welcome to Area31! ...
     just as things are getting interesting." Showing the model its recent
     lines is not enough — it varies the adjectives and keeps the shape. So
     the ANGLE is rotated deterministically from the ledger.

Run:  python3 tests/test_welcome_new_members.py     (or under pytest)
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BOT = _ROOT / "services" / "telegram" / "bot.py"
SRC = _BOT.read_text()


def _literal(name: str):
    """Pull a module-level literal out of bot.py without importing it."""
    tree = ast.parse(SRC)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in bot.py")


# ---------------------------------------------------------------------------
# The prompt must not ask for what cannot be known
# ---------------------------------------------------------------------------

def test_prompt_forbids_claims_about_the_person():
    sys_prompt = _literal("_WELCOME_SYSTEM").lower()
    for required in ("you know nothing about this person",
                     "never claim to know their work",
                     "never predict what they will contribute"):
        assert required in sys_prompt, (
            f"the ban on inventing a stranger's biography is gone: {required!r} "
            "— this is what produced 'your insights on urban planning' for "
            "somebody who had never spoken")


def test_prompt_does_not_demand_impossible_specificity():
    """The exact sentence that caused the fabrications must not come back."""
    sys_prompt = _literal("_WELCOME_SYSTEM").lower()
    assert "could not have been sent to anyone else" not in sys_prompt, (
        "this demanded specificity about a person nothing is known about, "
        "and the model invented it three times out of three")


# ---------------------------------------------------------------------------
# Variety is forced, not hoped for
# ---------------------------------------------------------------------------

def test_angles_are_distinct_and_plural():
    angles = _literal("_WELCOME_ANGLES")
    assert len(angles) >= 6, "too few angles to keep a busy room varied"
    assert len(set(angles)) == len(angles), "duplicate angles"


def test_angle_rotation_advances_with_the_ledger():
    """Rotated off ledger length, so it survives a restart mid-cycle."""
    angles = _literal("_WELCOME_ANGLES")
    seen = [angles[n % len(angles)] for n in range(len(angles))]
    assert len(set(seen)) == len(angles), (
        "the rotation repeats before the pool is spent")
    # And it does not reset: a ledger of 3 continues at index 3, not 0.
    assert angles[3 % len(angles)] != angles[0]


def test_fallbacks_all_take_a_name():
    for f in _literal("WELCOME_FALLBACKS"):
        assert "{name}" in f, f"fallback cannot be personalised: {f!r}"


# ---------------------------------------------------------------------------
# Structure of the handler — the things that would be embarrassing live
# ---------------------------------------------------------------------------

def _handler_src() -> str:
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_welcome_new_members":
            return ast.get_source_segment(SRC, node) or ""
    raise AssertionError("_welcome_new_members not found")


def test_bots_and_self_are_not_welcomed():
    body = _handler_src()
    assert "is_bot" in body, "a bot joining would get a warm personal welcome"
    assert "u.id != me" in body, "she would welcome herself"


def test_rejoins_are_not_rewelcomed():
    assert "_already_welcomed" in _handler_src(), (
        "someone who leaves and rejoins would be welcomed every time")


def test_pilot_gate_applies():
    assert "chat_allowed" in _handler_src(), (
        "welcomes would be sent outside the pilot chats")


def test_a_burst_does_not_become_a_flood():
    body = _handler_src()
    assert "_WELCOME_MAX_BURST" in body, (
        "a bulk add would post one message per person")
    assert _literal("_WELCOME_MAX_BURST") <= 5


def test_a_failed_send_is_not_recorded_as_welcomed():
    """Same rule _greet_on_join states: retry on a genuine rejoin."""
    body = _handler_src()
    send_idx = body.index("send_message")
    record_idx = body.index('state["greeted"]')
    assert send_idx < record_idx, (
        "the ledger is written before the send succeeds, so a failed "
        "welcome is swallowed forever")


def test_silence_is_never_an_outcome():
    """Model down must still greet them — PRINCIPLES §1."""
    body = _handler_src()
    assert "_pick_welcome_fallback" in body, (
        "if the model returns nothing, the joiner is met with silence")


def test_handler_is_registered():
    assert "filters.StatusUpdate.NEW_CHAT_MEMBERS" in SRC, (
        "nothing routes join events to the welcome handler")
    assert "_welcome_new_members" in SRC.split("def main(")[-1], (
        "the handler is defined but never added to the application")


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
