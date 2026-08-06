#!/usr/bin/env python3
"""Guards on the implicit-complaint detector (services/telegram/feedback.py).

The fixtures are real. `AREA31_TRIGGER` is the verbatim message that, at
01:46:13 on 2026-08-06, deleted Aura's messages and muted Area31 for two
hours — the owner explaining how she decides whether to speak. Talking about
the feature triggered the feature.

The asymmetry this file protects is the point: a missed complaint costs one
unwanted message, a false one costs two hours of a silent room. So a
complaint may only DELETE AND MUTE when she could plausibly be its subject
AND it is aimed at a "you". Anything merely plausible is recorded and learned
from, and touches nothing.

`REAL_TRAFFIC` below is every message in the 1058 recorded inbound messages
of data/telegram/dm_history.jsonl that fired the OLD detector — the whole
population, not a selection. Two of the seven are about farting and Taco
Bell. Those two are why "does the message contain the word you" is not the
test: both do.

Run:  python3 tests/test_complaint_guard.py     (or under pytest)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "services" / "telegram"))

import feedback as fb  # noqa: E402

OWNER = sorted(fb.OWNER_USER_IDS)[0]
STRANGER = 999_000_111
AREA31 = -1003025733750

# The message that caused the incident, verbatim from the bot log.
AREA31_TRIGGER = (
    'yes, it\'s dynamically adaptive. she will jump in if she feels it '
    'crosses a certain "value add" threshold but also doesn\'t breach the '
    '"i already spoke too much recently" threshold'
)

# The one that fired legitimately the same evening, 20:47:08.
REAL_COMPLAINT = "lol ok aura tone it down"


def _engine():
    """A FeedbackEngine writing to a throwaway directory."""
    tmp = Path(tempfile.mkdtemp(prefix="complaint_guard_"))
    fb.LEARNED_DIRECTIVES_FILE = tmp / "learned_directives.json"
    fb.FEEDBACK_AUDIT_FILE = tmp / "feedback_audit.json"
    fb.FEEDBACK_QUEUE_FILE = tmp / "feedback_queue.json"
    return fb.FeedbackEngine()


def _call(engine, text, *, user_id=STRANGER, reply=False, since=1):
    return engine.record_implicit(
        user_id, AREA31, "tester", text,
        aura_last_msg="something she said",
        is_reply_to_bot=reply,
        msgs_since_aura=since,
    )


# ---------------------------------------------------------------------------
# The incident
# ---------------------------------------------------------------------------

def test_area31_trigger_is_ignored():
    """The exact message that muted the room must now do nothing at all."""
    assert _call(_engine(), AREA31_TRIGGER, user_id=OWNER) is None


def test_area31_trigger_ignored_from_a_stranger_too():
    """It is quoted. Who said it does not matter."""
    assert _call(_engine(), AREA31_TRIGGER, user_id=STRANGER) is None


def test_real_complaint_still_retracts():
    """The guard must not cost her the complaints that were real."""
    c = _call(_engine(), REAL_COMPLAINT)
    assert c is not None, "a complaint saying her name must survive"
    assert c.category == "too_much"
    assert c.actionable, "naming her is the strong case — it may retract"


# ---------------------------------------------------------------------------
# Quoting — somebody discussing the phrase, not saying it
# ---------------------------------------------------------------------------

def test_quoted_phrase_ignored():
    assert _call(_engine(), 'she mutes on "too much", which is too broad') is None


def test_smart_quotes_and_backticks_ignored():
    for text in ['the rule is “tone it down” apparently',
                 'the pattern is `shut up` in feedback.py']:
        assert _call(_engine(), text) is None, text


def test_apostrophes_do_not_form_a_quote():
    """"don't" must not swallow the sentence and hide a real complaint."""
    c = _call(_engine(), "don't, aura — that's too much")
    assert c is not None and c.actionable


# ---------------------------------------------------------------------------
# Is it about her at all
# ---------------------------------------------------------------------------

def test_not_about_her_when_she_has_not_spoken():
    """Two people telling each other to relax is not feedback."""
    assert _call(_engine(), "mate just relax", since=40) is None


def test_recent_but_unaddressed_is_recorded_not_actioned():
    """The third outcome: plausible. Learn from it, break nothing."""
    c = _call(_engine(), "ok this is getting to be too much", since=1)
    assert c is not None, "she spoke one message ago — worth recording"
    assert not c.actionable, "but nobody addressed her; must not mute"
    assert c.strength == "ambient"


def test_reply_to_her_is_actionable_without_her_name():
    c = _call(_engine(), "that's too long", reply=True, since=1)
    assert c is not None and c.actionable and c.strength == "addressed"


# ---------------------------------------------------------------------------
# The owner
# ---------------------------------------------------------------------------

def test_owner_describing_her_is_ignored():
    """He demos her, and demoing means describing. Unquoted, still ignored."""
    assert _call(_engine(), "she goes quiet if you say it's too much",
                 user_id=OWNER, since=1) is None


def test_owner_addressing_her_is_obeyed():
    """The exemption is for description. When he means it, he says her name."""
    c = _call(_engine(), "aura, tone it down", user_id=OWNER, since=1)
    assert c is not None and c.actionable


# ---------------------------------------------------------------------------
# Nothing that was never a complaint became one
# ---------------------------------------------------------------------------

def test_clean_message_is_still_none():
    assert _call(_engine(), "what's the price of eth today") is None


def test_queue_records_strength():
    e = _engine()
    _call(e, "this is too much", since=1)
    assert e._queue[-1]["strength"] == "ambient"


# ---------------------------------------------------------------------------
# Real traffic — every old-detector fire in 1058 recorded inbound messages,
# with the verdict this guard gives it. Replayed 2026-08-06.
# ---------------------------------------------------------------------------

REAL_TRAFFIC = [
    # (text, must_be_actionable, why)
    ("Do you know, is too much farting good for my health? 😁", False,
     "has a 'you' and a 'too much' in different clauses; about farting"),
    ("And he likes taco bell too much", False,
     "third person; the complaint is about Taco Bell"),
    ("I would like a softer no response from you, if i am asking too much.  "
     "Your responses so far come off as passive aggressive", False,
     "a real complaint, but 'too much' is about his own asking — recorded, "
     "not actioned. The conceded miss: downgraded, never dropped."),
    ("Stop being evasive", True, "imperative, no pronoun needed"),
    ("Log data you hallucinated because there is not a single duplicate line",
     True, "'you hallucinated' — same clause"),
    ("I didn’t call you aweful because of the wrong assumptions of duplicated "
     "messages I’m calling you aweful because you are acting like a bossy "
     "bitch for several days now", True,
     "'you are acting like' — same clause as the match"),
    ("Stop that over shit", True, "imperative"),
]


def test_real_traffic_verdicts():
    for text, should_action, why in REAL_TRAFFIC:
        c = _call(_engine(), text, since=1)
        got = bool(c and c.actionable)
        assert got == should_action, (
            f"{'expected ACTION' if should_action else 'expected NO action'} "
            f"({why}) for {text[:60]!r}")


def test_real_traffic_still_learns_from_everything():
    """Downgraded is not dropped. Every one of the seven is still recorded."""
    for text, _, _ in REAL_TRAFFIC:
        assert _call(_engine(), text, since=1) is not None, text


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
