"""
End-to-end proof of the growth/experimentation pipeline with synthetic
events (2026-08-22). No Telegram, no LLM, no production data: everything
runs against a temp dir via AURA_TG_GROWTH_DIR.

Simulates two weeks in which chats assigned persona=edgy + length=terse
measurably out-engage persona=dry + length=standard (users react and come
back next day), then verifies that:

  1. events append and never contain message text (privacy rail)
  2. the daily report builds and contains the expected sections
  3. the composite reward separates the good chats from the bad
  4. --roll updates Beta posteriors so the bandit prefers the good arms
  5. rolling is idempotent (a day cannot be double-counted)
  6. structural rails (assert_rails) hold for every shipped variant
  7. the puck extract is aggregate-only (no ids, no text)

Run:  python3 tests/test_growth_engine.py     (from services/telegram)
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="tg_growth_test_"))
os.environ["AURA_TG_GROWTH_DIR"] = str(TMP)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import growth_events as ge          # noqa: E402
import strategy                     # noqa: E402
import growth_report as gr          # noqa: E402

DAY = 86400
# Synthetic history: 16..2 days ago, so every day's D+3 return window is
# closed and --roll can consume all of it.
T0 = time.time() - 16 * DAY

GOOD = {"persona": "edgy", "length": "terse", "media": "light",
        "proactivity": "witty", "onboarding": "playful",
        "referral_hook": "earned"}
BAD = {"persona": "dry", "length": "standard", "media": "off",
       "proactivity": "conservative", "onboarding": "warm",
       "referral_hook": "none"}


def _force_assign(chat_id: int, variant: dict) -> None:
    strategy._assignments[str(chat_id)] = {"variant": dict(variant),
                                           "ts": T0}


def _event(ts: float, event: str, **fields) -> None:
    rec = {"ts": round(ts, 2), "event": event}
    rec.update(fields)
    with open(ge.GROWTH_EVENTS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _engagement(ts: float, event: str, **fields) -> None:
    rec = {"ts": round(ts, 2), "event": event}
    rec.update(fields)
    with open(gr.ENGAGEMENT_METRICS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def build_world() -> None:
    # 6 DM chats: 100..102 assigned GOOD, 200..202 assigned BAD
    for cid in (100, 101, 102):
        _force_assign(cid, GOOD)
    for cid in (200, 201, 202):
        _force_assign(cid, BAD)
    strategy._save(strategy.STRATEGY_ASSIGNMENTS_FILE, strategy._assignments)

    for day in range(14):
        t = T0 + day * DAY + 3600
        for cid in (100, 101, 102):
            uid = cid + 1000
            # GOOD chats: exchange + reaction + user returns daily
            _event(t, "msg_in", chat_id=cid, user_id=uid,
                   chat_type="private", n_chars=40)
            _event(t + 30, "msg_out", chat_id=cid, user_id=uid, kind="dm",
                   latency_s=4.0, n_chars=90, n_sentences=2,
                   variant=GOOD, gif=False)
            _event(t + 90, "msg_in", chat_id=cid, user_id=uid,
                   chat_type="private", n_chars=25)
            _event(t + 150, "msg_in", chat_id=cid, user_id=uid,
                   chat_type="private", n_chars=30)
            _event(t + 200, "reaction", chat_id=cid, user_id=uid,
                   emoji=["\U0001F525"], on_message_id=day)
            _engagement(t + 95, "reply", chat_id=cid, user_id=uid,
                        message_id=day, text="haha amazing, love it")
        for cid in (200, 201, 202):
            uid = cid + 1000
            # BAD chats: user speaks only every 4th day, never reacts,
            # never continues, one files a complaint mid-run
            if day % 4 == 0:
                _event(t, "msg_in", chat_id=cid, user_id=uid,
                       chat_type="private", n_chars=40)
                _event(t + 40, "msg_out", chat_id=cid, user_id=uid,
                       kind="dm", latency_s=18.0, n_chars=300,
                       n_sentences=4, variant=BAD, gif=False)
            if day == 8 and cid == 200:
                _event(t + 300, "negative", chat_id=cid, user_id=uid,
                       kind="complaint")


def main() -> int:
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # 6. rails hold (raises on violation)
    strategy.assert_rails()
    check("rails: every shipped variant passes assert_rails", True)

    # 1. pipeline API writes events, and privacy: no text field ever
    ge.msg_in(999, 42, "private", 33)
    ge.msg_out(999, 42, "dm", 2.5, "Two sentences. Exactly two.",
               strategy.variant_for(999))
    lines = [json.loads(x) for x in
             ge.GROWTH_EVENTS_FILE.read_text().splitlines()]
    check("events: pipeline appends", len(lines) >= 2,
          f"{len(lines)} events so far")
    check("privacy: no 'text' key in any growth event",
          not any("text" in e for e in lines))
    out = [e for e in lines if e["event"] == "msg_out"][-1]
    check("events: msg_out carries variant + n_sentences",
          out.get("n_sentences") == 2 and isinstance(out.get("variant"), dict))

    build_world()
    events = gr.load_events()
    engagement = gr.load_engagement()

    # 2. report builds
    day5 = gr._day_of(T0 + 5 * DAY + 3600)
    report = gr.build_report(events, engagement, day5)
    for section in ("TG GROWTH REPORT", "Retention", "BANDIT POSTERIORS",
                    "Latency"):
        check(f"report: contains '{section}'", section in report)
    gr.GROWTH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (gr.GROWTH_REPORTS_DIR / f"{day5}.txt").write_text(report)
    check("report: file written",
          (gr.GROWTH_REPORTS_DIR / f"{day5}.txt").exists())

    # 3. reward separation
    rewards = gr.chat_rewards_for_day(events, engagement, day5)
    good_r = [rewards[str(c)] for c in (100, 101, 102) if str(c) in rewards]
    bad_r = [rewards[str(c)] for c in (200, 201, 202) if str(c) in rewards]
    check("reward: good chats scored", len(good_r) == 3, f"{good_r}")
    check("reward: separation good>bad",
          good_r and (not bad_r or min(good_r) > max(bad_r)),
          f"good={good_r} bad={bad_r}")

    # 4. roll + posterior preference
    rolled = gr.roll(events, engagement)
    check("roll: consumed synthetic days", len(rolled) >= 10,
          f"{len(rolled)} days rolled")
    post = {(d, v): m for d, v, m, n in strategy.posterior_table()}
    check("bandit: edgy > dry",
          post[("persona", "edgy")] > post[("persona", "dry")],
          f"edgy={post[('persona','edgy')]:.3f} dry={post[('persona','dry')]:.3f}")
    check("bandit: terse > standard",
          post[("length", "terse")] > post[("length", "standard")],
          f"terse={post[('length','terse')]:.3f} "
          f"standard={post[('length','standard')]:.3f}")

    # Thompson sampling now prefers the winners for NEW chats
    picks = [strategy._thompson_pick("persona") for _ in range(200)]
    edgy_share = picks.count("edgy") / len(picks)
    check("bandit: new assignments lean to the winner", edgy_share > 0.5,
          f"edgy drawn {edgy_share:.0%} of 200 samples")

    # 5. idempotency
    rolled2 = gr.roll(events, engagement)
    check("roll: idempotent (second pass rolls nothing)", rolled2 == [],
          f"second pass rolled {rolled2}")

    # 7. puck extract is aggregate-only
    extract = gr.puck_extract(events, engagement)
    check("puck extract: builds with all three sections",
          all(s in extract for s in ("RESPONSE LENGTH", "HUMOR TOLERANCE",
                                     "LATENCY TOLERANCE")))
    # "100" collides with "100%" rates, so probe the unambiguous tokens:
    # user ids, and id-bearing field names.
    check("puck extract: no chat/user ids leak",
          not any(tok in extract for tok in
                  ("1100", "1101", "1102", "1200", "1201", "1202",
                   "chat_id", "user_id")))

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILED — {failures}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    rc = main()
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(rc)
