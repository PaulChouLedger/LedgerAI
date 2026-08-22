"""
growth_report -- Daily rollup over the growth event pipeline, the composite
reward that feeds the bandit, and the aggregate puck-transfer extract
(2026-08-22).

Usage (from services/telegram, or anywhere -- paths are absolute):

  python3 growth_report.py                  # report for yesterday (UTC)
  python3 growth_report.py --date 2026-08-20
  python3 growth_report.py --days 7         # multi-day summary
  python3 growth_report.py --roll           # update bandit posteriors for
                                            # every fully-observed day not
                                            # yet rolled (idempotent)
  python3 growth_report.py --puck-extract   # aggregate findings that
                                            # transfer to puck voice behavior

Reports are printed and written to GROWTH_REPORTS_DIR/YYYY-MM-DD.txt.

════════════════════════════════════════════════════════════════════════════
THE COMPOSITE ENGAGEMENT SCORE (documented here and in docs/TG-GROWTH.md;
change it in ONE place -- this file -- and the doc says so)

Per chat, per day D, reward r in [0, 1]:

  +0.50  RETURN   DMs: the user sends anything on D+1..D+3.
                  Groups: fraction of day-D engaged users (replied to or
                  reacted to Aura) who are active again on D+1..D+3.
  +0.25  POSITIVE explicit positive signal on D: a reaction on one of her
                  messages, or a reply whose text matches the praise
                  pattern (read from engagement_metrics.jsonl -- reply text
                  is not duplicated into growth events).
  +0.15  DEPTH    a conversation continued: >= 2 human messages after one
                  of her messages within a 30-minute session.
  +0.10  SHARE    a referral click attributed to a user of this chat, a
                  /referral issued, or an accepted share hook.
  -0.60  each NEGATIVE event: complaint, "stop", mute, removal. Floor 0.

Raw message count is deliberately absent: the target is return visits and
explicit positive signal, not volume (strategy.py rail 5). A day with bot
output and none of the above scores 0 -- silence from users is evidence.

A day is "fully observed" for rolling only once D+3 has ended, because the
RETURN term looks three days ahead. --roll therefore lags four days behind
the calendar; the daily report still prints same-day counts immediately.
════════════════════════════════════════════════════════════════════════════

PUCK FEEDBACK LOOP (--puck-extract): aggregates ONLY -- rates, medians,
counts per variant. No user ids, no chat ids, no text ever leaves this
report. Raw event and transcript files stay on this machine.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from growth_events import (  # noqa: E402
    GROWTH_EVENTS_FILE, GROWTH_REPORTS_DIR, GROWTH_DIR,
)
import strategy  # noqa: E402

ENGAGEMENT_METRICS_FILE = GROWTH_DIR / "engagement_metrics.jsonl"

SESSION_GAP_S = 1800  # 30 min of silence ends a session

_PRAISE_RE = re.compile(
    r"\b(love (this|it|you)|great|amazing|awesome|so (good|helpful)|lol|"
    r"lmao|haha|hilarious|brilliant|incredible|thank(s| you)|nice one|"
    r"good (one|call|point))\b", re.I)


def _day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _day_bounds(day: str) -> tuple[float, float]:
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return d0.timestamp(), (d0 + timedelta(days=1)).timestamp()


def load_events(path: Path = None) -> list[dict]:
    path = path or GROWTH_EVENTS_FILE
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn tail line is expected on live files
    return out


def load_engagement() -> list[dict]:
    return load_events(ENGAGEMENT_METRICS_FILE)


# ---------------------------------------------------------------------------
# Sessions: interleave msg_in/msg_out per chat, split on 30-min gaps
# ---------------------------------------------------------------------------

def _sessions(events: list[dict], day: str) -> dict[int, list[list[dict]]]:
    t0, t1 = _day_bounds(day)
    per_chat: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e["event"] in ("msg_in", "msg_out") and t0 <= e["ts"] < t1:
            per_chat[e.get("chat_id")].append(e)
    out: dict[int, list[list[dict]]] = {}
    for cid, evs in per_chat.items():
        evs.sort(key=lambda e: e["ts"])
        sessions, cur = [], [evs[0]]
        for e in evs[1:]:
            if e["ts"] - cur[-1]["ts"] > SESSION_GAP_S:
                sessions.append(cur)
                cur = []
            cur.append(e)
        sessions.append(cur)
        out[cid] = sessions
    return out


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


# ---------------------------------------------------------------------------
# Composite reward per chat for one day (formula in module docstring)
# ---------------------------------------------------------------------------

def chat_rewards_for_day(events: list[dict], engagement: list[dict],
                         day: str) -> dict[str, float]:
    t0, t1 = _day_bounds(day)
    t_future = t1 + 3 * 86400

    day_ev = [e for e in events if t0 <= e["ts"] < t1]
    future_in = defaultdict(set)   # chat_id -> user_ids active D+1..D+3
    for e in events:
        if e["event"] == "msg_in" and t1 <= e["ts"] < t_future:
            future_in[e["chat_id"]].add(e.get("user_id"))

    out_chats = {e["chat_id"] for e in day_ev if e["event"] == "msg_out"}
    if not out_chats:
        return {}

    # Engaged users per chat on D (replied to or reacted to Aura)
    engaged = defaultdict(set)
    positive = defaultdict(bool)
    for e in day_ev:
        if e["event"] == "reaction":
            engaged[e["chat_id"]].add(e.get("user_id"))
            positive[e["chat_id"]] = True
    for m in engagement:
        if m.get("event") == "reply" and t0 <= m.get("ts", 0) < t1:
            engaged[m["chat_id"]].add(m.get("user_id"))
            if _PRAISE_RE.search(m.get("text", "")):
                positive[m["chat_id"]] = True
        if m.get("event") == "reaction" and t0 <= m.get("ts", 0) < t1:
            engaged[m["chat_id"]].add(m.get("user_id"))
            positive[m["chat_id"]] = True

    chat_type = {}
    for e in events:
        if e["event"] == "msg_in":
            chat_type.setdefault(e["chat_id"], e.get("chat_type"))

    share = defaultdict(bool)
    negatives = defaultdict(int)
    for e in day_ev:
        if e["event"] in ("referral_click", "referral_link_issued",
                          "share_hook_offered", "command") and \
                e.get("command", "referral") == "referral":
            if e.get("chat_id") is not None:
                share[e["chat_id"]] = True
        if e["event"] == "negative":
            negatives[e["chat_id"]] += 1

    sessions = _sessions(events, day)

    rewards: dict[str, float] = {}
    for cid in out_chats:
        r = 0.0
        # RETURN
        if chat_type.get(cid) == "private":
            if future_in.get(cid):
                r += 0.50
        else:
            eng = {u for u in engaged.get(cid, set()) if u}
            if eng:
                back = eng & future_in.get(cid, set())
                r += 0.50 * (len(back) / len(eng))
        # POSITIVE
        if positive.get(cid):
            r += 0.25
        # DEPTH: >=2 human msgs after a bot msg within one session
        for sess in sessions.get(cid, []):
            seen_out = False
            after = 0
            for e in sess:
                if e["event"] == "msg_out":
                    seen_out, after = True, 0
                elif seen_out and e["event"] == "msg_in":
                    after += 1
                    if after >= 2:
                        r += 0.15
                        break
            if after >= 2:
                break
        # SHARE
        if share.get(cid):
            r += 0.10
        # NEGATIVE
        r -= 0.60 * negatives.get(cid, 0)
        rewards[str(cid)] = max(0.0, min(1.0, r))
    return rewards


# ---------------------------------------------------------------------------
# Daily report
# ---------------------------------------------------------------------------

def build_report(events: list[dict], engagement: list[dict],
                 day: str) -> str:
    t0, t1 = _day_bounds(day)
    day_ev = [e for e in events if t0 <= e["ts"] < t1]
    by = defaultdict(list)
    for e in day_ev:
        by[e["event"]].append(e)

    lines = [f"TG GROWTH REPORT — {day} (UTC)", "=" * 44]

    ins, outs = by["msg_in"], by["msg_out"]
    users_today = {e.get("user_id") for e in ins if e.get("user_id")}
    chats_today = {e.get("chat_id") for e in ins}

    # Retention: of yesterday's users, who came back today; same for D-7
    def _active_users(d: str) -> set:
        a0, a1 = _day_bounds(d)
        return {e.get("user_id") for e in events
                if e["event"] == "msg_in" and a0 <= e["ts"] < a1
                and e.get("user_id")}

    prev_day = (datetime.strptime(day, "%Y-%m-%d")
                - timedelta(days=1)).strftime("%Y-%m-%d")
    week_ago = (datetime.strptime(day, "%Y-%m-%d")
                - timedelta(days=7)).strftime("%Y-%m-%d")
    y_users = _active_users(prev_day)
    w_users = _active_users(week_ago)
    d1 = (len(y_users & users_today), len(y_users))
    d7 = (len(w_users & users_today), len(w_users))

    lines += [
        f"Activity      : {len(ins)} msgs in / {len(outs)} out, "
        f"{len(users_today)} users across {len(chats_today)} chats",
        f"New           : {len(by['chat_first_seen'])} chats, "
        f"{len(by['user_first_seen'])} users first seen",
        f"Groups        : +{len(by['group_add'])} added, "
        f"-{len(by['group_remove'])} removed, "
        f"{len(by['member_join'])} member joins seen",
        f"Commands      : {len(by['command'])}  "
        f"({', '.join(sorted({e.get('command','?') for e in by['command']})) or '—'})",
        f"Reactions     : {len(by['reaction'])} on her messages",
        f"Referral      : {len(by['referral_click'])} clicks, "
        f"{len(by['referral_link_issued'])} links issued, "
        f"{len(by['share_hook_offered'])} earned hooks offered",
        f"Negative      : {len(by['negative'])} "
        f"({', '.join(sorted({e.get('kind','?') for e in by['negative']})) or 'none'})",
    ]

    lats = [e["latency_s"] for e in outs if e.get("latency_s") is not None]
    if lats:
        p90 = sorted(lats)[max(0, int(len(lats) * 0.9) - 1)]
        lines.append(f"Latency       : median {_median(lats):.1f}s, "
                     f"p90 {p90:.1f}s ({len(lats)} sends)")
    sessions = _sessions(events, day)
    sess_lens = [s[-1]["ts"] - s[0]["ts"]
                 for ss in sessions.values() for s in ss if len(s) > 1]
    if sess_lens:
        lines.append(f"Sessions      : {sum(len(s) for s in sessions.values())} "
                     f"total, median length "
                     f"{_median(sess_lens)/60:.1f} min")
    lines.append(
        f"Retention     : D1 {d1[0]}/{d1[1]} returned"
        + (f" ({d1[0]/d1[1]:.0%})" if d1[1] else "")
        + f", D7 {d7[0]}/{d7[1]}"
        + (f" ({d7[0]/d7[1]:.0%})" if d7[1] else ""))

    # Rewards for the day (provisional if the D+3 window is still open)
    rewards = chat_rewards_for_day(events, engagement, day)
    provisional = time.time() < t1 + 3 * 86400
    if rewards:
        tag = " (PROVISIONAL — return window still open)" if provisional else ""
        lines.append(f"Reward        : mean {sum(rewards.values())/len(rewards):.2f} "
                     f"across {len(rewards)} chat-days{tag}")

    # Variant table
    lines += ["", "BANDIT POSTERIORS (mean reward | chat-days observed)"]
    cur = None
    for dim, name, mean, n in strategy.posterior_table():
        if dim != cur:
            lines.append(f"  {dim}:")
            cur = dim
        lines.append(f"    {name:16s} {mean:.3f} | n={n}")

    counts = defaultdict(int)
    for v in strategy.assigned_variants().values():
        for d, name in v.items():
            counts[(d, name)] += 1
    lines += ["", "ASSIGNMENTS (chats per arm): " + ", ".join(
        f"{d}:{n}={c}" for (d, n), c in sorted(counts.items())) if counts
        else "ASSIGNMENTS: none yet"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bandit roll: every fully-observed day not yet rolled
# ---------------------------------------------------------------------------

def roll(events: list[dict], engagement: list[dict]) -> list[str]:
    days = sorted({_day_of(e["ts"]) for e in events
                   if e["event"] == "msg_out"})
    rolled = []
    for day in days:
        _, t1 = _day_bounds(day)
        if time.time() < t1 + 3 * 86400:
            continue  # return window still open
        rewards = chat_rewards_for_day(events, engagement, day)
        if rewards and strategy.batch_update(day, rewards):
            rolled.append(day)
    return rolled


# ---------------------------------------------------------------------------
# Puck transfer extract — aggregates only, by design
# ---------------------------------------------------------------------------

def puck_extract(events: list[dict], engagement: list[dict]) -> str:
    outs = [e for e in events if e["event"] == "msg_out"]
    if not outs:
        return "PUCK EXTRACT: no outbound data yet."

    # Which sent messages earned a reply/reaction (via engagement metrics)?
    replied_ids = set()
    for m in engagement:
        if m.get("event") in ("reply", "reaction"):
            replied_ids.add((m.get("chat_id"), m.get("message_id")))

    # Engagement rate by response length (n_sentences)
    by_len = defaultdict(lambda: [0, 0])   # n_sentences -> [sends, engaged]
    by_persona = defaultdict(lambda: [0, 0])
    lat_buckets = defaultdict(lambda: [0, 0])  # latency bucket -> [sends, engaged]

    # growth events don't carry message_id; approximate engagement per
    # chat-day: a send counts as engaged if its chat had any reply/reaction
    # that day. Coarse, aggregate, and honest about it.
    eng_days = set()
    for m in engagement:
        if m.get("event") in ("reply", "reaction"):
            eng_days.add((m.get("chat_id"), _day_of(m.get("ts", 0))))

    for e in outs:
        k = (e.get("chat_id"), _day_of(e["ts"]))
        hit = k in eng_days
        ns = min(e.get("n_sentences") or 0, 4)
        by_len[ns][0] += 1
        by_len[ns][1] += hit
        v = e.get("variant") or {}
        if v.get("persona"):
            by_persona[v["persona"]][0] += 1
            by_persona[v["persona"]][1] += hit
        lat = e.get("latency_s")
        if lat is not None:
            b = "<5s" if lat < 5 else "5-15s" if lat < 15 else ">15s"
            lat_buckets[b][0] += 1
            lat_buckets[b][1] += hit

    def _rate(v):
        return f"{v[1]/v[0]:.0%} of {v[0]}" if v[0] else "—"

    lines = [
        "PUCK TRANSFER EXTRACT — aggregates only; no user data leaves this "
        "report", "=" * 60,
        "",
        "1. RESPONSE LENGTH (puck: preferred reply discipline)",
        *(f"   {n} sentence(s): engaged-day rate {_rate(v)}"
          for n, v in sorted(by_len.items())),
        "",
        "2. HUMOR TOLERANCE (puck: how much edge the room rewards)",
        *(f"   persona={p:9s}: engaged-day rate {_rate(v)}"
          for p, v in sorted(by_persona.items())),
        "",
        "3. LATENCY TOLERANCE (puck: how long a hold-phrase must cover)",
        *(f"   {b:6s}: engaged-day rate {_rate(v)}"
          for b, v in sorted(lat_buckets.items())),
        "",
        "Read with the bandit posteriors in the daily report. Transfer the",
        "FINDING (e.g. 'two-sentence replies out-engage three'), never the",
        "data. Raw events, transcripts and ids stay on this machine.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    yesterday = (datetime.now(timezone.utc)
                 - timedelta(days=1)).strftime("%Y-%m-%d")
    ap.add_argument("--date", default=yesterday)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--roll", action="store_true")
    ap.add_argument("--puck-extract", action="store_true")
    args = ap.parse_args()

    events = load_events()
    engagement = load_engagement()

    if args.roll:
        rolled = roll(events, engagement)
        print(f"Rolled {len(rolled)} day(s) into the bandit: "
              f"{', '.join(rolled) or 'none pending'}")
        return

    if args.puck_extract:
        print(puck_extract(events, engagement))
        return

    end = datetime.strptime(args.date, "%Y-%m-%d")
    for i in range(args.days - 1, -1, -1):
        day = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        report = build_report(events, engagement, day)
        print(report)
        print()
        GROWTH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = GROWTH_REPORTS_DIR / f"{day}.txt"
        try:
            out.write_text(report + "\n")
        except OSError as e:
            print(f"REPORT WRITE FAILED ({e}) — output above is the only copy",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
