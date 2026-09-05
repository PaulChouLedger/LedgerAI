"""
test_room_priors -- the env-gated room-doctrine prior loading in
strategy.py (2026-09-04). Synthetic policy file, scratch growth dir; no
live services, no live data.

  env -u LD_LIBRARY_PATH python3 -u tests/test_room_priors.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, msg
    PASS += 1
    print(f"  ok  {msg}")


def _fresh_strategy():
    for m in ("strategy", "growth_events"):
        sys.modules.pop(m, None)
    import strategy
    return importlib.reload(strategy)


with tempfile.TemporaryDirectory() as td:
    os.environ["AURA_TG_GROWTH_DIR"] = td   # never touch production state

    policy = Path(td) / "policy.json"
    policy.write_text(json.dumps({
        "schema": "room-doctrine-v1",
        "recommendations": {"tg": {
            "priors_only": True,
            "prior_pseudo_n_cap": 20,
            "bandit_priors": {
                "length": {"terse": {"a": 70.0, "b": 30.0}},   # over cap
                "persona": {"dry": {"a": 7.0, "b": 3.0}},      # under cap
                "bogus_dim": {"x": {"a": 5, "b": 5}},          # unknown dim
                "media": {"nonexistent": {"a": 5, "b": 5}},    # unknown arm
            },
        }},
    }))

    # ---- flag OFF (default): nothing seeded --------------------------------
    os.environ.pop("AURA_TG_ROOM_PRIORS", None)
    os.environ["AURA_ROOM_POLICY"] = str(policy)
    s = _fresh_strategy()
    p = s._posterior("length", "terse")
    ok(p["a"] == 1.0 and p["b"] == 1.0,
       "flag off: posteriors untouched (default is OFF)")

    # ---- flag ON: seeded, capped, evidence respected -----------------------
    os.environ["AURA_TG_ROOM_PRIORS"] = "1"
    s = _fresh_strategy()
    p = s._posterior("persona", "dry")
    ok(abs(p["a"] - 8.0) < 1e-6 and abs(p["b"] - 4.0) < 1e-6,
       "under-cap prior seeded as-is (dry a=1+7, b=1+3)")
    p = s._posterior("length", "terse")
    ok(abs((p["a"] - 1.0) + (p["b"] - 1.0) - 20.0) < 1e-6,
       "over-cap prior clamped to 20 pseudo-observations")
    ok(abs((p["a"] - 1.0) - 14.0) < 1e-6,
       "clamp preserves the prior's mean (0.7)")
    ok("bogus_dim" not in s._bandit.get("dims", {}),
       "unknown dimension ignored")
    ok("nonexistent" not in s._bandit.get("dims", {}).get("media", {}),
       "unknown arm ignored")
    ok(s._posterior("persona", "edgy")["a"] == 1.0,
       "arms without a prior stay flat")

    # ---- an arm with real evidence is never overwritten --------------------
    os.environ["AURA_TG_ROOM_PRIORS"] = "1"
    s = _fresh_strategy()
    # simulate real evidence arriving before a (second) seed attempt
    p = s._posterior("persona", "dry")
    before = dict(p)
    s._seed_room_priors()
    ok(s._posterior("persona", "dry") == before,
       "re-seeding skips an already-informative posterior")

    # ---- unreadable policy with flag ON: loud, not fatal -------------------
    os.environ["AURA_ROOM_POLICY"] = str(Path(td) / "missing.json")
    s = _fresh_strategy()
    ok(s._posterior("length", "terse")["a"] == 1.0,
       "missing policy: runs without priors (error logged, not raised)")

    # cleanup env so nothing leaks into other tests run in-process
    for k in ("AURA_TG_ROOM_PRIORS", "AURA_ROOM_POLICY", "AURA_TG_GROWTH_DIR"):
        os.environ.pop(k, None)

print(f"\nALL {PASS} checks passed")
