#!/usr/bin/env python3
"""
Aura Stress Test — Hammer the LLM with diverse queries via API.

Sends queries through /chat-tg (non-streaming) and /chat-tts (streaming)
to test stability, garbage detection, and response quality.

Usage (run from dev machine, puck must be reachable):
    python3 tests/aura_stress_test.py [puck_ip]
"""

import json
import sys
import time
import requests
import random
import textwrap

PUCK_IP = sys.argv[1] if len(sys.argv) > 1 else "100.77.61.111"
BASE = f"http://{PUCK_IP}:11434"
TIMEOUT = 120

# ── Test Queries ──────────────────────────────────────────────────────────────

SIMPLE_QUERIES = [
    "Hello!",
    "How are you?",
    "Thanks!",
    "Good morning",
    "What's up?",
]

KNOWLEDGE_QUERIES = [
    "What is photosynthesis?",
    "Who was Albert Einstein?",
    "How far is the moon from Earth?",
    "What causes earthquakes?",
    "Why is the ocean salty?",
    "What is the speed of light?",
    "How do vaccines work?",
    "What is the capital of Japan?",
    "Who painted the Mona Lisa?",
    "What is DNA?",
]

COMPLEX_QUERIES = [
    "If you had to explain quantum mechanics to a child, how would you do it?",
    "What are the pros and cons of nuclear energy versus solar power?",
    "Can you explain why we dream and what purpose dreams might serve?",
    "What would happen to Earth if the moon suddenly disappeared?",
    "How does the internet actually work, from typing a URL to seeing a page?",
]

INSTRUCTION_QUERIES = [
    "How do I make scrambled eggs?",
    "How do I change a flat tire?",
    "What are the steps to make french press coffee?",
    "How do I tie a bowline knot?",
]

EDGE_CASES = [
    "What?",
    "Hmm",
    "I don't know",
    "Can you repeat that?",
    "",  # empty
    "a" * 500,  # very long single token
    "Tell me about " + " ".join(["everything"] * 50),  # long query
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_health():
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def is_garbage(text):
    if not text or len(text) < 10:
        return False
    cleaned = text.strip().replace(" ", "").lower()
    if len(cleaned) == 0:
        return False
    most_common = max(set(cleaned), key=cleaned.count)
    ratio = cleaned.count(most_common) / len(cleaned)
    return ratio > 0.8 and len(cleaned) > 20


def query_non_streaming(prompt, label=""):
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/chat-tg",
            json={"prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - t0
        resp = r.json().get("response", "")
        garbage = is_garbage(resp)
        empty = resp.strip() == ""
        status = "GARBAGE" if garbage else ("EMPTY" if empty else "OK")
        return status, resp[:150], elapsed
    except requests.exceptions.Timeout:
        return "TIMEOUT", "", time.time() - t0
    except Exception as e:
        return "ERROR", str(e)[:100], time.time() - t0


def query_streaming(prompt, label=""):
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/chat-tts",
            json={"prompt": prompt, "session_id": "stress_test"},
            stream=True,
            timeout=TIMEOUT,
        )
        text = ""
        first_chunk_time = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0
            content = line[5:].strip() if line.startswith("data:") else line.strip()
            if not content or "<sentence" in content:
                continue
            try:
                d = json.loads(content)
                t = d.get("text", d.get("token", d.get("response", "")))
                text += t
            except (json.JSONDecodeError, ValueError):
                text += content
        elapsed = time.time() - t0
        garbage = is_garbage(text)
        empty = text.strip() == ""
        status = "GARBAGE" if garbage else ("EMPTY" if empty else "OK")
        return status, text[:150], elapsed, first_chunk_time
    except requests.exceptions.Timeout:
        return "TIMEOUT", "", time.time() - t0, None
    except Exception as e:
        return "ERROR", str(e)[:100], time.time() - t0, None


# ── Test Phases ───────────────────────────────────────────────────────────────

def run_phase(name, queries, mode="non-streaming"):
    print(f"\n{'='*70}")
    print(f"  {name} ({mode}, {len(queries)} queries)")
    print(f"{'='*70}")
    results = {"OK": 0, "GARBAGE": 0, "EMPTY": 0, "TIMEOUT": 0, "ERROR": 0}

    for i, q in enumerate(queries, 1):
        display_q = q[:60] + "..." if len(q) > 60 else q
        if not q:
            display_q = "(empty)"

        if mode == "streaming":
            status, resp, elapsed, first = query_streaming(q)
            first_str = f", first={first:.1f}s" if first else ""
            print(f"  [{i:2d}] {status:7s} {elapsed:5.1f}s{first_str}  Q: {display_q}")
        else:
            status, resp, elapsed = query_non_streaming(q)
            print(f"  [{i:2d}] {status:7s} {elapsed:5.1f}s  Q: {display_q}")

        if status == "OK":
            # Print truncated response
            wrapped = textwrap.shorten(resp, width=80, placeholder="...")
            print(f"       -> {wrapped}")
        elif status in ("GARBAGE", "ERROR"):
            print(f"       !! {resp[:80]}")

        results[status] += 1

    return results


def run_rapid_fire(n=10):
    """Send n queries as fast as possible (non-streaming)."""
    print(f"\n{'='*70}")
    print(f"  RAPID FIRE ({n} queries, no delay)")
    print(f"{'='*70}")
    queries = random.sample(KNOWLEDGE_QUERIES + SIMPLE_QUERIES, min(n, len(KNOWLEDGE_QUERIES + SIMPLE_QUERIES)))
    results = {"OK": 0, "GARBAGE": 0, "EMPTY": 0, "TIMEOUT": 0, "ERROR": 0}

    for i, q in enumerate(queries, 1):
        status, resp, elapsed = query_non_streaming(q)
        tag = "OK" if status == "OK" else status
        short_resp = textwrap.shorten(resp, width=60, placeholder="...") if resp else ""
        print(f"  [{i:2d}] {tag:7s} {elapsed:5.1f}s  {q[:40]:40s} -> {short_resp}")
        results[status] += 1

    return results


def run_alternating(n=6):
    """Alternate between streaming and non-streaming to stress KV cache."""
    print(f"\n{'='*70}")
    print(f"  ALTERNATING STREAM/NON-STREAM ({n} queries)")
    print(f"{'='*70}")
    queries = random.sample(KNOWLEDGE_QUERIES, min(n, len(KNOWLEDGE_QUERIES)))
    results = {"OK": 0, "GARBAGE": 0, "EMPTY": 0, "TIMEOUT": 0, "ERROR": 0}

    for i, q in enumerate(queries, 1):
        if i % 2 == 1:
            # Streaming
            status, resp, elapsed, _ = query_streaming(q)
            mode = "STREAM"
        else:
            # Non-streaming
            status, resp, elapsed = query_non_streaming(q)
            mode = "NONSTR"
        tag = "OK" if status == "OK" else status
        short_resp = textwrap.shorten(resp, width=50, placeholder="...") if resp else ""
        print(f"  [{i:2d}] {mode} {tag:7s} {elapsed:5.1f}s  {q[:35]:35s} -> {short_resp}")
        results[status] += 1

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Aura Stress Test — Target: {BASE}")
    print(f"{'='*70}")

    if not check_health():
        print(f"ERROR: LLM not responding at {BASE}/health")
        sys.exit(1)
    print("Health check: OK\n")

    # Reset sessions
    try:
        requests.post(f"{BASE}/reset-session", json={"session_id": "__all__"}, timeout=5)
    except Exception:
        pass

    all_results = {}
    t_start = time.time()

    # Phase 1: Simple pleasantries (non-streaming)
    all_results["simple"] = run_phase("SIMPLE QUERIES", SIMPLE_QUERIES)

    # Phase 2: Knowledge queries (non-streaming)
    all_results["knowledge"] = run_phase("KNOWLEDGE QUERIES", KNOWLEDGE_QUERIES)

    # Phase 3: Complex queries (non-streaming)
    all_results["complex"] = run_phase("COMPLEX QUERIES", COMPLEX_QUERIES)

    # Phase 4: Instruction queries (non-streaming)
    all_results["instructions"] = run_phase("INSTRUCTION QUERIES", INSTRUCTION_QUERIES)

    # Phase 5: Streaming queries (chat-tts path)
    all_results["streaming"] = run_phase(
        "STREAMING QUERIES",
        random.sample(KNOWLEDGE_QUERIES + COMPLEX_QUERIES, 5),
        mode="streaming",
    )

    # Phase 6: Rapid fire
    all_results["rapid"] = run_rapid_fire(10)

    # Phase 7: Alternating stream/non-stream (KV cache stress)
    all_results["alternating"] = run_alternating(6)

    # Phase 8: Edge cases
    all_results["edge"] = run_phase("EDGE CASES", EDGE_CASES)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    total_ok = sum(r["OK"] for r in all_results.values())
    total_garbage = sum(r["GARBAGE"] for r in all_results.values())
    total_empty = sum(r["EMPTY"] for r in all_results.values())
    total_timeout = sum(r["TIMEOUT"] for r in all_results.values())
    total_error = sum(r["ERROR"] for r in all_results.values())
    total_queries = total_ok + total_garbage + total_empty + total_timeout + total_error

    for phase, results in all_results.items():
        phase_total = sum(results.values())
        phase_ok = results["OK"]
        pct = (phase_ok / phase_total * 100) if phase_total > 0 else 0
        issues = []
        if results["GARBAGE"]: issues.append(f"{results['GARBAGE']} garbage")
        if results["EMPTY"]: issues.append(f"{results['EMPTY']} empty")
        if results["TIMEOUT"]: issues.append(f"{results['TIMEOUT']} timeout")
        if results["ERROR"]: issues.append(f"{results['ERROR']} error")
        issue_str = f"  ({', '.join(issues)})" if issues else ""
        print(f"  {phase:15s}: {phase_ok}/{phase_total} OK ({pct:5.1f}%){issue_str}")

    score = (total_ok / total_queries * 100) if total_queries > 0 else 0
    print(f"\n  TOTAL: {total_ok}/{total_queries} OK  ({score:.1f}%)")
    print(f"  Time: {total_time:.0f}s")

    if total_garbage > 0:
        print(f"\n  ⚠️  {total_garbage} GARBAGE responses detected!")
    if score >= 90:
        print(f"\n  ✅ PASS")
    elif score >= 70:
        print(f"\n  ⚠️  MARGINAL")
    else:
        print(f"\n  ❌ FAIL")


if __name__ == "__main__":
    main()
