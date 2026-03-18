#!/usr/bin/env python3
"""
Aura Overnight QA — 8-hour automated conversational stress test.

Runs on the RTX workstation. Uses Qwen 32B to have natural multi-turn
conversations with Aura on the puck. Each conversation goes 5-8 turns deep
with real follow-up questions, then starts a new topic.

Logs every interaction, measures latency, tracks transcription accuracy,
detects self-echo, and generates a weakness report.

Usage:
    python3 tests/aura_overnight_qa.py [--hours 8] [--pause 30]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PIPER_MODEL = "/tmp/piper_test_voice/en_US-lessac-medium.onnx"
PIPER_LENGTH = "1.1"
PUCK_HOST = "ledger@192.168.1.94"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q5_1"
OLLAMA_URL = "http://localhost:11434/api/chat"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_LOG = LOG_DIR / f"qa_{RUN_ID}.csv"
REPORT_FILE = LOG_DIR / f"qa_{RUN_ID}_report.txt"
RAW_LOG = LOG_DIR / f"qa_{RUN_ID}_raw.log"

# Conversations are 5-8 turns each, with a 30-60s pause between conversations
CONVO_MIN_TURNS = 5
CONVO_MAX_TURNS = 8

# ---------------------------------------------------------------------------
# Conversation topics — the RTX picks one and dives deep
# ---------------------------------------------------------------------------

CONVERSATION_OPENERS = [
    # Science
    "Hey Aura, I've been wondering about black holes. What exactly happens if you fall into one?",
    "Aura, can you explain how photosynthesis works? Like, how do plants actually eat sunlight?",
    "I was reading about dark matter today. What is it exactly?",
    "Aura, how do stars actually form? Like, from the very beginning?",
    "What causes the northern lights? I've always wanted to understand that.",
    "Aura, I'm curious about DNA. How does it actually copy itself?",
    "Tell me about quantum mechanics. What makes it so weird?",
    "How do volcanoes form? What's happening under the surface?",
    # Technology
    "Aura, how does Wi-Fi actually work? Like the actual radio waves part?",
    "I keep hearing about quantum computing. What makes it different from regular computers?",
    "How do noise-cancelling headphones actually cancel sound? It seems like magic.",
    "Can you explain how GPS knows exactly where I am?",
    "What's the deal with blockchain? I still don't really get it.",
    "How do touch screens know where your finger is?",
    # History & Culture
    "Aura, how did they actually build the pyramids? The stones are massive.",
    "What caused the Roman Empire to fall? Was it one thing or many?",
    "Tell me about the Silk Road. Not the website, the ancient trade route.",
    "What was life like during the Renaissance? Why was it so special?",
    "How did ancient sailors navigate without GPS or even good maps?",
    # Daily Life & Health
    "Aura, what actually happens in your brain when you dream?",
    "Why do we need sleep? Like, what would happen if we just stopped?",
    "How does the immune system know which cells are invaders?",
    "What causes headaches? And why are there different kinds?",
    "How do broken bones heal themselves? That seems incredible.",
    # Nature & Animals
    "What's the smartest animal besides humans? I'm curious what you think.",
    "Aura, how deep is the deepest part of the ocean? What lives down there?",
    "How do birds know where to migrate? They fly thousands of miles somehow.",
    "Tell me about octopuses. I heard they're basically aliens.",
    "How do trees communicate with each other? I heard they actually do.",
    # Philosophy & Ideas
    "Do you think AI will ever truly be conscious? Like actually aware?",
    "What's more important for humanity, creativity or logic?",
    "Aura, is space exploration worth the enormous cost?",
    "What do you think makes someone truly intelligent?",
    "If you could change one thing about the world, what would it be?",
    # Fun & Creative
    "Aura, tell me the most interesting fun fact you know.",
    "If aliens visited Earth, what do you think would surprise them most?",
    "What would happen if the moon just disappeared one day?",
    "Can you make up a very short story? Just a few sentences.",
    "What's the most beautiful place on Earth in your opinion?",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log_raw(msg: str):
    """Append to raw log file and print."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(RAW_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def ollama_chat(messages: list[dict], temperature: float = 0.8) -> str:
    """Call local Ollama with conversation history."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 100}
    }
    try:
        result = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=60
        )
        resp = json.loads(result.stdout)
        return resp["message"]["content"].strip()
    except Exception as e:
        log_raw(f"ERROR: Ollama call failed: {e}")
        return ""


def speak(text: str) -> float:
    """Synthesize with Piper male voice, play from RTX. Returns duration in seconds."""
    wav_path = "/tmp/convo_line.wav"
    try:
        subprocess.run(
            f"echo {repr(text)} | piper --model {PIPER_MODEL} "
            f"--length-scale {PIPER_LENGTH} --output_file {wav_path}",
            shell=True, capture_output=True, timeout=15
        )
        result = subprocess.run(
            ["soxi", "-D", wav_path], capture_output=True, text=True, timeout=5
        )
        duration = float(result.stdout.strip()) if result.returncode == 0 else 3.0
    except Exception:
        duration = 3.0

    try:
        subprocess.run(["aplay", wav_path], capture_output=True, timeout=20)
    except Exception as e:
        log_raw(f"ERROR: aplay failed: {e}")

    return duration


def get_puck_journal(since_secs: int = 30) -> str:
    """Get recent puck journal entries."""
    try:
        result = subprocess.run(
            ["ssh", PUCK_HOST,
             f"sudo journalctl -u aura --no-pager --since '{since_secs} sec ago'"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except Exception as e:
        log_raw(f"ERROR: SSH journal failed: {e}")
        return ""


def wait_for_aura_response(play_duration: float, timeout: int = 35) -> dict:
    """Wait for Aura to respond and capture detailed metrics.

    Uses a timestamp marker to only capture NEW log entries (no overlap).
    """
    result = {
        "transcript": "",
        "response": "",
        "first_audio_ms": 0,
        "total_ms": 0,
        "clauses": 0,
        "self_echo": False,
        "whisper_heard": "",
        "error": "",
        "filler_played": "",
        "no_response": False,
    }

    t_start = time.time()

    # Wait for question audio to finish + VAD + Whisper processing
    time.sleep(play_duration + 1.5)

    # Poll for Aura's response — only look at entries AFTER we started
    seen_pipelined = set()
    responses = []
    mic_texts = []
    poll_deadline = t_start + timeout
    last_new_response_time = 0

    while time.time() < poll_deadline:
        elapsed = int(time.time() - t_start) + 2
        journal = get_puck_journal(since_secs=elapsed)
        lines = journal.strip().split('\n')

        for line in lines:
            # What the mic heard (first occurrence only)
            mic_match = re.search(r'\[mic\] "(.*?)"', line)
            if mic_match:
                heard = mic_match.group(1)
                if heard not in mic_texts:
                    mic_texts.append(heard)

            # Thinking filler
            filler_match = re.search(r'\[speaker\] Thinking filler.*?: (\S+)', line)
            if filler_match and not result["filler_played"]:
                result["filler_played"] = filler_match.group(1)

            # LLM streaming error
            if "Streaming error" in line or "Connection refused" in line:
                result["error"] = "LLM connection error"

            # Pipelined response (what she actually said)
            pipe_match = re.search(
                r'\[speaker\] Pipelined: (\d+)ms total.*?first=(\d+)ms, (\d+) clauses'
                r'.*?(?:→|->)\s*"(.*?)"', line
            )
            if pipe_match:
                said_text = pipe_match.group(4)
                line_key = said_text[:60]
                if line_key not in seen_pipelined:
                    seen_pipelined.add(line_key)
                    first_ms = int(pipe_match.group(2))
                    n_clauses = int(pipe_match.group(3))

                    if not responses:
                        result["first_audio_ms"] = first_ms
                    result["clauses"] += n_clauses

                    # Skip filler-like short responses
                    stripped = said_text.lower().strip().rstrip('!.,')
                    if stripped in ("sure", "great", "okay", "hello",
                                   "i'm here, ready to help", "what do you need"):
                        continue

                    responses.append(said_text)
                    last_new_response_time = time.time()

        # If we have responses and no new ones for 3 seconds, done
        if responses and last_new_response_time and (time.time() - last_new_response_time > 3):
            break

        # If we've been waiting >12s with no response at all, give up
        if not responses and (time.time() - t_start > play_duration + 12):
            break

        time.sleep(0.8)

    # Detect self-echo: mic heard something that matches Aura's response
    if len(mic_texts) > 1 and responses:
        for mic_t in mic_texts[1:]:  # skip first (that's the question)
            mic_words = set(mic_t.lower().split()[:6])
            for resp in responses:
                resp_words = set(resp.lower().split()[:6])
                if len(mic_words & resp_words) >= 3:
                    result["self_echo"] = True
                    break

    result["whisper_heard"] = mic_texts[0] if mic_texts else ""
    result["response"] = " ".join(responses) if responses else "[no response]"
    result["no_response"] = not responses
    result["total_ms"] = int((time.time() - t_start) * 1000)

    return result


def check_puck_alive() -> bool:
    """Verify Aura is still running on the puck."""
    try:
        result = subprocess.run(
            ["ssh", PUCK_HOST, "systemctl is-active aura"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() == "active"
    except Exception:
        return False


def check_llm_alive() -> bool:
    """Verify LLM is responding on the puck."""
    try:
        result = subprocess.run(
            ["ssh", PUCK_HOST, "curl -s http://localhost:11434/health"],
            capture_output=True, text=True, timeout=10
        )
        return '"status":"ok"' in result.stdout
    except Exception:
        return False


def reset_llm_session():
    """Clear the LLM conversation session on the puck to prevent context buildup."""
    try:
        result = subprocess.run(
            ["ssh", PUCK_HOST,
             'curl -s -X POST http://localhost:11434/reset-session '
             '-H "Content-Type: application/json" '
             '-d \'{"session_id": "__all__"}\''],
            capture_output=True, text=True, timeout=10
        )
        if "ok" in result.stdout:
            log_raw("  [Session reset OK]")
        else:
            log_raw(f"  [Session reset response: {result.stdout.strip()}]")
    except Exception as e:
        log_raw(f"  [Session reset failed: {e}]")


def restart_services_if_needed(stats: dict):
    """Restart Aura and/or LLM on the puck if they're not running."""
    if not check_puck_alive():
        log_raw("WARNING: Aura is not running! Restarting...")
        try:
            subprocess.run(
                ["ssh", PUCK_HOST, "sudo systemctl restart aura"],
                capture_output=True, timeout=30
            )
            time.sleep(25)
            stats["restarts"] += 1
        except Exception as e:
            log_raw(f"ERROR: Aura restart failed: {e}")

    if not check_llm_alive():
        log_raw("WARNING: LLM is not responding! Restarting...")
        try:
            # Kill any stale LLM processes first
            subprocess.run(
                ["ssh", PUCK_HOST,
                 "pkill -9 -f container_rest.py 2>/dev/null; sleep 2; "
                 "nohup bash /home/ledger/Aura4/run_llm_native.sh > /tmp/llm_native.log 2>&1 &"],
                capture_output=True, timeout=15
            )
            time.sleep(15)
            # Verify it came back
            if check_llm_alive():
                log_raw("LLM restarted successfully")
            else:
                log_raw("ERROR: LLM still not responding after restart")
            stats["llm_restarts"] += 1
        except Exception as e:
            log_raw(f"ERROR: LLM restart failed: {e}")


# ---------------------------------------------------------------------------
# Core: run one multi-turn conversation
# ---------------------------------------------------------------------------

def run_conversation(opener: str, n_turns: int, csv_writer, stats: dict):
    """Run a full multi-turn conversation with Aura."""

    log_raw(f"\n{'═' * 60}")
    log_raw(f"NEW CONVERSATION ({n_turns} turns)")
    log_raw(f"Opener: \"{opener[:80]}\"")
    log_raw(f"{'═' * 60}")

    messages = [
        {"role": "system", "content":
         "You are a curious, friendly person having a real conversation with an AI "
         "assistant named Aura. You're genuinely interested in what she has to say.\n\n"
         "Rules:\n"
         "- Keep your responses to 1-2 sentences MAX. This is a spoken conversation.\n"
         "- Ask follow-up questions based on what Aura actually said.\n"
         "- Be natural — react with surprise, curiosity, or thoughts of your own.\n"
         "- Sometimes share your own perspective before asking a follow-up.\n"
         "- If Aura says something wrong or confusing, gently push back.\n"
         "- Do NOT use markdown, emojis, lists, or special formatting.\n"
         "- Do NOT start with 'That's interesting' or 'Wow' every time. Vary your reactions.\n"
         "- Speak plainly, like a real person talking out loud."},
    ]

    for turn in range(n_turns):
        log_raw(f"\n  ── Turn {turn+1}/{n_turns} ──")

        # Generate what to say
        if turn == 0:
            rtx_says = opener
        else:
            rtx_says = ollama_chat(messages)
            if not rtx_says:
                log_raw("  ERROR: LLM returned empty, ending conversation")
                break

        messages.append({"role": "assistant", "content": rtx_says})
        log_raw(f"  [You]: \"{rtx_says}\"")

        # Speak it
        play_dur = speak(rtx_says)

        # Wait for Aura's response
        log_raw(f"  [Waiting for Aura...]")
        result = wait_for_aura_response(play_dur, timeout=35)

        log_raw(f"  [Whisper heard]: \"{result['whisper_heard']}\"")
        log_raw(f"  [Aura said]: \"{result['response'][:120]}\"")
        log_raw(f"  [Timing]: first={result['first_audio_ms']}ms, "
                f"total={result['total_ms']}ms, clauses={result['clauses']}")
        if result["self_echo"]:
            log_raw(f"  ⚠ SELF-ECHO detected")
        if result["error"]:
            log_raw(f"  ⚠ ERROR: {result['error']}")

        # Record stats
        stats["total_questions"] += 1
        if result["no_response"]:
            stats["no_response"] += 1
        if result["self_echo"]:
            stats["self_echoes"] += 1
        if result["error"]:
            stats["errors"] += 1
        if not result["whisper_heard"]:
            stats["whisper_misses"] += 1
        if result["first_audio_ms"] > 0:
            stats["first_audio_times"].append(result["first_audio_ms"])
        stats["total_times"].append(result["total_ms"])

        # Write CSV
        csv_writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "conversation": stats["conversations"],
            "turn": turn + 1,
            "question": rtx_says,
            "whisper_heard": result["whisper_heard"],
            "response": result["response"],
            "first_audio_ms": result["first_audio_ms"],
            "total_ms": result["total_ms"],
            "clauses": result["clauses"],
            "self_echo": result["self_echo"],
            "filler": result["filler_played"],
            "error": result["error"],
        })

        # Feed Aura's response to the LLM for follow-up
        aura_resp = result["response"]
        if result["no_response"]:
            messages.append({"role": "user", "content":
                "Aura didn't respond. Try rephrasing your question or ask something different."})
        else:
            messages.append({"role": "user", "content":
                f'Aura said: "{aura_resp}". '
                "Respond naturally and ask a follow-up question."})

        # Natural pause between turns (like a real conversation)
        pause = random.uniform(1.5, 3.0)
        time.sleep(pause)

        # If Aura failed to respond twice in a row, check services
        if turn > 0 and result["no_response"]:
            restart_services_if_needed(stats)

    stats["conversations"] += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aura Overnight QA")
    parser.add_argument("--hours", type=float, default=8.0, help="Duration in hours")
    parser.add_argument("--pause", type=int, default=45, help="Seconds between conversations")
    args = parser.parse_args()

    end_time = datetime.now() + timedelta(hours=args.hours)

    log_raw(f"{'=' * 60}")
    log_raw(f"AURA OVERNIGHT QA — {args.hours}h conversational stress test")
    log_raw(f"Run ID: {RUN_ID}")
    log_raw(f"End time: {end_time.strftime('%Y-%m-%d %H:%M')}")
    log_raw(f"CSV: {CSV_LOG}")
    log_raw(f"Report: {REPORT_FILE}")
    log_raw(f"{'=' * 60}")

    # Stats
    stats = {
        "total_questions": 0,
        "no_response": 0,
        "self_echoes": 0,
        "errors": 0,
        "first_audio_times": [],
        "total_times": [],
        "whisper_misses": 0,
        "conversations": 0,
        "restarts": 0,
        "llm_restarts": 0,
    }

    # Verify services
    restart_services_if_needed(stats)

    # CSV setup
    fieldnames = [
        "timestamp", "conversation", "turn", "question", "whisper_heard",
        "response", "first_audio_ms", "total_ms", "clauses", "self_echo",
        "filler", "error"
    ]
    csv_file = open(CSV_LOG, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Shuffle openers so we don't repeat in order
    openers = list(CONVERSATION_OPENERS)
    random.shuffle(openers)
    opener_idx = 0

    try:
        while datetime.now() < end_time:
            remaining = end_time - datetime.now()
            hrs = remaining.seconds // 3600
            mins = (remaining.seconds % 3600) // 60
            log_raw(f"\n{'━' * 60}")
            log_raw(f"CONVERSATION {stats['conversations']+1} — {hrs}h {mins}m remaining")
            log_raw(f"Stats so far: {stats['total_questions']} turns, "
                    f"{stats['no_response']} no-response, "
                    f"{stats['self_echoes']} self-echoes")
            log_raw(f"{'━' * 60}")

            # Health check + reset LLM session between conversations
            restart_services_if_needed(stats)
            reset_llm_session()

            # Pick opener
            if opener_idx >= len(openers):
                random.shuffle(openers)
                opener_idx = 0
            opener = openers[opener_idx]
            opener_idx += 1

            # Run conversation
            n_turns = random.randint(CONVO_MIN_TURNS, CONVO_MAX_TURNS)
            run_conversation(opener, n_turns, csv_writer, stats)

            csv_file.flush()

            # Pause between conversations
            pause = random.uniform(args.pause * 0.7, args.pause * 1.3)
            log_raw(f"\n  [Pausing {pause:.0f}s before next conversation...]")
            time.sleep(pause)

    except KeyboardInterrupt:
        log_raw("\nINTERRUPTED by user")
    finally:
        csv_file.close()

    generate_report(stats)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(stats: dict):
    """Generate the final weakness report."""
    log_raw(f"\n{'=' * 60}")
    log_raw("GENERATING REPORT...")

    rows = []
    try:
        with open(CSV_LOG, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        pass

    total = max(stats["total_questions"], 1)
    weaknesses = []

    # 1. Response rate
    no_resp_pct = stats["no_response"] / total * 100
    if no_resp_pct > 5:
        weaknesses.append(
            f"NO-RESPONSE RATE: {no_resp_pct:.1f}% ({stats['no_response']}/{total}). "
            f"Possible: VAD not triggering, Whisper failing, LLM timeout, or echo gate "
            f"suppressing the question.")

    # 2. Self-echo
    echo_pct = stats["self_echoes"] / total * 100
    if stats["self_echoes"] > 0:
        weaknesses.append(
            f"SELF-ECHO: {echo_pct:.1f}% ({stats['self_echoes']}/{total}). "
            f"Mic picks up Aura's own speech. Current holdoff may be too short, "
            f"or inter-clause gaps allow bleed-through.")

    # 3. Latency analysis
    if stats["first_audio_times"]:
        times = sorted(stats["first_audio_times"])
        avg_f = sum(times) / len(times)
        med_f = times[len(times) // 2]
        p95_f = times[int(len(times) * 0.95)]
        max_f = max(times)
        min_f = min(times)
        if avg_f > 1000:
            weaknesses.append(
                f"FIRST-AUDIO LATENCY: avg={avg_f:.0f}ms, p95={p95_f:.0f}ms. "
                f"Target <500ms. Check LLM response time or Piper model load.")

    # 4. Whisper accuracy
    whisper_issues = []
    for row in rows:
        q = row.get("question", "").lower()
        heard = row.get("whisper_heard", "").lower()
        if heard and q and len(q) > 10:
            q_words = set(re.findall(r'\w{3,}', q))  # words 3+ chars
            h_words = set(re.findall(r'\w{3,}', heard))
            if q_words:
                overlap = len(q_words & h_words) / len(q_words)
                if overlap < 0.35:
                    whisper_issues.append({
                        "sent": row["question"][:70],
                        "heard": heard[:70],
                        "overlap": f"{overlap:.0%}",
                    })
    if whisper_issues:
        weaknesses.append(
            f"WHISPER ACCURACY: {len(whisper_issues)}/{len(rows)} turns had <35% word match. "
            f"distil-small.en struggles with far-field + male synthetic voice.\n"
            + "\n".join(f"    Sent: \"{w['sent']}\"\n    Heard: \"{w['heard']}\" ({w['overlap']})"
                        for w in whisper_issues[:8]))

    # 5. Service stability
    if stats["restarts"] > 0 or stats["llm_restarts"] > 0:
        weaknesses.append(
            f"SERVICE INSTABILITY: Aura restarted {stats['restarts']}x, "
            f"LLM restarted {stats['llm_restarts']}x during the test.")

    # 6. Error patterns
    error_rows = [r for r in rows if r.get("error")]
    if error_rows:
        error_types = {}
        for r in error_rows:
            e = r["error"]
            error_types[e] = error_types.get(e, 0) + 1
        weaknesses.append(
            f"ERRORS: {len(error_rows)} total.\n" +
            "\n".join(f"    {e}: {c}x" for e, c in error_types.items()))

    # Build report
    report = []
    report.append("=" * 60)
    report.append("AURA OVERNIGHT QA REPORT")
    report.append(f"Run ID: {RUN_ID}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 60)

    report.append(f"\nSUMMARY:")
    report.append(f"  Total turns: {stats['total_questions']}")
    report.append(f"  Conversations: {stats['conversations']}")
    report.append(f"  Aura restarts: {stats['restarts']}")
    report.append(f"  LLM restarts: {stats['llm_restarts']}")

    report.append(f"\nRESPONSE RATE:")
    report.append(f"  Responded: {total - stats['no_response']}/{total} "
                  f"({(1 - stats['no_response']/total)*100:.1f}%)")
    report.append(f"  No response: {stats['no_response']}/{total} ({no_resp_pct:.1f}%)")

    if stats["first_audio_times"]:
        report.append(f"\nPIPER TTS LATENCY (first audio):")
        report.append(f"  Min:    {min_f}ms")
        report.append(f"  Avg:    {avg_f:.0f}ms")
        report.append(f"  Median: {med_f}ms")
        report.append(f"  P95:    {p95_f}ms")
        report.append(f"  Max:    {max_f}ms")

    report.append(f"\nSELF-ECHO: {stats['self_echoes']}/{total} ({echo_pct:.1f}%)")
    report.append(f"WHISPER MISSES: {stats['whisper_misses']} (mic heard nothing)")

    if weaknesses:
        report.append(f"\n{'─' * 60}")
        report.append("WEAKNESSES IDENTIFIED:")
        report.append(f"{'─' * 60}")
        for i, w in enumerate(weaknesses, 1):
            report.append(f"\n{i}. {w}")
    else:
        report.append(f"\nNO MAJOR WEAKNESSES DETECTED")

    report.append(f"\n{'=' * 60}")
    report.append(f"Full CSV: {CSV_LOG}")
    report.append(f"Raw log:  {RAW_LOG}")
    report.append(f"{'=' * 60}")

    report_text = "\n".join(report)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)
    print(report_text)
    log_raw(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
