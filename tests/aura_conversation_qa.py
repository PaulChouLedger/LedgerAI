#!/usr/bin/env python3
"""
Aura Conversation QA — RTX simulates a human talking to the puck for 5 minutes.

The RTX generates natural conversational prompts via Ollama (acting as "Paul"),
speaks them aloud via Piper, and polls the puck's journal for Aura's responses.
Logs detailed timing metrics (latency, speech duration, Whisper accuracy) and
produces a structured QA report with improvement recommendations.

Usage:
    python3 tests/aura_conversation_qa.py [--minutes 5]
"""

import argparse
import json
import re
import subprocess
import statistics
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PIPER_MODEL = "/tmp/piper_test_voice/en_US-lessac-medium.onnx"
PIPER_LENGTH = "1.1"
PUCK_HOST = "ledger@192.168.1.94"
PUCK_LLM_URL = "http://192.168.1.94:11434"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"
RTX_MIC_DEVICE = "plughw:1,0"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TRANSCRIPT_FILE = LOG_DIR / f"conversation_qa_{RUN_ID}.txt"
METRICS_FILE = LOG_DIR / f"conversation_qa_{RUN_ID}.json"
AUDIO_FILE = LOG_DIR / f"conversation_qa_{RUN_ID}.wav"

# ---------------------------------------------------------------------------
# Conversation topics — the RTX cycles through these to cover different
# capabilities: general knowledge, personal, creative, follow-up, etc.
# ---------------------------------------------------------------------------
CONVERSATION_PLAN = [
    {
        "topic": "greeting",
        "opener": "Hey Aura, how's it going?",
        "goal": "Test basic greeting and persona warmth",
    },
    {
        "topic": "general_knowledge",
        "instruction": "Ask Aura a simple general knowledge question, like about a famous landmark, a historical event, or a science fact. Keep it casual and conversational.",
        "goal": "Test factual recall and conciseness",
    },
    {
        "topic": "personal_opinion",
        "instruction": "Ask Aura for her personal opinion or preference on something — a favorite season, food, or type of music. Be genuinely curious.",
        "goal": "Test personality and creative expression",
    },
    {
        "topic": "follow_up",
        "instruction": "Follow up on what Aura just said. Ask a deeper question about her previous answer. Show genuine interest.",
        "goal": "Test multi-turn coherence and memory within session",
    },
    {
        "topic": "practical_help",
        "instruction": "Ask Aura for practical help — a recipe idea, a workout suggestion, or advice on something everyday. Keep it natural.",
        "goal": "Test helpfulness and structured responses",
    },
    {
        "topic": "follow_up_2",
        "instruction": "Follow up again on what Aura just said. Ask for more detail or a clarification. Keep the conversation flowing naturally.",
        "goal": "Test sustained multi-turn coherence",
    },
    {
        "topic": "creative",
        "instruction": "Ask Aura something creative — to tell a short joke, describe something poetically, or imagine a fun scenario. Keep it light.",
        "goal": "Test creative and playful responses",
    },
    {
        "topic": "emotional",
        "instruction": "Share something slightly personal or reflective with Aura — say you've had a long day, or you're thinking about something. See how she responds with empathy.",
        "goal": "Test emotional intelligence and empathy",
    },
    {
        "topic": "follow_up_3",
        "instruction": "Respond to Aura's empathetic answer. Thank her or share a bit more. Keep the emotional tone going.",
        "goal": "Test sustained emotional engagement",
    },
    {
        "topic": "closing",
        "instruction": "Wrap up the conversation naturally. Say goodbye or tell Aura you'll talk to her later. Be warm.",
        "goal": "Test graceful conversation closing",
    },
]

# ---------------------------------------------------------------------------
# The RTX persona — acts like a casual, friendly user
# ---------------------------------------------------------------------------
USER_SYSTEM_PROMPT = (
    "You are Paul, a friendly tech enthusiast having a casual conversation with "
    "your AI assistant Aura. RULES: "
    "1. Keep each message to ONE short sentence only (under 15 words). "
    "2. Ask a direct question — do NOT make statements or observations. "
    "3. Do NOT reference technical issues, errors, or processing problems. "
    "4. Ignore any mention of 'trouble processing' — just ask a new question. "
    "5. Be casual and warm. No emojis or special characters."
)


def ollama_chat(messages, timeout=60):
    """Call Ollama on the RTX for the simulated user's lines."""
    try:
        resp = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", json.dumps({
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            })],
            capture_output=True, text=True, timeout=timeout,
        )
        data = json.loads(resp.stdout)
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  [Ollama error: {e}]")
        return ""


def speak_on_rtx(text):
    """Synthesize and play text on the RTX speaker. Returns duration in seconds."""
    wav_path = "/tmp/qa_user_line.wav"
    subprocess.run(
        ["piper", "--model", PIPER_MODEL, "--length-scale", PIPER_LENGTH,
         "--output_file", wav_path],
        input=text, capture_output=True, text=True, timeout=15,
    )
    try:
        result = subprocess.run(
            ["soxi", "-D", wav_path], capture_output=True, text=True, timeout=5,
        )
        duration = float(result.stdout.strip())
    except Exception:
        duration = len(text) * 0.06
    subprocess.run(["aplay", wav_path], capture_output=True, timeout=30)
    return duration


def poll_journal_for_response(since_ts, timeout=40.0, poll_interval=2.0):
    """Poll puck journal until Aura finishes speaking all clauses, or timeout.

    Returns (said_text, whisper_heard, first_response_ts, final_response_ts).
    """
    deadline = time.time() + timeout
    time.sleep(5.0)  # Whisper needs time to process

    first_ts = None
    while time.time() < deadline:
        since_str = datetime.utcfromtimestamp(since_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            result = subprocess.run(
                ["ssh", PUCK_HOST,
                 f"sudo journalctl -u aura --no-pager --since '{since_str}' 2>/dev/null"],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            time.sleep(poll_interval)
            continue

        lines = result.stdout.strip().split("\n")
        said_parts = []
        heard_parts = []
        pipelined_lines = []
        for line in lines:
            mic_match = re.search(r'\[mic\] "(.*)"', line)
            pipe_match = re.search(r'-> "(.*)"', line)
            if mic_match:
                heard_parts.append(mic_match.group(1))
            if pipe_match:
                said_parts.append(pipe_match.group(1))
                pipelined_lines.append(line)
                if first_ts is None:
                    # Parse timestamp from journal line
                    ts_match = re.match(r'^(\w+ \d+ [\d:]+)', line)
                    if ts_match:
                        first_ts = ts_match.group(1)

        if said_parts:
            last_line = pipelined_lines[-1]
            has_pending = "[pending=" in last_line

            if has_pending:
                time.sleep(poll_interval)
                continue

            # Wait for final clause audio to finish
            dur_match = re.search(r'(\d+)ms audio', last_line)
            if dur_match:
                wait_s = int(dur_match.group(1)) / 1000.0 + 1.0
                print(f"  [Waiting {wait_s:.1f}s for final clause]")
                time.sleep(wait_s)

            return " ".join(said_parts), " ".join(heard_parts)

        time.sleep(poll_interval)

    return "[no response]", ""


def reset_puck_session():
    """Reset the puck's LLM session and clear any persona."""
    try:
        req = urllib.request.Request(
            f"{PUCK_LLM_URL}/reset-session",
            data=json.dumps({"session_id": "__all__"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

    try:
        req = urllib.request.Request(
            f"{PUCK_LLM_URL}/set-persona",
            data=json.dumps({"persona": None}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def check_puck_health():
    """Check if the puck LLM is responding."""
    try:
        resp = urllib.request.urlopen(f"{PUCK_LLM_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        return data.get("status") in ("ok", "healthy")
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=5.0)
    args = parser.parse_args()

    end_time = time.time() + args.minutes * 60
    transcript_lines = []
    turn_metrics = []

    def log(text):
        print(text)
        transcript_lines.append(text)

    log("=" * 70)
    log("AURA CONVERSATION QA TEST")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Duration: {args.minutes} minutes")
    log(f"User voice: Piper lessac-medium (RTX)")
    log(f"User LLM: Ollama {OLLAMA_MODEL} (RTX CPU)")
    log(f"Aura: Puck (Piper Olga + Qwen 1.5B)")
    log("=" * 70)
    log("")

    # Pre-flight checks
    log("[Pre-flight]")
    puck_ok = check_puck_health()
    log(f"  Puck LLM health: {'OK' if puck_ok else 'FAILED'}")
    if not puck_ok:
        log("  ERROR: Puck LLM not responding. Aborting.")
        return

    reset_puck_session()
    log("  Puck session reset")
    log("")

    # Silent countdown
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    # Start audio recording
    rec_proc = subprocess.Popen(
        ["arecord", "-D", RTX_MIC_DEVICE, "-f", "S16_LE", "-c", "1",
         "-r", "16000", str(AUDIO_FILE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log(f"[Recording: {AUDIO_FILE.name}]")
    log("")

    # Build Ollama conversation context
    ollama_messages = [
        {"role": "system", "content": USER_SYSTEM_PROMPT},
    ]

    turn = 0
    topic_idx = 0

    while time.time() < end_time and topic_idx < len(CONVERSATION_PLAN):
        topic = CONVERSATION_PLAN[topic_idx]
        turn += 1
        turn_start = time.time()

        log(f"── Turn {turn} ({topic['topic']}) ──")
        log(f"  [Goal: {topic['goal']}]")

        # Generate the user's line
        if "opener" in topic:
            user_line = topic["opener"]
        else:
            instruction = topic["instruction"]
            # Only include Aura's last reply if it was meaningful (not an error loop)
            last_said = turn_metrics[-1].get("aura_said", "") if turn_metrics else ""
            if last_said and "trouble processing" not in last_said and last_said != "[no response]":
                context = f"Aura just said: \"{last_said}\"\n\n{instruction}"
            else:
                context = instruction
            ollama_messages.append({"role": "user", "content": context})

            llm_start = time.time()
            user_line = ollama_chat(ollama_messages, timeout=90)
            llm_elapsed = time.time() - llm_start
            log(f"  [RTX LLM: {llm_elapsed:.1f}s]")

            if not user_line:
                log("  [RTX LLM failed — skipping turn]")
                topic_idx += 1
                continue

            ollama_messages.append({"role": "assistant", "content": user_line})

        log(f"  PAUL: {user_line}")
        log("")

        # Speak the line
        speak_start = time.time()
        speak_dur = speak_on_rtx(user_line)
        speech_done_ts = time.time()

        # Poll for Aura's response
        aura_said, whisper_heard = poll_journal_for_response(speak_start, timeout=40.0)
        response_done_ts = time.time()

        # Calculate latency (from end of RTX speech to first Aura audio)
        total_response_time = response_done_ts - speech_done_ts

        log(f"  AURA: {aura_said}")
        if whisper_heard:
            log(f"  [Whisper heard: {whisper_heard}]")
        log(f"  [Response time: {total_response_time:.1f}s | Speech: {speak_dur:.1f}s]")
        log("")

        # ---- Live transcript table ----
        print("  ┌─────────────────────────────────────────────────────────────")
        print(f"  │ RTX SAID:      {user_line[:70]}")
        print(f"  │ PUCK HEARD:    {whisper_heard[:70] if whisper_heard else '(nothing)'}")
        print(f"  │ AURA REPLIED:  {aura_said[:70]}")
        print(f"  │ LATENCY:       {total_response_time:.1f}s")
        print("  └─────────────────────────────────────────────────────────────")
        print("")

        # Collect metrics
        metric = {
            "turn": turn,
            "topic": topic["topic"],
            "goal": topic["goal"],
            "user_said": user_line,
            "aura_said": aura_said,
            "whisper_heard": whisper_heard,
            "speak_duration_s": round(speak_dur, 2),
            "response_time_s": round(total_response_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        # Check for transcription accuracy
        if whisper_heard and aura_said != "[no response]":
            # Simple word overlap ratio
            user_words = set(user_line.lower().split())
            whisper_words = set(whisper_heard.lower().split())
            if user_words:
                overlap = len(user_words & whisper_words) / len(user_words)
                metric["whisper_word_overlap"] = round(overlap, 2)

        turn_metrics.append(metric)
        topic_idx += 1

    # Stop recording
    rec_proc.terminate()
    try:
        rec_proc.wait(timeout=3)
    except Exception:
        rec_proc.kill()

    # ---------------------------------------------------------------------------
    # Analysis & Recommendations
    # ---------------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("QA ANALYSIS")
    log("=" * 70)
    log("")

    total_turns = len(turn_metrics)
    successful = [m for m in turn_metrics if m["aura_said"] != "[no response]"]
    failed = [m for m in turn_metrics if m["aura_said"] == "[no response]"]

    log(f"Total turns: {total_turns}")
    log(f"Successful responses: {len(successful)}/{total_turns}")
    log(f"No response: {len(failed)}/{total_turns}")
    log("")

    if successful:
        response_times = [m["response_time_s"] for m in successful]
        log("Response Time (from end of user speech to Aura done speaking):")
        log(f"  Mean:   {statistics.mean(response_times):.1f}s")
        log(f"  Median: {statistics.median(response_times):.1f}s")
        log(f"  Min:    {min(response_times):.1f}s")
        log(f"  Max:    {max(response_times):.1f}s")
        if len(response_times) > 1:
            log(f"  StdDev: {statistics.stdev(response_times):.1f}s")
        log("")

    # Whisper accuracy
    overlaps = [m["whisper_word_overlap"] for m in turn_metrics
                if "whisper_word_overlap" in m]
    if overlaps:
        log("Whisper Transcription Accuracy (word overlap with user speech):")
        log(f"  Mean overlap: {statistics.mean(overlaps):.0%}")
        log(f"  Min:          {min(overlaps):.0%}")
        log(f"  Max:          {max(overlaps):.0%}")
        log("")

    # Per-turn summary
    log("Per-Turn Summary:")
    for m in turn_metrics:
        status = "OK" if m["aura_said"] != "[no response]" else "FAIL"
        resp_t = f"{m['response_time_s']:.1f}s"
        aura_short = m["aura_said"][:60] + ("..." if len(m["aura_said"]) > 60 else "")
        log(f"  T{m['turn']:02d} [{status}] {resp_t:>6s} | {m['topic']:<18s} | {aura_short}")
    log("")

    # Aura response length analysis
    if successful:
        response_lengths = [len(m["aura_said"].split()) for m in successful]
        log("Aura Response Length (words):")
        log(f"  Mean: {statistics.mean(response_lengths):.0f}")
        log(f"  Min:  {min(response_lengths)}")
        log(f"  Max:  {max(response_lengths)}")
        log("")

    # ---------------------------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------------------------
    log("=" * 70)
    log("RECOMMENDATIONS")
    log("=" * 70)
    log("")

    recs = []

    # Response time
    if successful:
        mean_rt = statistics.mean(response_times)
        if mean_rt > 15:
            recs.append(
                "CRITICAL — Response latency is very high (mean {:.0f}s). "
                "Users will disengage. Investigate: Whisper transcription time, "
                "LLM inference time, TTS synthesis time. Profile each stage "
                "via journal timestamps.".format(mean_rt)
            )
        elif mean_rt > 10:
            recs.append(
                "HIGH — Response latency (mean {:.0f}s) is noticeable. "
                "Consider: (1) switching to distil-small.en Whisper for ~2x "
                "speedup, (2) reducing LLM max_tokens, (3) shorter system "
                "prompt to reduce prefill time.".format(mean_rt)
            )
        elif mean_rt > 6:
            recs.append(
                "MODERATE — Response time (mean {:.0f}s) is acceptable but "
                "could be tighter. Pipelined TTS is helping. Consider streaming "
                "Whisper (chunked) for faster first-token.".format(mean_rt)
            )
        else:
            recs.append(
                "GOOD — Response time (mean {:.0f}s) feels conversational. "
                "No action needed.".format(mean_rt)
            )

    # Failures
    if len(failed) > 0:
        fail_rate = len(failed) / total_turns
        if fail_rate > 0.3:
            recs.append(
                "CRITICAL — {:.0%} of turns got no response. Check: "
                "(1) Is the puck hearing the RTX? Increase speaker volume. "
                "(2) VAD too aggressive? Lower END_SILENCE_MS. "
                "(3) Whisper rejecting short utterances? Check spectral "
                "filters.".format(fail_rate)
            )
        elif fail_rate > 0:
            recs.append(
                "MINOR — {} turn(s) got no response. May be environmental "
                "noise or edge-case VAD behavior. Monitor but not urgent."
                .format(len(failed))
            )

    # Whisper accuracy
    if overlaps:
        mean_overlap = statistics.mean(overlaps)
        if mean_overlap < 0.3:
            recs.append(
                "HIGH — Whisper word overlap is low ({:.0%}). The puck may "
                "not be hearing the RTX clearly. Check: (1) speaker-to-mic "
                "distance, (2) echo cancellation in XVF3800, (3) try "
                "distil-large-v3 for better accuracy.".format(mean_overlap)
            )

    # Response length
    if successful:
        mean_len = statistics.mean(response_lengths)
        if mean_len > 80:
            recs.append(
                "MODERATE — Aura's responses are verbose (mean {:.0f} words). "
                "Consider tightening the system prompt: 'Keep responses to "
                "2-3 sentences unless asked for detail.'".format(mean_len)
            )
        elif mean_len < 5:
            recs.append(
                "MODERATE — Aura's responses are very terse (mean {:.0f} words). "
                "She may sound robotic. Consider warming up the system prompt "
                "or increasing temperature slightly.".format(mean_len)
            )

    # Multi-turn coherence (check if follow-up topics got relevant responses)
    followup_turns = [m for m in turn_metrics
                      if "follow_up" in m["topic"] and m["aura_said"] != "[no response]"]
    if followup_turns:
        recs.append(
            "INFO — {} follow-up turn(s) completed. Review transcript to "
            "verify Aura referenced her previous answers (session memory "
            "coherence).".format(len(followup_turns))
        )

    if not recs:
        recs.append("No specific issues detected. System performing well.")

    for i, rec in enumerate(recs, 1):
        log(f"  {i}. {rec}")
        log("")

    log("=" * 70)
    log("END OF QA REPORT")
    log("=" * 70)

    # Save files
    TRANSCRIPT_FILE.write_text("\n".join(transcript_lines))
    print(f"\nTranscript: {TRANSCRIPT_FILE}")

    # Save structured metrics
    report = {
        "run_id": RUN_ID,
        "duration_minutes": args.minutes,
        "total_turns": total_turns,
        "successful": len(successful),
        "failed": len(failed),
        "mean_response_time_s": round(statistics.mean(response_times), 2) if successful else None,
        "turns": turn_metrics,
        "recommendations": recs,
    }
    METRICS_FILE.write_text(json.dumps(report, indent=2))
    print(f"Metrics:    {METRICS_FILE}")
    print(f"Audio:      {AUDIO_FILE}")

    try:
        result = subprocess.run(
            ["soxi", "-D", str(AUDIO_FILE)], capture_output=True, text=True, timeout=5,
        )
        print(f"Duration:   {float(result.stdout.strip()):.0f}s")
    except Exception:
        pass


if __name__ == "__main__":
    main()
