#!/usr/bin/env python3
"""
Aura Political Debate — RTX is a Democrat, Puck is a Republican.

RTX generates Democrat arguments via Ollama, speaks them via Piper.
Puck's LLM gets a Republican persona injected via /set-persona API.
Polls puck journal for responses. Logs full transcript with live table.

Usage:
    python3 tests/aura_debate_scenario.py [--minutes 5]
"""

import argparse
import json
import random
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
PIPER_VOLUME = "1.05"  # 5% louder
PUCK_HOST = "ledger@192.168.1.94"
PUCK_LLM_URL = "http://192.168.1.94:11434"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"
RTX_MIC_DEVICE = "plughw:1,0"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TRANSCRIPT_FILE = LOG_DIR / f"debate_{RUN_ID}.txt"
AUDIO_FILE = LOG_DIR / f"debate_{RUN_ID}.wav"

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
REPUBLICAN_PERSONA = (
    "You are conservative. You disagree with the user. One short sentence only."
)

DEMOCRAT_SYSTEM_PROMPT = (
    "You are Alex, an aggressive Democrat debating a Republican named Aura. "
    "You are fired up about Trump, MAGA, billionaire tax cuts, Project 2025, "
    "DOGE gutting agencies, tariffs crashing the economy, and attacks on democracy. "
    "RULES: "
    "1. ONE sentence only, under 20 words. Be sharp and cutting. "
    "2. Call out hypocrisy. Mock MAGA talking points. Be ruthless. "
    "3. Reference real 2026 events: tariff recession, DOGE layoffs, TikTok ban, deportation raids. "
    "4. Do NOT be polite. This is a bare-knuckle political fight."
)

# Debate topics to cycle through
DEBATE_TOPICS = [
    "Open hot: attack Trump's tariffs for crashing the stock market and raising prices on working families.",
    "Attack DOGE for firing federal workers and gutting the VA, FDA, and CDC.",
    "Go after the billionaire tax cuts — Elon pays less tax than a nurse.",
    "Hit deportation raids — families ripped apart, economy losing workers.",
    "Attack Project 2025 — they want to ban abortion nationwide and gut public schools.",
    "Mock trickle-down — Kansas tried it, went bankrupt. Now they want it nationwide.",
    "Go after the TikTok ban — free speech for me but not for thee.",
    "Attack voter suppression — closing polling places, purging voter rolls.",
    "Hit climate denial — Miami is flooding, California is burning, and they say it's fake.",
    "Closing: call MAGA the biggest threat to democracy since the Confederacy.",
]

# RTX interjections — yelled over Aura while she's talking
DEMOCRAT_FILLERS = [
    "Oh, come on!",
    "That is a lie!",
    "Are you serious right now?",
    "That's ridiculous!",
    "Nobody believes that!",
    "Oh please!",
    "Wrong! Wrong!",
    "You can't be serious!",
    "Give me a break!",
    "That's complete nonsense!",
]


def set_puck_persona(persona):
    try:
        req = urllib.request.Request(
            f"{PUCK_LLM_URL}/set-persona",
            data=json.dumps({"persona": persona}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        body = resp.read().decode()
        print(f"  [Persona set: {body.strip()}]")
        return "ok" in body
    except Exception as e:
        print(f"  [Persona error: {e}]")
        return False


def clear_puck_persona():
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


def reset_puck_session():
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


def ollama_chat(messages, timeout=60):
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
    wav_path = "/tmp/debate_line.wav"
    boosted_path = "/tmp/debate_line_loud.wav"
    subprocess.run(
        ["piper", "--model", PIPER_MODEL, "--length-scale", PIPER_LENGTH,
         "--volume", PIPER_VOLUME, "--output_file", wav_path],
        input=text, capture_output=True, text=True, timeout=15,
    )
    # Moderate volume boost
    subprocess.run(
        ["sox", wav_path, boosted_path, "gain", "3", "norm", "-3"],
        capture_output=True, timeout=10,
    )
    try:
        result = subprocess.run(
            ["soxi", "-D", boosted_path], capture_output=True, text=True, timeout=5,
        )
        duration = float(result.stdout.strip())
    except Exception:
        duration = len(text) * 0.06
    subprocess.run(["aplay", boosted_path], capture_output=True, timeout=30)
    return duration


def poll_journal_for_response(since_ts, timeout=40.0, poll_interval=2.0):
    deadline = time.time() + timeout
    time.sleep(5.0)

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

        if said_parts:
            last_line = pipelined_lines[-1]
            if "[pending=" in last_line:
                time.sleep(poll_interval)
                continue

            dur_match = re.search(r'(\d+)ms audio', last_line)
            if dur_match:
                wait_s = int(dur_match.group(1)) / 1000.0 + 1.0
                print(f"  [Waiting {wait_s:.1f}s for final clause]")
                time.sleep(wait_s)

            return " ".join(said_parts), " ".join(heard_parts)

        time.sleep(poll_interval)

    return "[no response]", ""


def poll_journal_for_response_with_fillers(since_ts, timeout=40.0, poll_interval=2.0,
                                            speak_fn=None, fillers=None,
                                            first_filler_after=3.0, filler_interval=8.0):
    """Poll for Aura's response, speaking fillers during dead air."""
    deadline = time.time() + timeout
    wait_start = time.time()
    last_filler_time = wait_start
    filler_spoken = False
    filler_list = list(fillers or [])
    random.shuffle(filler_list)
    filler_idx = 0

    # Short initial wait (reduced from 5s to let fillers happen sooner)
    time.sleep(2.0)

    while time.time() < deadline:
        elapsed = time.time() - wait_start

        # Speak a filler if enough time has passed and Aura hasn't responded yet
        if (speak_fn and filler_list and filler_idx < len(filler_list)
                and elapsed > first_filler_after
                and time.time() - last_filler_time > filler_interval):
            speak_fn(filler_list[filler_idx])
            filler_idx += 1
            last_filler_time = time.time()
            filler_spoken = True

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

        if said_parts:
            last_line = pipelined_lines[-1]
            if "[pending=" in last_line:
                time.sleep(poll_interval)
                continue

            dur_match = re.search(r'(\d+)ms audio', last_line)
            if dur_match:
                wait_s = int(dur_match.group(1)) / 1000.0 + 1.0
                print(f"  [Waiting {wait_s:.1f}s for final clause]")
                time.sleep(wait_s)

            return " ".join(said_parts), " ".join(heard_parts)

        time.sleep(poll_interval)

    return "[no response]", ""


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
    log("POLITICAL DEBATE: DEMOCRAT vs REPUBLICAN")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Duration: {args.minutes} minutes")
    log(f"Democrat: Alex (RTX — Ollama {OLLAMA_MODEL} + Piper lessac)")
    log(f"Republican: Aura (Puck — Piper Olga + Qwen 3B)")
    log("=" * 70)
    log("")

    # Reset and inject persona
    reset_puck_session()
    time.sleep(1)
    log("  [Injecting Republican persona into Aura...]")
    set_puck_persona(REPUBLICAN_PERSONA)
    log("")

    # Countdown
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    # Start recording
    rec_proc = subprocess.Popen(
        ["arecord", "-D", RTX_MIC_DEVICE, "-f", "S16_LE", "-c", "1",
         "-r", "16000", str(AUDIO_FILE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log(f"  [Recording: {AUDIO_FILE.name}]")
    log("")

    # Build Ollama context
    ollama_messages = [
        {"role": "system", "content": DEMOCRAT_SYSTEM_PROMPT},
    ]

    turn = 0
    topic_idx = 0

    while time.time() < end_time and topic_idx < len(DEBATE_TOPICS):
        turn += 1
        topic = DEBATE_TOPICS[topic_idx]

        log(f"── Round {turn} ──")

        # Build context for Ollama
        last_said = turn_metrics[-1].get("aura_said", "") if turn_metrics else ""
        if last_said and last_said != "[no response]" and "trouble processing" not in last_said:
            context = f"Aura (Republican) just said: \"{last_said}\"\n\n{topic}"
        else:
            context = topic

        ollama_messages.append({"role": "user", "content": context})
        llm_start = time.time()
        democrat_line = ollama_chat(ollama_messages, timeout=60)
        llm_elapsed = time.time() - llm_start

        if not democrat_line:
            log("  [RTX LLM failed — skipping]")
            topic_idx += 1
            continue

        ollama_messages.append({"role": "assistant", "content": democrat_line})

        log(f"  ALEX (D): {democrat_line}")
        log(f"  [RTX LLM: {llm_elapsed:.1f}s]")
        log("")

        # Speak it
        speak_start = time.time()
        speak_dur = speak_on_rtx(democrat_line)
        speech_done_ts = time.time()

        # Poll for Aura's response — no fillers (they get picked up by puck mic and flood her)
        aura_said, whisper_heard = poll_journal_for_response(
            speak_start, timeout=40.0,
        )
        response_done_ts = time.time()
        total_response_time = response_done_ts - speech_done_ts

        log(f"  AURA (R): {aura_said}")
        if whisper_heard:
            log(f"  [Whisper heard: {whisper_heard}]")
        log(f"  [Response: {total_response_time:.1f}s | Speech: {speak_dur:.1f}s]")
        log("")

        # Live transcript table
        print("  ┌─────────────────────────────────────────────────────────────")
        print(f"  │ DEMOCRAT SAID:   {democrat_line[:65]}")
        print(f"  │ PUCK HEARD:     {whisper_heard[:65] if whisper_heard else '(nothing)'}")
        print(f"  │ REPUBLICAN:     {aura_said[:65]}")
        print(f"  │ LATENCY:        {total_response_time:.1f}s")
        print("  └─────────────────────────────────────────────────────────────")
        print("")

        turn_metrics.append({
            "turn": turn,
            "topic": topic,
            "democrat_said": democrat_line,
            "aura_said": aura_said,
            "whisper_heard": whisper_heard,
            "speak_duration_s": round(speak_dur, 2),
            "response_time_s": round(total_response_time, 2),
        })

        topic_idx += 1

    # Stop recording
    rec_proc.terminate()
    try:
        rec_proc.wait(timeout=3)
    except Exception:
        rec_proc.kill()

    # Restore normal persona
    clear_puck_persona()

    # Summary
    log("")
    log("=" * 70)
    log("DEBATE SUMMARY")
    log("=" * 70)

    successful = [m for m in turn_metrics if m["aura_said"] != "[no response]"]
    failed = [m for m in turn_metrics if m["aura_said"] == "[no response]"]

    log(f"Total rounds: {len(turn_metrics)}")
    log(f"Successful exchanges: {len(successful)}/{len(turn_metrics)}")

    if successful:
        rts = [m["response_time_s"] for m in successful]
        log(f"Response time — mean: {statistics.mean(rts):.1f}s, "
            f"median: {statistics.median(rts):.1f}s, "
            f"min: {min(rts):.1f}s, max: {max(rts):.1f}s")

    log("")
    log("Full Exchange Log:")
    for m in turn_metrics:
        status = "OK" if m["aura_said"] != "[no response]" else "FAIL"
        log(f"  R{m['turn']:02d} [{status}] {m['response_time_s']:>5.1f}s")
        log(f"       D: {m['democrat_said'][:70]}")
        log(f"       R: {m['aura_said'][:70]}")
        log("")

    log("=" * 70)
    log("END OF DEBATE")
    log("=" * 70)

    TRANSCRIPT_FILE.write_text("\n".join(transcript_lines))
    print(f"\nTranscript: {TRANSCRIPT_FILE}")
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
