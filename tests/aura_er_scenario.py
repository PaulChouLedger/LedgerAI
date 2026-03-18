#!/usr/bin/env python3
"""
Aura ER Doctor Scenario — RTX acts as an ER doctor, Aura is the patient.

Injects a patient persona into the puck's LLM via /set-persona API so Aura
answers in character. Polls puck journal for responses instead of fixed waits.
Records full audio from RTX ReSpeaker mic.

Usage:
    python3 tests/aura_er_scenario.py [--minutes 2]
"""

import argparse
import json
import random
import re
import subprocess
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
OLLAMA_MODEL = "qwen2.5:3b-instruct-q5_1"
OLLAMA_URL = "http://localhost:11434/api/chat"
RTX_MIC_DEVICE = "plughw:1,0"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
TRANSCRIPT_FILE = LOG_DIR / f"er_scenario_{RUN_ID}.txt"
AUDIO_FILE = LOG_DIR / f"er_scenario_{RUN_ID}.wav"

# Doctor filler phrases (spoken BEFORE the question, not during wait)
DOCTOR_FILLERS = [
    "Okay.", "I see.", "Mm hmm.", "Alright.", "Got it.",
]

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
PATIENT_PERSONA = (
    "You are Aura, a 34-year-old woman in the emergency room. "
    "You have sharp pain in your lower right abdomen that started 6 hours ago "
    "near your belly button and moved to the right side. You feel nauseous, "
    "you vomited once, and you have a slight fever. The pain gets worse when "
    "you walk or cough. You are scared and in pain. "
    "RULES: Answer the doctor's questions honestly as this patient. "
    "Keep answers to 1-2 short sentences. Do NOT diagnose yourself. "
    "Do NOT say what you think you have. Do NOT give medical advice. "
    "Do NOT say things like 'you will recover' or 'sounds like appendicitis'. "
    "You are the PATIENT, not a doctor. Just describe your symptoms when asked."
)

ER_DOCTOR_PROMPT = (
    "You are Dr. Rafael, an ER physician doing a patient intake. "
    "The patient's name is Aura. She has abdominal pain. "
    "Ask ONE question at a time. Keep it under 2 sentences. "
    "Be warm but professional. Address her by name sometimes. "
    "Start by introducing yourself and asking what brought her in."
)

DIAGNOSIS_LINE = (
    "Aura, based on everything you've told me — the pain that started near your "
    "belly button and moved to your lower right side, the nausea, vomiting, and "
    "low-grade fever — this is very consistent with appendicitis. We're going to "
    "run some blood work and get a C T scan right away to confirm, but I want you "
    "to know we're going to take very good care of you."
)


def set_puck_persona(persona):
    try:
        req = urllib.request.Request(
            "http://192.168.1.94:11434/set-persona",
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
            "http://192.168.1.94:11434/set-persona",
            data=json.dumps({"persona": None}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def ollama_chat(messages):
    try:
        resp = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", json.dumps({
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            })],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(resp.stdout)
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  [Ollama error: {e}]")
        return ""


def speak_on_rtx(text):
    wav_path = "/tmp/er_doctor_line.wav"
    subprocess.run(
        ["piper", "--model", PIPER_MODEL, "--length-scale", PIPER_LENGTH,
         "--output_file", wav_path],
        input=text, capture_output=True, text=True, timeout=15
    )
    try:
        result = subprocess.run(
            ["soxi", "-D", wav_path], capture_output=True, text=True, timeout=5
        )
        duration = float(result.stdout.strip())
    except Exception:
        duration = len(text) * 0.06
    subprocess.run(["aplay", wav_path], capture_output=True, timeout=30)
    return duration


def poll_journal_for_response(since_ts, timeout=35.0, poll_interval=2.0):
    """Poll puck journal until Aura finishes speaking ALL clauses, or timeout.

    Aura speaks in multiple Pipelined entries. Lines with [pending=N] mean
    more clauses are queued. We wait until the final clause (no pending tag)
    finishes playing before returning.
    """
    deadline = time.time() + timeout
    # Give the puck at least 5s before first poll (Whisper needs time)
    time.sleep(5.0)

    while time.time() < deadline:
        since_str = datetime.utcfromtimestamp(since_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
        result = subprocess.run(
            ["ssh", PUCK_HOST,
             f"sudo journalctl -u aura --no-pager --since '{since_str}' 2>/dev/null"],
            capture_output=True, text=True, timeout=10
        )

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
            # Check if last Pipelined line still has pending clauses
            last_line = pipelined_lines[-1]
            has_pending = "[pending=" in last_line

            if has_pending:
                # More clauses coming — keep polling
                time.sleep(poll_interval)
                continue

            # All clauses done — parse audio duration of LAST clause and wait
            dur_match = re.search(r'(\d+)ms audio', last_line)
            if dur_match:
                wait_s = int(dur_match.group(1)) / 1000.0 + 1.0
                print(f"  [Waiting {wait_s:.1f}s for final clause to finish]")
                time.sleep(wait_s)

            return " ".join(said_parts), " ".join(heard_parts)

        time.sleep(poll_interval)

    return "[no response]", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=2.0)
    args = parser.parse_args()

    # Reserve 25s at end for diagnosis
    end_time = time.time() + args.minutes * 60 - 25
    transcript_lines = []

    def log(text):
        print(text)
        transcript_lines.append(text)

    log("=" * 70)
    log("AURA ER DOCTOR SCENARIO — APPENDICITIS")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Duration: {args.minutes} minutes")
    log(f"Doctor: Dr. Rafael (Ollama {OLLAMA_MODEL} + Piper lessac)")
    log(f"Patient: Aura (Puck — Piper Olga epoch 1250)")
    log("=" * 70)
    log("")

    # Reset puck LLM and inject patient persona
    subprocess.run(
        ["ssh", PUCK_HOST,
         'curl -s -X POST http://localhost:11434/reset-session '
         '-H "Content-Type: application/json" '
         '-d \'{"session_id": "__all__"}\''],
        capture_output=True, text=True, timeout=10
    )
    time.sleep(2)

    log("  [Injecting patient persona...]")
    set_puck_persona(PATIENT_PERSONA)
    log("")

    # Silent countdown
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    # Start audio recording
    rec_proc = subprocess.Popen(
        ["arecord", "-D", RTX_MIC_DEVICE, "-f", "S16_LE", "-c", "1",
         "-r", "16000", str(AUDIO_FILE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log(f"  [Recording: {AUDIO_FILE.name}]")
    log("")

    # Doctor opening
    messages = [
        {"role": "system", "content": ER_DOCTOR_PROMPT},
        {"role": "user", "content": "Begin the patient intake."},
    ]
    doctor_line = ollama_chat(messages)
    if not doctor_line:
        doctor_line = ("Hi Aura, I'm Dr. Rafael. Can you tell me what brought you "
                       "into the emergency room today?")
    messages.append({"role": "assistant", "content": doctor_line})

    turn = 1

    while time.time() < end_time:
        log(f"── Turn {turn} ──")
        log(f"  DR. RAFAEL: {doctor_line}")
        log("")

        # Play doctor's question
        before_ts = time.time()
        speak_on_rtx(doctor_line)

        # Poll for Aura's response (no fixed wait — returns as soon as she responds)
        aura_said, whisper_heard = poll_journal_for_response(before_ts, timeout=30.0)

        log(f"  AURA: {aura_said}")
        if whisper_heard:
            log(f"  [Whisper heard: {whisper_heard}]")
        log("")

        # Brief acknowledgment filler before next question
        if turn > 1 and aura_said != "[no response]":
            filler = random.choice(DOCTOR_FILLERS)
            speak_on_rtx(filler)

        patient_response = aura_said if aura_said != "[no response]" else "The patient groans but doesn't answer clearly."
        messages.append({"role": "user", "content": f"[Patient says]: {patient_response}"})

        doctor_line = ollama_chat(messages)
        if not doctor_line:
            break
        messages.append({"role": "assistant", "content": doctor_line})

        turn += 1

    # Final diagnosis
    log(f"── Diagnosis ──")
    log(f"  DR. RAFAEL: {DIAGNOSIS_LINE}")
    log("")
    speak_on_rtx(DIAGNOSIS_LINE)

    # Wait a beat for dramatic effect
    time.sleep(2.0)

    # Stop recording
    rec_proc.terminate()
    try:
        rec_proc.wait(timeout=3)
    except Exception:
        rec_proc.kill()

    # Restore normal persona
    clear_puck_persona()

    log("")
    log("=" * 70)
    log("END OF ER SCENARIO")
    log(f"Total turns: {turn}")
    log("=" * 70)

    TRANSCRIPT_FILE.write_text("\n".join(transcript_lines))
    print(f"\nTranscript saved: {TRANSCRIPT_FILE}")
    print(f"Audio saved: {AUDIO_FILE}")
    try:
        result = subprocess.run(
            ["soxi", "-D", str(AUDIO_FILE)], capture_output=True, text=True, timeout=5
        )
        print(f"Audio duration: {float(result.stdout.strip()):.0f}s")
    except Exception:
        pass


if __name__ == "__main__":
    main()
