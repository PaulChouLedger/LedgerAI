#!/usr/bin/env python3
"""
Aura Speech Interaction Benchmark — Scores the puck 0-100.

Runs a series of calibrated tests against the puck measuring:
  1. Whisper Transcription Accuracy (25 pts)
  2. End-to-End Response Latency    (25 pts)
  3. LLM Response Coherence         (20 pts)
  4. TTS Pipeline Speed             (15 pts)
  5. Robustness / Error Handling     (15 pts)

Speaks test phrases via Piper on RTX, polls puck journal for results,
computes sub-scores, and prints a final scorecard.

Usage:
    python3 tests/aura_speech_benchmark.py
"""

import json
import re
import subprocess
import statistics
import time
import urllib.request
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PIPER_MODEL = "/tmp/piper_test_voice/en_US-lessac-medium.onnx"
PIPER_LENGTH = "1.1"
PUCK_HOST = "ledger@192.168.1.94"
PUCK_LLM_URL = "http://192.168.1.94:11434"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILE = LOG_DIR / f"benchmark_{RUN_ID}.txt"

# ---------------------------------------------------------------------------
# Test battery — carefully designed phrases
# ---------------------------------------------------------------------------

# Transcription accuracy: short, clear phrases with known words
TRANSCRIPTION_TESTS = [
    "Hello, how are you today?",
    "What is the weather like in San Francisco?",
    "Tell me about artificial intelligence.",
    "My name is Paul and I live in London.",
    "Can you set a timer for five minutes?",
    "What is the capital of France?",
    "I need to schedule a meeting for tomorrow at three.",
    "How do you make a good cup of coffee?",
]

# Latency tests: simple questions that should get fast responses
LATENCY_TESTS = [
    "What color is the sky?",
    "How many days are in a week?",
    "What is two plus two?",
    "Say hello.",
    "What year is it?",
]

# Coherence tests: questions with verifiable correct answers
COHERENCE_TESTS = [
    ("What is the capital of Japan?", ["tokyo"]),
    ("How many legs does a dog have?", ["four", "4"]),
    ("What planet do we live on?", ["earth"]),
    ("Is water wet?", ["yes", "wet"]),
    ("What language do people speak in Brazil?", ["portuguese"]),
]

# Robustness tests: edge cases — mumbling, noise words, long input, rapid-fire
ROBUSTNESS_TESTS = [
    ("Um, uh, so like, what do you think about, you know, stuff?", "vague"),
    ("Supercalifragilisticexpialidocious.", "unusual_word"),
    ("Hi.", "ultra_short"),
    ("Can you tell me a very long and detailed story about a knight who goes on a quest to find a magical sword that has been lost for a thousand years in a dark and dangerous forest?", "long_input"),
    ("What? Huh? Wait, never mind. Actually, yes. What time is it?", "confused"),
]


def speak_on_rtx(text):
    """Synthesize and play text on RTX speaker. Returns duration in seconds."""
    wav_path = "/tmp/bench_line.wav"
    boosted_path = "/tmp/bench_line_loud.wav"
    subprocess.run(
        ["piper", "--model", PIPER_MODEL, "--length-scale", PIPER_LENGTH,
         "--output_file", wav_path],
        input=text, capture_output=True, text=True, timeout=15,
    )
    # Loud enough for puck XVF3800 to reliably capture
    subprocess.run(
        ["sox", wav_path, boosted_path, "gain", "10", "norm", "-1"],
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


def poll_journal(since_ts, timeout=30.0, poll_interval=2.0):
    """Poll puck journal for Whisper transcription and Aura response.

    Returns (aura_said, whisper_heard, first_chunk_ms, total_audio_ms).
    """
    deadline = time.time() + timeout
    time.sleep(4.0)

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
        first_chunk_ms = None
        total_audio_ms = None

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

            # Extract timing from pipelined log: "Pipelined: 9121ms total, 8079ms audio, first=908ms"
            for pl in pipelined_lines:
                total_match = re.search(r'(\d+)ms total', pl)
                first_match = re.search(r'first=(\d+)ms', pl)
                audio_match = re.search(r'(\d+)ms audio', pl)
                if total_match and first_chunk_ms is None:
                    if first_match:
                        first_chunk_ms = int(first_match.group(1))
                if audio_match:
                    total_audio_ms = int(audio_match.group(1))

            # Wait for final clause to finish playing
            dur_match = re.search(r'(\d+)ms audio', last_line)
            if dur_match:
                wait_s = int(dur_match.group(1)) / 1000.0 + 0.5
                time.sleep(wait_s)

            return " ".join(said_parts), " ".join(heard_parts), first_chunk_ms, total_audio_ms

        time.sleep(poll_interval)

    return "[no response]", "", None, None


def word_overlap(spoken, heard):
    """Word-level overlap ratio (0.0 to 1.0)."""
    if not spoken or not heard:
        return 0.0
    spoken_words = set(re.findall(r'\w+', spoken.lower()))
    heard_words = set(re.findall(r'\w+', heard.lower()))
    if not spoken_words:
        return 0.0
    return len(spoken_words & heard_words) / len(spoken_words)


def sequence_similarity(a, b):
    """SequenceMatcher ratio (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def check_health():
    """Check puck LLM and Whisper are up."""
    try:
        resp = urllib.request.urlopen(f"{PUCK_LLM_URL}/health", timeout=5)
        llm = json.loads(resp.read().decode())
        llm_ok = llm.get("status") == "ok"
    except Exception:
        llm_ok = False

    try:
        resp = urllib.request.urlopen("http://192.168.1.94:5000/health", timeout=5)
        whisper = json.loads(resp.read().decode())
        whisper_ok = whisper.get("status") == "healthy"
    except Exception:
        whisper_ok = False

    return llm_ok, whisper_ok


def reset_puck():
    """Reset puck session and clear any persona."""
    try:
        req = urllib.request.Request(
            f"{PUCK_LLM_URL}/reset-session",
            data=json.dumps({"session_id": "__all__"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
    try:
        req = urllib.request.Request(
            f"{PUCK_LLM_URL}/set-persona",
            data=json.dumps({"persona": None}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def main():
    report = []

    def log(text):
        print(text)
        report.append(text)

    log("=" * 70)
    log("AURA SPEECH INTERACTION BENCHMARK")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Puck: {PUCK_HOST}")
    log("=" * 70)
    log("")

    # Pre-flight
    llm_ok, whisper_ok = check_health()
    log(f"  LLM:     {'OK' if llm_ok else 'DOWN'}")
    log(f"  Whisper: {'OK' if whisper_ok else 'DOWN'}")
    if not llm_ok or not whisper_ok:
        log("  ABORT: Services not healthy.")
        return
    log("")

    reset_puck()
    time.sleep(2)

    # =========================================================================
    # TEST 1: Transcription Accuracy (25 points)
    # =========================================================================
    log("━" * 70)
    log("TEST 1: WHISPER TRANSCRIPTION ACCURACY (25 pts)")
    log("━" * 70)

    overlap_scores = []
    similarity_scores = []

    for i, phrase in enumerate(TRANSCRIPTION_TESTS):
        log(f"  [{i+1}/{len(TRANSCRIPTION_TESTS)}] Speaking: \"{phrase}\"")
        since_ts = time.time()
        speak_on_rtx(phrase)
        _, whisper_heard, _, _ = poll_journal(since_ts, timeout=25.0)

        overlap = word_overlap(phrase, whisper_heard)
        similarity = sequence_similarity(phrase, whisper_heard)
        overlap_scores.append(overlap)
        similarity_scores.append(similarity)

        status = "GOOD" if overlap > 0.7 else "FAIR" if overlap > 0.4 else "POOR"
        log(f"           Heard: \"{whisper_heard}\"")
        log(f"           Overlap: {overlap:.0%} | Similarity: {similarity:.0%} [{status}]")
        log("")
        time.sleep(8)  # Wait for Aura to finish speaking before next phrase

    avg_overlap = statistics.mean(overlap_scores) if overlap_scores else 0
    avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0
    # Combined score: 60% word overlap + 40% sequence similarity
    transcription_raw = avg_overlap * 0.6 + avg_similarity * 0.4
    transcription_score = round(transcription_raw * 25, 1)

    log(f"  RESULT: Avg overlap={avg_overlap:.0%}, Avg similarity={avg_similarity:.0%}")
    log(f"  SCORE: {transcription_score}/25")
    log("")

    reset_puck()
    time.sleep(2)

    # =========================================================================
    # TEST 2: End-to-End Response Latency (25 points)
    # =========================================================================
    log("━" * 70)
    log("TEST 2: END-TO-END RESPONSE LATENCY (25 pts)")
    log("━" * 70)

    e2e_latencies = []
    first_chunk_times = []

    for i, phrase in enumerate(LATENCY_TESTS):
        log(f"  [{i+1}/{len(LATENCY_TESTS)}] Speaking: \"{phrase}\"")
        since_ts = time.time()
        speak_dur = speak_on_rtx(phrase)
        speech_done = time.time()

        aura_said, _, first_chunk_ms, total_audio_ms = poll_journal(since_ts, timeout=30.0)
        response_done = time.time()

        e2e = response_done - speech_done
        e2e_latencies.append(e2e)
        if first_chunk_ms is not None:
            first_chunk_times.append(first_chunk_ms)

        log(f"           Response: \"{aura_said[:60]}\"")
        fc_str = f"{first_chunk_ms}ms" if first_chunk_ms else "N/A"
        log(f"           E2E: {e2e:.1f}s | First chunk: {fc_str}")
        log("")
        time.sleep(8)

    avg_e2e = statistics.mean(e2e_latencies) if e2e_latencies else 30.0
    avg_first_chunk = statistics.mean(first_chunk_times) if first_chunk_times else 5000

    # Scoring: <5s E2E = perfect, >20s = 0
    # Note: E2E includes SSH polling overhead (~4-6s), so real latency is lower
    # Score based on first-chunk time which is more accurate
    if first_chunk_times:
        # First chunk: <500ms = perfect, >5000ms = 0
        fc_score = max(0, min(1.0, (5000 - avg_first_chunk) / 4500))
        # E2E as secondary signal: <8s = perfect (accounting for poll overhead), >25s = 0
        e2e_score = max(0, min(1.0, (25 - avg_e2e) / 17))
        latency_raw = fc_score * 0.6 + e2e_score * 0.4
    else:
        e2e_score = max(0, min(1.0, (25 - avg_e2e) / 17))
        latency_raw = e2e_score

    latency_score = round(latency_raw * 25, 1)

    log(f"  RESULT: Avg E2E={avg_e2e:.1f}s, Avg first chunk={avg_first_chunk:.0f}ms")
    log(f"  SCORE: {latency_score}/25")
    log("")

    reset_puck()
    time.sleep(2)

    # =========================================================================
    # TEST 3: LLM Response Coherence (20 points)
    # =========================================================================
    log("━" * 70)
    log("TEST 3: LLM RESPONSE COHERENCE (20 pts)")
    log("━" * 70)

    coherence_hits = 0

    for i, (question, expected_keywords) in enumerate(COHERENCE_TESTS):
        log(f"  [{i+1}/{len(COHERENCE_TESTS)}] Speaking: \"{question}\"")
        since_ts = time.time()
        speak_on_rtx(question)
        aura_said, whisper_heard, _, _ = poll_journal(since_ts, timeout=30.0)

        aura_lower = aura_said.lower()
        hit = any(kw in aura_lower for kw in expected_keywords)
        if hit:
            coherence_hits += 1

        no_response = aura_said == "[no response]"
        garbage = aura_said and len(set(aura_said)) < 5  # e.g. "GGGGGG"
        trouble = "trouble processing" in aura_lower

        if no_response:
            status = "NO RESPONSE"
        elif garbage:
            status = "GARBAGE"
        elif trouble:
            status = "ERROR"
        elif hit:
            status = "CORRECT"
        else:
            status = "WRONG"

        log(f"           Heard: \"{whisper_heard}\"")
        log(f"           Said:  \"{aura_said[:60]}\"")
        log(f"           Expected: {expected_keywords} [{status}]")
        log("")
        time.sleep(8)

    coherence_raw = coherence_hits / len(COHERENCE_TESTS) if COHERENCE_TESTS else 0
    coherence_score = round(coherence_raw * 20, 1)

    log(f"  RESULT: {coherence_hits}/{len(COHERENCE_TESTS)} correct")
    log(f"  SCORE: {coherence_score}/20")
    log("")

    reset_puck()
    time.sleep(2)

    # =========================================================================
    # TEST 4: TTS Pipeline Speed (15 points)
    # =========================================================================
    log("━" * 70)
    log("TEST 4: TTS PIPELINE SPEED (15 pts)")
    log("━" * 70)

    tts_ratios = []  # ratio of audio duration to total pipeline time

    for i, phrase in enumerate(LATENCY_TESTS[:3]):  # reuse simple phrases
        log(f"  [{i+1}/3] Speaking: \"{phrase}\"")
        since_ts = time.time()
        speak_on_rtx(phrase)
        _, _, first_chunk_ms, total_audio_ms = poll_journal(since_ts, timeout=30.0)

        if first_chunk_ms and total_audio_ms:
            # Pipeline total is in the log as "Xms total"
            # Good TTS: audio is most of the pipeline time (high ratio)
            # First chunk should be fast (<1000ms)
            fc_ratio = min(1.0, 1000 / max(first_chunk_ms, 1))
            tts_ratios.append(fc_ratio)
            log(f"           First chunk: {first_chunk_ms}ms | Audio: {total_audio_ms}ms")
        else:
            tts_ratios.append(0)
            log(f"           No timing data")
        log("")
        time.sleep(8)

    avg_tts = statistics.mean(tts_ratios) if tts_ratios else 0
    tts_score = round(avg_tts * 15, 1)

    log(f"  RESULT: Avg TTS efficiency={avg_tts:.0%}")
    log(f"  SCORE: {tts_score}/15")
    log("")

    reset_puck()
    time.sleep(2)

    # =========================================================================
    # TEST 5: Robustness (15 points)
    # =========================================================================
    log("━" * 70)
    log("TEST 5: ROBUSTNESS & EDGE CASES (15 pts)")
    log("━" * 70)

    robustness_passes = 0

    for i, (phrase, test_type) in enumerate(ROBUSTNESS_TESTS):
        log(f"  [{i+1}/{len(ROBUSTNESS_TESTS)}] [{test_type}] \"{phrase[:50]}{'...' if len(phrase)>50 else ''}\"")
        since_ts = time.time()
        speak_on_rtx(phrase)
        aura_said, whisper_heard, _, _ = poll_journal(since_ts, timeout=35.0)

        no_response = aura_said == "[no response]"
        garbage = aura_said and len(set(aura_said)) < 5
        trouble = "trouble processing" in aura_said.lower()

        if no_response:
            status = "FAIL (no response)"
        elif garbage:
            status = "FAIL (garbage)"
        elif trouble:
            # "trouble processing" is actually a graceful fallback — partial credit
            status = "PARTIAL (graceful error)"
            robustness_passes += 0.5
        else:
            status = "PASS"
            robustness_passes += 1

        log(f"           Heard: \"{whisper_heard[:60]}\"")
        log(f"           Said:  \"{aura_said[:60]}\"")
        log(f"           [{status}]")
        log("")
        time.sleep(8)

    robustness_raw = robustness_passes / len(ROBUSTNESS_TESTS) if ROBUSTNESS_TESTS else 0
    robustness_score = round(robustness_raw * 15, 1)

    log(f"  RESULT: {robustness_passes}/{len(ROBUSTNESS_TESTS)} passed")
    log(f"  SCORE: {robustness_score}/15")
    log("")

    # =========================================================================
    # FINAL SCORECARD
    # =========================================================================
    total = transcription_score + latency_score + coherence_score + tts_score + robustness_score

    if total >= 85:
        grade = "A"
    elif total >= 70:
        grade = "B"
    elif total >= 55:
        grade = "C"
    elif total >= 40:
        grade = "D"
    else:
        grade = "F"

    log("=" * 70)
    log("FINAL SCORECARD")
    log("=" * 70)
    log(f"  1. Transcription Accuracy:  {transcription_score:>5.1f} / 25")
    log(f"  2. Response Latency:        {latency_score:>5.1f} / 25")
    log(f"  3. LLM Coherence:           {coherence_score:>5.1f} / 20")
    log(f"  4. TTS Pipeline Speed:      {tts_score:>5.1f} / 15")
    log(f"  5. Robustness:              {robustness_score:>5.1f} / 15")
    log(f"  {'─' * 40}")
    log(f"  TOTAL:                      {total:>5.1f} / 100  [{grade}]")
    log("")

    # Recommendations
    log("RECOMMENDATIONS:")
    if transcription_score < 18:
        log("  - Whisper transcription needs work. Consider upgrading model or")
        log("    improving mic gain / noise filtering.")
    if latency_score < 18:
        log("  - Response latency is high. Consider smaller LLM or optimizing")
        log("    the TTS pipeline / filler timing.")
    if coherence_score < 14:
        log("  - LLM responses are often wrong or incoherent. Consider a")
        log("    larger model or better system prompts.")
    if tts_score < 10:
        log("  - TTS first-chunk time is slow. Check GPU utilization and")
        log("    consider pre-warming the TTS engine.")
    if robustness_score < 10:
        log("  - Edge cases cause failures. Review garbage detector thresholds")
        log("    and add better input sanitization.")
    if total >= 85:
        log("  - System performing well. Focus on polish and UX.")

    log("")
    log("=" * 70)

    REPORT_FILE.write_text("\n".join(report))
    print(f"\nReport saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
