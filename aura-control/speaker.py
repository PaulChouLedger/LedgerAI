# === speaker.py — Streaming TTS playback with ElevenLabs ===

import os
import re
import time
import queue
import threading
import subprocess
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from state import set_playing, is_playing  # ✅ Global playback state

# === Load environment ===
load_dotenv()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
assert ELEVEN_API_KEY and ELEVEN_VOICE_ID, "Missing ElevenLabs credentials"

# === ElevenLabs client ===
client = ElevenLabs(api_key=ELEVEN_API_KEY)

# === TTS Config ===
PCM_SAMPLE_RATE = 22050
PCM_FORMAT = "pcm_22050"
SENTENCE_QUEUE = queue.Queue()

# === Dynamic Speech Control ===
USE_SSML = True
INSERT_BREAKS = True
INSERT_SENTENCE_PAUSE = True
EMPHASIZE_WORDS = ["really", "important", "please", "must", "urgent"]

# === Voice Modulation ===
TTS_VOLUME = 85
DEFAULT_EMOTION = "neutral"
RATE = "100%"
PITCH = "100%"

# === Volume control (optional) ===
def set_volume(percent=TTS_VOLUME):
    try:
        subprocess.run(
            ["amixer", "-c", "1", "sset", "PCM", f"{percent}%"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass  # Ignore volume errors

# === Emotion Detection ===
def detect_emotion(text):
    lowered = text.lower()
    if any(w in lowered for w in ["awesome", "great", "yay", "excited", "love", "happy", "cool"]):
        return "excited"
    if any(w in lowered for w in ["sorry", "unfortunately", "apologize", "sad", "can't", "regret"]):
        return "disappointed"
    if "?" in text:
        return "neutral"
    return DEFAULT_EMOTION

# === SSML Wrapper ===
def ssml_wrap(text):
    if not USE_SSML:
        return text

    # Insert short breaks after commas/semicolons
    if INSERT_BREAKS:
        text = re.sub(r"([,;])", r"\1<break time='300ms'/>", text)

    # Insert longer breaks after end of sentence
    if INSERT_SENTENCE_PAUSE:
        text = re.sub(r"([.?!])", r"\1<break time='600ms'/>", text)

    # Emphasize selected words
    for word in EMPHASIZE_WORDS:
        text = re.sub(rf"\b({word})\b", r"<emphasis>\1</emphasis>", text, flags=re.IGNORECASE)

    emotion = detect_emotion(text)
    print(f"[Speaker] 🎭 Detected emotion: {emotion}")

    return (
        f"<speak>"
        f"<voice emotion='{emotion}'>"
        f"<prosody rate='{RATE}' pitch='{PITCH}'>"
        f"{text}"
        f"</prosody>"
        f"</voice>"
        f"</speak>"
    )

# === Async queue fill ===
def enqueue_tts_chunk(text):
    if text and not re.match(r"^[\s.,!?]+$", text):
        SENTENCE_QUEUE.put(text.strip())

# === Playback Thread ===
def playback_loop():
    while True:
        sentence = SENTENCE_QUEUE.get()
        if not sentence:
            continue

        print(f"[Speaker] 🔈 Speaking: \"{sentence}\"")
        set_playing(True)
        set_volume()

        try:
            proc = subprocess.Popen(
                ["paplay", "--raw", "--rate=22050", "--channels=1", "--format=s16le"],
                stdin=subprocess.PIPE
            )

            stream = client.text_to_speech.convert(
                text=ssml_wrap(sentence),
                voice_id=ELEVEN_VOICE_ID,
                output_format=PCM_FORMAT
            )

            start = time.time()
            for chunk in stream:
                if chunk:
                    proc.stdin.write(chunk)

            proc.stdin.close()
            proc.wait()
            print(f"⏱️ TTS latency: {time.time() - start:.2f}s")

        except Exception as e:
            print(f"[Speaker] ❌ TTS error: {e}")
        finally:
            set_playing(False)

# === Stream LLM and queue chunks ===
def speak_llm_response(prompt):
    import requests
    print(f"[LLM] ✅ Sending prompt to LLM container: {prompt}")

    try:
        response = requests.post(
            "http://localhost:11434/chat",
            json={"prompt": prompt},
            stream=True,
            timeout=60
        )

        buffer = []
        for line in response.iter_lines(decode_unicode=True):
            token = line.strip()
            if not token:
                continue

            print(f"[LLM] 🧠 {token}")
            buffer.append(token)

            if token in {".", "!", "?"}:
                chunk = " ".join(buffer).strip()
                enqueue_tts_chunk(chunk)
                buffer.clear()

        if buffer:
            enqueue_tts_chunk(" ".join(buffer).strip())

    except Exception as e:
        print(f"[LLM] ❌ Error streaming: {e}")

# === ElevenLabs Warm-Up ===
def warm_up_tts():
    print("[Speaker] 🔧 Warming up ElevenLabs with intro message...")
    enqueue_tts_chunk("Welcome to Aura Vision, system initializing, please wait")

# === Startup ===
threading.Thread(target=playback_loop, daemon=True).start()
