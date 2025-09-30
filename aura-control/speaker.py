import os
import re
import time
import queue
import threading
import subprocess
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from state import set_playing, is_playing

# === Load API credentials ===
load_dotenv()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
assert ELEVEN_API_KEY and ELEVEN_VOICE_ID, "Missing ElevenLabs credentials"
client = ElevenLabs(api_key=ELEVEN_API_KEY)

# === Audio settings ===
PCM_SAMPLE_RATE = 22050
PCM_FORMAT = "pcm_22050"
VOLUME_SET = False
TTS_VOLUME = 100  # percent

# Device identification
DEVICE_NAME = "UACDemoV1.0"   # part of the USB device name from `aplay -l`
ALSA_CONTROLS = ["PCM", "Speaker", "Master"]  # try these in order

# === TTS config ===
SENTENCE_QUEUE = queue.Queue()
playback_lock = threading.Lock()
TTS_TOKEN_LIMIT = int(os.getenv("TTS_TOKEN_LIMIT", "15"))  # Max tokens before forcing sentence split
USE_SSML = True
INSERT_BREAKS = True
INSERT_SENTENCE_PAUSE = True
EMPHASIZE_WORDS = ["really", "important", "please", "must", "urgent"]
DEFAULT_EMOTION = "neutral"
RATE = "100%"
PITCH = "100%"

# === Detect ALSA card index dynamically ===
def detect_card_index(device_name: str) -> int:
    try:
        output = subprocess.check_output(["aplay", "-l"], text=True)
        for line in output.splitlines():
            if device_name in line:
                match = re.search(r"card (\d+):", line)
                if match:
                    return int(match.group(1))
    except Exception as e:
        print(f"[Speaker] ⚠️ Failed to detect ALSA card index: {e}")
    return 0  # fallback

# === Set playback volume once ===
def set_volume_once():
    global VOLUME_SET
    if not VOLUME_SET:
        card_index = detect_card_index(DEVICE_NAME)
        for ctrl in ALSA_CONTROLS:
            try:
                subprocess.run(
                    ["amixer", "-c", str(card_index), "sset", ctrl, f"{TTS_VOLUME}%"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                print(f"[Speaker] 🔊 Volume set to {TTS_VOLUME}% on card {card_index}:{ctrl}")
                VOLUME_SET = True
                return
            except Exception:
                continue
        print("[Speaker] ⚠️ Could not set volume — check ALSA controls")

def detect_emotion(text):
    lowered = text.lower()
    if any(w in lowered for w in ["awesome", "great", "yay", "excited", "love"]):
        return "excited"
    if any(w in lowered for w in ["sorry", "unfortunately", "regret", "sad"]):
        return "disappointed"
    return DEFAULT_EMOTION

def normalize_units(text):
    return text.replace("°C", "degrees Celsius").replace("°F", "degrees Fahrenheit")

def ssml_wrap(text):
    if not USE_SSML:
        return text
    if INSERT_BREAKS:
        text = re.sub(r"([,;])", r"\1<break time='300ms'/>", text)
    if INSERT_SENTENCE_PAUSE:
        text = re.sub(r"([.?!])", r"\1<break time='600ms'/>", text)
    for word in EMPHASIZE_WORDS:
        text = re.sub(rf"\b({word})\b", r"<emphasis>\1</emphasis>", text, flags=re.IGNORECASE)
    emotion = detect_emotion(text)
    print(f"[Speaker] 🎭 Detected emotion: {emotion}")
    return (
        f"<speak><voice emotion='{emotion}'>"
        f"<prosody rate='{RATE}' pitch='{PITCH}'>{text}</prosody>"
        f"</voice></speak>"
    )

def preprocess_for_tts(text):
    return re.sub(r"<sentence_start>|<sentence_end>", "", text).strip()

def enqueue_tts_chunk(text):
    if text and not re.match(r"^[\s.,!?]+$", text):
        SENTENCE_QUEUE.put(text.strip())

# === TTS playback using aplay ===
def tts_playback_thread(text):
    with playback_lock:
        set_playing(True)
        try:
            stream = client.text_to_speech.convert(
                text=ssml_wrap(normalize_units(text)),
                voice_id=ELEVEN_VOICE_ID,
                output_format=PCM_FORMAT,
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.0,
                    "style": 0.0,
                    "use_speaker_boost": False,
                    "optimize_streaming_latency": True
                }
            )

            first_chunk = next(stream, None)
            if not first_chunk:
                raise RuntimeError("No audio received")

            proc = subprocess.Popen(
                ["aplay", "-f", "S16_LE", "-r", str(PCM_SAMPLE_RATE), "-c", "1"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Calculate TTS latency from transcription end to TTS initiation
            tts_latency = time.time() - tts_start_time
            print(f"⏱️ TTS latency: {tts_latency:.2f}s")
            
            proc.stdin.write(first_chunk)
            proc.stdin.flush()

            for chunk in stream:
                if chunk:
                    proc.stdin.write(chunk)
                    proc.stdin.flush()

            proc.stdin.close()
            proc.wait()

        except Exception as e:
            print(f"[Speaker] ❌ TTS error: {e}")
        finally:
            set_playing(False)

# === Playback loop ===
def playback_loop():
    set_volume_once()
    while True:
        sentence = SENTENCE_QUEUE.get()
        sentence = preprocess_for_tts(sentence)
        if not sentence or sentence.lower() in {"uh", "hmm", "um", "<silence>"}:
            print(f"[Speaker] ⚠️ Skipping filler: \"{sentence}\"")
            continue
        print(f"[Speaker] 🔈 Speaking: \"{sentence}\"")
        threading.Thread(target=tts_playback_thread, args=(sentence,), daemon=True).start()
        time.sleep(0.1)

# === Stream LLM output ===
def speak_llm_response(prompt, context=""):
    import requests
    print(f"[LLM] ✅ Prompt to LLM: {prompt}")
    
    # Start TTS latency measurement
    tts_start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/chat",
            json={"prompt": prompt, "context": context, "chat_id": "voice_session"},
            stream=True, timeout=60
        )
        buffer = []
        for line in response.iter_lines(decode_unicode=True):
            token = line.strip()
            if not token:
                continue
            print(f"[LLM] 🧠 {token}")
            buffer.append(token)
            
            # Count total words in buffer for accurate limit checking
            total_words = sum(len(t.split()) for t in buffer)
            
            # Check for sentence endings, but avoid splitting on abbreviations/initials
            ends = any(token.endswith(p) for p in [".", "!", "?"])
            # Don't split on single letters followed by period (initials like "K.")
            is_initial = len(token) == 2 and token.endswith('.') and token[0].isupper()
            # Don't split on common abbreviations
            is_abbreviation = token.lower() in ['mr.', 'mrs.', 'dr.', 'prof.', 'st.', 'ave.', 'blvd.', 'inc.', 'ltd.', 'corp.', 'etc.', 'vs.', 'jr.', 'sr.']
            
            # Check if this looks like a name continuation (single capitalized word ending with period)
            # Only match single words, not full sentences
            is_name_continuation = (token[0].isupper() and token.endswith('.') and 
                                 len(token) > 2 and len(token.split()) == 1 and 
                                 not token.lower() in ['the.', 'and.', 'or.', 'but.', 'for.', 'nor.', 'yet.', 'so.'])
            
            # Special case: if previous token was an initial (like "J.K.") and current token ends with period,
            # treat it as a name continuation to prevent splitting
            prev_token_was_initial = len(buffer) > 0 and len(buffer[-1]) == 2 and buffer[-1].endswith('.') and buffer[-1][0].isupper()
            is_following_initial = prev_token_was_initial and token.endswith('.') and len(token) > 2
            
            # Don't split if current token is an initial/abbreviation OR if it looks like a name continuation
            # OR if it's following an initial (like "Rowling." after "J.K.")
            should_split = (ends and not is_initial and not is_abbreviation and not is_name_continuation and not is_following_initial) or total_words >= TTS_TOKEN_LIMIT
            
            if should_split:
                enqueue_tts_chunk(" ".join(buffer).strip())
                buffer.clear()
        if buffer:
            enqueue_tts_chunk(" ".join(buffer).strip())
    except Exception as e:
        print(f"[LLM] ❌ Streaming error: {e}")

# === Warmup ===
def warm_up_tts():
    print("[Speaker] 🔧 Warming up...")
    enqueue_tts_chunk("AuraVision is initializing, please wait.")

# === Start thread ===
threading.Thread(target=playback_loop, daemon=True).start()
