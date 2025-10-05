import os
import re
import time
import queue
import threading
import subprocess
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from state import set_playing, is_playing
import numpy as np

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
TTS_VOLUME = 50  # percent

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
    global pending_initials
    
    print(f"[TTS Debug] Enqueueing: '{text}'")
    
    # Check if this should be merged with pending initials
    if pending_initials:
        # Check if current text is a name that should be merged
        if (text.endswith('.') and 
            len(text) > 2 and 
            len(text.split()) == 1 and 
            text[0].isupper() and 
            not text.lower() in ['the.', 'and.', 'or.', 'but.', 'for.', 'nor.', 'yet.', 'so.']):
            
            # Merge the pending initials with this name
            merged_text = pending_initials + " " + text
            print(f"[TTS Debug] MERGED: '{pending_initials}' + '{text}' = '{merged_text}'")
            pending_initials = None
            if merged_text and not re.match(r"^[\s.,!?]+$", merged_text):
                SENTENCE_QUEUE.put(merged_text.strip())
                print(f"[TTS Debug] Added merged to queue: '{merged_text.strip()}'")
            return
    
    # Check if this text ends with initials
    initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*\s*$'
    if re.search(initials_pattern, text):
        print(f"[TTS Debug] PENDING: Text ends with initials: '{text}'")
        pending_initials = text
        return  # Don't enqueue this yet, wait for the name
    
    # Normal enqueue
    if text and not re.match(r"^[\s.,!?]+$", text):
        SENTENCE_QUEUE.put(text.strip())
        print(f"[TTS Debug] Added to queue: '{text.strip()}'")
    else:
        print(f"[TTS Debug] Skipped (empty or filler): '{text}'")

def merge_initials_with_names(text):
    """Post-process text to merge initials with names that might have been split"""
    # Look for patterns like "J.K." followed by "Rowling." in the same text
    # This handles cases where the LLM already merged them but they're in separate sentences
    
    # Pattern: initials at end of sentence, followed by name at start of next sentence
    # Example: "written by J.K." + "Rowling." -> "written by J.K. Rowling."
    
    # Find all initials patterns
    initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*'
    initials_matches = re.findall(initials_pattern, text)
    
    for initials in initials_matches:
        # Look for this initials pattern at the end of a sentence
        pattern = re.escape(initials) + r'\s*$'
        if re.search(pattern, text):
            print(f"[TTS Debug] Found initials at end: '{initials}'")
            # This text ends with initials, might need to be merged with next chunk
            # We'll handle this in the buffer logic above
    
    return text

# Global variable to store pending initials
pending_initials = None

def check_for_initials_merge(text):
    """Check if this text should be merged with pending initials"""
    global pending_initials
    
    if pending_initials:
        # Check if current text is a name that should be merged
        if (text.endswith('.') and 
            len(text) > 2 and 
            len(text.split()) == 1 and 
            text[0].isupper() and 
            not text.lower() in ['the.', 'and.', 'or.', 'but.', 'for.', 'nor.', 'yet.', 'so.']):
            
            # Merge the pending initials with this name
            merged_text = pending_initials + " " + text
            print(f"[TTS Debug] MERGED: '{pending_initials}' + '{text}' = '{merged_text}'")
            pending_initials = None
            return merged_text
    
    # Check if this text ends with initials
    initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*\s*$'
    if re.search(initials_pattern, text):
        print(f"[TTS Debug] PENDING: Text ends with initials: '{text}'")
        pending_initials = text
        return None  # Don't enqueue this yet, wait for the name
    
    pending_initials = None
    return text

def analyze_audio_frequency(audio_chunk):
    """Analyze audio chunk to extract dominant frequency for GUI pulsation"""
    try:
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Skip if audio is too short
        if len(audio_data) < 100:
            return 0.15  # Default speed
            
        # Apply FFT to get frequency spectrum
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/PCM_SAMPLE_RATE)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft)
        
        # Find dominant frequency (excluding DC component)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Find peak frequency
        peak_idx = np.argmax(positive_magnitude[1:]) + 1  # Skip DC
        dominant_freq = positive_freqs[peak_idx]
        
        # Normalize frequency to GUI pulsation speed (0.1 to 0.5)
        # Human speech is typically 85-255 Hz, map to 0.1-0.5 speed
        normalized_speed = 0.1 + (dominant_freq / 255) * 0.4
        normalized_speed = max(0.1, min(0.5, normalized_speed))  # Clamp to range
        
        print(f"[Audio Analysis] 🎵 Freq: {dominant_freq:.1f}Hz -> Speed: {normalized_speed:.3f}")
        return normalized_speed
        
    except Exception as e:
        print(f"[Audio Analysis] ❌ Error analyzing frequency: {e}")
        return 0.15  # Default speed

def update_gui_frequency(frequency_speed):
    """Update GUI with current audio frequency for pulsation"""
    try:
        from aura_gui import set_tts_frequency
        set_tts_frequency(frequency_speed)
    except ImportError:
        pass  # GUI not available

# === TTS playback using aplay ===
def tts_playback_thread(text, tts_start_time):
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
            
            # Skip frequency analysis during warm-up and setup
            pass

            for chunk in stream:
                if chunk:
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
                    
                    # Skip frequency analysis during warm-up and setup
                    pass

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
        # Use current time as tts_start_time for playback loop
        tts_start_time = time.time()
        threading.Thread(target=tts_playback_thread, args=(sentence, tts_start_time), daemon=True).start()
        time.sleep(0.1)

# === Stream LLM output ===
def speak_llm_response(prompt, context=""):
    global pending_initials
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
            
            # Debug: Show buffer state
            print(f"[TTS Debug] Token: '{token}', Buffer: {buffer}")
            
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
            
            # Debug logging
            if prev_token_was_initial:
                print(f"[TTS Debug] Previous token was initial: '{buffer[-1]}', current: '{token}', is_following_initial: {is_following_initial}")
            
            # Don't split if current token is an initial/abbreviation OR if it looks like a name continuation
            # OR if it's following an initial (like "Rowling." after "J.K.")
            should_split = (ends and not is_initial and not is_abbreviation and not is_name_continuation and not is_following_initial) or total_words >= TTS_TOKEN_LIMIT
            
            # Dynamic pattern: detect if current token is a name following initials
            # Pattern: "A.B." or "A.B.C." followed by "Name."
            if (len(buffer) > 1 and 
                token.endswith('.') and 
                len(token) > 2 and 
                len(token.split()) == 1 and  # Single word
                token[0].isupper() and  # Capitalized
                not token.lower() in ['the.', 'and.', 'or.', 'but.', 'for.', 'nor.', 'yet.', 'so.']):  # Not common words
                
                # Check if any previous sentence in buffer ends with initials pattern (A.B. or A.B.C.)
                print(f"[TTS Debug] Checking pattern for: '{token}'")
                print(f"[TTS Debug] Full buffer: {buffer}")
                
                # Look for initials pattern in any previous sentence
                initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*\s*$'
                found_initials = False
                
                for i in range(len(buffer) - 1, -1, -1):
                    if i < len(buffer) - 1:  # Don't check the current token
                        sentence = buffer[i]
                        print(f"[TTS Debug] Checking sentence {i}: '{sentence}'")
                        if re.search(initials_pattern, sentence):
                            print(f"[TTS Debug] MERGING: Found initials in sentence {i}: '{sentence}' + '{token}'")
                            # Don't split, let it continue to build the full name
                            should_split = False
                            found_initials = True
                            break
                
                if not found_initials:
                    print(f"[TTS Debug] No initials pattern found in any previous sentence")
            
            
            if should_split:
                chunk_text = " ".join(buffer).strip()
                # Remove sentence tags before TTS
                clean_text = re.sub(r'<sentence_start>|<sentence_end>', '', chunk_text).strip()
                print(f"[TTS Debug] SPLITTING! Chunk: '{chunk_text}' -> Clean: '{clean_text}'")
                if clean_text:
                    enqueue_tts_chunk(clean_text)
                buffer.clear()
            else:
                # Check if we have a pending initials and current token is a name
                if (pending_initials and 
                    token.endswith('.') and 
                    len(token) > 2 and 
                    len(token.split()) == 1 and 
                    token[0].isupper() and 
                    not token.lower() in ['the.', 'and.', 'or.', 'but.', 'for.', 'nor.', 'yet.', 'so.']):
                    
                    # Merge the pending initials with this name
                    merged_text = pending_initials + " " + token
                    print(f"[TTS Debug] MERGED: '{pending_initials}' + '{token}' = '{merged_text}'")
                    pending_initials = None
                    if merged_text and not re.match(r"^[\s.,!?]+$", merged_text):
                        SENTENCE_QUEUE.put(merged_text.strip())
                        print(f"[TTS Debug] Added merged to queue: '{merged_text.strip()}'")
                    buffer.clear()
        if buffer:
            chunk_text = " ".join(buffer).strip()
            clean_text = re.sub(r'<sentence_start>|<sentence_end>', '', chunk_text).strip()
            if clean_text:
                enqueue_tts_chunk(clean_text)
    except Exception as e:
        print(f"[LLM] ❌ Streaming error: {e}")

# === Warmup ===
def warm_up_tts():
    print("[Speaker] 🔧 Warming up...")
    enqueue_tts_chunk("AuraVision is initializing, please wait.")

# === Start thread ===
threading.Thread(target=playback_loop, daemon=True).start()
