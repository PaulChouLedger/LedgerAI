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
# Load .env from workspace root (2 levels up from this file)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(workspace_root, '.env')
load_dotenv(dotenv_path)

# Try both old and new variable names for backwards compatibility
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID") or "default"

if not ELEVEN_API_KEY or ELEVEN_API_KEY == "your_elevenlabs_api_key_here":
    raise RuntimeError(
        "❌ Missing ElevenLabs API key!\n"
        "   Run: ./aura_config.sh\n"
        "   Choose option 5 to configure TTS\n"
        "   Or edit .env and set: ELEVENLABS_API_KEY=your_key_here"
    )
client = ElevenLabs(api_key=ELEVEN_API_KEY)

# === Audio settings ===
PCM_SAMPLE_RATE = 22050
PCM_FORMAT = "pcm_22050"
VOLUME_SET = False

# Require TTS volume from .env (no default)
tts_volume_raw = os.getenv("TTS_VOLUME")
if tts_volume_raw is None or not tts_volume_raw.strip().isdigit() or not (0 <= int(tts_volume_raw) <= 100):
    raise RuntimeError(
        "❌ Missing or invalid TTS_VOLUME. Set it in .env via aura_config.sh → TTS, then restart."
    )
TTS_VOLUME = int(tts_volume_raw)  # percent

# Device identification - auto-detect connected output device
ALSA_CONTROLS = ["PCM", "Speaker", "Master"]  # try these in order

# Auto-detect output device (prefer UACDemoV1.0, fallback to any USB audio, then default)
def detect_output_device():
    """Auto-detect audio output device - prefer UACDemoV1.0, fallback to any USB audio or default"""
    try:
        output = subprocess.check_output(["aplay", "-l"], text=True)
        # First, try to find UACDemoV1.0
        for line in output.splitlines():
            if "UACDemoV1.0" in line:
                match = re.search(r"card (\d+):", line)
                if match:
                    card_num = int(match.group(1))
                    return f"UACDemoV1.0", card_num
        
        # Fallback: find any USB audio device with output (0 in, X out)
        for line in output.splitlines():
            if "USB Audio" in line and ("0 in" in line or "out" in line):
                match = re.search(r"card (\d+):", line)
                if match:
                    card_num = int(match.group(1))
                    # Extract device name from line
                    device_match = re.search(r"card \d+: (\w+)", line)
                    device_name = device_match.group(1) if device_match else f"USB_Audio_{card_num}"
                    return device_name, card_num
        
        # No USB device found - return None to use default
        return None, None
    except Exception as e:
        print(f"[Speaker] ⚠️ Failed to detect output device: {e}")
        return None, None

# Detect output device on startup
OUTPUT_DEVICE_NAME, OUTPUT_CARD_INDEX = detect_output_device()
if OUTPUT_DEVICE_NAME:
    print(f"[Speaker] 🔍 Auto-detected output device: {OUTPUT_DEVICE_NAME} (card {OUTPUT_CARD_INDEX})")
else:
    print(f"[Speaker] 🔍 Using default ALSA device (no specific device detected)")

# === TTS config ===
SENTENCE_QUEUE = queue.Queue()
playback_lock = threading.Lock()
# Batching: Accumulate chunks before sending to TTS (reduces API calls)
TTS_BATCH_ENABLED = os.getenv("TTS_BATCH_ENABLED", "true").lower() == "true"
TTS_BATCH_MAX_WORDS = int(os.getenv("TTS_BATCH_MAX_WORDS", "50"))  # Max words per batch (very aggressive for low latency)
TTS_BATCH_MIN_WORDS = int(os.getenv("TTS_BATCH_MIN_WORDS", "3"))  # Very low for immediate first audio (was 12)
TTS_BATCH_MAX_CHUNKS = int(os.getenv("TTS_BATCH_MAX_CHUNKS", "2"))  # Keep small batches
TTS_BATCH_TIMEOUT = float(os.getenv("TTS_BATCH_TIMEOUT", "0.02"))  # Very short timeout for low latency (was 0.05)
_batch_buffer = []  # Buffer for batching chunks
_batch_lock = threading.Lock()
_batch_timer = None  # Timer for delayed flush
_batch_started = False  # Track if we've sent the first batch (for low-latency start)
_llm_request_start_time = None  # Track when LLM request started for accurate latency measurement
TTS_TOKEN_LIMIT = 200  # Max tokens before forcing sentence split (hardcoded)
USE_SSML = True
INSERT_BREAKS = True
INSERT_SENTENCE_PAUSE = True
EMPHASIZE_WORDS = ["really", "important", "please", "must", "urgent"]
DEFAULT_EMOTION = "neutral"
RATE = "100%"
PITCH = "100%"

# Note: detect_output_device() is called at module load time above
# These functions use the pre-detected OUTPUT_CARD_INDEX

# === Set playback volume once ===
def set_volume_once():
    global VOLUME_SET
    if not VOLUME_SET:
        if OUTPUT_CARD_INDEX is not None:
            for ctrl in ALSA_CONTROLS:
                try:
                    subprocess.run(
                        ["amixer", "-c", str(OUTPUT_CARD_INDEX), "sset", ctrl, f"{TTS_VOLUME}%"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                    print(f"[Speaker] 🔊 Volume set to {TTS_VOLUME}% on card {OUTPUT_CARD_INDEX}:{ctrl}")
                    VOLUME_SET = True
                    return
                except Exception:
                    continue
            print("[Speaker] ⚠️ Could not set volume — check ALSA controls")
        else:
            # Using default device - volume control may not be available
            print(f"[Speaker] 🔊 Using default ALSA device (volume: {TTS_VOLUME}%)")
            VOLUME_SET = True

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
    # Remove control tags for clean TTS output
    text = re.sub(r"<sentence_start>|<sentence_end>|<pause>", "", text)
    return text.strip()

def enqueue_tts_chunk(text):
    """
    Enqueue TTS chunk with optional batching to reduce API calls.
    If batching is enabled, accumulates chunks until threshold or timeout.
    """
    global pending_initials, _batch_buffer
    
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
            pending_initials = None
            if merged_text and not re.match(r"^[\s.,!?]+$", merged_text):
                text = merged_text.strip()
            else:
                return
    
    # Check if this text ends with initials
    initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*\s*$'
    if re.search(initials_pattern, text):
        pending_initials = text
        return  # Don't enqueue this yet, wait for the name
    
    # Normal enqueue
    if not text or re.match(r"^[\s.,!?]+$", text):
        return
    
    text = text.strip()
    
    # Batching: Accumulate chunks to reduce API calls (with low-latency start)
    if TTS_BATCH_ENABLED:
        global _batch_timer, _batch_started
        with _batch_lock:
            _batch_buffer.append(text)
            total_words = sum(len(chunk.split()) for chunk in _batch_buffer)
            total_chunks = len(_batch_buffer)
            
            # LOW-LATENCY START: Flush immediately on first chunk (or after very few words)
            # This starts TTS ASAP even if more text is coming
            if not _batch_started and (total_words >= TTS_BATCH_MIN_WORDS or total_chunks >= 1):
                # First batch - flush immediately to start TTS as fast as possible
                # Even with just 1 chunk, start TTS immediately for lowest latency
                batched_text = " ".join(_batch_buffer)
                SENTENCE_QUEUE.put(batched_text)
                _batch_buffer = []
                _batch_started = True
                if _batch_timer:
                    _batch_timer.cancel()
                    _batch_timer = None
                print(f"[Speaker] 🚀 First batch ({total_chunks} chunks, {total_words} words) - flushing immediately (low-latency start)")
                return
            
            # Check if we should flush the batch immediately (subsequent batches)
            should_flush = (
                total_words >= TTS_BATCH_MAX_WORDS or
                total_chunks >= TTS_BATCH_MAX_CHUNKS
            )
            
            if should_flush:
                # Cancel any pending timer
                if _batch_timer:
                    _batch_timer.cancel()
                    _batch_timer = None
                
                # Join chunks with spaces
                batched_text = " ".join(_batch_buffer)
                SENTENCE_QUEUE.put(batched_text)
                _batch_buffer = []
                print(f"[Speaker] 📦 Batched {total_chunks} chunks ({total_words} words) - flushing immediately (threshold reached)")
            else:
                # For single chunk, flush immediately to avoid timeout delay (even if just 1 word)
                if total_chunks == 1:
                    batched_text = " ".join(_batch_buffer)
                    SENTENCE_QUEUE.put(batched_text)
                    _batch_buffer = []
                    _batch_started = True
                    if _batch_timer:
                        _batch_timer.cancel()
                        _batch_timer = None
                    print(f"[Speaker] 📦 Single chunk ({total_words} words) - flushing immediately")
                else:
                    # Cancel existing timer and start a new one (reset timeout)
                    if _batch_timer:
                        _batch_timer.cancel()
                    _batch_timer = threading.Timer(TTS_BATCH_TIMEOUT, _flush_batch_if_ready)
                    _batch_timer.start()
    else:
        # No batching - send immediately
        SENTENCE_QUEUE.put(text)

def _flush_batch_if_ready():
    """Flush the batch buffer if it has content"""
    global _batch_buffer, _batch_timer, _batch_started
    with _batch_lock:
        if _batch_buffer:
            batched_text = " ".join(_batch_buffer)
            total_words = sum(len(chunk.split()) for chunk in _batch_buffer)
            total_chunks = len(_batch_buffer)
            SENTENCE_QUEUE.put(batched_text)
            _batch_buffer = []
            _batch_started = True
            _batch_timer = None
            print(f"[Speaker] 📦 Flushed batch ({total_chunks} chunks, {total_words} words) after timeout")

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
            # This text ends with initials, might need to be merged with next chunk
            # We'll handle this in the buffer logic above
            pass
    
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
            pending_initials = None
            return merged_text
    
    # Check if this text ends with initials
    initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*\s*$'
    if re.search(initials_pattern, text):
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
        from gui.aura_gui import set_tts_frequency
        set_tts_frequency(frequency_speed)
    except ImportError:
        pass  # GUI not available

# === TTS playback using aplay ===
def tts_playback_thread(text, tts_start_time):
    with playback_lock:
        set_playing(True)
        
        # Track TTS generation token usage (based on text length)
        try:
            from wallet.wallet_integration import get_usage_tracker
            tracker = get_usage_tracker()
            # Approximate speech duration: ~150 words per minute = 2.5 words per second
            words = len(text.split())
            speech_duration_seconds = words / 2.5
            tracker.record_usage('tts_generation', multiplier=speech_duration_seconds)
        except Exception as e:
            print(f"[TokenUsage] ⚠️ Failed to track TTS usage: {e}")
        
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

            # Use ALSA 'plug' plugin for automatic format conversion (handles sample rate and channels)
            # This ensures proper conversion from ElevenLabs mono 22050 Hz to device's native format
            proc = None
            if OUTPUT_CARD_INDEX is not None:
                # Use plughw: instead of hw: to enable automatic format conversion
                device_spec = f"plughw:{OUTPUT_CARD_INDEX},0"
                try:
                    # Let plug plugin handle conversion - specify input format (mono 22050)
                    # Output will be automatically converted to device's native format
                    proc = subprocess.Popen(
                        ["aplay", "-D", device_spec, "-f", "S16_LE", "-r", str(PCM_SAMPLE_RATE), "-c", "1"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )
                    # Wait a moment to see if process starts successfully
                    time.sleep(0.1)
                    if proc.poll() is not None:
                        # Process died immediately - try default device
                        try:
                            if proc.stderr:
                                stderr_output = proc.stderr.read().decode().strip()
                                if stderr_output:
                                    print(f"[Speaker] ⚠️ Device {device_spec} failed: {stderr_output}")
                        except Exception:
                            pass
                        print(f"[Speaker] 🔄 Falling back to default ALSA device...")
                        proc = None  # Will be set to default below
                except Exception as e:
                    print(f"[Speaker] ⚠️ Failed to start aplay with device {device_spec}: {e}")
                    print(f"[Speaker] 🔄 Falling back to default ALSA device...")
                    proc = None  # Will be set to default below
            
            # Use default ALSA device with plug plugin (either no device detected or explicit device failed)
            if proc is None:
                # Use default device with plug plugin for automatic conversion
                proc = subprocess.Popen(
                    ["aplay", "-D", "plug:default", "-f", "S16_LE", "-r", str(PCM_SAMPLE_RATE), "-c", "1"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Calculate TTS latency from transcription end to TTS initiation
            tts_latency = time.time() - tts_start_time
            print(f"⏱️ TTS latency: {tts_latency:.2f}s")
            
            # Write chunks directly (ALSA plug plugin handles mono->stereo and sample rate conversion)
            try:
                # Verify process is still running before writing first chunk
                if proc.poll() is not None:
                    raise BrokenPipeError(f"aplay process terminated before writing (exit code: {proc.returncode})")
                
                # Write first chunk directly (plug plugin handles conversion)
                proc.stdin.write(first_chunk)
                proc.stdin.flush()

                for chunk in stream:
                    if chunk:
                        # Check if process is still alive before writing
                        if proc.poll() is not None:
                            raise BrokenPipeError(f"aplay process terminated during playback (exit code: {proc.returncode})")
                        try:
                            # Write chunk directly (plug plugin handles conversion)
                            proc.stdin.write(chunk)
                            proc.stdin.flush()
                        except BrokenPipeError:
                            # Process terminated during write
                            if proc.poll() is not None:
                                raise BrokenPipeError(f"aplay process terminated (exit code: {proc.returncode})")
                            raise
                
                # Close stdin and wait for process to finish
                proc.stdin.close()
                
                # Wait for process to finish (with timeout if available)
                try:
                    proc.wait(timeout=10)  # Wait up to 10 seconds for process to finish
                except TypeError:
                    # Python < 3.3 doesn't support timeout
                    proc.wait()
                except subprocess.TimeoutExpired:
                    print(f"[Speaker] ⚠️ aplay process timeout - killing...")
                    if proc:
                        proc.kill()
                        proc.wait()
                
            except BrokenPipeError as e:
                print(f"[Speaker] ❌ Audio pipe broken: {e}")
                # Try to get stderr if available for better diagnostics
                if proc and proc.stderr:
                    try:
                        stderr = proc.stderr.read().decode().strip()
                        if stderr:
                            print(f"[Speaker] aplay stderr: {stderr}")
                    except Exception:
                        pass
                if proc:
                    proc.kill()
                    proc.wait()
                # Don't re-raise - just log the error
                print(f"[Speaker] ⚠️ TTS playback failed - continuing...")
            except Exception as e:
                print(f"[Speaker] ❌ Unexpected TTS error: {e}")
                if proc:
                    proc.kill()
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
        # Use LLM request start time if available, otherwise use current time
        global _llm_request_start_time
        tts_start_time = _llm_request_start_time if _llm_request_start_time is not None else time.time()
        # Reset after first use to avoid using stale time for subsequent chunks
        if _llm_request_start_time is not None:
            _llm_request_start_time = None
        threading.Thread(target=tts_playback_thread, args=(sentence, tts_start_time), daemon=True).start()
        # Removed sleep to reduce latency - threads are daemon so they won't block

# === Stream LLM output ===
def speak_llm_response(prompt, context=""):
    global pending_initials, _batch_started
    import requests
    print(f"[LLM] ✅ Prompt to LLM: {prompt}")
    
    # Reset batch tracking for new response
    _batch_started = False
    early_flush_done = False  # Track if we've done early flush for this response
    
    # Track token usage for this query
    try:
        from wallet.wallet_integration import get_usage_tracker
        tracker = get_usage_tracker()
        
        # Determine query complexity
        prompt_length = len(prompt.split())
        has_context = bool(context)
        
        if prompt_length > 50 or has_context:
            # Complex query with context/RAG
            tracker.record_usage('complex_query', multiplier=1.0 + (prompt_length / 100))
        elif prompt_length > 20:
            # Medium complexity RAG query
            tracker.record_usage('rag_query')
        else:
            # Simple query
            tracker.record_usage('simple_query')
    except Exception as e:
        print(f"[TokenUsage] ⚠️ Failed to track usage: {e}")
    
    # Start TTS latency measurement (measured from LLM request start)
    global _llm_request_start_time
    _llm_request_start_time = time.time()
    
    # Get the correct LLM port based on global state (default medical)
    # Use a single endpoint to avoid port/fallback warnings.
    # Both medical and generic containers bind to the same host port; only one runs at a time.
    # Get LLM port from settings (not hardcoded)
    try:
        from state import get_llm_mode
        llm_mode = get_llm_mode()
        # Both medical and generic now use port 11434
        primary_port = "11434"
    except Exception:
        # Fallback to medical port (default)
        primary_port = "11434"
    
    def _post_stream(port: str):
        return requests.post(
            f"http://localhost:{port}/chat-tts",
            json={"prompt": prompt, "context": context, "chat_id": "voice_session"},
            stream=True,
            timeout=20
        )

    try:
        response = _post_stream(primary_port)
        if response.status_code != 200:
            raise RuntimeError(f"LLM HTTP {response.status_code} on port {primary_port}")
        # Process streaming tokens
        buffer = []
        for line in response.iter_lines(decode_unicode=True):
            token = line.strip()
            if not token:
                continue

            # Handle sentence control markers
            if token == '<sentence_start>':
                # Mark start of new sentence
                continue
            elif token == '<sentence_end>':
                # Flush buffer immediately for separate TTS playback
                if buffer:
                    chunk_text = " ".join(buffer).strip()
                    clean_text = re.sub(r'<sentence_start>|<sentence_end>', '', chunk_text).strip()
                    if clean_text:
                        print(f"[Speaker] 🎙️ Flushing sentence on <sentence_end>: '{clean_text}'")
                        enqueue_tts_chunk(clean_text)
                    buffer.clear()
                continue

            print(f"[LLM] 🧠 {token}")
            buffer.append(token)
            
            # Count total words in buffer for accurate limit checking
            total_words = sum(len(t.split()) for t in buffer)
            
            # AGGRESSIVE EARLY FLUSH: Send first chunk immediately after 2-3 words for ultra-low latency
            # This starts TTS generation while more text is still streaming
            if not early_flush_done and total_words >= 2:
                # Flush immediately with first few words to start TTS ASAP
                chunk_text = " ".join(buffer).strip()
                clean_text = re.sub(r'<sentence_start>|<sentence_end>', '', chunk_text).strip()
                if clean_text:
                    enqueue_tts_chunk(clean_text)
                    buffer.clear()  # Clear buffer after flushing
                    early_flush_done = True  # Mark as done to prevent multiple early flushes
                    # Continue accumulating for next chunk
            
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
            
            # Dynamic pattern: detect if current token is a name following initials
            # Pattern: "A.B." or "A.B.C." followed by "Name."
            if (len(buffer) > 1 and 
                token.endswith('.') and 
                len(token) > 2 and 
                len(token.split()) == 1 and  # Single word
                token[0].isupper() and  # Capitalized
                not token.lower() in ['the.', 'and.', 'or.', 'but.', 'for.', 'nor.', 'yet.', 'so.']):  # Not common words
                
                # Check if any previous sentence in buffer ends with initials pattern (A.B. or A.B.C.)
                
                # Look for initials pattern in any previous sentence
                initials_pattern = r'\b[A-Z]\.(?:[A-Z]\.)*\s*$'
                found_initials = False
                
                for i in range(len(buffer) - 1, -1, -1):
                    if i < len(buffer) - 1:  # Don't check the current token
                        sentence = buffer[i]
                        if re.search(initials_pattern, sentence):
                            # Don't split, let it continue to build the full name
                            should_split = False
                            found_initials = True
                            break
                
                if not found_initials:
                    pass
            
            if should_split:
                chunk_text = " ".join(buffer).strip()
                # Remove sentence tags before TTS
                clean_text = re.sub(r'<sentence_start>|<sentence_end>', '', chunk_text).strip()
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
                    buffer.clear()
        if buffer:
            chunk_text = " ".join(buffer).strip()
            clean_text = re.sub(r'<sentence_start>|<sentence_end>', '', chunk_text).strip()
            if clean_text:
                enqueue_tts_chunk(clean_text)
        # Flush any remaining batched chunks when streaming ends
        if TTS_BATCH_ENABLED:
            _flush_batch_if_ready()
    except Exception as e:
        print(f"[LLM] ❌ Streaming error: {e}")
        # Flush batch even on error to avoid losing buffered chunks
        if TTS_BATCH_ENABLED:
            _flush_batch_if_ready()

# === Warmup ===
def warm_up_tts():
    print("[Speaker] 🔧 Warming up...")
    enqueue_tts_chunk("AuraVision is initializing, please wait.")

# === Start thread ===
threading.Thread(target=playback_loop, daemon=True).start()
