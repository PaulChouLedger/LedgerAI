import os
import re
import time
import queue
import threading
import subprocess
import inspect
from dotenv import load_dotenv
from state import set_playing, is_playing, get_tts_engine, get_chatterbox_voice_cloning_enabled
import numpy as np

# === Load API credentials ===
# Load .env from workspace root (2 levels up from this file)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(workspace_root, '.env')
load_dotenv(dotenv_path)

# Try both old and new variable names for backwards compatibility
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID") or "default"

# ChatterboxTTS voice cloning - path to reference audio file (optional)
# If set, ChatterboxTTS will clone the voice from this audio sample
# Should be a WAV file with at least 5 seconds of clear speech
CHATTERBOX_VOICE_SAMPLE = os.getenv("CHATTERBOX_VOICE_SAMPLE", "")
# Default location: assets/voice_samples/ (relative to workspace root)
if not CHATTERBOX_VOICE_SAMPLE:
    default_voice_sample = os.path.join(workspace_root, "assets", "voice_samples", "sample.wav")
    if os.path.exists(default_voice_sample):
        CHATTERBOX_VOICE_SAMPLE = default_voice_sample

# Voice embedding cache path (for faster synthesis)
VOICE_EMBEDDING_CACHE_DIR = os.path.join(workspace_root, "data", "voice_cache")
os.makedirs(VOICE_EMBEDDING_CACHE_DIR, exist_ok=True)

# Initialize TTS engines (lazy loading)
_elevenlabs_client = None
_chatterbox_tts = None
_chatterbox_voice_embedding = None  # Cached voice embedding for faster synthesis

# Voice embedding cache path (for faster synthesis)
VOICE_EMBEDDING_CACHE_DIR = os.path.join(workspace_root, "data", "voice_cache")
os.makedirs(VOICE_EMBEDDING_CACHE_DIR, exist_ok=True)
_chatterbox_voice_embedding = None  # Cached voice embedding for faster synthesis

def _get_elevenlabs_client():
    """Lazy load ElevenLabs client (only when needed)"""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        if not ELEVEN_API_KEY or ELEVEN_API_KEY == "your_elevenlabs_api_key_here":
            raise RuntimeError(
                "❌ Missing ElevenLabs API key!\n"
                "   Run: ./aura_config.sh\n"
                "   Choose option 5 to configure TTS\n"
                "   Or edit .env and set: ELEVENLABS_API_KEY=your_key_here"
            )
        from elevenlabs.client import ElevenLabs
        _elevenlabs_client = ElevenLabs(api_key=ELEVEN_API_KEY)
    return _elevenlabs_client

def _get_chatterbox_tts():
    """Lazy load ChatterboxTTS (only when needed)"""
    global _chatterbox_tts
    if _chatterbox_tts is None:
        try:
            # Try different import paths for ChatterboxTTS
            try:
                from chatterbox.tts import ChatterboxTTS
            except ImportError:
                from chatterbox import ChatterboxTTS
            
            # Try from_pretrained first (recommended method)
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _chatterbox_tts = ChatterboxTTS.from_pretrained(device=device)
                print(f"[Speaker] ✅ ChatterboxTTS initialized successfully (device: {device})")
            except (AttributeError, TypeError):
                # Fallback to simple constructor
                _chatterbox_tts = ChatterboxTTS()
                print("[Speaker] ✅ ChatterboxTTS initialized successfully")
        except ImportError:
            raise RuntimeError(
                "❌ ChatterboxTTS not installed!\n"
                "   Install with: pip install setuptools && pip install chatterbox-tts\n"
                "   If pkuseg build fails, see: docs/CHATTERBOX_INSTALLATION_FIX.md"
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize ChatterboxTTS: {e}")
    return _chatterbox_tts

def _get_or_create_voice_embedding(chatterbox, voice_sample_path):
    """
    Get or create cached voice embedding for faster synthesis.
    This pre-processes the voice sample once and caches it, eliminating
    real-time processing overhead on subsequent TTS calls.
    
    Returns:
        Voice embedding object or None if caching not supported
    """
    global _chatterbox_voice_embedding, VOICE_EMBEDDING_CACHE_DIR
    
    # Check if we already have a cached embedding in memory
    if _chatterbox_voice_embedding is not None:
        return _chatterbox_voice_embedding
    
    # Try to load from disk cache
    import hashlib
    import pickle
    
    # Create cache key from file path and modification time
    sample_stat = os.stat(voice_sample_path)
    cache_key = hashlib.md5(
        f"{voice_sample_path}:{sample_stat.st_mtime}:{sample_stat.st_size}".encode()
    ).hexdigest()
    cache_path = os.path.join(VOICE_EMBEDDING_CACHE_DIR, f"voice_embedding_{cache_key}.pkl")
    
    # Try loading from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                _chatterbox_voice_embedding = pickle.load(f)
            print(f"[Speaker] ✅ Loaded cached voice embedding from: {cache_path}")
            return _chatterbox_voice_embedding
        except Exception as e:
            print(f"[Speaker] ⚠️ Failed to load cached embedding: {e}")
            # Continue to create new embedding
    
    # Try to extract and cache voice embedding
    # Note: This depends on ChatterboxTTS API - may not be available in all versions
    try:
        # Method 1: Try to get voice embedding directly (if API supports it)
        if hasattr(chatterbox, 'extract_voice_embedding') or hasattr(chatterbox, 'get_voice_embedding'):
            extract_method = getattr(chatterbox, 'extract_voice_embedding', None) or \
                           getattr(chatterbox, 'get_voice_embedding', None)
            if extract_method:
                print(f"[Speaker] 🔧 Extracting voice embedding from sample...")
                _chatterbox_voice_embedding = extract_method(voice_sample_path)
                
                # Cache to disk
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(_chatterbox_voice_embedding, f)
                    print(f"[Speaker] ✅ Cached voice embedding to: {cache_path}")
                except Exception as e:
                    print(f"[Speaker] ⚠️ Failed to cache embedding: {e}")
                
                return _chatterbox_voice_embedding
    except Exception as e:
        print(f"[Speaker] ⚠️ Voice embedding extraction not available: {e}")
    
    # If embedding extraction not available, return None
    # The system will fall back to real-time cloning (with audio_prompt_path)
    return None

# === Audio settings ===
PCM_SAMPLE_RATE = 22050
PCM_FORMAT = "pcm_22050"
VOLUME_SET = False

# TTS volume - load from .env if available, otherwise default to 100
TTS_VOLUME = int(os.getenv("TTS_VOLUME", "100"))  # percent (default: 100%)
# Ensure volume is in valid range
TTS_VOLUME = max(0, min(100, TTS_VOLUME))

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
# All values hardcoded - no .env needed (only API keys use .env)
TTS_BATCH_ENABLED = True
TTS_BATCH_MAX_WORDS = 50  # Max words per batch (very aggressive for low latency)
TTS_BATCH_MIN_WORDS = 3  # Very low for immediate first audio (was 12)
TTS_BATCH_MAX_CHUNKS = 2  # Keep small batches
TTS_BATCH_TIMEOUT = 0.02  # Very short timeout for low latency (was 0.05)
_batch_buffer = []  # Buffer for batching chunks
_batch_lock = threading.Lock()
_batch_timer = None  # Timer for delayed flush
_batch_started = False  # Track if we've sent the first batch (for low-latency start)
_llm_request_start_time = None  # Track when LLM request started for accurate latency measurement
_sentence_enqueue_time = None  # Track when sentence was enqueued for latency measurement
TTS_TOKEN_LIMIT = 200  # Max tokens before forcing sentence split (hardcoded)
USE_SSML = True
INSERT_BREAKS = True
INSERT_SENTENCE_PAUSE = True
EMPHASIZE_WORDS = ["really", "important", "please", "must", "urgent"]
DEFAULT_EMOTION = "neutral"
RATE = "100%"

# Early chunking disabled - we respect LLM's sentence boundaries only
EARLY_CHUNKING_ENABLED = False
PITCH = "100%"

# Sentence-based TTS: Use LLM sentence tags to control TTS boundaries
# This ensures sentences are spoken as complete units, preserving meaning
SENTENCE_BASED_TTS_ENABLED = True  # Enable sentence-based TTS (respects LLM sentence boundaries)

# Note: detect_output_device() is called at module load time above
# These functions use the pre-detected OUTPUT_CARD_INDEX

# === Set playback volume ===
def set_volume_once():
    """Set volume once on startup (for backward compatibility)"""
    global VOLUME_SET
    if not VOLUME_SET:
        set_volume()

def find_pulseaudio_sink_for_device(device_name="UACDemoV1.0"):
    """Find PulseAudio sink that matches the specified device name (output device, not input)"""
    try:
        # Get detailed sink information
        result = subprocess.run(
            ["pactl", "list", "sinks"],
            capture_output=True, text=True, check=True, timeout=2
        )
        
        current_sink = None
        sink_info_lines = []
        in_sink_block = False
        
        # Parse sink information - look for device name in sink description or properties
        for line in result.stdout.splitlines():
            line_stripped = line.strip()
            
            # Start of a new sink block
            if line_stripped.startswith("Sink #"):
                # Process previous sink block if we have one
                if in_sink_block and current_sink and sink_info_lines:
                    # Check if this sink matches our device (and is not the microphone)
                    sink_info_text = " ".join(sink_info_lines)
                    # Look for device name but exclude microphone/input devices
                    if device_name in sink_info_text and "XVF3800" not in sink_info_text:
                        return current_sink
                    # Also check for "UACDemo" (partial match) and exclude microphone
                    if "UACDemo" in sink_info_text and "XVF3800" not in sink_info_text:
                        return current_sink
                
                # Start new sink block
                in_sink_block = True
                current_sink = None
                sink_info_lines = []
            elif line_stripped.startswith("Name:"):
                current_sink = line_stripped.split(":", 1)[1].strip()
                sink_info_lines.append(line_stripped)
            elif in_sink_block and current_sink:
                sink_info_lines.append(line_stripped)
        
        # Process last sink block
        if in_sink_block and current_sink and sink_info_lines:
            sink_info_text = " ".join(sink_info_lines)
            if device_name in sink_info_text and "XVF3800" not in sink_info_text:
                return current_sink
            if "UACDemo" in sink_info_text and "XVF3800" not in sink_info_text:
                return current_sink
        
        # If not found by device name, try to find by ALSA card number (but exclude microphone)
        if OUTPUT_CARD_INDEX is not None:
            result = subprocess.run(
                ["pactl", "list", "sinks"],
                capture_output=True, text=True, check=True, timeout=2
            )
            
            current_sink = None
            sink_info_lines = []
            in_sink_block = False
            
            for line in result.stdout.splitlines():
                line_stripped = line.strip()
                
                if line_stripped.startswith("Sink #"):
                    if in_sink_block and current_sink and sink_info_lines:
                        sink_info_text = " ".join(sink_info_lines)
                        # Look for card number but exclude microphone
                        if (f"card {OUTPUT_CARD_INDEX}" in sink_info_text.lower() or 
                            f"card{OUTPUT_CARD_INDEX}" in sink_info_text.lower()) and "XVF3800" not in sink_info_text:
                            return current_sink
                    
                    in_sink_block = True
                    current_sink = None
                    sink_info_lines = []
                elif line_stripped.startswith("Name:"):
                    current_sink = line_stripped.split(":", 1)[1].strip()
                    sink_info_lines.append(line_stripped)
                elif in_sink_block and current_sink:
                    sink_info_lines.append(line_stripped)
            
            # Process last sink
            if in_sink_block and current_sink and sink_info_lines:
                sink_info_text = " ".join(sink_info_lines)
                if (f"card {OUTPUT_CARD_INDEX}" in sink_info_text.lower() or 
                    f"card{OUTPUT_CARD_INDEX}" in sink_info_text.lower()) and "XVF3800" not in sink_info_text:
                    return current_sink
        
        return None
    except Exception as e:
        print(f"[Speaker] ⚠️ Error finding PulseAudio sink: {e}")
        return None

def set_volume():
    """Set TTS volume - supports both ALSA and PulseAudio/PipeWire. Can be called multiple times."""
    global VOLUME_SET
    
    # Try PulseAudio/PipeWire first (modern systems)
    try:
        # Find the correct sink for UACDemoV1.0 (output device), not the default sink
        sink_name = find_pulseaudio_sink_for_device("UACDemoV1.0")
        
        if not sink_name:
            # Fallback: try to find any sink with UACDemo in the name
            sink_name = find_pulseaudio_sink_for_device("UACDemo")
        
        if not sink_name:
            # Last resort: use default sink (may be wrong device, but better than nothing)
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, check=True, timeout=2
            )
            sink_name = result.stdout.strip()
            if sink_name:
                print(f"[Speaker] ⚠️ Using default PulseAudio sink (may not be correct device): {sink_name}")
        
        if sink_name:
            # Set volume to TTS_VOLUME%
            result = subprocess.run(
                ["pactl", "set-sink-volume", sink_name, f"{TTS_VOLUME}%"],
                capture_output=True, text=True, check=True, timeout=2
            )
            print(f"[Speaker] 🔊 Volume set to {TTS_VOLUME}% via PulseAudio (sink: {sink_name})")
            VOLUME_SET = True
            return
        else:
            print(f"[Speaker] ⚠️ Could not find PulseAudio sink for UACDemoV1.0")
    except FileNotFoundError:
        print(f"[Speaker] ⚠️ pactl not found - PulseAudio not available")
    except subprocess.CalledProcessError as e:
        print(f"[Speaker] ⚠️ PulseAudio command failed: {e.stderr if e.stderr else e}")
    except subprocess.TimeoutExpired:
        print(f"[Speaker] ⚠️ PulseAudio command timed out")
    except Exception as e:
        print(f"[Speaker] ⚠️ PulseAudio error: {e}")
    
    # Fallback to ALSA
    if OUTPUT_CARD_INDEX is not None:
        for ctrl in ALSA_CONTROLS:
            try:
                subprocess.run(
                    ["amixer", "-c", str(OUTPUT_CARD_INDEX), "sset", ctrl, f"{TTS_VOLUME}%"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=2
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
        # Add pauses for numbered list items (e.g., "1.", "2.", "3.")
        # First, add a pause BEFORE the number to separate it from preceding text
        # This makes "Albert Soler 2." sound like "Albert Soler [pause] 2."
        # Handles both cases: "text 2." and "text: 2." (after spacing fix)
        text = re.sub(r"(\s+)(\d+\.)", r"\1<break time='500ms'/>\2", text)
        # Also handle numbers that appear right after colons/semicolons (if spacing fix missed them)
        text = re.sub(r"([:;])(\d+\.)", r"\1<break time='500ms'/>\2", text)
        # Then add a longer pause AFTER the number for natural list pacing
        text = re.sub(r"(\d+\.)", r"\1<break time='900ms'/>", text)
        # Add standard pauses for other sentence endings
        # Only add breaks to periods that don't already have a break immediately after them
        text = re.sub(r"([.?!])(?!<break)", r"\1<break time='600ms'/>", text)
        # Clean up: if we accidentally added both breaks, keep only the longer one
        text = re.sub(r"<break time='600ms'/><break time='900ms'/>", r"<break time='900ms'/>", text)
        text = re.sub(r"<break time='900ms'/><break time='600ms'/>", r"<break time='900ms'/>", text)
        # Clean up: remove duplicate breaks before numbers
        text = re.sub(r"<break time='500ms'/><break time='500ms'/>", r"<break time='500ms'/>", text)
    for word in EMPHASIZE_WORDS:
        text = re.sub(rf"\b({word})\b", r"<emphasis>\1</emphasis>", text, flags=re.IGNORECASE)
    emotion = detect_emotion(text)
    print(f"[Speaker] 🎭 Detected emotion: {emotion}")
    return (
        f"<speak><voice emotion='{emotion}'>"
        f"<prosody rate='{RATE}' pitch='{PITCH}'>{text}</prosody>"
        f"</voice></speak>"
    )

# Import shared text cleaning utility
try:
    from utils.text_cleaning import clean_text_formatting
    _clean_text_for_tts = clean_text_formatting  # Alias for backwards compatibility
except ImportError:
    # Fallback if utils module not available
    def _clean_text_for_tts(text):
        """
        Clean and normalize text for TTS playback.
        Removes markdown formatting, fixes spacing, removes hashtags and asterisks.
        """
        if not text:
            return text
        
        # Remove markdown headers (hashtags at start of line)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove standalone hashtags
        text = re.sub(r'#{1,6}(?=\s|$)', '', text)
        
        # Remove markdown bold/italic (asterisks)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
        text = re.sub(r'\*([^*\n]+)\*', r'\1', text)  # *text* -> text
        # Remove standalone asterisks (markdown formatting artifacts)
        text = re.sub(r'\*\*+', '', text)  # Remove multiple asterisks
        text = re.sub(r'(?<!\w)\*(?!\w)', '', text)  # Remove single asterisks not part of words
        
        # Fix missing spaces after punctuation
        text = re.sub(r'([a-zA-Z0-9])([.!?])([a-zA-Z-])', r'\1\2 \3', text)  # word.word -> word. word
        text = re.sub(r'([,.!?:;])([a-zA-Z])', r'\1 \2', text)  # word,word -> word, word
        text = re.sub(r'([a-zA-Z0-9])(\()', r'\1 \2', text)  # word(word -> word (word
        text = re.sub(r'(\))([a-zA-Z0-9])', r'\1 \2', text)  # word)word -> word) word
        
        # Fix missing spaces before numbered list items (e.g., "text1." -> "text 1.")
        # Matches patterns like "text1.", "text2.", "are:1.", etc.
        text = re.sub(r'([a-zA-Z0-9:;,])(\d+\.)', r'\1 \2', text)  # text1. -> text 1.
        
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

def preprocess_for_tts(text):
    # Remove control tags for clean TTS output
    text = re.sub(r"<sentence_start>|<sentence_end>|<pause>", "", text)
    # Apply text cleaning to remove markdown and fix spacing
    text = _clean_text_for_tts(text)
    return text.strip()

def enqueue_tts_chunk(text, bypass_batching=False):
    """
    Enqueue TTS chunk with optional batching to reduce API calls.
    If batching is enabled, accumulates chunks until threshold or timeout.
    
    Args:
        text: Text to send to TTS
        bypass_batching: If True, send immediately without batching (for continuous streaming)
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
    
    # Bypass batching for continuous streaming chunks (they should play immediately)
    if bypass_batching:
        SENTENCE_QUEUE.put(text)
        return
    
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

# === TTS audio generation (supports both engines) ===
def _is_network_error(exception):
    """Check if exception is a network/DNS error that might be retryable"""
    error_str = str(exception).lower()
    error_type = type(exception).__name__
    
    # Check for DNS resolution failures
    if "temporary failure in name resolution" in error_str or "name resolution" in error_str:
        return True
    if "errno -3" in error_str or "errno -2" in error_str:  # DNS errors
        return True
    
    # Check for connection errors
    if "connect" in error_type.lower() or "connection" in error_str:
        return True
    if "timeout" in error_str or "timed out" in error_str:
        return True
    
    # Check for network-related httpx/httpcore errors
    if "httpx" in error_type.lower() or "httpcore" in error_type.lower():
        if "connect" in error_str or "network" in error_str or "dns" in error_str:
            return True
    
    return False

def _generate_tts_audio(text):
    """
    Generate TTS audio using the selected engine with fallback.
    Returns a generator that yields audio chunks (bytes).
    """
    tts_engine = get_tts_engine()
    use_chatterbox = (tts_engine == "chatterbox")
    
    # Try primary engine first
    if use_chatterbox:
        try:
            chatterbox = _get_chatterbox_tts()
            print(f"[Speaker] 🎙️ Using ChatterboxTTS")
            # ChatterboxTTS may not support SSML, so use plain normalized text
            # Remove SSML tags if present and use clean text
            clean_text = normalize_units(text)
            # Remove any SSML tags that might have been added
            clean_text = re.sub(r'<[^>]+>', '', clean_text)
            
            # Use voice cloning if enabled and a reference sample is configured
            voice_cloning_enabled = get_chatterbox_voice_cloning_enabled()
            global CHATTERBOX_VOICE_SAMPLE, _chatterbox_voice_embedding
            if voice_cloning_enabled and CHATTERBOX_VOICE_SAMPLE and os.path.exists(CHATTERBOX_VOICE_SAMPLE):
                # Try to use cached voice embedding first (fastest - no real-time processing)
                voice_embedding = _get_or_create_voice_embedding(chatterbox, CHATTERBOX_VOICE_SAMPLE)
                
                try:
                    # Method 1: Use cached embedding if available (lowest latency)
                    if voice_embedding is not None:
                        print(f"[Speaker] 🎭 Using cached voice embedding (low latency, ~100-150ms)")
                        # Try to use embedding directly (if API supports it)
                        if hasattr(chatterbox, 'generate'):
                            sig = inspect.signature(chatterbox.generate)
                            params = list(sig.parameters.keys())
                            if 'voice_embedding' in params or 'embedding' in params:
                                param_name = 'voice_embedding' if 'voice_embedding' in params else 'embedding'
                                audio = chatterbox.generate(clean_text, **{param_name: voice_embedding, 'exaggeration': 0.6})
                            else:
                                # Embedding parameter not available, fall through to audio_prompt_path
                                voice_embedding = None
                        elif hasattr(chatterbox, 'synthesize'):
                            sig = inspect.signature(chatterbox.synthesize)
                            params = list(sig.parameters.keys())
                            if 'voice_embedding' in params or 'embedding' in params:
                                param_name = 'voice_embedding' if 'voice_embedding' in params else 'embedding'
                                audio = chatterbox.synthesize(clean_text, **{param_name: voice_embedding})
                            else:
                                # Embedding parameter not available, fall through to audio_prompt_path
                                voice_embedding = None
                        else:
                            voice_embedding = None
                    
                    # Method 2: Use audio_prompt_path (real-time cloning, adds latency)
                    if voice_embedding is None:
                        print(f"[Speaker] 🎭 Using voice cloning from: {CHATTERBOX_VOICE_SAMPLE} (adds ~50-100ms latency)")
                        if hasattr(chatterbox, 'generate'):
                            audio = chatterbox.generate(
                                clean_text,
                                audio_prompt_path=CHATTERBOX_VOICE_SAMPLE,
                                exaggeration=0.6  # Emotion intensity (0.3-0.7)
                            )
                        elif hasattr(chatterbox, 'synthesize'):
                            sig = inspect.signature(chatterbox.synthesize)
                            if 'audio_prompt_path' in sig.parameters:
                                audio = chatterbox.synthesize(
                                    clean_text,
                                    audio_prompt_path=CHATTERBOX_VOICE_SAMPLE
                                )
                            else:
                                # synthesize doesn't support voice cloning
                                print(f"[Speaker] ⚠️ Voice cloning not supported by this API, using default voice")
                                audio = chatterbox.synthesize(clean_text)
                        else:
                            # No known synthesis method
                            raise AttributeError("No synthesis method found")
                except Exception as e:
                    # If voice cloning fails, fall back to default voice
                    print(f"[Speaker] ⚠️ Voice cloning failed ({e}), using default voice")
                    if hasattr(chatterbox, 'generate'):
                        audio = chatterbox.generate(clean_text)
                    else:
                        audio = chatterbox.synthesize(clean_text)
            else:
                # No voice cloning - use default voice
                if hasattr(chatterbox, 'generate'):
                    audio = chatterbox.generate(clean_text)
                else:
                    audio = chatterbox.synthesize(clean_text)
            
            # Convert to bytes if needed (Chatterbox may return numpy array)
            if isinstance(audio, np.ndarray):
                # Handle different dtypes
                if audio.dtype == np.float32 or audio.dtype == np.float64:
                    # Normalize float audio to int16 range (-1.0 to 1.0 -> -32768 to 32767)
                    # Clamp to prevent clipping
                    audio = np.clip(audio, -1.0, 1.0)
                    audio = (audio * 32767).astype(np.int16)
                elif audio.dtype != np.int16:
                    # Convert other integer types to int16
                    audio = audio.astype(np.int16)
                
                # Check if resampling is needed (Chatterbox might use different sample rate)
                # For now, assume Chatterbox uses 22050 Hz (same as ElevenLabs PCM format)
                # If different, we'd need to resample here
                audio_bytes = audio.tobytes()
            elif isinstance(audio, bytes):
                audio_bytes = audio
            else:
                # Try to convert to bytes
                audio_bytes = bytes(audio)
            
            # Yield audio in chunks (simulate streaming for compatibility)
            chunk_size = 4096  # 4KB chunks
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]
            return
        except Exception as e:
            print(f"[Speaker] ⚠️ ChatterboxTTS failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Speaker] 🔄 Falling back to ElevenLabs...")
            # Fall through to ElevenLabs
    
    # Use ElevenLabs (either as primary or fallback) with retry logic for network errors
    max_retries = 3
    initial_delay = 1.0  # Start with 1 second
    max_delay = 10.0  # Cap at 10 seconds
    
    for attempt in range(max_retries):
        try:
            client = _get_elevenlabs_client()
            if attempt == 0:
                print(f"[Speaker] 🎙️ Using ElevenLabs")
            else:
                print(f"[Speaker] 🔄 Retrying ElevenLabs (attempt {attempt + 1}/{max_retries})")
            
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
            # Successfully got stream - yield chunks
            for chunk in stream:
                if chunk:
                    yield chunk
            return  # Success - exit retry loop
            
        except Exception as e:
            is_network_err = _is_network_error(e)
            error_msg = str(e)
            
            if is_network_err and attempt < max_retries - 1:
                # Calculate exponential backoff delay
                delay = min(initial_delay * (2 ** attempt), max_delay)
                print(f"[Speaker] ⚠️ Network error (DNS/connection failure): {error_msg}")
                print(f"[Speaker] 🔄 Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                print(f"[Speaker] 💡 Check network connectivity and DNS configuration")
                time.sleep(delay)
                continue  # Retry
            else:
                # Not a network error, or all retries exhausted
                print(f"[Speaker] ❌ ElevenLabs TTS failed: {e}")
                import traceback
                traceback.print_exc()
                
                # If it's a network error after all retries, try ChatterboxTTS as fallback
                if is_network_err and not use_chatterbox:
                    print(f"[Speaker] 🔄 Network connectivity issue - trying ChatterboxTTS as fallback...")
                    try:
                        chatterbox = _get_chatterbox_tts()
                        print(f"[Speaker] 🎙️ Using ChatterboxTTS (offline fallback)")
                        clean_text = normalize_units(text)
                        clean_text = re.sub(r'<[^>]+>', '', clean_text)
                        
                        if hasattr(chatterbox, 'generate'):
                            audio = chatterbox.generate(clean_text)
                        else:
                            audio = chatterbox.synthesize(clean_text)
                        
                        # Convert to bytes
                        if isinstance(audio, np.ndarray):
                            if audio.dtype == np.float32 or audio.dtype == np.float64:
                                audio = np.clip(audio, -1.0, 1.0)
                                audio = (audio * 32767).astype(np.int16)
                            elif audio.dtype != np.int16:
                                audio = audio.astype(np.int16)
                            audio_bytes = audio.tobytes()
                        elif isinstance(audio, bytes):
                            audio_bytes = audio
                        else:
                            audio_bytes = bytes(audio)
                        
                        # Yield audio in chunks
                        chunk_size = 4096
                        for i in range(0, len(audio_bytes), chunk_size):
                            yield audio_bytes[i:i + chunk_size]
                        return  # Success with fallback
                    except Exception as fallback_error:
                        print(f"[Speaker] ❌ ChatterboxTTS fallback also failed: {fallback_error}")
                        # Fall through to raise original error
                
                # If Chatterbox was used as primary and also failed, report both failures
                if use_chatterbox:
                    raise RuntimeError(f"Both TTS engines failed. Chatterbox: already tried as primary, ElevenLabs: {e}")
                else:
                    if is_network_err:
                        raise RuntimeError(
                            f"ElevenLabs TTS failed after {max_retries} retries due to network error: {e}\n"
                            f"💡 Check network connectivity, DNS configuration, and internet connection"
                        )
                    else:
                        raise RuntimeError(f"ElevenLabs TTS failed: {e}")

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
            tts_api_start_time = time.time()
            stream = _generate_tts_audio(text)
            first_chunk = next(stream, None)
            tts_api_latency = time.time() - tts_api_start_time
            if tts_api_latency > 0.5:  # Only log if significant
                print(f"[Speaker] ⏱️ ElevenLabs API latency: {tts_api_latency:.3f}s (time to first chunk)")
            if not first_chunk:
                raise RuntimeError("No audio received")

            # Use detected output device (UACDemoV1.0) with plug plugin for automatic format conversion
            # This ensures proper conversion from TTS mono 22050 Hz to device's native format
            # The plug plugin handles all format conversions automatically
            if OUTPUT_CARD_INDEX is not None:
                # Use detected output device (UACDemoV1.0)
                alsa_device = f"plughw:{OUTPUT_CARD_INDEX},0"
                print(f"[Speaker] 🔊 Playing TTS on detected device: {OUTPUT_DEVICE_NAME} (card {OUTPUT_CARD_INDEX})")
            else:
                # Fallback to default if device not detected
                alsa_device = "plug:default"
                print(f"[Speaker] 🔊 Playing TTS on default device (output device not detected)")
            
            proc = subprocess.Popen(
                ["aplay", "-D", alsa_device, "-f", "S16_LE", "-r", str(PCM_SAMPLE_RATE), "-c", "1"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE  # Capture stderr for diagnostics
            )

            # Give aplay a brief moment to initialize and catch immediate failures
            time.sleep(0.01)
            
            # Verify process is still running after startup
            if proc.poll() is not None:
                # Process already terminated - get error details
                aplay_stderr = None
                if proc.stderr:
                    try:
                        aplay_stderr = proc.stderr.read().decode().strip()
                    except Exception:
                        pass
                error_msg = f"aplay process terminated immediately (exit code: {proc.returncode})"
                if aplay_stderr:
                    error_msg += f": {aplay_stderr}"
                raise BrokenPipeError(error_msg)
            
            # Validate first chunk
            if not first_chunk or len(first_chunk) == 0:
                raise ValueError("First audio chunk is empty")
            
            # Calculate TTS latency from transcription end to TTS initiation
            tts_latency = time.time() - tts_start_time
            print(f"⏱️ TTS latency: {tts_latency:.2f}s")
            
            # Write chunks directly (ALSA plug plugin handles mono->stereo and sample rate conversion)
            try:
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
                # Get stderr for diagnostics
                aplay_stderr = None
                if proc and proc.stderr:
                    try:
                        aplay_stderr = proc.stderr.read().decode().strip()
                    except Exception as stderr_err:
                        print(f"[Speaker] ⚠️ Could not read aplay stderr: {stderr_err}")
                
                # Get process return code for diagnostics
                exit_code = None
                if proc:
                    try:
                        exit_code = proc.poll()
                        if exit_code is None:
                            # Process still running, try to wait for it
                            try:
                                exit_code = proc.wait(timeout=1)
                            except (subprocess.TimeoutExpired, TypeError):
                                # Force kill if it's hanging
                                proc.kill()
                                exit_code = proc.wait()
                    except Exception as poll_err:
                        print(f"[Speaker] ⚠️ Could not get exit code: {poll_err}")
                
                # Print detailed diagnostics
                if aplay_stderr:
                    print(f"[Speaker] 🔍 aplay error details: {aplay_stderr}")
                if exit_code is not None:
                    print(f"[Speaker] 🔍 aplay exit code: {exit_code}")
                if not aplay_stderr and exit_code is not None:
                    # Common aplay exit code meanings
                    if exit_code == 1:
                        print(f"[Speaker] 💡 aplay exit code 1 usually means: invalid format/device/permissions or audio data issue")
                    elif exit_code == 2:
                        print(f"[Speaker] 💡 aplay exit code 2 usually means: device busy or unavailable")
                
                # Clean up process
                if proc and proc.poll() is None:
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
                
                # Don't re-raise - just log the error
                print(f"[Speaker] ⚠️ TTS playback failed - continuing...")
            except Exception as e:
                print(f"[Speaker] ❌ Unexpected TTS error: {e}")
                import traceback
                traceback.print_exc()
                # Try to get diagnostics
                if proc:
                    try:
                        if proc.stderr:
                            stderr_output = proc.stderr.read().decode().strip()
                            if stderr_output:
                                print(f"[Speaker] 🔍 aplay stderr: {stderr_output}")
                        exit_code = proc.poll()
                        if exit_code is None:
                            proc.kill()
                            exit_code = proc.wait()
                        print(f"[Speaker] 🔍 aplay exit code: {exit_code}")
                    except Exception as cleanup_err:
                        print(f"[Speaker] ⚠️ Error during cleanup: {cleanup_err}")
                    finally:
                        if proc.poll() is None:
                            try:
                                proc.kill()
                                proc.wait()
                            except Exception:
                                pass

        except Exception as e:
            print(f"[Speaker] ❌ TTS error: {e}")
        finally:
            set_playing(False)

# === Playback loop ===
def playback_loop():
    set_volume_once()
    while True:
        queue_get_time = time.time()
        sentence = SENTENCE_QUEUE.get()
        sentence = preprocess_for_tts(sentence)
        if not sentence or sentence.lower() in {"uh", "hmm", "um", "<silence>"}:
            print(f"[Speaker] ⚠️ Skipping filler: \"{sentence}\"")
            continue
        print(f"[Speaker] 🔈 Speaking: \"{sentence}\"")
        # Use LLM request start time if available, otherwise use current time
        global _llm_request_start_time, _sentence_enqueue_time
        tts_start_time = _llm_request_start_time if _llm_request_start_time is not None else time.time()
        # Reset after first use to avoid using stale time for subsequent chunks
        if _llm_request_start_time is not None:
            _llm_request_start_time = None
        
        # Track queue processing latency
        if _sentence_enqueue_time is not None:
            queue_latency = queue_get_time - _sentence_enqueue_time
            if queue_latency > 0.1:  # Only log if significant
                print(f"[Speaker] ⏱️ Queue processing latency: {queue_latency:.3f}s")
            _sentence_enqueue_time = None
        
        threading.Thread(target=tts_playback_thread, args=(sentence, tts_start_time), daemon=True).start()
        # Removed sleep to reduce latency - threads are daemon so they won't block

# === Stream LLM output ===
def speak_llm_response(prompt, context=""):
    global pending_initials, _batch_started
    import requests
    print(f"[LLM] ✅ Prompt to LLM: {prompt}")
    
    # Reset batch tracking for new response
    _batch_started = False
    
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
            timeout=30  # Safety net for RAG queries (with truncation, responses should be much faster)
        )

    try:
        response = _post_stream(primary_port)
        if response.status_code != 200:
            raise RuntimeError(f"LLM HTTP {response.status_code} on port {primary_port}")
        # Process streaming tokens - ONLY using sentence tags, NO fallbacks
        # IMPROVED: Intelligently batch sentences based on semantic relationships and structure
        sentence_buffer = []  # Buffer for current sentence (between sentence_start and sentence_end)
        in_sentence = False  # Track if we're inside a sentence block
        sentence_batch = []  # Buffer for batching related sentences together
        last_sentence_text = None  # Track last sentence sent to TTS (to detect questions)
        MIN_SENTENCE_WORDS = 5  # Sentences with fewer words will be batched
        MIN_SENTENCE_CHARS = 20  # Sentences with fewer chars will be batched
        MAX_BATCH_SIZE = 5  # Increased from 3 for better grouping of related content
        MAX_BATCH_WORDS = 40  # Maximum words in a batch before forcing flush
        
        def _is_empty_sentence(text):
            """Check if sentence is empty or just whitespace/punctuation"""
            if not text:
                return True
            # Remove all whitespace first
            text_no_ws = re.sub(r'\s+', '', text)
            if not text_no_ws:
                return True
            # Remove markdown formatting and check if anything meaningful remains
            text_clean = re.sub(r'\*\*|\*|_|`|#', '', text_no_ws)
            # Remove punctuation-only content (including dashes, colons, etc.)
            text_clean = re.sub(r'^[\-:;,.!?()\[\]{}]+$', '', text_clean)
            # Check if anything meaningful remains
            return not text_clean or len(text_clean.strip()) == 0
        
        def _is_incomplete_sentence(text):
            """Detect if sentence appears incomplete and should be grouped with next sentence"""
            if not text:
                return False
            text_stripped = text.strip()
            # Ends with opening parenthesis (likely continuation, e.g., "Piperacillin-Tazobactam (3.")
            if text_stripped.endswith('('):
                return True
            # Ends with colon (likely introducing a list or continuation)
            if text_stripped.endswith(':'):
                return True
            # Ends with a number followed by period (likely part of a dosage, e.g., "3." or "(3.")
            # This catches patterns like "-Piperacillin-Tazobactam (3."
            if re.search(r'\(?\d+\.\s*$', text_stripped):
                return True
            # Ends with incomplete dosage pattern without closing paren
            # Check for dosage units at the end that suggest continuation (but not complete sentences)
            if not text_stripped.endswith(('.', '!', '?', ')')):
                # Pattern: ends with dosage unit but no period before it (incomplete)
                if re.search(r'\b(mg|g|kg|ml|IV|PO|Q\d+|q\d+)\s*$', text_stripped, re.IGNORECASE):
                    # Exclude if it's a complete sentence (has period before the unit)
                    if not re.search(r'\.\s+(mg|g|kg|ml|IV|PO)', text_stripped, re.IGNORECASE):
                        return True
            # Very short sentences that look like fragments (less than 15 chars, no proper ending)
            if len(text_stripped) < 15 and not text_stripped.endswith(('.', '!', '?', ')')):
                return True
            return False
        
        def _is_list_item(text):
            """Detect if sentence is a list item (bullet, numbered, or dash-prefixed)"""
            if not text:
                return False
            text_stripped = text.strip()
            # Starts with bullet, dash, or number pattern
            if re.match(r'^[-•*]\s+', text_stripped):
                return True
            if re.match(r'^\d+[.)]\s+', text_stripped):
                return True
            # Starts with medication/dosage pattern (common in medical lists)
            if re.match(r'^[A-Z][a-z]+(-[A-Z][a-z]+)*\s+', text_stripped):
                # Check if it contains dosage info
                if re.search(r'\d+\s*(mg|g|kg|ml|IV|PO)', text_stripped, re.IGNORECASE):
                    return True
            return False
        
        def _should_batch_with_previous(prev_text, current_text):
            """Determine if current sentence should be batched with previous based on semantic relationship"""
            if not prev_text or not current_text:
                return False
            
            prev_stripped = prev_text.strip()
            current_stripped = current_text.strip()
            
            # If previous sentence is incomplete, definitely batch
            if _is_incomplete_sentence(prev_text):
                return True
            
            # Medical dosage continuation patterns
            # Previous ends with number and period (e.g., "3."), current starts with number (e.g., "375 mg")
            # This catches cases like "-Piperacillin-Tazobactam (3." followed by "375 mg Q6H IV)"
            if re.search(r'\d+\.\s*$', prev_stripped):
                if re.match(r'\d+', current_stripped):
                    return True
            
            # Previous ends with opening paren, current continues (e.g., "(" followed by "375 mg")
            if prev_stripped.endswith('('):
                return True
            
            # If both are list items, batch them
            if _is_list_item(prev_text) and _is_list_item(current_text):
                return True
            
            # If previous ends with colon and current is a list item, batch
            if prev_stripped.endswith(':') and _is_list_item(current_text):
                return True
            
            # Medical dosage continuation: Previous ends with medication name, current starts with dosage
            if re.search(r'[A-Z][a-z]+(-[A-Z][a-z]+)*\s*$', prev_stripped):
                if re.match(r'\(?\d+', current_stripped):
                    return True
            
            # Both are very short (likely related fragments)
            if len(prev_stripped) < 15 and len(current_stripped) < 15:
                return True
            
            return False
        
        def _is_short_sentence(text):
            """Check if sentence is short enough to batch"""
            # Don't batch if it's empty
            if _is_empty_sentence(text):
                return False
            word_count = len(text.split())
            char_count = len(text)
            return word_count < MIN_SENTENCE_WORDS and char_count < MIN_SENTENCE_CHARS
        
        def _should_flush_batch(current_text=None):
            """Determine if current batch should be flushed"""
            if not sentence_batch:
                return False
            
            # Count total words in batch
            total_words = sum(len(s.split()) for s in sentence_batch)
            if current_text:
                total_words += len(current_text.split())
            
            # Flush if batch is getting too large
            if total_words >= MAX_BATCH_WORDS:
                return True
            
            # Flush if we have too many sentences
            if len(sentence_batch) >= MAX_BATCH_SIZE:
                return True
            
            return False
        
        def _flush_sentence_batch():
            """Send batched sentences as one chunk"""
            nonlocal last_sentence_text
            if sentence_batch:
                combined = " ".join(sentence_batch).strip()
                if combined and not _is_empty_sentence(combined):
                    print(f"[Speaker] 📦 Batching {len(sentence_batch)} related sentences: '{combined[:80]}...'")
                    enqueue_tts_chunk(combined)
                    # Track last batched text (for question detection)
                    last_sentence_text = combined
                sentence_batch.clear()
        
        for line in response.iter_lines(decode_unicode=True):
            token = line.rstrip("\r\n")
            if not token:
                continue

            # Debug: Log all control tags
            if token.startswith('<') and token.endswith('>'):
                print(f"[Speaker] 🏷️  Control tag received: '{token}'")

            # Handle sentence control markers - REQUIRED, no fallback
            if token == '<sentence_start>':
                # Start of new sentence - reset buffer and wait for tokens
                sentence_buffer = []
                in_sentence = True
                print(f"[Speaker] 🎬 <sentence_start> detected - buffering tokens until <sentence_end>")
                continue
            elif token == '<sentence_end>':
                # Send remaining buffer to TTS when sentence_end tag is received
                sentence_end_received_time = time.time()
                if sentence_buffer:
                    # Join all buffered tokens into complete sentence
                    chunk_text = "".join(sentence_buffer).strip()
                    clean_text = re.sub(r'<sentence_start>|<sentence_end>', '', chunk_text).strip()
                    # Apply text cleaning to remove markdown and fix spacing
                    clean_text = _clean_text_for_tts(clean_text)
                    # Normalize whitespace (collapse multiple spaces to single space)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    
                    if clean_text and not _is_empty_sentence(clean_text):
                        # OPTIMIZED: Send each sentence immediately when <sentence_end> is received
                        # This allows TTS to start playing while LLM continues generating next sentence
                        # No batching - each sentence is sent as soon as it's complete for lowest latency
                        
                        # Flush any pending batch first (maintain order)
                        if sentence_batch:
                            _flush_sentence_batch()
                        
                        # Send this sentence immediately - don't wait for LLM to finish generating
                        print(f"[Speaker] 🎙️ <sentence_end> received - sending sentence to TTS immediately: '{clean_text[:60]}...'")
                        enqueue_tts_chunk(clean_text)
                        
                        # Track last sentence sent to TTS (for question detection)
                        last_sentence_text = clean_text
                        
                        # Store time when first sentence was enqueued for latency tracking
                        global _sentence_enqueue_time
                        if _sentence_enqueue_time is None:
                            _sentence_enqueue_time = sentence_end_received_time
                    else:
                        # Empty or whitespace-only sentence - skip it
                        if clean_text:
                            print(f"[Speaker] ⏭️  Skipping empty/whitespace sentence: '{clean_text[:50]}'")
                else:
                    print(f"[Speaker] ⚠️ <sentence_end> received but sentence_buffer is empty")
                # Reset state for next sentence
                sentence_buffer = []
                in_sentence = False
                continue

            # Only process tokens if we're in a sentence block (tags required)
            if not in_sentence:
                print(f"[Speaker] ⚠️ Token '{token}' received outside sentence block - IGNORING (waiting for <sentence_start>)")
                continue

            # Accumulate tokens in buffer - send to TTS only when <sentence_end> is received
            # This ensures sentences are spoken as complete units, preserving meaning
            print(f"[LLM] 🧠 {token}")
            sentence_buffer.append(token)
        
        # Flush any remaining batched sentences at end of stream
        _flush_sentence_batch()
        
        # Check if last sentence ended with a question mark (for natural conversation flow)
        # This allows VAD to remain active so user can respond without wake word
        if last_sentence_text:
            # Check if last sentence ends with "?" (after stripping whitespace and markdown)
            last_sentence_stripped = last_sentence_text.strip()
            # Remove trailing markdown/punctuation artifacts and check for question mark
            last_char = last_sentence_stripped[-1] if last_sentence_stripped else ""
            ended_with_question = last_char == "?"
            
            # Update state to track if response ended with question
            try:
                from state import set_last_response_ended_with_question
                set_last_response_ended_with_question(ended_with_question)
                if ended_with_question:
                    print(f"[Speaker] ❓ Last response ended with question - VAD will remain active for natural conversation")
            except Exception as e:
                print(f"[Speaker] ⚠️ Failed to set question flag: {e}")
        else:
            # No sentences were sent, clear the flag
            try:
                from state import set_last_response_ended_with_question
                set_last_response_ended_with_question(False)
            except Exception:
                pass
        
        # No fallback: if stream ends without sentence_end tag, tokens are lost (tags are required)
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
