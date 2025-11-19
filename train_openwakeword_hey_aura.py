#!/usr/bin/env python3
"""
Train OpenWakeWord model for "hey aura" with TTS echo handling.

This script trains a custom openwakeword model that can detect "hey aura"
while ignoring TTS echo from the speaker output.

Usage:
    python3 train_openwakeword_hey_aura.py --mode collect    # Collect training data
    python3 train_openwakeword_hey_aura.py --mode train     # Train the model
    python3 train_openwakeword_hey_aura.py --mode full      # Collect + Train
"""

import os
import sys
import argparse
import json
import time
import subprocess
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple
import soundfile as sf
from dotenv import load_dotenv

# Add aura-control to path for imports
workspace_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(workspace_root / "aura-control" / "core"))

# Load environment variables
dotenv_path = workspace_root / ".env"
load_dotenv(dotenv_path)

# Training configuration
WAKE_PHRASE = "hey aura"
SAMPLE_RATE = 16000  # OpenWakeWord uses 16kHz
TRAINING_DATA_DIR = workspace_root / "data" / "wake_word_training"
POSITIVE_DIR = TRAINING_DATA_DIR / "positive"
NEGATIVE_DIR = TRAINING_DATA_DIR / "negative"
TTS_NEGATIVE_DIR = TRAINING_DATA_DIR / "negative_tts"
MODEL_OUTPUT_DIR = workspace_root / "data" / "models" / "wake_words"
MODEL_NAME = "hey_aura_v0.1"

# Create directories
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
TTS_NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import openwakeword
        from openwakeword import Model
    except ImportError:
        missing.append("openwakeword (pip install openwakeword)")
    
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile (pip install soundfile)")
    
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio (pip install pyaudio)")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        return False
    
    print("✅ All dependencies installed")
    return True


def detect_output_device():
    """Detect audio output device (similar to speaker.py)."""
    try:
        output = subprocess.check_output(["aplay", "-l"], text=True)
        # First, try to find UACDemoV1.0
        for line in output.splitlines():
            if "UACDemoV1.0" in line:
                match = re.search(r"card (\d+):", line)
                if match:
                    card_num = int(match.group(1))
                    return f"plughw:{card_num},0"
        
        # Fallback: find any USB audio device with output
        for line in output.splitlines():
            if "USB Audio" in line and ("0 in" in line or "out" in line):
                match = re.search(r"card (\d+):", line)
                if match:
                    card_num = int(match.group(1))
                    return f"plughw:{card_num},0"
        
        # No USB device found - use default with plug plugin
        return "plug:default"
    except Exception as e:
        print(f"⚠️ Failed to detect output device: {e}")
        return "default"


def play_and_record_tts(text: str, device_index: int = None, duration_padding: float = 0.5) -> np.ndarray:
    """
    Generate TTS, play it through speakers, and record it back through microphone.
    This captures real echo/reverb that occurs in actual use.
    """
    import pyaudio
    import threading
    from elevenlabs.client import ElevenLabs
    
    global _selected_device_index
    
    if device_index is None:
        device_index = _selected_device_index
    
    ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
    ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID") or "default"
    
    if not ELEVEN_API_KEY or ELEVEN_API_KEY == "your_elevenlabs_api_key_here":
        raise RuntimeError("ElevenLabs API key not configured")
    
    client = ElevenLabs(api_key=ELEVEN_API_KEY)
    PCM_SAMPLE_RATE = 22050
    PCM_FORMAT = "pcm_22050"
    
    # Generate TTS audio
    stream = client.text_to_speech.convert(
        text=text,
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
    
    # Collect all audio chunks
    audio_chunks = []
    for chunk in stream:
        if chunk:
            audio_chunks.append(chunk)
    
    if not audio_chunks:
        raise RuntimeError("No audio received from TTS")
    
    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(audio_chunks), dtype=np.int16)
    audio_duration = len(audio_data) / PCM_SAMPLE_RATE
    
    # Detect output device
    output_device = detect_output_device()
    
    # Set up recording
    p = pyaudio.PyAudio()
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    recorded_frames = []
    recording_active = threading.Event()
    recording_done = threading.Event()
    
    def record_audio():
        """Record audio in a separate thread."""
        stream = p.open(
            format=format,
            channels=channels,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk
        )
        
        recording_active.set()
        
        # Record for audio duration + padding
        num_chunks = int(SAMPLE_RATE / chunk * (audio_duration + duration_padding))
        
        for _ in range(num_chunks):
            if not recording_active.is_set():
                break
            data = stream.read(chunk, exception_on_overflow=False)
            recorded_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        recording_done.set()
    
    # Start recording thread
    record_thread = threading.Thread(target=record_audio, daemon=True)
    record_thread.start()
    
    # Wait for recording to start
    recording_active.wait(timeout=1.0)
    time.sleep(0.1)  # Small delay to ensure recording is active
    
    # Play TTS through speakers
    try:
        proc = subprocess.Popen(
            ["aplay", "-D", output_device, "-f", "S16_LE", "-r", str(PCM_SAMPLE_RATE), "-c", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Write audio data
        proc.stdin.write(audio_data.tobytes())
        proc.stdin.close()
        
        # Wait for playback to complete
        proc.wait()
    except Exception as e:
        print(f"   ⚠️  Playback error: {e}")
        recording_active.clear()
    
    # Wait a bit for echo to be captured
    time.sleep(duration_padding)
    recording_active.clear()
    
    # Wait for recording to finish
    recording_done.wait(timeout=5.0)
    
    p.terminate()
    
    if not recorded_frames:
        raise RuntimeError("No audio recorded")
    
    # Convert recorded audio to numpy array
    recorded_audio = np.frombuffer(b''.join(recorded_frames), dtype=np.int16)
    recorded_float = recorded_audio.astype(np.float32) / 32768.0
    
    return recorded_float


def generate_tts_samples(text: str, num_samples: int = 10, output_dir: Path = None, play_through_speakers: bool = True) -> List[Path]:
    """
    Generate TTS audio samples using ElevenLabs.
    
    If play_through_speakers=True (recommended):
    - Plays TTS through speakers and records it back through microphone
    - Captures real echo/reverb that occurs in actual use
    - These are NEGATIVE samples (to teach model NOT to trigger on TTS)
    
    If play_through_speakers=False:
    - Just generates TTS audio directly (no echo/reverb)
    - Less realistic but faster
    """
    global _selected_device_index
    
    ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
    ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID") or "default"
    
    if not ELEVEN_API_KEY or ELEVEN_API_KEY == "your_elevenlabs_api_key_here":
        print("⚠️  ElevenLabs API key not configured - skipping TTS sample generation")
        print("   Set ELEVENLABS_API_KEY in .env file")
        return []
    
    output_dir = output_dir or TTS_NEGATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    if play_through_speakers:
        print(f"🔊 Generating {num_samples} TTS samples of '{text}' (NEGATIVE samples)...")
        print(f"   Playing through speakers and recording echo/reverb")
        print(f"   Make sure speakers are on and microphone can hear them!")
        
        # Ensure microphone device is selected
        if _selected_device_index is None:
            _selected_device_index = select_microphone_device()
            if _selected_device_index is None:
                print("⚠️  No microphone device selected - cannot record echo")
                return []
        
        print(f"   Using microphone device index: {_selected_device_index}")
        print(f"   Starting in 2 seconds...")
        time.sleep(2)
    else:
        print(f"🎤 Generating {num_samples} TTS samples of '{text}' (NEGATIVE samples)...")
        print(f"   Direct generation (no echo/reverb)")
    
    for i in range(num_samples):
        try:
            if play_through_speakers:
                # Play through speakers and record echo
                print(f"\n   [{i+1}/{num_samples}] Playing and recording...", end="", flush=True)
                recorded_audio = play_and_record_tts(text, device_index=_selected_device_index)
                print(" ✅")
                
                # Save recorded audio (already at 16kHz from recording)
                output_file = output_dir / f"tts_echo_{i+1:03d}.wav"
                sf.write(str(output_file), recorded_audio, SAMPLE_RATE)
                generated_files.append(output_file)
                
                print(f"      ✅ Saved: {output_file.name}")
                time.sleep(0.5)  # Brief pause between samples
            else:
                # Direct generation (old method)
                from elevenlabs.client import ElevenLabs
                client = ElevenLabs(api_key=ELEVEN_API_KEY)
                PCM_SAMPLE_RATE = 22050
                PCM_FORMAT = "pcm_22050"
                
                stream = client.text_to_speech.convert(
                    text=text,
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
                
                audio_chunks = []
                for chunk in stream:
                    if chunk:
                        audio_chunks.append(chunk)
                
                if not audio_chunks:
                    print(f"   ⚠️  No audio received for sample {i+1}")
                    continue
                
                audio_data = np.frombuffer(b''.join(audio_chunks), dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                from scipy import signal
                num_samples_resampled = int(len(audio_float) * SAMPLE_RATE / PCM_SAMPLE_RATE)
                audio_resampled = signal.resample(audio_float, num_samples_resampled)
                
                output_file = output_dir / f"tts_echo_{i+1:03d}.wav"
                sf.write(str(output_file), audio_resampled, SAMPLE_RATE)
                generated_files.append(output_file)
                print(f"   ✅ Generated: {output_file.name}")
                time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error generating TTS sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Generated {len(generated_files)} TTS samples")
    if play_through_speakers:
        print(f"   These samples include real echo/reverb from your environment")
    return generated_files


def detect_preferred_microphone(p):
    """Auto-detect preferred microphone (reSpeaker, USB audio, etc.)."""
    preferred_keywords = ["reSpeaker", "respeaker", "USB Audio", "USB", "XVF3800", "UAC"]
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            name = info['name'].lower()
            for keyword in preferred_keywords:
                if keyword.lower() in name:
                    print(f"   ✅ Auto-detected: [{i}] {info['name']}")
                    return i
    
    return None


def select_microphone_device():
    """Select microphone device once and save preference."""
    import pyaudio
    
    config_file = TRAINING_DATA_DIR / "device_config.json"
    
    # Try to load saved preference
    saved_device = None
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                saved_device = config.get('device_index')
                saved_device_name = config.get('device_name', 'Unknown')
                print(f"📱 Saved device preference: [{saved_device}] {saved_device_name}")
        except Exception:
            pass
    
    p = pyaudio.PyAudio()
    
    try:
        # List available devices
        print("\n📱 Available audio input devices:")
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append((i, info))
                marker = " ⭐" if saved_device == i else ""
                print(f"   [{i}] {info['name']} ({info['maxInputChannels']} channels){marker}")
        
        if not devices:
            print("   ❌ No input devices found!")
            return None
        
        # Auto-detect preferred device
        auto_device = detect_preferred_microphone(p)
        
        # Use saved device, auto-detected device, or ask user
        device_index = None
        if saved_device is not None:
            # Verify saved device still exists
            if any(d[0] == saved_device for d in devices):
                use_saved = input(f"\nUse saved device [{saved_device}]? (y/n, default=y): ").strip().lower()
                if use_saved != 'n':
                    device_index = saved_device
                    device_name = next(d[1]['name'] for d in devices if d[0] == saved_device)
                    print(f"✅ Using saved device: [{device_index}] {device_name}")
        
        if device_index is None and auto_device is not None:
            use_auto = input(f"\nUse auto-detected device [{auto_device}]? (y/n, default=y): ").strip().lower()
            if use_auto != 'n':
                device_index = auto_device
                device_name = next(d[1]['name'] for d in devices if d[0] == auto_device)
                print(f"✅ Using auto-detected device: [{device_index}] {device_name}")
        
        if device_index is None:
            # Ask user to select
            try:
                device_str = input("\nEnter device index (or press Enter for default): ").strip()
                if device_str:
                    device_index = int(device_str)
                else:
                    device_index = None
                    device_name = "Default"
            except (ValueError, KeyboardInterrupt):
                device_index = None
                device_name = "Default"
        
        # Save preference
        if device_index is not None:
            device_name = next(d[1]['name'] for d in devices if d[0] == device_index)
            config = {
                'device_index': device_index,
                'device_name': device_name,
                'saved_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"💾 Device preference saved to {config_file}")
        
        return device_index
        
    finally:
        p.terminate()


# Global variable to store selected device (set once, reused for all recordings)
_selected_device_index = None


def record_audio_sample(duration: float = 3.0, sample_rate: int = SAMPLE_RATE, device_index: int = None) -> np.ndarray:
    """Record audio from microphone using the selected device."""
    import pyaudio
    
    global _selected_device_index
    
    # Use provided device_index, or the globally selected one
    if device_index is None:
        device_index = _selected_device_index
    
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk
        )
        
        print(f"\n🎤 Recording {duration} seconds... (speak '{WAKE_PHRASE}')")
        print("   Recording...", end="", flush=True)
        
        frames = []
        num_chunks = int(sample_rate / chunk * duration)
        
        for i in range(num_chunks):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            if i % 10 == 0:
                print(".", end="", flush=True)
        
        print("\n✅ Recording complete")
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        return audio_float
        
    finally:
        p.terminate()


def collect_positive_samples(num_samples: int = 20):
    """Collect positive training samples (human speech saying 'hey aura')."""
    global _selected_device_index
    
    print(f"\n{'='*60}")
    print(f"📝 COLLECTING POSITIVE SAMPLES")
    print(f"{'='*60}")
    print(f"Wake phrase: '{WAKE_PHRASE}'")
    print(f"Target: {num_samples} samples")
    print(f"Output directory: {POSITIVE_DIR}")
    print("\n💡 Tips:")
    print("   - Speak naturally, as you would in real use")
    print("   - Vary your tone, speed, and volume")
    print("   - Include samples from different distances")
    print("   - Press Ctrl+C to stop early\n")
    
    # Select device once at the start
    if _selected_device_index is None:
        _selected_device_index = select_microphone_device()
        if _selected_device_index is None:
            print("⚠️  No device selected, using default")
    
    collected = 0
    existing_files = list(POSITIVE_DIR.glob("*.wav"))
    start_index = len(existing_files) + 1
    
    try:
        while collected < num_samples:
            print(f"\n--- Sample {collected + 1}/{num_samples} ---")
            
            # Record audio (device already selected)
            audio = record_audio_sample(duration=2.5, device_index=_selected_device_index)
            
            # Save sample
            output_file = POSITIVE_DIR / f"positive_{start_index + collected:03d}.wav"
            sf.write(str(output_file), audio, SAMPLE_RATE)
            print(f"✅ Saved: {output_file.name}")
            
            collected += 1
            
            # Ask if user wants to continue
            if collected < num_samples:
                try:
                    response = input(f"\nContinue? (y/n, default=y): ").strip().lower()
                    if response == 'n':
                        break
                except KeyboardInterrupt:
                    break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    
    print(f"\n✅ Collected {collected} positive samples")
    print(f"   Total positive samples: {len(list(POSITIVE_DIR.glob('*.wav')))}")


def collect_negative_samples(num_samples: int = 30):
    """Collect negative training samples (other phrases, background noise)."""
    global _selected_device_index
    
    print(f"\n{'='*60}")
    print(f"📝 COLLECTING NEGATIVE SAMPLES")
    print(f"{'='*60}")
    print(f"Target: {num_samples} samples")
    print(f"Output directory: {NEGATIVE_DIR}")
    print("\n💡 Tips:")
    print("   - Say other phrases (NOT 'hey aura')")
    print("   - Include background noise, silence, music")
    print("   - Include similar-sounding phrases")
    print("   - Press Ctrl+C to stop early\n")
    
    # Select device once at the start (if not already selected)
    if _selected_device_index is None:
        _selected_device_index = select_microphone_device()
        if _selected_device_index is None:
            print("⚠️  No device selected, using default")
    
    negative_phrases = [
        "hey there",
        "hey you",
        "hey siri",
        "hey google",
        "hey alexa",
        "hello",
        "hi there",
        "what's up",
        "good morning",
        "how are you",
        "aura",
        "hey",
    ]
    
    collected = 0
    existing_files = list(NEGATIVE_DIR.glob("*.wav"))
    start_index = len(existing_files) + 1
    
    try:
        while collected < num_samples:
            print(f"\n--- Sample {collected + 1}/{num_samples} ---")
            
            # Suggest a phrase
            if collected < len(negative_phrases):
                suggested = negative_phrases[collected]
                print(f"💡 Suggested phrase: '{suggested}' (or say anything else)")
            else:
                print("💡 Say any phrase (NOT 'hey aura')")
            
            # Record audio (device already selected)
            audio = record_audio_sample(duration=2.5, device_index=_selected_device_index)
            
            # Save sample
            output_file = NEGATIVE_DIR / f"negative_{start_index + collected:03d}.wav"
            sf.write(str(output_file), audio, SAMPLE_RATE)
            print(f"✅ Saved: {output_file.name}")
            
            collected += 1
            
            # Ask if user wants to continue
            if collected < num_samples:
                try:
                    response = input(f"\nContinue? (y/n, default=y): ").strip().lower()
                    if response == 'n':
                        break
                except KeyboardInterrupt:
                    break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    
    print(f"\n✅ Collected {collected} negative samples")
    print(f"   Total negative samples: {len(list(NEGATIVE_DIR.glob('*.wav')))}")


def generate_tts_negative_samples(num_samples: int = 20, play_through_speakers: bool = True):
    """Generate TTS echo samples (critical for handling TTS echo)."""
    print(f"\n{'='*60}")
    print(f"📝 GENERATING TTS ECHO SAMPLES (NEGATIVE)")
    print(f"{'='*60}")
    print(f"These samples teach the model NOT to trigger on TTS output")
    
    if play_through_speakers:
        print(f"\n🔊 Mode: Play through speakers + Record echo (RECOMMENDED)")
        print(f"   - Plays TTS through your speakers")
        print(f"   - Records it back through microphone")
        print(f"   - Captures real echo/reverb from your environment")
        print(f"   - More realistic for training")
    else:
        print(f"\n🎤 Mode: Direct generation (no echo)")
        print(f"   - Generates TTS audio directly")
        print(f"   - Faster but less realistic")
    
    print(f"\nGenerating {num_samples} TTS samples of '{WAKE_PHRASE}'...")
    
    files = generate_tts_samples(WAKE_PHRASE, num_samples, TTS_NEGATIVE_DIR, play_through_speakers=play_through_speakers)
    
    print(f"\n✅ Generated {len(files)} TTS echo samples")
    print(f"   These are NEGATIVE samples - model should NOT trigger on them")
    return files


def prepare_training_data():
    """Prepare training dataset from collected samples."""
    print(f"\n{'='*60}")
    print(f"📊 PREPARING TRAINING DATA")
    print(f"{'='*60}")
    
    # Count samples
    positive_files = list(POSITIVE_DIR.glob("*.wav"))
    negative_files = list(NEGATIVE_DIR.glob("*.wav"))
    tts_negative_files = list(TTS_NEGATIVE_DIR.glob("*.wav"))
    
    print(f"\n📁 Sample counts:")
    print(f"   Positive (human 'hey aura'): {len(positive_files)}")
    print(f"   Negative (other phrases): {len(negative_files)}")
    print(f"   TTS Negative (TTS 'hey aura'): {len(tts_negative_files)}")
    print(f"   Total: {len(positive_files) + len(negative_files) + len(tts_negative_files)}")
    
    if len(positive_files) < 10:
        print("\n⚠️  WARNING: Need at least 10 positive samples for training")
        print("   Run with --mode collect to gather more samples")
        return False
    
    if len(negative_files) + len(tts_negative_files) < 20:
        print("\n⚠️  WARNING: Need at least 20 negative samples for training")
        print("   Run with --mode collect to gather more samples")
        return False
    
    # Create dataset manifest
    manifest = {
        "wake_phrase": WAKE_PHRASE,
        "sample_rate": SAMPLE_RATE,
        "positive_samples": [str(f) for f in positive_files],
        "negative_samples": [str(f) for f in negative_files] + [str(f) for f in tts_negative_files],
        "created": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    manifest_file = TRAINING_DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Dataset manifest saved: {manifest_file}")
    return True


def train_model():
    """Train the OpenWakeWord model."""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING OPENWAKEWORD MODEL")
    print(f"{'='*60}")
    
    # Check if training data exists
    manifest_file = TRAINING_DATA_DIR / "dataset_manifest.json"
    if not manifest_file.exists():
        print("❌ Dataset manifest not found. Run data collection first:")
        print("   python3 train_openwakeword_hey_aura.py --mode collect")
        return False
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    positive_files = [Path(f) for f in manifest["positive_samples"]]
    negative_files = [Path(f) for f in manifest["negative_samples"]]
    
    print(f"\n📊 Training dataset:")
    print(f"   Positive samples: {len(positive_files)}")
    print(f"   Negative samples: {len(negative_files)}")
    
    if len(positive_files) < 10 or len(negative_files) < 20:
        print("\n❌ Insufficient training data")
        return False
    
    try:
        print(f"\n🔄 Preparing training data...")
        
        # OpenWakeWord training is done via Google Colab notebooks
        # We'll format the data and provide instructions
        
        # Load positive and negative samples
        positive_clips = []
        negative_clips = []
        
        print("   Loading positive samples...")
        for f in positive_files:
            try:
                audio, sr = sf.read(str(f))
                if sr != SAMPLE_RATE:
                    # Resample if needed
                    from scipy import signal
                    num_samples = int(len(audio) * SAMPLE_RATE / sr)
                    audio = signal.resample(audio, num_samples)
                positive_clips.append(audio)
            except Exception as e:
                print(f"   ⚠️  Error loading {f.name}: {e}")
        
        print("   Loading negative samples...")
        for f in negative_files:
            try:
                audio, sr = sf.read(str(f))
                if sr != SAMPLE_RATE:
                    # Resample if needed
                    from scipy import signal
                    num_samples = int(len(audio) * SAMPLE_RATE / sr)
                    audio = signal.resample(audio, num_samples)
                negative_clips.append(audio)
            except Exception as e:
                print(f"   ⚠️  Error loading {f.name}: {e}")
        
        print(f"\n✅ Loaded {len(positive_clips)} positive and {len(negative_clips)} negative clips")
        
        # Save training data in format expected by openwakeword
        training_data_dir = TRAINING_DATA_DIR / "formatted"
        training_data_dir.mkdir(exist_ok=True)
        
        pos_dir = training_data_dir / "positive"
        neg_dir = training_data_dir / "negative"
        pos_dir.mkdir(exist_ok=True)
        neg_dir.mkdir(exist_ok=True)
        
        # Copy/save clips in correct format
        print("   Saving formatted training data...")
        for i, clip in enumerate(positive_clips):
            sf.write(pos_dir / f"pos_{i:04d}.wav", clip, SAMPLE_RATE)
        
        for i, clip in enumerate(negative_clips):
            sf.write(neg_dir / f"neg_{i:04d}.wav", clip, SAMPLE_RATE)
        
        print(f"\n✅ Training data formatted and saved to {training_data_dir}")
        print(f"\n{'='*60}")
        print(f"📝 TRAINING INSTRUCTIONS")
        print(f"{'='*60}")
        print(f"\nOpenWakeWord training is done via Google Colab notebooks.")
        print(f"The training data has been prepared and formatted for you.\n")
        print(f"📁 Training data location:")
        print(f"   {training_data_dir}")
        print(f"   ├── positive/ ({len(positive_clips)} files)")
        print(f"   └── negative/ ({len(negative_clips)} files)\n")
        print(f"🚀 Next steps:")
        print(f"   1. Visit OpenWakeWord's GitHub repository:")
        print(f"      https://github.com/dscripka/openWakeWord")
        print(f"      - Check the README for training instructions")
        print(f"      - Look for notebooks/ folder or training scripts")
        print(f"      - Follow their training documentation")
        print(f"      - They may provide Colab links or Python scripts")
        print(f"\n   2. Upload the training data folder to Colab:")
        print(f"      - Upload the entire '{training_data_dir.name}' folder")
        print(f"      - Or upload 'positive' and 'negative' folders separately")
        print(f"\n   3. Follow the notebook instructions to:")
        print(f"      - Load your training data")
        print(f"      - Configure training parameters")
        print(f"      - Train the model")
        print(f"      - Export the trained model (.onnx file)")
        print(f"\n   4. Download the trained model and save it to:")
        print(f"      {MODEL_OUTPUT_DIR}/hey_aura_v0.1.onnx")
        print(f"\n   5. Update openwakeword_wake_word.py to use the new model:")
        print(f"      DEFAULT_MODEL = 'hey_aura_v0.1'")
        print(f"\n💡 Tip: The notebook will handle data augmentation and model training")
        print(f"   automatically. Training typically takes 10-30 minutes.\n")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing openwakeword: {e}")
        print("   Install with: pip install openwakeword")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train OpenWakeWord model for 'hey aura' with TTS echo handling"
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "full", "tts-only"],
        default="full",
        help="Operation mode: collect (data only), train (model only), full (both), tts-only (generate TTS samples)"
    )
    parser.add_argument(
        "--positive-samples",
        type=int,
        default=20,
        help="Number of positive samples to collect (default: 20)"
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=30,
        help="Number of negative samples to collect (default: 30)"
    )
    parser.add_argument(
        "--tts-samples",
        type=int,
        default=20,
        help="Number of TTS echo samples to generate (default: 20)"
    )
    parser.add_argument(
        "--tts-direct",
        action="store_true",
        help="Generate TTS samples directly (no echo/reverb). Default: play through speakers and record echo"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎤 OpenWakeWord Training for 'hey aura'")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return 1
    
    success = True
    
    if args.mode in ["collect", "full"]:
        # Collect positive samples
        collect_positive_samples(args.positive_samples)
        
        # Collect negative samples
        collect_negative_samples(args.negative_samples)
        
        # Generate TTS negative samples (critical for echo handling)
        generate_tts_negative_samples(args.tts_samples, play_through_speakers=not args.tts_direct)
        
        # Prepare training data
        if not prepare_training_data():
            success = False
    
    if args.mode == "tts-only":
        # Only generate TTS samples
        generate_tts_negative_samples(args.tts_samples, play_through_speakers=not args.tts_direct)
        prepare_training_data()
    
    if args.mode in ["train", "full"]:
        if success:
            # Train model
            if not train_model():
                success = False
    
    if success:
        print(f"\n{'='*60}")
        print("✅ Training process completed!")
        print(f"{'='*60}")
        print(f"\n📁 Training data: {TRAINING_DATA_DIR}")
        print(f"📁 Model output: {MODEL_OUTPUT_DIR}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review training data quality")
        print(f"   2. Complete model training (may require Colab notebook)")
        print(f"   3. Test the trained model")
        print(f"   4. Update openwakeword_wake_word.py to use the new model")
        return 0
    else:
        print(f"\n{'='*60}")
        print("⚠️  Training process completed with warnings")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)

