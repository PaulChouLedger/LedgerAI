#!/usr/bin/env python3
"""
Generate TTS negative samples for OpenWakeWord training.

This script generates TTS echo samples to train a custom openwakeword model
that can detect "hey aura" while ignoring TTS echo from the speaker output.

Usage:
    python3 collect_negative_tts_samples.py --tts-samples 300
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
WAKE_PHRASE_PHONEMES = "[HH][EY][AO][ER][AH]"  # Phoneme notation matching Colab notebook format
SAMPLE_RATE = 16000  # OpenWakeWord uses 16kHz

# TTS volume range (matching Colab training requirements)
TTS_MIN_VOLUME = 0.5  # 50% minimum volume
TTS_MAX_VOLUME = 1.5  # 150% maximum volume
TRAINING_DATA_DIR = workspace_root / "data" / "wake_word_training"
TTS_NEGATIVE_DIR = TRAINING_DATA_DIR / "negative_tts"
MODEL_OUTPUT_DIR = workspace_root / "data" / "models" / "wake_words"
MODEL_NAME = "hey_aura_v0.1"

# Create directories
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
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


def find_device_by_name(p, device_name="reSpeaker"):
    """
    Find microphone device by name (like listener.py does).
    Searches for device name in device names, not by index.
    
    Args:
        p: PyAudio instance
        device_name: Device name to search for (default: "reSpeaker")
    
    Returns:
        Device index if found, None otherwise
    """
    device_name_lower = device_name.lower()
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            if device_name_lower in info['name'].lower():
                return i
    
    return None


def play_and_record_tts(text: str, device_index: int = None, duration_padding: float = 0.5, volume: float = 1.0) -> np.ndarray:
    """
    Generate TTS, play it through speakers, and record it back through microphone.
    This captures real echo/reverb that occurs in actual use.
    
    Args:
        text: Text to generate TTS for
        device_index: Microphone device index
        duration_padding: Extra recording time after playback
        volume: Volume multiplier (0.0-1.0, default: 1.0)
    """
    import pyaudio
    import threading
    from elevenlabs.client import ElevenLabs
    
    # Find reSpeaker dynamically if device_index not provided
    if device_index is None:
        import pyaudio
        p = pyaudio.PyAudio()
        try:
            device_index = find_device_by_name(p, "reSpeaker")
            if device_index is None:
                raise RuntimeError("reSpeaker not found")
        finally:
            p.terminate()
    
    ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
    ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID") or "default"
    
    if not ELEVEN_API_KEY or ELEVEN_API_KEY == "your_elevenlabs_api_key_here":
        raise RuntimeError("ElevenLabs API key not configured")
    
    client = ElevenLabs(api_key=ELEVEN_API_KEY)
    PCM_SAMPLE_RATE = 22050
    PCM_FORMAT = "pcm_22050"
    
    # Generate TTS audio using phonetic text matching Colab format
    # Phonemes [HH][EY][AO][ER][AH] map to "hey_orah" pronunciation
    # Use "hey_orah" as text input (matches phonemes) since ElevenLabs uses text input
    if text.startswith("[") and "]" in text:
        # Text is in phoneme format: [HH][EY][AO][ER][AH] -> use "hey_orah" 
        tts_text = "hey_orah"  # Phonetic text matching the phoneme notation
    elif text.lower().strip() == WAKE_PHRASE.lower() or WAKE_PHRASE_PHONEMES in text:
        # Use phonetic text for consistency with Colab training
        tts_text = "hey_orah"  # Matches phonemes [HH][EY][AO][ER][AH]
    else:
        # Use text as-is for other cases
        tts_text = text
    
    stream = client.text_to_speech.convert(
        text=tts_text,
        voice_id=ELEVEN_VOICE_ID,
        output_format=PCM_FORMAT,
        voice_settings={
            "stability": 1.0,  # Maximum consistency - we want consistent TTS signature for negative samples
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
    
    # Apply volume scaling with proper handling to preserve variation
    # Convert to float for better precision, scale, then convert back
    audio_float = audio_data.astype(np.float32)
    audio_float = audio_float * volume
    # Clamp to prevent clipping (preserves volume differences even when clipping occurs)
    audio_float = np.clip(audio_float, -32768.0, 32767.0)
    audio_data = audio_float.astype(np.int16)
    
    audio_duration = len(audio_data) / PCM_SAMPLE_RATE
    
    # Detect output device
    output_device = detect_output_device()
    
    # Set up recording
    p = pyaudio.PyAudio()
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    # Validate device - if device_index is provided, verify it's still valid
    # If invalid, search for reSpeaker by name (like listener.py)
    DEVICE_NAME = "reSpeaker"
    try:
        device_info = p.get_device_info_by_index(device_index)
        if device_info.get('maxInputChannels', 0) == 0:
            # Device has no input channels, search for reSpeaker by name
            print(f"   ⚠️  Device {device_index} has no input channels, searching for reSpeaker by name...")
            device_index = find_device_by_name(p, DEVICE_NAME)
            if device_index is None:
                raise RuntimeError(f"reSpeaker not found and device {device_index} is invalid")
    except (IndexError, OSError) as e:
        # Device index invalid, search for reSpeaker by name
        print(f"   ⚠️  Device {device_index} invalid: {e}, searching for reSpeaker by name...")
        device_index = find_device_by_name(p, DEVICE_NAME)
        if device_index is None:
            raise RuntimeError(f"reSpeaker not found and saved device is invalid")
    
    # Verify final device
    device_info = p.get_device_info_by_index(device_index)
    if device_info.get('maxInputChannels', 0) == 0:
        raise RuntimeError(f"Selected device {device_index} ({device_info.get('name', 'Unknown')}) has no input channels")
    
    recorded_frames = []
    recording_active = threading.Event()
    recording_done = threading.Event()
    recording_error = [None]  # Use list to allow modification from nested function
    
    def record_audio():
        """Record audio in a separate thread."""
        try:
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
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    recorded_frames.append(data)
                except Exception as e:
                    print(f"   ⚠️  Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
        except Exception as e:
            recording_error[0] = str(e)
            print(f"   ⚠️  Recording thread error: {e}")
        finally:
            recording_done.set()
    
    # Start recording thread
    record_thread = threading.Thread(target=record_audio, daemon=False)  # Changed to non-daemon so we can wait for it
    record_thread.start()
    
    # Wait for recording to start
    if not recording_active.wait(timeout=2.0):
        raise RuntimeError(f"Recording failed to start - device {device_index} may be invalid or busy")
    
    if recording_error[0]:
        raise RuntimeError(f"Recording error: {recording_error[0]}")
    
    time.sleep(0.2)  # Small delay to ensure recording is active and capturing
    
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
    
    # Wait for recording to finish (give it more time)
    if not recording_done.wait(timeout=10.0):
        print(f"   ⚠️  Recording thread did not finish in time")
    
    # Wait for thread to actually complete
    record_thread.join(timeout=2.0)
    
    try:
        p.terminate()
    except Exception:
        pass  # Ignore termination errors
    
    if recording_error[0]:
        raise RuntimeError(f"Recording failed: {recording_error[0]}")
    
    if not recorded_frames:
        raise RuntimeError(f"No audio recorded - check device {device_index} and ensure microphone is working")
    
    # Convert recorded audio to numpy array
    recorded_audio = np.frombuffer(b''.join(recorded_frames), dtype=np.int16)
    recorded_float = recorded_audio.astype(np.float32) / 32768.0
    
    return recorded_float


def generate_tts_samples(text: str, num_samples: int = 10, output_dir: Path = None, play_through_speakers: bool = True, start_index: int = 1) -> List[Path]:
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
    ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
    ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID") or "default"
    
    if not ELEVEN_API_KEY or ELEVEN_API_KEY == "your_elevenlabs_api_key_here":
        print("⚠️  ElevenLabs API key not configured - skipping TTS sample generation")
        print("   Set ELEVENLABS_API_KEY in .env file")
        return []
    
    output_dir = output_dir or TTS_NEGATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Define volume levels to cycle through (matching Colab training: min 50%, max 150%)
    # Distribute volumes evenly across the range
    num_volume_steps = max(8, num_samples // 5)  # At least 8 different volumes, or 1 per 5 samples
    volume_levels = np.linspace(TTS_MIN_VOLUME, TTS_MAX_VOLUME, num_volume_steps).tolist()
    
    # Debug: Print volume levels to verify they're different
    if play_through_speakers:
        print(f"   Volume levels: {[f'{v:.2f}' for v in volume_levels]}")
    
    # Create volume labels for logging
    def volume_to_label(vol):
        if vol < 0.8:
            return "quiet"
        elif vol < 1.0:
            return "normal"
        elif vol < 1.5:
            return "loud"
        elif vol < 2.0:
            return "very-loud"
        else:
            return "maximum"
    
    volume_labels = [volume_to_label(v) for v in volume_levels]
    
    if play_through_speakers:
        print(f"🔊 Generating {num_samples} TTS samples (NEGATIVE samples)...")
        if text.lower().strip() == WAKE_PHRASE.lower() or WAKE_PHRASE_PHONEMES in text:
            print(f"   Text: '{WAKE_PHRASE}' (phoneme notation: {WAKE_PHRASE_PHONEMES})")
        else:
            print(f"   Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"   Playing through speakers and recording echo/reverb")
        print(f"   Volume range: {int(TTS_MIN_VOLUME*100)}% to {int(TTS_MAX_VOLUME*100)}% ({len(volume_levels)} different levels)")
        print(f"   Make sure speakers are on and microphone can hear them!")
        
        print(f"   Starting in 2 seconds...")
        time.sleep(2)
    else:
        print(f"🎤 Generating {num_samples} TTS samples of '{text}' (NEGATIVE samples)...")
        print(f"   Direct generation (no echo/reverb)")
    
    for i in range(num_samples):
        # Cycle through volume levels
        volume_idx = i % len(volume_levels)
        volume = volume_levels[volume_idx]
        volume_label = volume_labels[volume_idx]
        try:
            if play_through_speakers:
                # Play through speakers and record echo
                # Use the text parameter (could be wake phrase or varied phrases)
                # For wake phrase, use phoneme notation; for other text, use as-is
                if text.lower().strip() == WAKE_PHRASE.lower() or WAKE_PHRASE_PHONEMES in text:
                    tts_input = WAKE_PHRASE_PHONEMES  # Use phonemes for wake phrase
                else:
                    tts_input = text  # Use varied phrases as-is
                print(f"\n   [{i+1}/{num_samples}] Playing and recording ({volume_label}, {int(volume*100)}%, vol={volume:.2f})...", end="", flush=True)
                # Find reSpeaker dynamically for each recording
                import pyaudio
                p = pyaudio.PyAudio()
                try:
                    current_device_index = find_device_by_name(p, "reSpeaker")
                    if current_device_index is None:
                        raise RuntimeError("reSpeaker not found")
                finally:
                    p.terminate()
                recorded_audio = play_and_record_tts(tts_input, device_index=current_device_index, volume=volume)
                print(" ✅")
                
                # Save recorded audio (already at 16kHz from recording)
                # Use start_index to avoid overwriting files when generating multiple phrase sets
                file_index = start_index + i
                output_file = output_dir / f"tts_echo_{file_index:03d}.wav"
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
                        "stability": 1.0,  # Maximum consistency - we want consistent TTS signature for negative samples
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
                
                # Use start_index to avoid overwriting files when generating multiple phrase sets
                file_index = start_index + i
                output_file = output_dir / f"tts_echo_{file_index:03d}.wav"
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


def select_microphone_device():
    """Find reSpeaker device by name dynamically (like listener.py does)."""
    import pyaudio
    
    DEVICE_NAME = "reSpeaker"  # Match listener.py
    
    p = pyaudio.PyAudio()
    
    try:
        # List available devices
        print("\n📱 Available audio input devices:")
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append((i, info))
                marker = " ⭐ (reSpeaker - DEFAULT)" if DEVICE_NAME.lower() in info['name'].lower() else ""
                print(f"   [{i}] {info['name']} ({info['maxInputChannels']} channels){marker}")
        
        if not devices:
            print("   ❌ No input devices found!")
            return None
        
        # Search for reSpeaker by name (like listener.py does)
        device_index = find_device_by_name(p, DEVICE_NAME)
        
        if device_index is not None:
            device_name = p.get_device_info_by_index(device_index)['name']
            print(f"\n✅ Found reSpeaker by name: [{device_index}] {device_name}")
            print(f"   This is the recommended device for wake word training")
            return device_index
        else:
            # reSpeaker not found - use first available device as fallback
            if devices:
                device_index = devices[0][0]
                device_name = devices[0][1]['name']
                print(f"\n⚠️  reSpeaker not found, using first available device: [{device_index}] {device_name}")
                return device_index
            else:
                return None
        
    finally:
        p.terminate()


# No need to store device index - we'll find reSpeaker dynamically each time


def record_audio_sample(duration: float = 3.0, sample_rate: int = SAMPLE_RATE, device_index: int = None) -> np.ndarray:
    """Record audio from microphone - finds reSpeaker dynamically if device_index not provided."""
    import pyaudio
    
    # Find reSpeaker dynamically if device_index not provided
    if device_index is None:
        p = pyaudio.PyAudio()
        try:
            device_index = find_device_by_name(p, "reSpeaker")
            if device_index is None:
                raise RuntimeError("reSpeaker not found")
        finally:
            p.terminate()
    
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


def generate_tts_negative_samples(num_samples: int = 100, play_through_speakers: bool = True):
    """Generate TTS echo samples (critical for handling TTS echo)."""
    print(f"\n{'='*60}")
    print(f"📝 GENERATING TTS ECHO SAMPLES (NEGATIVE)")
    print(f"{'='*60}")
    print(f"These samples teach the model NOT to trigger on TTS output")
    print(f"\n⚠️  CRITICAL: TTS echo is the #1 source of false positives!")
    print(f"   Without these samples, the model will trigger on TTS playback")
    
    print(f"\n📋 Configuration:")
    print(f"   Wake phrase: '{WAKE_PHRASE}'")
    print(f"   Phonemes: {WAKE_PHRASE_PHONEMES} (matching Colab format: 'hey_orah')")
    print(f"   Volume range: {int(TTS_MIN_VOLUME*100)}% - {int(TTS_MAX_VOLUME*100)}%")
    
    if play_through_speakers:
        print(f"\n🔊 Mode: Play through speakers + Record echo (RECOMMENDED)")
        print(f"   - Plays TTS through your speakers at varying volumes")
        print(f"   - Records it back through microphone")
        print(f"   - Captures real echo/reverb from your environment")
        print(f"   - Volume range: {int(TTS_MIN_VOLUME*100)}% to {int(TTS_MAX_VOLUME*100)}% (distributed across samples)")
        print(f"   - More realistic for training")
    else:
        print(f"\n🎤 Mode: Direct generation (no echo)")
        print(f"   - Generates TTS audio directly")
        print(f"   - Faster but less realistic")
        print(f"   - Not recommended for production models")
    
    # Generate varied phrases to capture general TTS sound signature
    # The real issue is TTS leakage from responses to user questions, not TTS saying "hey aura"
    # Using varied phrases teaches the model to ignore TTS sound signature in general
    tts_phrases = [
        "Sure, I can help you with that.",
        "That's a great question.",
        "Let me think about that for a moment.",
        "I understand what you're asking.",
        "Here's what I found.",
        "Based on the information available.",
        "That makes sense.",
        "I see what you mean.",
        "Let me explain that.",
        "Here's how that works.",
        "That's correct.",
        "I can help you with that.",
        "Let me provide some context.",
        "That's an interesting point.",
        "Here's what you need to know.",
        "I'll break that down for you.",
        "That's a good observation.",
        "Let me clarify that.",
        "Here's the answer to your question.",
        "I can assist with that.",
        # Also include some "hey aura" samples for direct contrast (20% of samples)
        WAKE_PHRASE,  # "hey aura" - for direct contrast with positive samples
    ]
    
    # Calculate distribution: 80% varied phrases, 20% "hey aura"
    num_varied = int(num_samples * 0.8)
    num_wake_phrase = num_samples - num_varied
    
    print(f"\nGenerating {num_samples} TTS samples with varied phrases...")
    print(f"   - {num_varied} samples: Varied phrases (typical LLM responses - captures general TTS signature)")
    print(f"   - {num_wake_phrase} samples: '{WAKE_PHRASE}' (for direct contrast with positive samples)")
    print(f"   💡 Using varied phrases is better - teaches model to ignore TTS in general, not just 'hey aura' from TTS")
    
    all_files = []
    file_counter = 1  # Start file numbering from 1
    
    # Generate varied phrase samples
    if num_varied > 0:
        samples_per_phrase = max(1, num_varied // len(tts_phrases))
        for phrase in tts_phrases:
            if len(all_files) >= num_varied:
                break
            phrase_samples = min(samples_per_phrase, num_varied - len(all_files))
            files = generate_tts_samples(phrase, phrase_samples, TTS_NEGATIVE_DIR, play_through_speakers=play_through_speakers, start_index=file_counter)
            all_files.extend(files)
            file_counter += len(files)  # Update counter for next phrase set
    
    # Generate "hey aura" samples for direct contrast
    if num_wake_phrase > 0:
        files = generate_tts_samples(WAKE_PHRASE_PHONEMES, num_wake_phrase, TTS_NEGATIVE_DIR, play_through_speakers=play_through_speakers, start_index=file_counter)
        all_files.extend(files)
    
    files = all_files
    
    print(f"\n✅ Generated {len(files)} TTS echo samples")
    print(f"   These are NEGATIVE samples - model should NOT trigger on them")
    print(f"   Samples include various volume levels for robust training")
    return files


def prepare_training_data():
    """Prepare training dataset manifest from TTS negative samples."""
    print(f"\n{'='*60}")
    print(f"📊 PREPARING TRAINING DATA")
    print(f"{'='*60}")
    
    # Count TTS negative samples
    tts_negative_files = list(TTS_NEGATIVE_DIR.glob("*.wav"))
    
    print(f"\n📁 Sample counts:")
    print(f"   TTS Negative samples: {len(tts_negative_files)}")
    
    if len(tts_negative_files) < 20:
        print("\n⚠️  WARNING: Need at least 20 TTS negative samples for training")
        print("   Run the script again to generate more samples")
        return False
    
    # Create dataset manifest
    manifest = {
        "wake_phrase": WAKE_PHRASE,
        "sample_rate": SAMPLE_RATE,
        "tts_negative_samples": [str(f) for f in tts_negative_files],
        "created": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    manifest_file = TRAINING_DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Dataset manifest saved: {manifest_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS negative samples for OpenWakeWord training"
    )
    parser.add_argument(
        "--tts-samples",
        type=int,
        default=300,
        help="Number of TTS echo samples to generate (default: 300)"
    )
    parser.add_argument(
        "--tts-direct",
        action="store_true",
        help="Generate TTS samples directly (no echo/reverb). Default: play through speakers and record echo"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎤 TTS Negative Sample Generation for 'hey aura'")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return 1
    
    # Generate TTS negative samples
    generate_tts_negative_samples(args.tts_samples, play_through_speakers=not args.tts_direct)
    
    # Update manifest with generated samples
    if not prepare_training_data():
        print("\n⚠️  Warning: Not enough samples generated")
        return 1
    
    print(f"\n{'='*60}")
    print("✅ TTS sample generation completed!")
    print(f"{'='*60}")
    print(f"\n📁 Training data: {TRAINING_DATA_DIR}")
    print(f"📁 TTS negative samples: {TTS_NEGATIVE_DIR}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review TTS sample quality")
    print(f"   2. Use these samples with OpenWakeWord training (Colab notebook)")
    print(f"   3. Test the trained model")
    print(f"   4. Update openwakeword_wake_word.py to use the new model")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)

