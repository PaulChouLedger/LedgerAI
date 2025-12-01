#!/usr/bin/env python3
"""
Find Optimal RMS for Whisper Transcription

Tests the same audio sample at different RMS levels to determine
the minimum RMS needed for accurate transcription.

This helps avoid over-amplifying near-field speech while ensuring
far-field speech is amplified enough.

Usage:
    1. Record a test phrase when prompted
    2. Script will test it at different RMS levels (0.05, 0.08, 0.10, 0.12, 0.15, 0.18)
    3. Shows which levels produce accurate transcription
"""

import sys
import os
import time
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import requests
import io

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 4.0  # Maximum recording duration (fallback)
DEVICE_NAME = "XVF3800 4-Mic Array"

# === VAD Configuration ===
VAD_START_THRESHOLD = 0.25  # Match test_transcription.py and listener.py
VAD_SILENCE_THRESHOLD = 0.10  # Match test_transcription.py and listener.py
SILENCE_TIMEOUT = 0.2  # 200ms of silence before stopping
MIN_AUDIO_SAMPLES = 2000  # Minimum samples to consider valid

# Test RMS levels
TEST_RMS_LEVELS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

def find_device_index():
    """Find the XVF3800 device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower():
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

def record_test_audio(device_index, max_duration, model_vad):
    """Record a test audio sample using VAD (matches production pipeline)"""
    print(f"\n{'='*80}")
    print(f"  🎤 RECORDING TEST AUDIO (VAD-Controlled)")
    print(f"{'='*80}")
    print(f"\n  Please speak a clear test phrase when recording starts.")
    print(f"  Suggested: 'My name is Raphael and I'm testing the microphone.'")
    print(f"\n  Recording will:")
    print(f"    1. Wait for speech (VAD > {VAD_START_THRESHOLD})")
    print(f"    2. Record until silence ({SILENCE_TIMEOUT*1000:.0f}ms of silence)")
    print(f"    3. Maximum duration: {max_duration} seconds")
    print(f"{'='*80}\n")
    
    input("Press ENTER when ready to record...")
    
    print("\n  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)
    print("\n  🎤 Listening for speech...\n")
    
    buffer = []
    silence_start = None
    max_samples = int(SAMPLE_RATE * max_duration)
    samples_recorded = 0
    
    try:
        with sd.InputStream(device=device_index, channels=2, samplerate=SAMPLE_RATE,
                            blocksize=FRAME_SIZE, dtype="float32") as stream:
            
            # === Stage 1: Wait for speech ===
            print("  Waiting for speech...", end="", flush=True)
            while samples_recorded < max_samples:
                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                
                if channel_0.size < 512:
                    continue
                
                # Check VAD
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                
                if vad_prob > VAD_START_THRESHOLD:
                    print(f"\n  🔊 Speech detected (VAD={vad_prob:.2f}) - recording...")
                    buffer.append(audio_block)
                    samples_recorded += FRAME_SIZE
                    break
                else:
                    print(".", end="", flush=True)
            
            # === Stage 2: Record speech until silence ===
            vad_history = []  # Track VAD levels for analysis
            frame_count = 0
            while samples_recorded < max_samples:
                audio_block, _ = stream.read(FRAME_SIZE)
                channel_0 = audio_block[:, 0]
                
                if channel_0.size < 512:
                    continue
                
                buffer.append(audio_block)
                samples_recorded += FRAME_SIZE
                frame_count += 1
                
                # Check for silence
                vad_prob = model_vad(torch.from_numpy(channel_0), SAMPLE_RATE).item()
                vad_history.append(vad_prob)
                
                if vad_prob < VAD_SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        print(f"\n  ⏹️  Speech ended (silence detected, VAD={vad_prob:.2f} < {VAD_SILENCE_THRESHOLD})")
                        break
                else:
                    silence_start = None
                
                # Show VAD level every 10 frames to track trends
                if frame_count % 10 == 0:
                    print(f" VAD:{vad_prob:.2f}", end="", flush=True)
                else:
                    print(".", end="", flush=True)
            
            # Show VAD statistics
            if vad_history:
                min_vad = min(vad_history)
                max_vad = max(vad_history)
                avg_vad = sum(vad_history) / len(vad_history)
                print(f"\n  📊 VAD Statistics: min={min_vad:.2f}, max={max_vad:.2f}, avg={avg_vad:.2f}")
                # Show if VAD dropped below threshold during speech
                low_vad_frames = [v for v in vad_history if v < VAD_SILENCE_THRESHOLD]
                if low_vad_frames:
                    print(f"  ⚠️  VAD dropped below {VAD_SILENCE_THRESHOLD} in {len(low_vad_frames)}/{len(vad_history)} frames")
                    print(f"      This may cause early cutoff if silence timeout is too short")
            
            if samples_recorded >= max_samples:
                print(f"\n  ⏹️  Maximum duration reached ({max_duration}s)")
        
        print(f"\n  ✅ Recording complete!")
        
    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  Recording cancelled")
        return None
    
    # Concatenate all frames
    if not buffer:
        print(f"\n  ⚠️  No audio recorded!")
        return None
    
    audio = np.concatenate(buffer)
    audio = audio[:, 0]  # Channel 0 only
    
    # Check minimum length
    if len(audio) < MIN_AUDIO_SAMPLES:
        print(f"\n  ⚠️  Audio too short ({len(audio)/SAMPLE_RATE:.2f}s < {MIN_AUDIO_SAMPLES/SAMPLE_RATE:.2f}s)")
        return None
    
    # Show statistics
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    
    print(f"\n  📊 Recorded Audio:")
    print(f"     Duration: {len(audio) / SAMPLE_RATE:.2f}s")
    print(f"     RMS: {rms:.6f}")
    print(f"     Peak: {peak:.4f}")
    
    return audio

def normalize_audio(audio, target_rms):
    """
    Normalize audio to target RMS level (raw, no limiting)
    This gives us true test data - if peaks exceed 1.0, that's useful information
    It tells us which RMS levels cause clipping in real scenarios
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    
    if current_rms < 1e-6:
        return audio
    
    gain = target_rms / current_rms
    normalized = audio * gain
    
    # No limiting - raw audio for accurate testing
    # If peaks exceed 1.0, that's data we need to know
    # This tells us which RMS levels are achievable without clipping
    
    return normalized

def transcribe_audio(audio):
    """Send audio to Whisper for transcription"""
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, SAMPLE_RATE, format="WAV")
    wav_io.seek(0)
    
    try:
        response = requests.post(
            "http://localhost:5000/transcribe",
            files={"audio": ("speech.wav", wav_io, "audio/wav")},
            timeout=10
        )
        result = response.json()
        text = result["text"].get("text", "").strip() if isinstance(result["text"], dict) else result.get("text", "").strip()
        return text
    except Exception as e:
        return f"ERROR: {e}"

def test_rms_levels(original_audio, test_levels):
    """Test transcription at different RMS levels"""
    print(f"\n{'='*80}")
    print(f"  🧪 TESTING DIFFERENT RMS LEVELS")
    print(f"{'='*80}\n")
    
    results = []
    
    for target_rms in test_levels:
        print(f"\n  Testing RMS={target_rms:.2f}...")
        
        # Normalize to target RMS
        normalized = normalize_audio(original_audio, target_rms)
        
        # Check actual levels
        actual_rms = np.sqrt(np.mean(normalized ** 2))
        actual_peak = np.max(np.abs(normalized))
        
        # Warn if clipping (peak > 1.0)
        clip_warning = ""
        if actual_peak > 1.0:
            clip_warning = " ⚠️ CLIPPING"
        
        print(f"    Actual: RMS={actual_rms:.4f}, Peak={actual_peak:.4f}{clip_warning}")
        print(f"    Transcribing...", end=" ")
        
        # Transcribe
        text = transcribe_audio(normalized)
        
        if text and not text.startswith("ERROR"):
            print(f"✅")
            print(f"    Result: '{text}'")
            success = True
        else:
            print(f"❌")
            print(f"    Result: {text if text else '(empty)'}")
            success = False
        
        results.append({
            'target_rms': target_rms,
            'actual_rms': actual_rms,
            'peak': actual_peak,
            'text': text,
            'success': success
        })
        
        time.sleep(0.5)  # Brief pause between tests
    
    return results

def print_summary(results):
    """Print summary of results"""
    print(f"\n{'='*80}")
    print(f"  📊 RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Target RMS':<12} {'Actual RMS':<12} {'Peak':<8} {'Status':<8} {'Transcription':<40}")
    print(f"{'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*40}")
    
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        text_preview = r['text'][:37] + "..." if len(r['text']) > 40 else r['text']
        print(f"{r['target_rms']:<12.2f} {r['actual_rms']:<12.4f} {r['peak']:<8.4f} {status:<8} {text_preview:<40}")
    
    print(f"\n{'='*80}")
    print(f"  💡 RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    # Find minimum working RMS
    successful = [r for r in results if r['success']]
    
    if successful:
        # Find best RMS (full transcription, no clipping)
        best_rms = None
        for r in successful:
            # Prefer full transcriptions without clipping
            if r['peak'] <= 1.0 and len(r['text']) > 20:  # Full transcription
                if best_rms is None or r['target_rms'] < best_rms['target_rms']:
                    best_rms = r
        
        if best_rms:
            print(f"  ✅ Recommended RMS: {best_rms['target_rms']:.2f}")
            print(f"     Peak: {best_rms['peak']:.4f} (no clipping)")
            print(f"     Transcription: '{best_rms['text'][:50]}...'")
        else:
            # Fallback to minimum
            min_rms = min(successful, key=lambda x: x['target_rms'])
            print(f"  ✅ Minimum working RMS: {min_rms['target_rms']:.2f}")
            print(f"     Peak: {min_rms['peak']:.4f}")
        
        # Check for clipping issues
        clipped = [r for r in results if r.get('peak', 0) > 1.0]
        if clipped:
            clipped_levels = [f"{r['target_rms']:.2f}" for r in clipped]
            print(f"\n  ⚠️  Clipping detected at RMS levels: {', '.join(clipped_levels)}")
            print(f"     Peaks exceed 1.0 - these levels will cause distortion in production")
        
        # Check for cutoffs
        cutoffs = [r for r in successful if len(r['text']) < 20 and r.get('peak', 0) <= 1.0]
        if cutoffs:
            cutoff_levels = [f"{r['target_rms']:.2f}" for r in cutoffs]
            print(f"\n  ⚠️  Partial transcriptions at RMS levels: {', '.join(cutoff_levels)}")
            print(f"     These levels may cause Whisper to cut off speech")
        
        if min_rms['target_rms'] < 0.10:
            print(f"\n  ⚠️  Very low minimum ({min_rms['target_rms']:.2f}) - may be too quiet for consistency")
            print(f"     Recommend using 0.08-0.15 for reliability")
    else:
        print(f"  ❌ No successful transcriptions!")
        print(f"     Try recording louder or checking microphone setup")
    
    print(f"\n{'='*80}\n")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🧪 WHISPER RMS OPTIMIZER (VAD-Controlled)")
    print("  Find the minimum RMS level for accurate transcription")
    print("="*80)
    
    # Load VAD model
    print("\n[VAD] 🔄 Loading Silero VAD model...")
    try:
        model_vad, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", onnx=False)
        print("[VAD] ✅ VAD model loaded")
    except Exception as e:
        print(f"[VAD] ❌ Failed to load VAD model: {e}")
        print("[VAD] 💡 Falling back to fixed-duration recording (no VAD)")
        model_vad = None
    
    # Show current hardware configuration
    print("\n" + "="*70)
    print("[Hardware] Current XVF3800 DSP Configuration:")
    print("="*70)
    try:
        # Load hardware config from listener's saved state
        import json
        config_path = os.path.expanduser("~/LedgerAI/data/xvf3800_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Display config
            agc_status = "✅ ON" if config.get('config', {}).get('PP_AGCONOFF', 0) == 1 else "❌ OFF"
            agc_target = config.get('config', {}).get('PP_AGCDESIREDLEVEL', 0.0)
            agc_gain = config.get('config', {}).get('PP_AGCMAXGAIN', 0.0)
            hpf_status = "✅ ON" if config.get('config', {}).get('AEC_HPFONOFF', 0) == 1 else "❌ OFF"
            ec_status = "✅ ON" if config.get('config', {}).get('PP_ECHOONOFF', 0) == 1 else "❌ OFF"
            
            print(f"  Hardware AGC:           {agc_status} (target={agc_target:.2f}, max={agc_gain:.0f} linear)")
            print(f"  High-Pass Filter:       {hpf_status}")
            print(f"  Echo Cancellation:      {ec_status}")
            print(f"\n  ℹ️  This test uses whatever hardware settings are currently active.")
            print(f"  ℹ️  Results only valid for THESE settings!")
            if model_vad:
                print(f"  ℹ️  Using VAD-controlled recording (matches production pipeline)")
        else:
            print(f"  ⚠️  No saved config found - using current hardware state")
    except Exception as e:
        print(f"  ⚠️  Could not load config: {e}")
    print("="*70 + "\n")
    
    try:
        # Find microphone
        print(f"[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Record test audio (VAD required)
        if not model_vad:
            print("\n  ❌ VAD model failed to load - cannot proceed")
            print("  💡 Please ensure PyTorch and Silero VAD are installed")
            return 1
        
        audio = record_test_audio(device_index, RECORDING_DURATION, model_vad)
        
        if audio is None:
            return 1
        
        # Test at different RMS levels
        results = test_rms_levels(audio, TEST_RMS_LEVELS)
        
        # Print summary
        print_summary(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[Audio] ⚠️  Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n[Audio] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

