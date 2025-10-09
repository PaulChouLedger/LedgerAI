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
import sounddevice as sd
import soundfile as sf
import requests
import io

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 4.0  # 3 seconds for test phrase
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"

# Test RMS levels
TEST_RMS_LEVELS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

def find_device_index():
    """Find the ReSpeaker device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

def record_test_audio(device_index, duration):
    """Record a test audio sample"""
    print(f"\n{'='*80}")
    print(f"  🎤 RECORDING TEST AUDIO")
    print(f"{'='*80}")
    print(f"\n  Please speak a clear test phrase when recording starts.")
    print(f"  Suggested: 'My name is Raphael and I'm testing the microphone.'")
    print(f"\n  Recording duration: {duration} seconds")
    print(f"{'='*80}\n")
    
    input("Press ENTER when ready to record...")
    
    print("\n  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)
    print("\n  🔴 RECORDING... (speak now!)\n")
    
    samples_needed = int(SAMPLE_RATE * duration)
    recording = []
    
    try:
        with sd.InputStream(device=device_index, channels=6, samplerate=SAMPLE_RATE,
                            blocksize=FRAME_SIZE, dtype="float32") as stream:
            samples_recorded = 0
            while samples_recorded < samples_needed:
                audio_block, _ = stream.read(FRAME_SIZE)
                recording.append(audio_block[:, 0])  # Channel 0
                samples_recorded += FRAME_SIZE
                
                # Progress indicator
                progress = (samples_recorded / samples_needed) * 100
                print(f"  Recording... {progress:.0f}%", end="\r")
        
        print(f"\n\n  ✅ Recording complete!")
        
    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  Recording cancelled")
        return None
    
    # Concatenate all frames
    audio = np.concatenate(recording)[:samples_needed]
    
    # Show statistics
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    
    print(f"\n  📊 Recorded Audio:")
    print(f"     Duration: {len(audio) / SAMPLE_RATE:.2f}s")
    print(f"     RMS: {rms:.6f}")
    print(f"     Peak: {peak:.4f}")
    
    return audio

def normalize_audio(audio, target_rms):
    """Normalize audio to target RMS level"""
    current_rms = np.sqrt(np.mean(audio ** 2))
    
    if current_rms < 1e-6:
        return audio
    
    gain = target_rms / current_rms
    normalized = audio * gain
    normalized = np.clip(normalized, -0.95, 0.95)
    
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
        
        print(f"    Actual: RMS={actual_rms:.4f}, Peak={actual_peak:.4f}")
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
        min_rms = min(successful, key=lambda x: x['target_rms'])
        print(f"  ✅ Minimum working RMS: {min_rms['target_rms']:.2f}")
        print(f"     Use this as AGC_TARGET_RMS for minimal amplification")
        print(f"     Good for near-field speech (reduces over-amplification)")
        
        if min_rms['target_rms'] < 0.10:
            print(f"\n  ⚠️  Very low minimum ({min_rms['target_rms']:.2f}) - may be too quiet for consistency")
            print(f"     Recommend using 0.10-0.12 for reliability")
    else:
        print(f"  ❌ No successful transcriptions!")
        print(f"     Try recording louder or checking microphone setup")
    
    print(f"\n{'='*80}\n")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🧪 WHISPER RMS OPTIMIZER")
    print("  Find the minimum RMS level for accurate transcription")
    print("="*80)
    
    try:
        # Find microphone
        print(f"\n[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Record test audio
        audio = record_test_audio(device_index, RECORDING_DURATION)
        
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

