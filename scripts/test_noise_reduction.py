#!/usr/bin/env python3
"""
Real-Time Noise Reduction Monitor

This script processes audio in real-time and displays frame-by-frame:
- Raw audio properties (RMS, peak, dominant frequencies)
- Noise reduction stats (energy removed, reduction %)
- Cleaned audio properties
- Live comparison before/after

Usage:
    python3 scripts/test_noise_reduction.py
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
from scipy import interpolate, signal

# === Config (mirroring listener.py) ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 30.0  # Monitor for 30 seconds
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"

# Audio processing (matches listener.py exactly)
USE_AUTO_GAIN = True  # Enable automatic gain control
AGC_TARGET_RMS = 0.18  # Target RMS for Whisper (increased for far-field)
AGC_MAX_GAIN = 40.0  # Maximum gain to apply (increased for 16m range)
AGC_SOFT_CLIP_THRESHOLD = 0.95  # Start soft-clipping above this level
ENABLE_NOISE_REDUCTION = True  # Enable noise reduction

# Filter options: "highpass", "bandpass", or "none"
FILTER_TYPE = "bandpass"  # bandpass filters speech range (80-3400 Hz)
HIGHPASS_CUTOFF = 60  # Hz - Low cutoff (lowered to preserve more speech energy)
LOWPASS_CUTOFF = 3400  # Hz - High cutoff (removes hiss/noise above speech)

DISPLAY_EVERY_N_FRAMES = 5  # Update display every N frames (~0.16s)

def find_device_index():
    """Find the ReSpeaker device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

def highpass_filter(audio, cutoff=200, order=5):
    """
    Apply high-pass filter to remove low-frequency fan noise
    - Removes everything below cutoff frequency
    - Preserves speech frequencies above cutoff
    """
    nyquist = SAMPLE_RATE / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure frequency is in valid range (0 < Wn < 1)
    normalized_cutoff = max(0.01, min(normalized_cutoff, 0.99))
    
    try:
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ High-pass filter failed: {e}")
        return audio

def lowpass_filter(audio, cutoff=3400, order=5):
    """
    Apply low-pass filter to remove high-frequency noise
    - Removes everything above cutoff frequency
    - Preserves speech frequencies below cutoff
    """
    nyquist = SAMPLE_RATE / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure frequency is in valid range (0 < Wn < 1)
    normalized_cutoff = max(0.01, min(normalized_cutoff, 0.99))
    
    try:
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ Low-pass filter failed: {e}")
        return audio

def bandpass_filter(audio, low_cutoff=80, high_cutoff=3400, order=5):
    """
    Apply band-pass filter to isolate speech frequencies
    - Removes frequencies below low_cutoff (rumble/fan noise)
    - Removes frequencies above high_cutoff (hiss/high-freq noise)
    - Preserves speech band (typically 80-3400 Hz)
    """
    nyquist = SAMPLE_RATE / 2
    low_norm = low_cutoff / nyquist
    high_norm = high_cutoff / nyquist
    
    # Ensure frequencies are in valid range
    low_norm = max(0.01, min(low_norm, 0.98))
    high_norm = max(0.02, min(high_norm, 0.99))
    
    # Ensure low < high
    if low_norm >= high_norm:
        print(f"[Audio] ⚠️ Invalid bandpass range: {low_cutoff}-{high_cutoff} Hz")
        return audio
    
    try:
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception as e:
        print(f"[Audio] ⚠️ Band-pass filter failed: {e}")
        return audio

def soft_clip(audio, threshold=0.85, max_peak=0.98):
    """
    Two-stage soft clipping with dynamic range compression
    Stage 1: Gradual compression above threshold (0.85)
    Stage 2: Hard limit at max_peak (0.98) to prevent distortion
    
    This preserves waveform shape while preventing peaks from destroying audio
    """
    # Stage 1: Soft compression for peaks above threshold
    mask = np.abs(audio) > threshold
    if np.any(mask):
        # Use tanh for smooth compression
        excess = audio[mask] - np.sign(audio[mask]) * threshold
        compressed = threshold + np.tanh(excess / (max_peak - threshold)) * (max_peak - threshold)
        audio[mask] = np.sign(audio[mask]) * compressed
    
    # Stage 2: Safety hard limit (should rarely trigger after soft compression)
    audio = np.clip(audio, -max_peak, max_peak)
    
    return audio

def auto_gain_control(audio):
    """
    Automatic gain control with two-stage soft clipping
    - Adapts gain based on input level (far-field support up to 40x)
    - Uses progressive soft clipping to preserve waveform shape
    - Prevents distortion while maximizing signal strength
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:  # Avoid division by zero
        return audio, 1.0
    
    # Calculate required gain to reach target RMS
    required_gain = AGC_TARGET_RMS / rms
    
    # Limit gain to maximum
    actual_gain = min(required_gain, AGC_MAX_GAIN)
    
    # Apply gain
    audio = audio * actual_gain
    
    # Apply two-stage soft clipping to preserve waveform
    # Stage 1: Soft compression starts at 0.85 (gradual)
    # Stage 2: Hard limit at 0.98 (prevents peak distortion)
    audio = soft_clip(audio, threshold=0.85, max_peak=0.98)
    
    return audio, actual_gain

def process_audio(audio):
    """
    Process audio exactly like listener.py:
    1. Frequency filtering (highpass/bandpass/none)
    2. Automatic gain control (adapts to speech distance)
    
    Returns: (processed_audio, applied_gain)
    """
    # Step 1: Frequency filtering
    if ENABLE_NOISE_REDUCTION:
        if FILTER_TYPE == "bandpass":
            audio = bandpass_filter(audio, low_cutoff=HIGHPASS_CUTOFF, high_cutoff=LOWPASS_CUTOFF)
        elif FILTER_TYPE == "highpass":
            audio = highpass_filter(audio, cutoff=HIGHPASS_CUTOFF)
        # "none" or other values = no filtering
    
    # Step 2: Automatic gain control
    if USE_AUTO_GAIN:
        audio, applied_gain = auto_gain_control(audio)
    else:
        audio = np.clip(audio, -1.0, 1.0)
        applied_gain = 1.0
    
    return audio, applied_gain

def get_top_frequencies(audio, top_n=3):
    """Get top N dominant frequencies from audio"""
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    
    sorted_indices = np.argsort(magnitude)[::-1]
    freqs = []
    for idx in sorted_indices[:top_n]:
        freq = idx * (SAMPLE_RATE / 2) / len(magnitude)
        mag = magnitude[idx]
        freqs.append((freq, mag))
    return freqs

def format_freq_list(freqs):
    """Format frequency list for display"""
    return ", ".join([f"{freq:.0f}Hz" for freq, _ in freqs])

def process_audio_realtime(device_index, duration):
    """Process audio in real-time and display properties"""
    print(f"\n{'='*90}")
    print(f"  🎤 REAL-TIME AUDIO PROCESSING MONITOR (Mirrors listener.py)")
    print(f"{'='*90}")
    print(f"  Duration: {duration}s")
    if FILTER_TYPE == "bandpass":
        print(f"  Pipeline: Band-Pass ({HIGHPASS_CUTOFF}-{LOWPASS_CUTOFF} Hz) → AGC (Target={AGC_TARGET_RMS}, Max={AGC_MAX_GAIN}x) → Soft Clip")
    elif FILTER_TYPE == "highpass":
        print(f"  Pipeline: High-Pass ({HIGHPASS_CUTOFF} Hz) → AGC (Target={AGC_TARGET_RMS}, Max={AGC_MAX_GAIN}x) → Soft Clip")
    else:
        print(f"  Pipeline: No filtering → AGC (Target={AGC_TARGET_RMS}, Max={AGC_MAX_GAIN}x) → Soft Clip")
    print(f"  Optimized for: Far-field speech (up to 16 meters)")
    print(f"  Display Update: Every {DISPLAY_EVERY_N_FRAMES} frames (~{DISPLAY_EVERY_N_FRAMES * FRAME_DURATION:.2f}s)")
    print(f"{'='*90}\n")
    
    print("\n  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)
    print("\n  🔴 MONITORING...\n")
    
    # Print header
    print(f"\n{'='*90}")
    print(f"{'Frame':<8} {'RAW':<30} │ {'PROCESSED':<30} │ {'AGC':<15}")
    print(f"{'-'*8} {'-'*30} │ {'-'*30} │ {'-'*15}")
    print(f"{'#':<8} {'RMS':<10} {'Peak':<10} {'Top Hz':<10} │ {'RMS':<10} {'Peak':<10} {'Top Hz':<10} │ {'Gain':<15}")
    print(f"{'='*90}")
    
    # Running statistics
    frame_count = 0
    total_raw_rms = 0
    total_clean_rms = 0
    total_gain = 0
    
    frames_needed = int(SAMPLE_RATE * duration / FRAME_SIZE)
    
    try:
        with sd.InputStream(device=device_index, channels=6, samplerate=SAMPLE_RATE,
                            blocksize=FRAME_SIZE, dtype="float32") as stream:
            for i in range(frames_needed):
                # Read frame
                audio_block, _ = stream.read(FRAME_SIZE)
                raw_audio = audio_block[:, 0]
                
                # Analyze raw audio
                raw_rms = np.sqrt(np.mean(raw_audio ** 2))
                raw_peak = np.max(np.abs(raw_audio))
                raw_freqs = get_top_frequencies(raw_audio, top_n=1)
                
                # Apply full processing pipeline (same as listener.py)
                clean_audio, applied_gain = process_audio(raw_audio)
                
                # Analyze clean audio
                clean_rms = np.sqrt(np.mean(clean_audio ** 2))
                clean_peak = np.max(np.abs(clean_audio))
                clean_freqs = get_top_frequencies(clean_audio, top_n=1)
                
                # Update statistics
                frame_count += 1
                total_raw_rms += raw_rms
                total_clean_rms += clean_rms
                total_gain += applied_gain
                
                # Display every N frames
                if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                    raw_freq_str = f"{raw_freqs[0][0]:.0f}" if raw_freqs else "N/A"
                    clean_freq_str = f"{clean_freqs[0][0]:.0f}" if clean_freqs else "N/A"
                    
                    print(f"{frame_count:<8} "
                          f"{raw_rms:.6f}  {raw_peak:.4f}  {raw_freq_str:<10} │ "
                          f"{clean_rms:.6f}  {clean_peak:.4f}  {clean_freq_str:<10} │ "
                          f"{applied_gain:.2f}x")
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*110}")
        print("  ⚠️  Monitoring stopped by user")
        print(f"{'='*110}\n")
    
    # Print summary statistics
    if frame_count > 0:
        avg_raw_rms = total_raw_rms / frame_count
        avg_clean_rms = total_clean_rms / frame_count
        avg_gain = total_gain / frame_count
        amplification_ratio = avg_clean_rms / avg_raw_rms if avg_raw_rms > 0 else 0
        
        print(f"\n{'='*90}")
        print(f"  📊 SUMMARY STATISTICS (from {frame_count} frames)")
        print(f"{'='*90}")
        print(f"  Average Raw RMS:       {avg_raw_rms:.6f}")
        print(f"  Average Processed RMS: {avg_clean_rms:.6f}")
        print(f"  Average AGC Gain:      {avg_gain:.2f}x")
        print(f"  Amplification Ratio:   {amplification_ratio:.2f}x")
        
        print(f"\n  💡 INTERPRETATION:")
        
        if avg_clean_rms < 0.12:
            print(f"     ⚠️  Low output RMS (<0.12) - may be too quiet for Whisper")
            print(f"        → Increase AGC_MAX_GAIN or AGC_TARGET_RMS")
        elif avg_clean_rms > 0.25:
            print(f"     ⚠️  High output RMS (>0.25) - risk of excessive soft clipping")
            print(f"        → Decrease AGC_TARGET_RMS or check for over-amplification")
        else:
            print(f"     ✅ Output RMS looks good (0.12-0.25 range)")
            print(f"        → Optimal for Whisper transcription")
        
        if avg_gain > AGC_MAX_GAIN * 0.9:
            print(f"\n     ⚠️  AGC frequently hitting max gain ({AGC_MAX_GAIN}x)")
            print(f"        → Consider increasing AGC_MAX_GAIN for even more far-field speech")
        
        print(f"\n     📏 FAR-FIELD CAPABILITY:")
        print(f"        At current settings (Max Gain: {AGC_MAX_GAIN}x):")
        print(f"        - Near speech (RMS ~0.020): Gain ~{AGC_TARGET_RMS/0.020:.1f}x → Good")
        print(f"        - Far speech  (RMS ~0.007): Gain ~{min(AGC_TARGET_RMS/0.007, AGC_MAX_GAIN):.1f}x → {'Good' if AGC_TARGET_RMS/0.007 < AGC_MAX_GAIN else 'Max Gain'}")
        
        print(f"{'='*90}\n")

def main():
    """Main execution"""
    print("\n" + "="*90)
    print("  🧪 REAL-TIME AUDIO PROCESSING MONITOR")
    print("="*90)
    
    try:
        # Find microphone
        print(f"\n[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Show configuration
        print(f"\n[Audio] 📋 Configuration:")
        print(f"  Filter Type: {FILTER_TYPE}")
        if FILTER_TYPE == "bandpass":
            print(f"  Band-Pass Range: {HIGHPASS_CUTOFF}-{LOWPASS_CUTOFF} Hz")
        elif FILTER_TYPE == "highpass":
            print(f"  High-Pass Cutoff: {HIGHPASS_CUTOFF} Hz")
        print(f"  AGC Target RMS: {AGC_TARGET_RMS}")
        print(f"  AGC Max Gain: {AGC_MAX_GAIN}x (far-field optimized)")
        print(f"  Soft Clip Threshold: {AGC_SOFT_CLIP_THRESHOLD}")
        
        # Start real-time processing
        process_audio_realtime(device_index, RECORDING_DURATION)
        
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
