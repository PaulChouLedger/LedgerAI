#!/usr/bin/env python3
"""
Test and Debug Noise Reduction

This script records audio and shows detailed before/after analysis:
- Raw audio properties (RMS, peak, frequency spectrum)
- Noise profile characteristics
- Post-processed audio properties
- Noise reduction effectiveness

Usage:
    python3 scripts/test_noise_reduction.py
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
from scipy import signal, interpolate

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 10.0  # Record 5 seconds for testing
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
NOISE_REDUCTION_STRENGTH = 0.85

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NOISE_PROFILE_PATH = os.path.join(PROJECT_DIR, "data", "noise_profile.npy")

def find_device_index():
    """Find the ReSpeaker device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

def analyze_audio(audio, label="Audio"):
    """Analyze and display audio properties"""
    # Time-domain statistics
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    
    # Frequency-domain analysis
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    
    # Find dominant frequencies
    sorted_indices = np.argsort(magnitude)[::-1]
    top_freqs = []
    for idx in sorted_indices[:5]:
        freq = idx * (SAMPLE_RATE / 2) / len(magnitude)
        mag = magnitude[idx]
        top_freqs.append((freq, mag))
    
    # Energy distribution
    total_energy = np.sum(magnitude ** 2)
    low_freq_energy = np.sum(magnitude[:len(magnitude)//4] ** 2)  # 0-2000 Hz
    mid_freq_energy = np.sum(magnitude[len(magnitude)//4:len(magnitude)//2] ** 2)  # 2000-4000 Hz
    high_freq_energy = np.sum(magnitude[len(magnitude)//2:] ** 2)  # 4000-8000 Hz
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  📊 Time Domain:")
    print(f"     RMS:          {rms:.6f}")
    print(f"     Peak:         {peak:.6f}")
    print(f"     Dynamic Range: {20 * np.log10(peak / (rms + 1e-10)):.1f} dB")
    print(f"\n  📈 Frequency Domain:")
    print(f"     FFT bins:     {len(magnitude)}")
    print(f"     Total energy: {total_energy:.2f}")
    print(f"     Low (0-2k):   {low_freq_energy:.2f} ({low_freq_energy/total_energy*100:.1f}%)")
    print(f"     Mid (2-4k):   {mid_freq_energy:.2f} ({mid_freq_energy/total_energy*100:.1f}%)")
    print(f"     High (4-8k):  {high_freq_energy:.2f} ({high_freq_energy/total_energy*100:.1f}%)")
    print(f"\n  🎵 Top 5 Frequencies:")
    for i, (freq, mag) in enumerate(top_freqs, 1):
        print(f"     {i}. {freq:7.1f} Hz → magnitude: {mag:.6f}")
    print(f"{'='*70}\n")
    
    return {
        'rms': rms,
        'peak': peak,
        'magnitude': magnitude,
        'total_energy': total_energy
    }

def spectral_noise_subtraction(audio, noise_profile, strength=0.85):
    """
    Apply spectral noise subtraction
    """
    # Compute FFT of signal
    fft_signal = np.fft.rfft(audio)
    magnitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)
    
    # Match noise profile length to signal length
    if len(noise_profile) != len(magnitude):
        x_old = np.linspace(0, 1, len(noise_profile))
        x_new = np.linspace(0, 1, len(magnitude))
        f = interpolate.interp1d(x_old, noise_profile, kind='linear', fill_value='extrapolate')
        noise_profile_matched = f(x_new)
    else:
        noise_profile_matched = noise_profile
    
    # Calculate stats
    signal_energy = np.sum(magnitude ** 2)
    noise_energy = np.sum(noise_profile_matched ** 2)
    
    print(f"\n{'='*70}")
    print(f"  🔧 Spectral Subtraction Process")
    print(f"{'='*70}")
    print(f"  Signal FFT bins:     {len(magnitude)}")
    print(f"  Noise profile bins:  {len(noise_profile)} → {len(noise_profile_matched)} (matched)")
    print(f"  Subtraction strength: {strength}")
    print(f"  Signal energy:       {signal_energy:.2f}")
    print(f"  Noise energy:        {noise_energy:.2f}")
    print(f"  Noise/Signal ratio:  {noise_energy/signal_energy*100:.1f}%")
    
    # Subtract noise profile from magnitude
    magnitude_clean = np.maximum(magnitude - strength * noise_profile_matched, 0)
    
    # Calculate reduction
    clean_energy = np.sum(magnitude_clean ** 2)
    energy_removed = signal_energy - clean_energy
    reduction_db = 20 * np.log10(signal_energy / (clean_energy + 1e-10))
    
    print(f"\n  After subtraction:")
    print(f"  Clean energy:        {clean_energy:.2f}")
    print(f"  Energy removed:      {energy_removed:.2f}")
    print(f"  Reduction:           {energy_removed/signal_energy*100:.1f}%")
    print(f"  Reduction (dB):      {reduction_db:.1f} dB")
    print(f"{'='*70}\n")
    
    # Reconstruct signal
    fft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = np.fft.irfft(fft_clean, n=len(audio))
    
    return audio_clean

def record_audio(device_index, duration):
    """Record audio for analysis"""
    print(f"[Audio] 🎤 Recording {duration}s of audio...")
    print(f"[Audio] 💡 Please speak normally or let it capture ambient noise")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"[Audio] Starting in {i}...")
        time.sleep(1)
    
    print("[Audio] 🔴 RECORDING...")
    
    frames_needed = int(SAMPLE_RATE * duration / FRAME_SIZE)
    audio_samples = []
    
    with sd.InputStream(device=device_index, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        for i in range(frames_needed):
            audio_block, _ = stream.read(FRAME_SIZE)
            channel_0 = audio_block[:, 0]
            audio_samples.append(channel_0)
            
            # Real-time audio level indicator
            rms = np.sqrt(np.mean(channel_0 ** 2))
            bar_length = int(rms * 100)
            bar = '█' * min(bar_length, 50)
            progress = (i + 1) / frames_needed * 100
            print(f"[Audio] {progress:3.0f}% │{bar:<50}│ RMS: {rms:.4f}", end='\r')
    
    print(f"\n[Audio] ✅ Recording complete!")
    
    audio = np.concatenate(audio_samples)
    return audio

def main():
    """Main test execution"""
    print("\n" + "="*70)
    print("  🧪 Noise Reduction Test & Debug Tool")
    print("="*70)
    
    try:
        # Step 1: Load noise profile
        print(f"\n[Audio] 🔍 Loading noise profile from: {NOISE_PROFILE_PATH}")
        if not os.path.exists(NOISE_PROFILE_PATH):
            print(f"[Audio] ❌ Noise profile not found!")
            print(f"[Audio] 💡 Run: python3 scripts/record_noise_profile.py")
            return 1
        
        noise_profile = np.load(NOISE_PROFILE_PATH)
        print(f"[Audio] ✅ Loaded noise profile: {len(noise_profile)} frequency bins")
        
        # Analyze noise profile
        noise_energy = np.sum(noise_profile ** 2)
        noise_mean = np.mean(noise_profile)
        noise_std = np.std(noise_profile)
        
        print(f"\n{'='*70}")
        print(f"  📊 Noise Profile Characteristics")
        print(f"{'='*70}")
        print(f"  Frequency bins:  {len(noise_profile)}")
        print(f"  Mean magnitude:  {noise_mean:.6f}")
        print(f"  Std deviation:   {noise_std:.6f}")
        print(f"  Total energy:    {noise_energy:.2f}")
        
        # Find dominant noise frequencies
        sorted_indices = np.argsort(noise_profile)[::-1]
        print(f"\n  🎵 Top 5 Noise Frequencies:")
        for i, idx in enumerate(sorted_indices[:5], 1):
            freq = idx * (SAMPLE_RATE / 2) / len(noise_profile)
            mag = noise_profile[idx]
            print(f"     {i}. {freq:7.1f} Hz → magnitude: {mag:.6f}")
        print(f"{'='*70}\n")
        
        # Step 2: Find microphone
        print("\n[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Step 3: Record test audio
        raw_audio = record_audio(device_index, RECORDING_DURATION)
        
        # Step 4: Analyze raw audio
        print("\n" + "="*70)
        print("  BEFORE NOISE REDUCTION")
        print("="*70)
        raw_stats = analyze_audio(raw_audio, "📥 Raw Audio Signal")
        
        # Step 5: Apply noise reduction
        clean_audio = spectral_noise_subtraction(raw_audio, noise_profile, NOISE_REDUCTION_STRENGTH)
        
        # Step 6: Analyze cleaned audio
        print("\n" + "="*70)
        print("  AFTER NOISE REDUCTION")
        print("="*70)
        clean_stats = analyze_audio(clean_audio, "📤 Cleaned Audio Signal")
        
        # Step 7: Compare before/after
        rms_reduction = (1 - clean_stats['rms'] / raw_stats['rms']) * 100
        energy_reduction = (1 - clean_stats['total_energy'] / raw_stats['total_energy']) * 100
        
        print("\n" + "="*70)
        print("  📊 BEFORE vs AFTER COMPARISON")
        print("="*70)
        print(f"  RMS Reduction:    {rms_reduction:.1f}%")
        print(f"  Energy Reduction: {energy_reduction:.1f}%")
        print(f"  Peak Before:      {raw_stats['peak']:.6f}")
        print(f"  Peak After:       {clean_stats['peak']:.6f}")
        print("="*70)
        
        # Step 8: Recommendations
        print("\n" + "="*70)
        print("  💡 RECOMMENDATIONS")
        print("="*70)
        
        if energy_reduction < 20:
            print("  ⚠️  Low noise reduction (<20%)")
            print("     → Increase NOISE_REDUCTION_STRENGTH (try 0.90-0.95)")
            print("     → Or re-record noise profile with device enclosed")
        elif energy_reduction > 70:
            print("  ⚠️  Very high noise reduction (>70%)")
            print("     → May be removing speech content")
            print("     → Decrease NOISE_REDUCTION_STRENGTH (try 0.60-0.75)")
        else:
            print("  ✅ Noise reduction looks good (20-70% range)")
            print("     → Current setting seems optimal")
        
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[Audio] ⚠️  Test cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n[Audio] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

