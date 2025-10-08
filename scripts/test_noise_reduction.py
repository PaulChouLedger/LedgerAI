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
from scipy import interpolate

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 30.0  # Monitor for 30 seconds
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
NOISE_REDUCTION_STRENGTH = 0.85
DISPLAY_EVERY_N_FRAMES = 5  # Update display every N frames (~0.16s)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NOISE_PROFILE_PATH = os.path.join(PROJECT_DIR, "data", "noise_profile.npy")

# Global noise profile
noise_profile = None

def find_device_index():
    """Find the ReSpeaker device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

def spectral_noise_subtraction(audio, noise_profile_global, strength=0.85):
    """Apply spectral noise subtraction to a single frame"""
    # Compute FFT of signal
    fft_signal = np.fft.rfft(audio)
    magnitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)
    
    # Match noise profile length to signal length
    if len(noise_profile_global) != len(magnitude):
        x_old = np.linspace(0, 1, len(noise_profile_global))
        x_new = np.linspace(0, 1, len(magnitude))
        f = interpolate.interp1d(x_old, noise_profile_global, kind='linear', fill_value='extrapolate')
        noise_profile_matched = f(x_new)
    else:
        noise_profile_matched = noise_profile_global
    
    # Calculate energies before subtraction
    signal_energy = np.sum(magnitude ** 2)
    noise_energy = np.sum(noise_profile_matched ** 2)
    
    # Subtract noise profile from magnitude
    magnitude_clean = np.maximum(magnitude - strength * noise_profile_matched, 0)
    
    # Calculate energies after subtraction
    clean_energy = np.sum(magnitude_clean ** 2)
    energy_removed = signal_energy - clean_energy
    
    # Reconstruct signal
    fft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = np.fft.irfft(fft_clean, n=len(audio))
    
    return audio_clean, {
        'signal_energy': signal_energy,
        'noise_energy': noise_energy,
        'clean_energy': clean_energy,
        'energy_removed': energy_removed,
        'reduction_pct': (energy_removed / signal_energy * 100) if signal_energy > 0 else 0
    }

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

def process_audio_realtime(device_index, noise_profile, duration):
    """Process audio in real-time and display properties"""
    print(f"\n{'='*90}")
    print(f"  🎤 REAL-TIME AUDIO PROCESSING MONITOR")
    print(f"{'='*90}")
    print(f"  Duration: {duration}s | Noise Reduction Strength: {NOISE_REDUCTION_STRENGTH}")
    print(f"  Display Update: Every {DISPLAY_EVERY_N_FRAMES} frames (~{DISPLAY_EVERY_N_FRAMES * FRAME_DURATION:.2f}s)")
    print(f"{'='*90}\n")
    
    print("  Please speak or let ambient noise be captured...")
    print("\n  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)
    print("\n  🔴 MONITORING...")
    
    # Print header
    print(f"\n{'='*90}")
    print(f"{'Frame':<8} {'RAW':<30} │ {'REDUCTION':<25} │ {'CLEAN':<25}")
    print(f"{'-'*8} {'-'*30} │ {'-'*25} │ {'-'*25}")
    print(f"{'#':<8} {'RMS':<10} {'Peak':<10} {'Top Hz':<10} │ {'Removed':<12} {'%':<12} │ {'RMS':<10} {'Peak':<10} {'Top Hz':<5}")
    print(f"{'='*90}")
    
    # Running statistics
    frame_count = 0
    total_raw_rms = 0
    total_clean_rms = 0
    total_reduction = 0
    
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
                
                # Apply noise reduction
                clean_audio, stats = spectral_noise_subtraction(raw_audio, noise_profile, NOISE_REDUCTION_STRENGTH)
                
                # Analyze clean audio
                clean_rms = np.sqrt(np.mean(clean_audio ** 2))
                clean_peak = np.max(np.abs(clean_audio))
                clean_freqs = get_top_frequencies(clean_audio, top_n=1)
                
                # Update statistics
                frame_count += 1
                total_raw_rms += raw_rms
                total_clean_rms += clean_rms
                total_reduction += stats['reduction_pct']
                
                # Display every N frames
                if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                    raw_freq_str = f"{raw_freqs[0][0]:.0f}" if raw_freqs else "N/A"
                    clean_freq_str = f"{clean_freqs[0][0]:.0f}" if clean_freqs else "N/A"
                    
                    print(f"{frame_count:<8} "
                          f"{raw_rms:.6f}  {raw_peak:.4f}  {raw_freq_str:<10} │ "
                          f"{stats['energy_removed']:<12.2f} {stats['reduction_pct']:<12.1f} │ "
                          f"{clean_rms:.6f}  {clean_peak:.4f}  {clean_freq_str:<5}")
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*90}")
        print("  ⚠️  Monitoring stopped by user")
        print(f"{'='*90}\n")
    
    # Print summary statistics
    if frame_count > 0:
        avg_raw_rms = total_raw_rms / frame_count
        avg_clean_rms = total_clean_rms / frame_count
        avg_reduction = total_reduction / frame_count
        overall_reduction = (1 - avg_clean_rms / avg_raw_rms) * 100 if avg_raw_rms > 0 else 0
        
        print(f"\n{'='*90}")
        print(f"  📊 SUMMARY STATISTICS (from {frame_count} frames)")
        print(f"{'='*90}")
        print(f"  Average Raw RMS:       {avg_raw_rms:.6f}")
        print(f"  Average Clean RMS:     {avg_clean_rms:.6f}")
        print(f"  Average Reduction:     {avg_reduction:.1f}%")
        print(f"  Overall RMS Reduction: {overall_reduction:.1f}%")
        print(f"\n  💡 INTERPRETATION:")
        
        if avg_reduction < 20:
            print(f"     ⚠️  Low noise reduction (<20%)")
            print(f"        → Noise profile may not match current environment")
            print(f"        → Try increasing NOISE_REDUCTION_STRENGTH")
            print(f"        → Or re-record noise profile with device fully enclosed")
        elif avg_reduction > 70:
            print(f"     ⚠️  Very high reduction (>70%)")
            print(f"        → May be removing speech content")
            print(f"        → Try decreasing NOISE_REDUCTION_STRENGTH to 0.65-0.75")
        else:
            print(f"     ✅ Noise reduction looks good (20-70% range)")
            print(f"        → Current settings appear optimal")
        
        print(f"{'='*90}\n")

def main():
    """Main execution"""
    print("\n" + "="*90)
    print("  🧪 REAL-TIME NOISE REDUCTION MONITOR")
    print("="*90)
    
    try:
        # Load noise profile
        global noise_profile
        print(f"\n[Audio] 🔍 Loading noise profile...")
        
        if not os.path.exists(NOISE_PROFILE_PATH):
            print(f"[Audio] ❌ Noise profile not found: {NOISE_PROFILE_PATH}")
            print(f"[Audio] 💡 Run: python3 scripts/record_noise_profile.py")
            return 1
        
        noise_profile = np.load(NOISE_PROFILE_PATH)
        print(f"[Audio] ✅ Loaded: {len(noise_profile)} frequency bins")
        
        # Show noise profile stats
        noise_mean = np.mean(noise_profile)
        noise_energy = np.sum(noise_profile ** 2)
        print(f"[Audio] 📊 Noise profile energy: {noise_energy:.2f}, mean: {noise_mean:.6f}")
        
        # Find microphone
        print(f"\n[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Start real-time processing
        process_audio_realtime(device_index, noise_profile, RECORDING_DURATION)
        
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
