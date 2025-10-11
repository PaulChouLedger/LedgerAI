#!/usr/bin/env python3
"""
Simple frequency analyzer (no plotting - terminal output only)
Perfect for headless Jetson or systems without matplotlib

Usage:
    python3 scripts/record_frequencies_simple.py
"""

import numpy as np
import sounddevice as sd
from scipy.fft import rfft, rfftfreq
from scipy import signal
import time
import sys

# === Configuration ===
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
SAMPLE_RATE = 16000
DURATION = 8  # seconds
CHANNELS = 6
CHANNEL_TO_ANALYZE = 0

# === Find Device ===
def find_device():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower():
            print(f"✅ Found: {device['name']} (index {i})")
            return i
    print("❌ ReSpeaker not found!")
    sys.exit(1)

# === Main ===
def main():
    print("="*70)
    print("  ReSpeaker Frequency Analyzer (Simple)")
    print("="*70)
    
    device_index = find_device()
    
    print(f"\n🎙️  Recording for {DURATION} seconds from channel {CHANNEL_TO_ANALYZE}...")
    
    # Record
    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=device_index,
        dtype='float32'
    )
    
    # Show progress
    for i in range(DURATION):
        time.sleep(1)
        print(f"⏱️  {i+1}/{DURATION} seconds", end='\r')
    
    sd.wait()
    print(f"\n✅ Recording complete!")
    
    # Extract channel
    channel_data = recording[:, CHANNEL_TO_ANALYZE]
    
    # Calculate stats
    rms = np.sqrt(np.mean(channel_data ** 2))
    peak = np.max(np.abs(channel_data))
    
    print(f"\n📊 Audio Statistics:")
    print(f"   RMS: {rms:.6f}")
    print(f"   Peak: {peak:.4f}")
    print(f"   Samples: {len(channel_data)}")
    
    # FFT Analysis
    print(f"\n🔬 Performing FFT analysis...")
    window = signal.windows.hann(len(channel_data))
    windowed = channel_data * window
    
    fft_values = rfft(windowed)
    fft_freqs = rfftfreq(len(windowed), 1/SAMPLE_RATE)
    magnitude = np.abs(fft_values)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Top frequencies
    top_indices = np.argsort(magnitude)[-20:][::-1]
    
    print(f"\n🎵 Top 20 Detected Frequencies:")
    print(f"{'Rank':<6} {'Freq (Hz)':<12} {'Mag (dB)':<12} {'Type'}")
    print("-" * 55)
    
    for rank, idx in enumerate(top_indices, 1):
        freq = fft_freqs[idx]
        mag_db = magnitude_db[idx]
        
        if freq < 60:
            freq_type = "Bass/Noise"
        elif freq < 250:
            freq_type = "Low"
        elif freq < 2000:
            freq_type = "Mid (Voice)"
        elif freq < 4000:
            freq_type = "High (Voice)"
        elif freq < 8000:
            freq_type = "Presence"
        else:
            freq_type = "Treble"
        
        print(f"{rank:<6} {freq:<12.1f} {mag_db:<12.1f} {freq_type}")
    
    # Save results
    output_file = f"freq_analysis_{int(time.time())}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Frequency Analysis Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"RMS: {rms:.6f}\n")
        f.write(f"Peak: {peak:.4f}\n")
        f.write(f"Sample Rate: {SAMPLE_RATE} Hz\n")
        f.write(f"Duration: {DURATION} seconds\n\n")
        f.write(f"Top Frequencies:\n")
        f.write(f"{'Rank':<6} {'Freq (Hz)':<12} {'Mag (dB)':<12}\n")
        f.write("-" * 40 + "\n")
        for rank, idx in enumerate(top_indices, 1):
            f.write(f"{rank:<6} {fft_freqs[idx]:<12.1f} {magnitude_db[idx]:<12.1f}\n")
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

