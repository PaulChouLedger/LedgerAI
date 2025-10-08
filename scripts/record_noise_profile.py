#!/usr/bin/env python3
"""
Record Background Noise Profile for Fan Noise Reduction

This script records pure background noise (no speech) to create a noise profile
that will be used for spectral subtraction during transcription.

Usage:
    python3 scripts/record_noise_profile.py

The noise profile will be saved to: data/noise_profile.npy
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 60.0  # Record 10 seconds of pure background noise
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"

# Output path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "noise_profile.npy")

def find_device_index():
    """Find the ReSpeaker device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found. Available devices:\n{sd.query_devices()}")

def record_noise_profile(device_index, duration):
    """
    Record background noise for the specified duration
    
    Returns:
        numpy array of noise samples
    """
    print(f"\n{'='*70}")
    print(f"[Audio] 🎤 Recording {duration}s of background noise...")
    print(f"[Audio] ⚠️  IMPORTANT: Please remain COMPLETELY SILENT!")
    print(f"[Audio] 💡 This will capture your Jetson fan noise pattern.")
    print(f"{'='*70}\n")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"[Audio] Starting in {i}...")
        time.sleep(1)
    
    print("[Audio] 🔴 RECORDING... (stay silent)")
    
    # Calculate number of frames needed
    frames_needed = int(SAMPLE_RATE * duration / FRAME_SIZE)
    noise_samples = []
    
    with sd.InputStream(device=device_index, channels=6, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE, dtype="float32") as stream:
        for i in range(frames_needed):
            audio_block, _ = stream.read(FRAME_SIZE)
            channel_0 = audio_block[:, 0]  # Use channel 0 (same as listener.py)
            noise_samples.append(channel_0)
            
            # Progress indicator
            progress = (i + 1) / frames_needed * 100
            elapsed = (i + 1) * FRAME_DURATION
            print(f"[Audio] Recording: {elapsed:.1f}s / {duration:.1f}s ({progress:.0f}%)   ", end='\r')
    
    print(f"\n[Audio] ✅ Recording complete!")
    
    # Concatenate all samples
    noise_audio = np.concatenate(noise_samples)
    return noise_audio

def compute_noise_profile(noise_audio):
    """
    Compute the noise spectrum from the recorded audio
    
    Returns:
        numpy array containing the FFT magnitude (noise profile)
    """
    print(f"[Audio] 🔧 Computing noise spectrum via FFT...")
    
    # Compute FFT
    noise_fft = np.fft.rfft(noise_audio)
    noise_magnitude = np.abs(noise_fft)
    
    print(f"[Audio] ✅ Noise profile computed: {len(noise_magnitude)} frequency bins")
    return noise_magnitude

def save_noise_profile(noise_profile, output_path):
    """Save the noise profile to disk"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as numpy array
    np.save(output_path, noise_profile)
    
    file_size = os.path.getsize(output_path)
    print(f"[Audio] 💾 Noise profile saved to: {output_path}")
    print(f"[Audio] 📊 File size: {file_size / 1024:.1f} KB")

def analyze_noise_profile(noise_profile):
    """Analyze and display statistics about the noise profile"""
    print(f"\n{'='*70}")
    print("[Audio] 📊 Noise Profile Analysis:")
    print(f"{'='*70}")
    print(f"  Frequency bins:     {len(noise_profile)}")
    print(f"  Mean magnitude:     {np.mean(noise_profile):.6f}")
    print(f"  Max magnitude:      {np.max(noise_profile):.6f}")
    print(f"  Min magnitude:      {np.min(noise_profile):.6f}")
    print(f"  Std deviation:      {np.std(noise_profile):.6f}")
    
    # Find dominant frequencies
    sorted_indices = np.argsort(noise_profile)[::-1]
    top_5_bins = sorted_indices[:5]
    
    print(f"\n  Top 5 dominant frequencies:")
    for i, bin_idx in enumerate(top_5_bins, 1):
        freq = bin_idx * (SAMPLE_RATE / 2) / len(noise_profile)
        magnitude = noise_profile[bin_idx]
        print(f"    {i}. {freq:.1f} Hz (magnitude: {magnitude:.6f})")
    
    print(f"{'='*70}\n")

def main():
    """Main script execution"""
    print("\n" + "="*70)
    print("  🎙️  Background Noise Profile Recorder")
    print("  For Jetson Fan Noise Reduction")
    print("="*70)
    
    try:
        # Step 1: Find microphone
        print("\n[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Step 2: Record noise
        noise_audio = record_noise_profile(device_index, RECORDING_DURATION)
        duration = len(noise_audio) / SAMPLE_RATE
        print(f"[Audio] 📊 Recorded {len(noise_audio)} samples ({duration:.2f}s)")
        
        # Step 3: Compute noise profile
        noise_profile = compute_noise_profile(noise_audio)
        
        # Step 4: Analyze profile
        analyze_noise_profile(noise_profile)
        
        # Step 5: Save to disk
        save_noise_profile(noise_profile, OUTPUT_PATH)
        
        # Success message
        print("\n" + "="*70)
        print("✅ SUCCESS! Noise profile created successfully!")
        print("="*70)
        print(f"\n📁 Saved to: {OUTPUT_PATH}")
        print(f"\n💡 Next steps:")
        print(f"   1. Restart your Aura application")
        print(f"   2. The noise reduction will automatically use this profile")
        print(f"   3. Enjoy cleaner transcription with fan noise filtered out!")
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[Audio] ⚠️  Recording cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n[Audio] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

