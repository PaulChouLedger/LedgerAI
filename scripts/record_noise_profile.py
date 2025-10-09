#!/usr/bin/env python3
"""
Record Noise Profile - Capture device noise signature

Records 5 seconds of silence to create a noise profile (digital signature)
of the fan/turbulent noise in the device. This profile is then used for
spectral subtraction during speech recognition.

Usage:
    python3 scripts/record_noise_profile.py

The noise profile will be saved to: data/noise_profile.npy
"""

import sys
import os
import time
import numpy as np
import sounddevice as sd

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 5.0  # 5 seconds of silence
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
OUTPUT_PATH = os.path.expanduser("~/LedgerAI/data/noise_profile.npy")

def find_device_index():
    """Find the ReSpeaker device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

def record_noise_profile(device_index, duration):
    """Record noise profile from silent environment"""
    print(f"\n{'='*80}")
    print(f"  🎙️  NOISE PROFILE RECORDING")
    print(f"{'='*80}")
    print(f"  This will record {duration} seconds of SILENCE to capture the")
    print(f"  noise signature of your device (fan, turbulence, etc.)")
    print(f"\n  ⚠️  IMPORTANT: Do NOT speak during recording!")
    print(f"  ⚠️  Let the device run idle for {duration} seconds.")
    print(f"{'='*80}\n")
    
    input("Press ENTER when ready to start recording...")
    
    print("\n  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)
    print("\n  🔴 RECORDING SILENCE... (stay quiet!)\n")
    
    frames_needed = int(SAMPLE_RATE * duration / FRAME_SIZE)
    all_frames = []
    
    try:
        with sd.InputStream(device=device_index, channels=6, samplerate=SAMPLE_RATE,
                            blocksize=FRAME_SIZE, dtype="float32") as stream:
            for i in range(frames_needed):
                # Read frame
                audio_block, _ = stream.read(FRAME_SIZE)
                raw_audio = audio_block[:, 0]  # Channel 0
                all_frames.append(raw_audio)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    elapsed = (i + 1) * FRAME_DURATION
                    print(f"  Recording... {elapsed:.1f}s / {duration}s", end="\r")
        
        print(f"\n\n  ✅ Recording complete!")
        
    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  Recording cancelled by user")
        return None
    
    # Concatenate all frames
    full_audio = np.concatenate(all_frames)
    
    # Compute noise profile (average power spectrum)
    print(f"\n  🔬 Analyzing noise spectrum...")
    
    # Use FFT to get frequency spectrum
    fft = np.fft.rfft(full_audio)
    magnitude = np.abs(fft)
    
    # Average magnitude spectrum is our noise profile
    noise_profile = magnitude
    
    # Show statistics
    avg_rms = np.sqrt(np.mean(full_audio ** 2))
    peak = np.max(np.abs(full_audio))
    
    print(f"\n  📊 Noise Profile Statistics:")
    print(f"     Duration: {duration}s")
    print(f"     Samples: {len(full_audio)}")
    print(f"     Average RMS: {avg_rms:.6f}")
    print(f"     Peak: {peak:.4f}")
    print(f"     Spectrum bins: {len(noise_profile)}")
    
    # Find dominant noise frequencies
    freqs = np.fft.rfftfreq(len(full_audio), 1/SAMPLE_RATE)
    top_indices = np.argsort(magnitude)[-5:][::-1]
    
    print(f"\n  🎵 Top 5 Noise Frequencies:")
    for idx in top_indices:
        freq = freqs[idx]
        mag = magnitude[idx]
        print(f"     {freq:7.1f} Hz - magnitude {mag:.2f}")
    
    return noise_profile

def save_noise_profile(noise_profile, output_path):
    """Save noise profile to disk"""
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as numpy array
    np.save(output_path, noise_profile)
    
    file_size = os.path.getsize(output_path)
    print(f"\n  💾 Noise profile saved to: {output_path}")
    print(f"     File size: {file_size} bytes")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🎙️  NOISE PROFILE RECORDER")
    print("  Creates a digital signature of device noise for spectral subtraction")
    print("="*80)
    
    try:
        # Find microphone
        print(f"\n[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Record noise profile
        noise_profile = record_noise_profile(device_index, RECORDING_DURATION)
        
        if noise_profile is None:
            return 1
        
        # Save to disk
        save_noise_profile(noise_profile, OUTPUT_PATH)
        
        print(f"\n{'='*80}")
        print(f"  ✅ SUCCESS!")
        print(f"  Your noise profile has been saved.")
        print(f"  Restart your listener to use spectral noise reduction.")
        print(f"{'='*80}\n")
        
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

