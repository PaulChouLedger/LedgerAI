#!/usr/bin/env python3
"""
Raw Audio Monitor - Simple 20-second recording test

Shows raw audio statistics from the ReSpeaker microphone:
- RMS (signal strength)
- Peak (loudest sample)
- Top 3 dominant frequencies

No processing, just raw data capture.
"""

import sys
import time
import numpy as np
import sounddevice as sd

# === Config ===
SAMPLE_RATE = 16000
FRAME_DURATION = 0.032
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
RECORDING_DURATION = 20.0  # 20 seconds
DEVICE_NAME = "ReSpeaker 4 Mic Array (UAC1.0)"
DISPLAY_EVERY_N_FRAMES = 5  # Update display every N frames (~0.16s)

def find_device_index():
    """Find the ReSpeaker device index"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if DEVICE_NAME.lower() in device["name"].lower() and device["max_input_channels"] >= 6:
            print(f"[Audio] 🎧 Found device: {device['name']} (index {i})")
            return i
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

def get_top_frequencies(audio, top_n=3):
    """Get top N dominant frequencies from audio"""
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
    
    sorted_indices = np.argsort(magnitude)[::-1]
    top_freqs = []
    for idx in sorted_indices[:top_n]:
        if magnitude[idx] > 0.01:  # Only significant peaks
            top_freqs.append(freqs[idx])
    
    return top_freqs

def monitor_audio(device_index, duration):
    """Monitor raw audio for specified duration"""
    print(f"\n{'='*80}")
    print(f"  🎤 RAW AUDIO MONITOR - 20 Second Test")
    print(f"{'='*80}")
    print(f"  Device: {DEVICE_NAME}")
    print(f"  Duration: {duration}s")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Update: Every {DISPLAY_EVERY_N_FRAMES} frames (~{DISPLAY_EVERY_N_FRAMES * FRAME_DURATION:.2f}s)")
    print(f"{'='*80}\n")
    
    print("\n  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)
    print("\n  🔴 RECORDING...\n")
    
    # Print header
    print(f"\n{'='*80}")
    print(f"{'Frame':<8} {'RMS':<12} {'Peak':<12} {'Top 3 Frequencies (Hz)':<40}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*40}")
    print(f"{'='*80}")
    
    # Running statistics
    frame_count = 0
    total_rms = 0
    total_peak = 0
    max_rms = 0
    max_peak = 0
    
    frames_needed = int(SAMPLE_RATE * duration / FRAME_SIZE)
    
    try:
        with sd.InputStream(device=device_index, channels=6, samplerate=SAMPLE_RATE,
                            blocksize=FRAME_SIZE, dtype="float32") as stream:
            for i in range(frames_needed):
                # Read frame
                audio_block, _ = stream.read(FRAME_SIZE)
                raw_audio = audio_block[:, 0]  # Channel 0
                
                # Analyze raw audio
                rms = np.sqrt(np.mean(raw_audio ** 2))
                peak = np.max(np.abs(raw_audio))
                
                # Update statistics
                frame_count += 1
                total_rms += rms
                total_peak += peak
                max_rms = max(max_rms, rms)
                max_peak = max(max_peak, peak)
                
                # Display every N frames
                if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                    freqs = get_top_frequencies(raw_audio, top_n=3)
                    freq_str = ", ".join([f"{f:.0f}" for f in freqs]) if freqs else "none"
                    
                    print(f"{frame_count:<8} "
                          f"{rms:<12.6f} "
                          f"{peak:<12.4f} "
                          f"{freq_str:<40}")
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("  ⚠️  Recording stopped by user")
        print(f"{'='*80}\n")
    
    # Print summary statistics
    if frame_count > 0:
        avg_rms = total_rms / frame_count
        avg_peak = total_peak / frame_count
        
        print(f"\n{'='*80}")
        print(f"  📊 SUMMARY STATISTICS (from {frame_count} frames)")
        print(f"{'='*80}")
        print(f"  Average RMS:       {avg_rms:.6f}")
        print(f"  Average Peak:      {avg_peak:.4f}")
        print(f"  Maximum RMS:       {max_rms:.6f}")
        print(f"  Maximum Peak:      {max_peak:.4f}")
        
        print(f"\n  💡 INTERPRETATION:")
        
        if avg_rms < 0.003:
            print(f"     🔇 Very quiet (RMS < 0.003) - silence or very far speech")
        elif avg_rms < 0.010:
            print(f"     🔉 Quiet (RMS 0.003-0.010) - far-field speech or background noise")
        elif avg_rms < 0.030:
            print(f"     🔊 Moderate (RMS 0.010-0.030) - mid-field speech")
        else:
            print(f"     📢 Loud (RMS > 0.030) - near-field speech")
        
        if max_peak > 0.8:
            print(f"\n     ⚠️  Peak > 0.8 detected - may clip with high gain")
        elif max_peak > 0.5:
            print(f"\n     ✅ Good signal level (peak 0.5-0.8)")
        else:
            print(f"\n     ℹ️  Low signal level (peak < 0.5)")
        
        print(f"{'='*80}\n")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🧪 RAW AUDIO MONITOR - No Processing")
    print("="*80)
    
    try:
        # Find microphone
        print(f"\n[Audio] 🔍 Searching for microphone...")
        device_index = find_device_index()
        
        # Start monitoring
        monitor_audio(device_index, RECORDING_DURATION)
        
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

