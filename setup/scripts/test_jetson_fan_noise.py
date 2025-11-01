#!/usr/bin/env python3
"""
Test script to measure fan noise impact on microphone.
Compares audio features with Jetson under different fan speeds.
"""

import time
import subprocess
import numpy as np
import sounddevice as sd
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE = 16000
DURATION = 3  # seconds
DEVICE_NAME = "XVF3800 4-Mic Array"

def get_fan_speed():
    """Get current Jetson fan speed"""
    try:
        result = subprocess.run(
            ["cat", "/sys/devices/pwm-fan/target_pwm"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return None

def record_sample(duration=DURATION):
    """Record audio sample"""
    print(f"  Recording {duration}s sample...")
    audio = sd.rec(int(duration * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=2,  # XVF3800 has 2 channels
                   device=DEVICE_NAME,
                   dtype='float32')
    sd.wait()
    return audio[:, 0]  # Channel 0

def analyze_audio(audio):
    """Analyze audio for fan noise characteristics"""
    # RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    # FFT
    fft_vals = rfft(audio)
    fft_freq = rfftfreq(len(audio), 1/SAMPLE_RATE)
    magnitude = np.abs(fft_vals)
    
    # Find peak frequencies (likely fan noise)
    top_5_indices = np.argsort(magnitude)[-5:][::-1]
    top_5_freqs = fft_freq[top_5_indices]
    top_5_mags = magnitude[top_5_indices]
    
    # Low frequency energy (fan rumble)
    low_freq_mask = fft_freq <= 200
    low_freq_energy = np.sum(magnitude[low_freq_mask] ** 2)
    total_energy = np.sum(magnitude ** 2)
    low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
    
    return {
        'rms': rms,
        'low_freq_ratio': low_freq_ratio,
        'peak_freqs': top_5_freqs[:3],  # Top 3
        'peak_mags': top_5_mags[:3]
    }

def main():
    print("\n" + "="*70)
    print("  🔍 JETSON FAN NOISE DIAGNOSTIC")
    print("="*70)
    print("\nThis test measures fan noise impact on microphone.")
    print("Speak the SAME phrase after each test for comparison.\n")
    
    fan_speed = get_fan_speed()
    if fan_speed:
        print(f"Current Jetson fan speed: {fan_speed} PWM\n")
    
    # Test 1: Silence (measure baseline fan noise)
    print("\n[Test 1/3] SILENCE TEST - Measuring fan noise baseline")
    print("           Keep quiet for 3 seconds...")
    time.sleep(2)
    silence_audio = record_sample()
    silence_stats = analyze_audio(silence_audio)
    
    print(f"  Baseline RMS:           {silence_stats['rms']:.6f}")
    print(f"  Low-freq ratio:         {silence_stats['low_freq_ratio']:.3f}")
    print(f"  Peak frequencies:       {silence_stats['peak_freqs']}")
    
    # Test 2: Speech
    print("\n[Test 2/3] SPEECH TEST")
    print("           Say: 'Hello, testing microphone with Jetson fan'")
    print("           Starting in 3 seconds...")
    time.sleep(3)
    speech_audio = record_sample()
    speech_stats = analyze_audio(speech_audio)
    
    print(f"  Speech RMS:             {speech_stats['rms']:.6f}")
    print(f"  Low-freq ratio:         {speech_stats['low_freq_ratio']:.3f}")
    print(f"  Peak frequencies:       {speech_stats['peak_freqs']}")
    
    # Analysis
    print("\n" + "="*70)
    print("  📊 ANALYSIS")
    print("="*70)
    
    noise_floor = silence_stats['rms']
    snr = 20 * np.log10(speech_stats['rms'] / noise_floor) if noise_floor > 0 else 0
    
    print(f"\n  Signal-to-Noise Ratio:  {snr:.1f} dB")
    if snr < 10:
        print("  ⚠️  SNR is LOW - Fan noise is significant problem!")
    elif snr < 20:
        print("  ⚠️  SNR is MODERATE - Fan noise affecting quality")
    else:
        print("  ✅ SNR is GOOD - Fan noise minimal")
    
    print(f"\n  Noise Floor (silence):  {noise_floor:.6f} RMS")
    print(f"  Speech Level:           {speech_stats['rms']:.6f} RMS")
    
    # Check for fan frequencies
    common_fan_freqs = [50, 60, 100, 120, 150, 200]  # Hz
    print(f"\n  Potential Fan Noise Frequencies:")
    for freq in silence_stats['peak_freqs']:
        nearby_fan_freq = min(common_fan_freqs, key=lambda x: abs(x - freq))
        if abs(freq - nearby_fan_freq) < 10:
            print(f"    {freq:.1f} Hz ⚠️  (near {nearby_fan_freq} Hz - likely fan)")
        else:
            print(f"    {freq:.1f} Hz")
    
    # Recommendations
    print("\n" + "="*70)
    print("  💡 RECOMMENDATIONS")
    print("="*70)
    
    if silence_stats['low_freq_ratio'] > 0.3:
        print("\n  ⚠️  HIGH low-frequency energy detected (>30%)")
        print("     This indicates fan/HVAC noise or poor microphone placement")
        print("\n  Solutions:")
        print("     1. Add foam vibration isolation between Jetson and mic")
        print("     2. Reposition microphone away from Jetson")
        print("     3. Use aggressive HPF: sudo python3 scripts/tune_respeaker.py beamforming_aggressive")
    
    if snr < 15:
        print("\n  ⚠️  LOW Signal-to-Noise Ratio")
        print("     Fan noise is significantly affecting microphone")
        print("\n  Solutions:")
        print("     1. PRIORITY: Physically separate mic from Jetson")
        print("     2. Add acoustic foam to enclosure walls")
        print("     3. Test with Jetson fan at lower speed (if possible)")
        print("     4. Use beamforming: sudo python3 scripts/tune_respeaker.py beamforming_aggressive")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

