#!/usr/bin/env python3
"""
Test script to measure EM noise impact at different Jetson power modes.
Compares audio quality (noise floor, SNR) across different power levels.

EM Noise Mechanisms:
1. Power Supply Ripple: Higher power = more current = voltage fluctuations
2. USB Power Rail Noise: USB devices share power with Jetson - noise can couple
3. EMI from CPU/GPU: Higher clocks = more electromagnetic interference
4. Ground Loops: Power differences can create ground potential issues
5. Voltage Instability: Too low power = insufficient/fluctuating USB power
"""

import time
import subprocess
import numpy as np
import sounddevice as sd
from scipy.fft import rfft, rfftfreq
import sys

SAMPLE_RATE = 16000
DURATION = 3  # seconds for silence test
DEVICE_NAME = "XVF3800 4-Mic Array"

def get_power_mode():
    """Get current Jetson power mode"""
    try:
        result = subprocess.run(
            ["sudo", "nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5
        )
        # Parse output like "NV Power Mode: 1"
        for line in result.stdout.split('\n'):
            if 'NV Power Mode:' in line:
                mode = line.split(':')[-1].strip()
                return int(mode)
        return None
    except Exception as e:
        print(f"⚠️  Could not get power mode: {e}")
        return None

def set_power_mode(mode):
    """Set Jetson power mode"""
    try:
        result = subprocess.run(
            ["sudo", "nvpmodel", "-m", str(mode)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            time.sleep(3)  # Wait for power mode to stabilize
            return True
        return False
    except Exception as e:
        print(f"⚠️  Could not set power mode {mode}: {e}")
        return False

def get_fan_speed():
    """Get current Jetson fan speed (acoustic noise indicator)"""
    try:
        result = subprocess.run(
            ["cat", "/sys/devices/pwm-fan/target_pwm"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return None

def record_silence(duration=DURATION):
    """Record silence to measure noise floor"""
    print(f"  Recording {duration}s of silence...", end="", flush=True)
    try:
        # Find device
        devices = sd.query_devices()
        device_index = None
        for i, dev in enumerate(devices):
            if DEVICE_NAME in dev['name']:
                device_index = i
                break
        
        if device_index is None:
            print(f"\n  ❌ Device '{DEVICE_NAME}' not found")
            return None
        
        # Record silence
        audio = sd.rec(
            int(SAMPLE_RATE * duration),
            samplerate=SAMPLE_RATE,
            channels=2,
            device=device_index,
            dtype='float32'
        )
        sd.wait()
        print(" ✅")
        
        # Use channel 0 (beamformed output)
        return audio[:, 0]
    except Exception as e:
        print(f"\n  ❌ Recording failed: {e}")
        return None

def analyze_noise(audio):
    """Analyze noise characteristics"""
    if audio is None or len(audio) == 0:
        return None
    
    # Time domain
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    
    # Frequency domain
    fft = rfft(audio)
    freqs = rfftfreq(len(audio), 1/SAMPLE_RATE)
    magnitude = np.abs(fft)
    
    # Power spectral density
    psd = magnitude ** 2
    
    # Low frequency noise (0-200 Hz) - power supply ripple, fan noise
    low_freq_mask = (freqs >= 0) & (freqs <= 200)
    low_freq_energy = np.sum(psd[low_freq_mask])
    total_energy = np.sum(psd)
    low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
    
    # Mid frequency noise (200-2000 Hz) - USB noise, switching noise
    mid_freq_mask = (freqs >= 200) & (freqs <= 2000)
    mid_freq_energy = np.sum(psd[mid_freq_mask])
    mid_freq_ratio = mid_freq_energy / total_energy if total_energy > 0 else 0
    
    # High frequency noise (2000+ Hz) - digital switching, EMI
    high_freq_mask = freqs >= 2000
    high_freq_energy = np.sum(psd[high_freq_mask])
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # Find peak noise frequencies
    peak_indices = np.argsort(magnitude)[-5:][::-1]
    peak_freqs = [freqs[i] for i in peak_indices if magnitude[i] > np.max(magnitude) * 0.1]
    
    return {
        'rms': rms,
        'peak': peak,
        'low_freq_ratio': low_freq_ratio,
        'mid_freq_ratio': mid_freq_ratio,
        'high_freq_ratio': high_freq_ratio,
        'peak_freqs': peak_freqs[:5],
        'total_energy': total_energy
    }

def main():
    print("\n" + "="*70)
    print("  🔌 JETSON POWER MODE EM NOISE TEST")
    print("="*70)
    print("\nThis test measures electromagnetic noise at different power levels.")
    print("Higher power = more current = potential for more EM interference.")
    print("\n⚠️  Requires sudo access to change power modes")
    print("="*70 + "\n")
    
    # Get current mode
    current_mode = get_power_mode()
    if current_mode is not None:
        print(f"Current Power Mode: {current_mode}")
    else:
        print("⚠️  Could not detect current power mode")
    
    # Test modes (0=MAXN, 1=10W, 2=15W, 3=20W typically)
    test_modes = [1, 2, 0, 3]  # Test 10W, 15W, MAXN, 20W
    results = {}
    
    for mode in test_modes:
        print(f"\n{'='*70}")
        print(f"  Testing Power Mode {mode}")
        print(f"{'='*70}")
        
        # Set power mode
        print(f"\n  Setting power mode to {mode}...", end="", flush=True)
        if set_power_mode(mode):
            print(" ✅")
            time.sleep(2)  # Additional stabilization
        else:
            print(" ❌ Failed - skipping this mode")
            continue
        
        # Get fan speed (indicator of power/heat)
        fan_speed = get_fan_speed()
        if fan_speed:
            print(f"  Fan Speed: {fan_speed} PWM")
        
        # Record silence
        audio = record_silence()
        if audio is None:
            continue
        
        # Analyze
        stats = analyze_noise(audio)
        if stats:
            results[mode] = {
                'stats': stats,
                'fan_speed': fan_speed
            }
            
            print(f"\n  📊 Noise Analysis:")
            print(f"     RMS (noise floor):     {stats['rms']:.6f}")
            print(f"     Peak:                  {stats['peak']:.4f}")
            print(f"     Low-freq ratio (0-200Hz):  {stats['low_freq_ratio']:.3f} (power supply/fan)")
            print(f"     Mid-freq ratio (200-2kHz): {stats['mid_freq_ratio']:.3f} (USB/switching)")
            print(f"     High-freq ratio (2kHz+):   {stats['high_freq_ratio']:.3f} (digital EMI)")
            print(f"     Peak noise frequencies: {[f'{f:.1f}Hz' for f in stats['peak_freqs']]}")
    
    # Restore original mode
    if current_mode is not None:
        print(f"\n{'='*70}")
        print(f"  Restoring original power mode {current_mode}...")
        set_power_mode(current_mode)
        print("  ✅ Restored")
    
    # Summary
    print(f"\n{'='*70}")
    print("  📊 SUMMARY - EM NOISE BY POWER MODE")
    print(f"{'='*70}\n")
    
    if not results:
        print("  ⚠️  No data collected")
        return
    
    # Sort by mode
    sorted_modes = sorted(results.keys())
    
    print(f"{'Mode':<6} {'Power':<8} {'RMS':<12} {'Low-Freq':<10} {'Mid-Freq':<10} {'High-Freq':<10}")
    print("-" * 70)
    
    mode_power_map = {
        0: "~25W (MAXN)",
        1: "~10W",
        2: "~15W",
        3: "~20W"
    }
    
    for mode in sorted_modes:
        stats = results[mode]['stats']
        power = mode_power_map.get(mode, f"Mode {mode}")
        print(f"{mode:<6} {power:<8} {stats['rms']:<12.6f} {stats['low_freq_ratio']:<10.3f} "
              f"{stats['mid_freq_ratio']:<10.3f} {stats['high_freq_ratio']:<10.3f}")
    
    # Find best mode (lowest noise floor)
    best_mode = min(results.keys(), key=lambda m: results[m]['stats']['rms'])
    best_stats = results[best_mode]['stats']
    
    print(f"\n{'='*70}")
    print("  💡 RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    print(f"  ✅ Best Mode (Lowest Noise): Mode {best_mode} ({mode_power_map.get(best_mode, 'Unknown')})")
    print(f"     Noise Floor: {best_stats['rms']:.6f} RMS")
    
    # Check for EM noise issues
    for mode in sorted_modes:
        stats = results[mode]['stats']
        issues = []
        
        if stats['mid_freq_ratio'] > 0.3:
            issues.append("High USB/switching noise")
        if stats['high_freq_ratio'] > 0.2:
            issues.append("High digital EMI")
        if stats['rms'] > 0.001:
            issues.append("High overall noise floor")
        
        if issues:
            print(f"\n  ⚠️  Mode {mode} ({mode_power_map.get(mode, 'Unknown')}):")
            for issue in issues:
                print(f"     - {issue}")
    
    print(f"\n  💡 Power Mode Impact on EM Noise:")
    print(f"     - Lower power (10W): May have voltage instability → USB noise")
    print(f"     - Medium power (15W): Often optimal balance")
    print(f"     - High power (25W): More EMI from CPU/GPU, but stable USB power")
    print(f"\n  💡 If transcription is poor, try:")
    print(f"     1. Test at Mode 2 (15W) - often the sweet spot")
    print(f"     2. Use USB isolator to break ground loops")
    print(f"     3. Use shielded USB cable")
    print(f"     4. Physically separate microphone from Jetson")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        # Try to restore original mode
        current_mode = get_power_mode()
        if current_mode is not None:
            set_power_mode(current_mode)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

