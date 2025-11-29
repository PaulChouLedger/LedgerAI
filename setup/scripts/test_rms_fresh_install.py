#!/usr/bin/env python3
"""
Quick RMS Level Test for Fresh Jetson Install

Tests microphone input RMS levels to verify device is working correctly
before running install_aura_bootable.sh. Helps identify if installation
script causes microphone issues.

Usage:
    python3 test_rms_fresh_install.py
    
Requirements:
    - sounddevice (pip install sounddevice)
    - numpy (pip install numpy)
"""

import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("❌ ERROR: sounddevice not installed")
    print("   Install with: pip install sounddevice")
    sys.exit(1)

# === Config ===
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
DEVICE_NAMES = ["reSpeaker", "XVF3800", "ArrayUAC10", "UACDemo"]

def find_device():
    """Find microphone device"""
    print("\n" + "="*70)
    print("  🔍 SEARCHING FOR MICROPHONE DEVICE")
    print("="*70 + "\n")
    
    devices = sd.query_devices()
    print(f"Found {len(devices)} audio devices:\n")
    
    microphone_devices = []
    for i, device in enumerate(devices):
        name = device["name"]
        max_input_channels = device.get("max_input_channels", 0)
        
        # Check if it's an input device
        if max_input_channels > 0:
            print(f"  [{i:2d}] {name}")
            print(f"       Input channels: {max_input_channels}, Sample rate: {device.get('default_samplerate', 'unknown')}")
            
            # Check if it matches our device names
            for dev_name in DEVICE_NAMES:
                if dev_name.lower() in name.lower():
                    microphone_devices.append((i, device))
                    print(f"       ✅ MATCHES: {dev_name}")
            print()
    
    if not microphone_devices:
        print("\n⚠️  No ReSpeaker/XVF3800 device found!")
        print("   Available input devices listed above.")
        print("   Make sure the microphone is connected via USB.")
        return None
    
    # Use first matching device
    device_index, device = microphone_devices[0]
    print(f"\n✅ Using device: {device['name']} (index {device_index})")
    return device_index

def test_rms_levels(device_index):
    """Test RMS levels with continuous recording"""
    print("\n" + "="*70)
    print("  🎤 TESTING RMS LEVELS")
    print("="*70 + "\n")
    print("This will record 3 seconds of audio.")
    print("Please speak clearly when recording starts.\n")
    
    input("Press ENTER when ready to record...")
    
    print("\n  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)
    print("\n  🔴 RECORDING... (speak now!)\n")
    
    try:
        # Record audio
        audio = sd.rec(
            int(SAMPLE_RATE * DURATION),
            samplerate=SAMPLE_RATE,
            channels=2,
            device=device_index,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        print("  ✅ Recording complete!\n")
        
        # Calculate statistics for each channel
        print("="*70)
        print("  📊 RMS LEVEL ANALYSIS")
        print("="*70 + "\n")
        
        for channel in range(2):
            channel_audio = audio[:, channel]
            
            # Calculate metrics
            rms = np.sqrt(np.mean(channel_audio ** 2))
            peak = np.max(np.abs(channel_audio))
            mean = np.mean(np.abs(channel_audio))
            
            # Determine status
            if rms > 0.05:
                status = "✅ GOOD (speech detected)"
            elif rms > 0.02:
                status = "⚠️  LOW (weak signal)"
            else:
                status = "❌ VERY LOW (silence/noise only)"
            
            print(f"Channel {channel}:")
            print(f"  RMS:     {rms:.6f} {status}")
            print(f"  Peak:    {peak:.4f}")
            print(f"  Mean:    {mean:.6f}")
            print()
        
        # Overall assessment
        overall_rms = np.sqrt(np.mean(audio ** 2))
        overall_peak = np.max(np.abs(audio))
        
        print("="*70)
        print("  📈 OVERALL ASSESSMENT")
        print("="*70 + "\n")
        
        print(f"Overall RMS:  {overall_rms:.6f}")
        print(f"Overall Peak: {overall_peak:.4f}\n")
        
        if overall_rms > 0.05:
            print("✅ SUCCESS: Microphone is working correctly!")
            print("   RMS levels indicate speech was captured.")
        elif overall_rms > 0.02:
            print("⚠️  WARNING: Low RMS levels detected.")
            print("   Microphone may be working but signal is weak.")
            print("   Check:")
            print("   - Microphone distance")
            print("   - Volume/gain settings")
            print("   - USB connection")
        else:
            print("❌ FAILURE: Very low RMS levels.")
            print("   Microphone may not be capturing audio correctly.")
            print("   Check:")
            print("   - USB connection")
            print("   - Device permissions")
            print("   - Try unplugging/replugging USB")
        
        print("\n" + "="*70)
        print("  💡 EXPECTED VALUES")
        print("="*70 + "\n")
        print("Normal speech (1-2 feet away):")
        print("  RMS:  0.05 - 0.15")
        print("  Peak: 0.20 - 0.80")
        print("\nQuiet speech or far-field:")
        print("  RMS:  0.02 - 0.05")
        print("  Peak: 0.10 - 0.30")
        print("\nSilence/background noise:")
        print("  RMS:  < 0.02")
        print("  Peak: < 0.10")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during recording: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_system_info():
    """Check system information"""
    print("\n" + "="*70)
    print("  💻 SYSTEM INFORMATION")
    print("="*70 + "\n")
    
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check sounddevice
    try:
        print(f"sounddevice version: {sd.__version__}")
    except:
        print("sounddevice version: unknown")
    
    # Check if running on Jetson
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            print(f"Device: {model}")
    except:
        print("Device: Unknown (not a Jetson?)")
    
    # Check USB devices
    print("\nUSB Audio Devices:")
    try:
        import subprocess
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=2)
        usb_lines = [line for line in result.stdout.split('\n') if 'Audio' in line or 'XVF' in line or 'UAC' in line]
        if usb_lines:
            for line in usb_lines:
                print(f"  {line}")
        else:
            print("  (No USB audio devices found in lsusb)")
    except:
        print("  (Could not check USB devices)")
    
    print()

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("  🎤 RMS LEVEL TEST - FRESH JETSON INSTALL")
    print("="*70)
    print("\nThis script tests microphone RMS levels to verify device")
    print("is working correctly before running install_aura_bootable.sh")
    print()
    
    # Check system info
    check_system_info()
    
    try:
        # Find device
        device_index = find_device()
        if device_index is None:
            print("\n❌ Cannot proceed without microphone device.")
            print("   Please connect the ReSpeaker microphone and try again.")
            return 1
        
        # Test RMS levels
        success = test_rms_levels(device_index)
        
        if success:
            print("\n✅ Test completed successfully!")
            print("   If RMS levels are good, you can proceed with install_aura_bootable.sh")
            return 0
        else:
            print("\n❌ Test failed. Check microphone connection and try again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

