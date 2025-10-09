#!/usr/bin/env python3
"""
ReSpeaker 4 Mic Array Tuner

Configures the hardware DSP on the ReSpeaker for optimal
far-field speech recognition (ASR).

Usage:
    sudo python3 scripts/tune_respeaker.py [preset]
    
Presets:
    - far_field: Optimized for 8-16 feet (default)
    - near_field: Optimized for 1-6 feet
    - reset: Restore factory defaults
    - show: Show current settings
"""

import sys
import os

# Add tuning module path (expand ~ properly even with sudo)
home_dir = os.path.expanduser('~aura') if os.geteuid() == 0 else os.path.expanduser('~')
tuning_path = os.path.join(home_dir, 'usb_4_mic_array')
sys.path.insert(0, tuning_path)

from tuning import Tuning

def configure_far_field():
    """Configure for far-field speech recognition (8-16 feet)"""
    import usb.core
    
    # Find ReSpeaker USB device
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found (Vendor: 0x2886, Product: 0x0018)")
        print("     Check USB connection and run: lsusb | grep 2886\n")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 CONFIGURING FOR FAR-FIELD SPEECH RECOGNITION (8-16 feet)")
    print("="*80 + "\n")
    
    # Disable hardware high-pass filter (we found it hurts far-field)
    print("[1/8] High-Pass Filter: OFF (preserves all speech frequencies)")
    dev.write("HPFONOFF", 0)
    
    # Enable hardware AGC for automatic gain
    print("[2/8] Hardware AGC: ON")
    dev.write("AGCONOFF", 1)
    
    # Set AGC desired level (target output power)
    # -23 dBov = 0.005 RMS, -20 dBov = 0.01 RMS, -16 dBov = 0.025 RMS
    # For Whisper, we want ~0.10 RMS = -10 dBov
    print("[3/8] AGC Desired Level: -10 dBov (0.10 RMS, optimal for Whisper)")
    dev.write("AGCDESIREDLEVEL", 0.10)  # 0.10 = -10 dBov
    
    # Set max AGC gain (30 dB = 31.6x, good for far-field)
    print("[4/8] AGC Max Gain: 30 dB (31.6x, for far-field)")
    dev.write("AGCMAXGAIN", 31.6)
    
    # Enable stationary noise suppression for ASR
    print("[5/8] Stationary Noise Suppression (ASR): ON")
    dev.write("STATNOISEONOFF_SR", 1)
    
    # Enable non-stationary noise suppression for ASR
    print("[6/8] Non-Stationary Noise Suppression (ASR): ON")
    dev.write("NONSTATNOISEONOFF_SR", 1)
    
    # Set over-subtraction factors (gentle for ASR)
    print("[7/8] Noise over-subtraction factors: 1.0 (gentle)")
    dev.write("GAMMA_NS_SR", 1.0)  # Stationary noise
    dev.write("GAMMA_NN_SR", 1.1)  # Non-stationary noise
    
    # Disable echo cancellation (not needed for voice assistant)
    print("[8/8] Echo Cancellation: OFF (not needed)")
    dev.write("ECHOONOFF", 0)
    
    print("\n" + "="*80)
    print("  ✅ FAR-FIELD CONFIGURATION COMPLETE")
    print("="*80)
    print("\n  Hardware is now optimized for 8-16 feet speech recognition!")
    print("  The ReSpeaker DSP will:")
    print("    - Apply hardware AGC (faster than software)")
    print("    - Remove stationary noise (fan hum, etc.)")
    print("    - Remove non-stationary noise (air conditioning, etc.)")
    print("    - Preserve all speech frequencies")
    print("\n  Restart your listener to use the new hardware settings.\n")
    
    return True

def configure_near_field():
    """Configure for near-field speech recognition (1-6 feet)"""
    import usb.core
    
    # Find ReSpeaker USB device
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 CONFIGURING FOR NEAR-FIELD SPEECH RECOGNITION (1-6 feet)")
    print("="*80 + "\n")
    
    # Enable gentle high-pass filter (70 Hz)
    print("[1/8] High-Pass Filter: 70 Hz (removes subsonic noise)")
    dev.write("HPFONOFF", 1)
    
    # Enable hardware AGC
    print("[2/8] Hardware AGC: ON")
    dev.write("AGCONOFF", 1)
    
    # Set AGC desired level
    print("[3/8] AGC Desired Level: -12 dBov (0.08 RMS)")
    dev.write("AGCDESIREDLEVEL", 0.08)
    
    # Lower max gain (don't need as much for near-field)
    print("[4/8] AGC Max Gain: 24 dB (15.8x)")
    dev.write("AGCMAXGAIN", 15.8)
    
    # Enable noise suppression
    print("[5/8] Stationary Noise Suppression (ASR): ON")
    dev.write("STATNOISEONOFF_SR", 1)
    
    print("[6/8] Non-Stationary Noise Suppression (ASR): ON")
    dev.write("NONSTATNOISEONOFF_SR", 1)
    
    # Gentler over-subtraction for near-field
    print("[7/8] Noise over-subtraction factors: 0.8 (gentle)")
    dev.write("GAMMA_NS_SR", 0.8)
    dev.write("GAMMA_NN_SR", 0.8)
    
    # Disable echo cancellation
    print("[8/8] Echo Cancellation: OFF")
    dev.write("ECHOONOFF", 0)
    
    print("\n" + "="*80)
    print("  ✅ NEAR-FIELD CONFIGURATION COMPLETE")
    print("="*80 + "\n")
    
    return True

def reset_defaults():
    """Reset to factory defaults"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🔄 RESETTING TO FACTORY DEFAULTS")
    print("="*80 + "\n")
    
    dev.write("HPFONOFF", 0)
    dev.write("AGCONOFF", 0)
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("  ✅ Reset complete\n")
    
    return True

def show_current_settings():
    """Show current ReSpeaker settings"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  📊 CURRENT RESPEAKER SETTINGS")
    print("="*80 + "\n")
    
    settings = {
        "High-Pass Filter": ("HPFONOFF", ["OFF", "70 Hz", "125 Hz", "180 Hz"]),
        "Hardware AGC": ("AGCONOFF", ["OFF", "ON"]),
        "AGC Desired Level": ("AGCDESIREDLEVEL", None),
        "AGC Max Gain": ("AGCMAXGAIN", None),
        "Stationary Noise (ASR)": ("STATNOISEONOFF_SR", ["OFF", "ON"]),
        "Non-Stat Noise (ASR)": ("NONSTATNOISEONOFF_SR", ["OFF", "ON"]),
        "Stationary Noise Factor": ("GAMMA_NS_SR", None),
        "Non-Stat Noise Factor": ("GAMMA_NN_SR", None),
        "Echo Cancellation": ("ECHOONOFF", ["OFF", "ON"]),
        "Voice Activity": ("VOICEACTIVITY", ["NO", "YES"]),
        "Speech Detected": ("SPEECHDETECTED", ["NO", "YES"])
    }
    
    for label, (param, options) in settings.items():
        try:
            value = dev.read(param)
            if options and isinstance(value, int) and value < len(options):
                display = f"{options[value]} ({value})"
            else:
                display = f"{value}"
            print(f"  {label:<25} {display}")
        except:
            print(f"  {label:<25} ERROR")
    
    print("\n" + "="*80 + "\n")
    
    return True

def main():
    """Main execution"""
    # Check if running with sudo
    if os.geteuid() != 0:
        print("\n" + "="*80)
        print("  ⚠️  PERMISSION ERROR")
        print("="*80)
        print("\n  USB device access requires root permissions.")
        print("\n  Please run with sudo:")
        print(f"    sudo python3 {' '.join(sys.argv)}")
        print("\n" + "="*80 + "\n")
        return 1
    
    preset = sys.argv[1] if len(sys.argv) > 1 else "far_field"
    
    print("\n" + "="*80)
    print("  🎙️  RESPEAKER 4 MIC ARRAY TUNER")
    print("  Hardware DSP Configuration for Speech Recognition")
    print("="*80)
    
    try:
        success = False
        
        if preset == "far_field":
            success = configure_far_field()
        elif preset == "near_field":
            success = configure_near_field()
        elif preset == "reset":
            success = reset_defaults()
        elif preset == "show":
            success = show_current_settings()
        else:
            print(f"\n  ❌ Unknown preset: {preset}")
            print(f"\n  Available presets:")
            print(f"    - far_field  : Optimized for 8-16 feet")
            print(f"    - near_field : Optimized for 1-6 feet")
            print(f"    - reset      : Factory defaults")
            print(f"    - show       : Show current settings\n")
            return 1
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

