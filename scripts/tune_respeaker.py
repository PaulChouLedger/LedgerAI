#!/usr/bin/env python3
"""
ReSpeaker 4 Mic Array Tuner

Configures the hardware DSP on the ReSpeaker for optimal
speech recognition (ASR).

Usage:
    sudo python3 scripts/tune_respeaker.py [preset]
    
Presets:
    - clean: HPF + Conservative AGC (0.1 RMS target) - DEFAULT
    - far_field: Optimized for 8-16 feet (high AGC + noise suppression)
    - near_field: Optimized for 1-6 feet (moderate AGC)
    - reset: Factory defaults (all OFF)
    - show: Show current settings
"""

import sys
import os
import json

# Add tuning module path (expand ~ properly even with sudo)
home_dir = os.path.expanduser('~aura') if os.geteuid() == 0 else os.path.expanduser('~')
tuning_path = os.path.join(home_dir, 'usb_4_mic_array')
sys.path.insert(0, tuning_path)

from tuning import Tuning

# State file for listener to read (no permissions needed)
CONFIG_STATE_FILE = os.path.join(home_dir, 'LedgerAI', 'data', 'respeaker_config.json')

def save_config_state(preset, config_dict):
    """Save current configuration to a file for listener to read"""
    try:
        state = {
            'preset': preset,
            'timestamp': __import__('time').time(),
            'config': config_dict
        }
        os.makedirs(os.path.dirname(CONFIG_STATE_FILE), exist_ok=True)
        with open(CONFIG_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\n  📄 Configuration saved to: {CONFIG_STATE_FILE}")
    except Exception as e:
        print(f"\n  ⚠️  Could not save config state: {e}")

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
    # Conservative level to prevent hardware clipping while avoiding drift
    # Software will do the main boosting to prevent distortion
    print("[3/8] AGC Desired Level: 0.03 RMS (gentle, prevents clipping & drift)")
    dev.write("AGCDESIREDLEVEL", 0.03)  # Gentle but stable
    
    # Set max AGC gain (26 dB = 20x, conservative to prevent clipping)
    print("[4/8] AGC Max Gain: 26 dB (20x, conservative to prevent clipping)")
    dev.write("AGCMAXGAIN", 20.0)
    
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
    
    # Save configuration state for listener
    config_dict = {
        'HPFONOFF': 0,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.03,
        'AGCMAXGAIN': 20.0,
        'STATNOISEONOFF_SR': 1,
        'NONSTATNOISEONOFF_SR': 1,
        'ECHOONOFF': 0
    }
    save_config_state('far_field', config_dict)
    
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
    
    # Save configuration state for listener
    config_dict = {
        'HPFONOFF': 1,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 15.8,
        'STATNOISEONOFF_SR': 1,
        'NONSTATNOISEONOFF_SR': 1,
        'ECHOONOFF': 0
    }
    save_config_state('near_field', config_dict)
    
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
    
    # Save configuration state for listener
    config_dict = {
        'HPFONOFF': 0,
        'AGCONOFF': 0,
        'AGCDESIREDLEVEL': 0.03,
        'AGCMAXGAIN': 20.0,
        'STATNOISEONOFF_SR': 0,
        'NONSTATNOISEONOFF_SR': 0,
        'ECHOONOFF': 0
    }
    save_config_state('reset', config_dict)
    
    return True

def configure_clean():
    """Clean audio with conservative AGC - HPF + dynamic gain to 0.1 RMS"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧹 CONFIGURING CLEAN AUDIO (HPF + Conservative AGC)")
    print("="*80 + "\n")
    
    # Enable 70Hz high-pass filter to remove 60Hz hum and low-frequency noise
    print("[1/4] High-Pass Filter: ON (70 Hz - removes EM noise)")
    dev.write("HPFONOFF", 1)  # 1 = 70Hz cutoff
    
    # Enable AGC with conservative settings
    print("[2/4] Hardware AGC: ON")
    dev.write("AGCONOFF", 1)
    
    print("[3/4] AGC Target Level: 0.1 RMS (conservative to prevent clipping)")
    dev.write("AGCDESIREDLEVEL", 0.1)
    dev.write("AGCMAXGAIN", 20.0)  # 20dB max boost
    
    # Disable all noise suppression
    print("[4/4] Noise Suppression: OFF (no artifacts)")
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n" + "="*80)
    print("  ✅ CLEAN CONFIGURATION COMPLETE")
    print("="*80)
    print("\n  Clean audio with conservative AGC:")
    print("    - 70Hz high-pass filter removes 60Hz AC hum")
    print("    - Conservative AGC (target=0.1 RMS, max=20dB)")
    print("    - No noise suppression (no artifacts)")
    print("\n  This balances clean audio with consistent levels.")
    print("  Lower AGC target prevents clipping/hallucinations.\n")
    
    # Save configuration state for listener
    config_dict = {
        'HPFONOFF': 1,
        'AGCONOFF': 1,  # Enabled
        'AGCDESIREDLEVEL': 0.1,
        'AGCMAXGAIN': 20.0,
        'STATNOISEONOFF_SR': 0,
        'NONSTATNOISEONOFF_SR': 0,
        'ECHOONOFF': 0
    }
    save_config_state('clean', config_dict)
    
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
        except Exception as e:
            print(f"  {label:<25} ERROR: {e}")
    
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
    
    preset = sys.argv[1] if len(sys.argv) > 1 else "clean"
    
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
        elif preset == "clean":
            success = configure_clean()
        elif preset == "reset":
            success = reset_defaults()
        elif preset == "show":
            success = show_current_settings()
        else:
            print(f"\n  ❌ Unknown preset: {preset}")
            print(f"\n  Available presets:")
            print(f"    - clean      : HPF + Conservative AGC (0.1 RMS target)")
            print(f"    - far_field  : High AGC + noise suppression (8-16 feet)")
            print(f"    - near_field : Moderate AGC (1-6 feet)")
            print(f"    - reset      : Factory defaults (all OFF)")
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

