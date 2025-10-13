#!/usr/bin/env python3
"""
ReSpeaker 4 Mic Array Tuner

Configures the hardware DSP on the ReSpeaker for optimal
speech recognition (ASR).

Usage:
    sudo python3 scripts/tune_respeaker.py [preset]
    
STANDARD PRESETS:
    - clean                   : HPF + AGC (0.08) + NS (3.0) - DEFAULT
    - balanced                : HPF 70Hz + NS (2.0) + AGC (0.08, 30dB)
    - balanced_beam (bb)      : Balanced + Beamforming ⭐ BEST FOR FAN NOISE
    - beamforming             : Adaptive beamforming + DOA (30dB, balanced)
    - beamforming_light       : Beamforming + light NS (30dB, clean environments)
    - beamforming_aggressive  : Beamforming + max NS (45dB, noisy/far-field)
    - beamforming_ultra       : Beamforming + extreme gain (50dB, max range)
    - far_field               : Optimized for 8-16 feet
    - near_field              : Optimized for 1-6 feet
    - reset                   : Factory defaults (all OFF)
    - show                    : Show current settings

TEST PROFILES (systematic optimization):
    - bf         : Beamforming ONLY (pure test)
    - hpbf       : Beamforming + HPF 70Hz
    - hp         : High-pass filter ONLY (70Hz)
    - hp1        : HPF + NS gamma=1.0 (mild)
    - hp2        : HPF + NS gamma=2.0 (moderate)
    - hp3        : HPF + NS gamma=3.0 (aggressive)
    - agc1       : AGC ONLY (0.05 RMS, 20dB max)
    - agc2       : AGC ONLY (0.08 RMS, 30dB max)
    - agc3       : AGC ONLY (0.12 RMS, 40dB max)

FAN NOISE PROFILES (Beamforming + HPF 70Hz + NS, NO AGC):
    - bf_ns0     : Beamforming + HPF 70Hz, NS OFF (baseline)
    - bf_ns1     : Beamforming + HPF 70Hz, NS gamma=1.0 (mild)
    - bf_ns2     : Beamforming + HPF 70Hz, NS gamma=2.0 (moderate)
    - bf_ns3     : Beamforming + HPF 70Hz, NS gamma=3.0 (maximum)

BALANCED PROFILES (HPF + NS + AGC):
    - balanced       : HPF 70Hz + NS gamma=2.0 + AGC (0.08, 30dB) - No beamforming
    - balanced_beam  : HPF 70Hz + NS gamma=2.0 + AGC (0.08, 30dB) + Beamforming ⭐
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
    """HPF + Stationary NS - removes EM noise and constant tones, no AGC"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧹 CONFIGURING CLEAN AUDIO (HPF + Balanced AGC + Stationary NS)")
    print("="*80 + "\n")
    
    # Enable 70Hz high-pass filter to remove 60Hz hum and low-frequency noise
    print("[1/4] High-Pass Filter: ON (70 Hz - removes low-freq EM noise)")
    dev.write("HPFONOFF", 1)  # 1 = 70Hz cutoff
    
    # Enable optimal AGC for Whisper's sweet spot (0.05-0.08 RMS)
    print("[2/4] Hardware AGC: ON (optimal - 0.08 RMS target for Whisper)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)  # Whisper's sweet spot
    dev.write("AGCMAXGAIN", 30.0)  # Max 30dB gain (31x amplification)
    dev.write("AGCTIME", 0.2)  # Moderate attack (prevents distortion)
    print("         - Target: 0.08 RMS (Whisper optimal), Max Gain: 30dB, Attack: 0.2s")
    
    # Enable stationary noise suppression to remove constant electrical tones
    print("[3/4] Stationary Noise Suppression: ON (removes 120Hz, 601Hz interference)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 3.0)  # Maximum aggressiveness to fight 601Hz
    print("         - Aggressiveness: 3.0 (maximum - targets 120Hz + 601Hz)")
    
    # Disable other processing
    print("[4/4] Other Processing: OFF")
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n" + "="*80)
    print("  ✅ CLEAN CONFIGURATION COMPLETE")
    print("="*80)
    print("\n  Processing enabled:")
    print("    - 70Hz high-pass filter removes low-freq noise (<70Hz)")
    print("    - Optimal AGC (0.08 RMS target, 30dB max, 0.2s attack)")
    print("      • Targets Whisper's sweet spot (0.05-0.08 RMS)")
    print("      • Testing shows >0.10 RMS causes hallucinations")
    print("      • 30dB = 31x amplification capability")
    print("      • Moderate attack prevents distortion")
    print("    - Stationary noise suppression (gamma=3.0) removes interference:")
    print("      • Removes 120Hz (AC harmonic)")
    print("      • Removes 601Hz (USB/display interference)")
    print("      • Maximum aggressiveness")
    print("\n  Pipeline: HPF → Hardware AGC (0.08 target) → Stationary NS (max)")
    print("  Whisper optimal: 0.05-0.08 RMS (testing one variable at a time)\n")
    
    # Save configuration state for listener
    config_dict = {
        'HPFONOFF': 1,
        'AGCONOFF': 1,  # Enabled - optimal for Whisper
        'AGCDESIREDLEVEL': 0.08,  # Whisper sweet spot
        'AGCMAXGAIN': 30.0,
        'AGCTIME': 0.2,
        'STATNOISEONOFF_SR': 1,  # Enabled - maximum
        'GAMMA_NS_SR': 3.0,  # Maximum aggressiveness
        'NONSTATNOISEONOFF_SR': 0,
        'ECHOONOFF': 0
    }
    save_config_state('clean', config_dict)
    
    return True

# === ISOLATED TEST PROFILES ===
# These profiles test ONE component at a time for systematic optimization

def configure_hp():
    """HP - High-Pass Filter ONLY (70Hz)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: HP - High-Pass Filter ONLY")
    print("="*80 + "\n")
    
    # Enable ONLY HPF
    print("[1/1] High-Pass Filter: ON (70 Hz)")
    dev.write("HPFONOFF", 1)
    
    # Disable everything else
    dev.write("AGCONOFF", 0)
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ HP Profile Complete")
    print("  Only removes <70Hz noise - all else untouched\n")
    
    save_config_state('hp', {
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 0,
        'GAMMA_NS_SR': 0.0
    })
    return True

def configure_hp1():
    """HP1 - High-Pass Filter + Mild Noise Suppression (gamma=1.0)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: HP1 - HPF + Mild Noise Suppression (gamma=1.0)")
    print("="*80 + "\n")
    
    print("[1/2] High-Pass Filter: ON (70 Hz)")
    dev.write("HPFONOFF", 1)
    
    print("[2/2] Stationary Noise Suppression: ON (gamma=1.0 - mild)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 1.0)
    
    # Disable other processing
    dev.write("AGCONOFF", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ HP1 Profile Complete")
    print("  HPF + gentle noise removal\n")
    
    save_config_state('hp1', {
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 1.0
    })
    return True

def configure_hp2():
    """HP2 - High-Pass Filter + Moderate Noise Suppression (gamma=2.0)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: HP2 - HPF + Moderate Noise Suppression (gamma=2.0)")
    print("="*80 + "\n")
    
    print("[1/2] High-Pass Filter: ON (70 Hz)")
    dev.write("HPFONOFF", 1)
    
    print("[2/2] Stationary Noise Suppression: ON (gamma=2.0 - moderate)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 2.0)
    
    # Disable other processing
    dev.write("AGCONOFF", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ HP2 Profile Complete")
    print("  HPF + moderate noise removal\n")
    
    save_config_state('hp2', {
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 2.0
    })
    return True

def configure_hp3():
    """HP3 - High-Pass Filter + Aggressive Noise Suppression (gamma=3.0)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: HP3 - HPF + Aggressive Noise Suppression (gamma=3.0)")
    print("="*80 + "\n")
    
    print("[1/2] High-Pass Filter: ON (70 Hz)")
    dev.write("HPFONOFF", 1)
    
    print("[2/2] Stationary Noise Suppression: ON (gamma=3.0 - aggressive)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 3.0)
    
    # Disable other processing
    dev.write("AGCONOFF", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ HP3 Profile Complete")
    print("  HPF + aggressive noise removal\n")
    
    save_config_state('hp3', {
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 3.0
    })
    return True

def configure_bf():
    """BF - Beamforming ONLY (no other processing)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: BF - Beamforming ONLY")
    print("="*80 + "\n")
    
    # Enable ONLY adaptive beamforming
    print("[1/1] Adaptive Beamforming: ON (tracks speaker)")
    dev.write("FREEZEONOFF", 0)  # 0 = Adaptive tracking enabled
    
    # Disable everything else
    dev.write("HPFONOFF", 0)
    dev.write("AGCONOFF", 0)
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    # Read DOA
    try:
        doa = dev.read("DOAANGLE")
        print(f"\n  📍 Speaker direction: {doa}°")
    except:
        pass
    
    print("\n  ✅ BF Profile Complete")
    print("  Pure beamforming - no other processing\n")
    
    save_config_state('bf', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 0,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 0,
        'GAMMA_NS_SR': 0.0
    })
    return True

def configure_hpbf():
    """HPBF - Beamforming + High-Pass Filter 70Hz"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: HPBF - Beamforming + High-Pass Filter")
    print("="*80 + "\n")
    
    # Enable adaptive beamforming
    print("[1/2] Adaptive Beamforming: ON (tracks speaker)")
    dev.write("FREEZEONOFF", 0)
    
    # Enable HPF
    print("[2/2] High-Pass Filter: ON (70 Hz)")
    dev.write("HPFONOFF", 1)
    
    # Disable other processing
    dev.write("AGCONOFF", 0)
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    # Read DOA
    try:
        doa = dev.read("DOAANGLE")
        print(f"\n  📍 Speaker direction: {doa}°")
    except:
        pass
    
    print("\n  ✅ HPBF Profile Complete")
    print("  Beamforming + HPF (removes low-freq noise)\n")
    
    save_config_state('hpbf', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 0,
        'GAMMA_NS_SR': 0.0
    })
    return True

def configure_beamforming_light():
    """Beamforming + Light noise suppression (best for clean environments)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 BEAMFORMING + LIGHT NS (Clean Environment)")
    print("="*80 + "\n")
    
    print("[1/5] Adaptive Beamforming: ON")
    dev.write("FREEZEONOFF", 0)
    
    print("[2/5] High-Pass Filter: ON (70 Hz)")
    dev.write("HPFONOFF", 1)
    
    print("[3/5] Hardware AGC: ON (0.08 RMS)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)
    dev.write("AGCMAXGAIN", 30.0)
    
    print("[4/5] Stationary Noise Suppression: LIGHT (gamma=1.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 1.0)  # Light
    
    print("[5/5] Non-Stationary Noise Suppression: LIGHT (gamma=1.0)")
    dev.write("NONSTATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NN_SR", 1.0)  # Light
    
    dev.write("ECHOONOFF", 0)
    
    try:
        doa = dev.read("DOAANGLE")
        print(f"\n  📍 Speaker direction: {doa}°")
    except:
        pass
    
    print("\n  ✅ Light beamforming optimized for clean environments")
    print("  Use when background noise is minimal\n")
    
    save_config_state('beamforming_light', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 30.0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 1.0,
        'NONSTATNOISEONOFF_SR': 1,
        'GAMMA_NN_SR': 1.0,
        'ECHOONOFF': 0
    })
    return True

def configure_beamforming_aggressive():
    """Beamforming + Aggressive noise suppression + High gain (best for noisy/far-field)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 BEAMFORMING + AGGRESSIVE (Noisy/Far-Field Optimized)")
    print("="*80 + "\n")
    
    print("[1/5] Adaptive Beamforming: ON")
    dev.write("FREEZEONOFF", 0)
    
    print("[2/5] High-Pass Filter: ON (125 Hz - aggressive)")
    dev.write("HPFONOFF", 2)  # 125Hz instead of 70Hz
    
    print("[3/5] Hardware AGC: ON (0.08 RMS, 45 dB MAX - far-field optimized)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)
    dev.write("AGCMAXGAIN", 45.0)  # Increased from 30 dB for far-field
    dev.write("AGCTIME", 0.2)
    print("         - Target: 0.08 RMS, Max Gain: 45dB (177x), Attack: 0.2s")
    
    print("[4/5] Stationary Noise Suppression: MAX (gamma=3.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 3.0)  # Max
    
    print("[5/5] Non-Stationary Noise Suppression: HIGH (gamma=2.5)")
    dev.write("NONSTATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NN_SR", 2.5)  # High
    
    dev.write("ECHOONOFF", 0)
    
    try:
        doa = dev.read("DOAANGLE")
        print(f"\n  📍 Speaker direction: {doa}°")
    except:
        pass
    
    print("\n  ✅ Aggressive beamforming optimized for noisy/far-field")
    print("  Benefits:")
    print("    • 45 dB max gain = 177x amplification (far-field capable)")
    print("    • Beamforming focuses on speaker, suppresses noise from other directions")
    print("    • Max noise suppression removes background noise before AGC amplifies it")
    print("    • 125 Hz HPF removes more low-frequency rumble")
    print("    • Software filter rejects low-energy bursts (RMS<0.035, Peak<0.15)")
    print("\n  This is the most aggressive setting - use for maximum range!\n")
    
    save_config_state('beamforming_aggressive', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 2,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 45.0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 3.0,
        'NONSTATNOISEONOFF_SR': 1,
        'GAMMA_NN_SR': 2.5,
        'ECHOONOFF': 0
    })
    return True

def configure_beamforming_ultra():
    """Beamforming + Ultra-aggressive settings (maximum far-field range)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🚀 BEAMFORMING ULTRA (Maximum Far-Field Range)")
    print("="*80 + "\n")
    
    print("[1/5] Adaptive Beamforming: ON")
    dev.write("FREEZEONOFF", 0)
    
    print("[2/5] High-Pass Filter: ON (125 Hz)")
    dev.write("HPFONOFF", 2)
    
    print("[3/5] Hardware AGC: ON (0.08 RMS, 50 dB MAX - EXTREME)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)
    dev.write("AGCMAXGAIN", 50.0)  # 316x amplification!
    dev.write("AGCTIME", 0.15)  # Faster attack for far-field
    print("         - Target: 0.08 RMS, Max Gain: 50dB (316x!), Attack: 0.15s")
    print("         ⚠️  WARNING: May amplify noise if beamforming/NS fail")
    
    print("[4/5] Stationary Noise Suppression: MAXIMUM (gamma=3.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 3.0)
    
    print("[5/5] Non-Stationary Noise Suppression: MAXIMUM (gamma=3.0)")
    dev.write("NONSTATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NN_SR", 3.0)  # Increased to max
    
    dev.write("ECHOONOFF", 0)
    
    try:
        doa = dev.read("DOAANGLE")
        print(f"\n  📍 Speaker direction: {doa}°")
    except:
        pass
    
    print("\n  ✅ Ultra-aggressive beamforming configured")
    print("  ⚠️  CAUTION: This is EXTREME gain!")
    print("  Benefits:")
    print("    • 50 dB = 316x amplification (captures very quiet/distant speech)")
    print("    • Maximum noise suppression before amplification")
    print("    • Beamforming isolates speaker direction")
    print("    • Software filter prevents noise amplification")
    print("\n  Use Cases:")
    print("    ✅ 15+ feet speaking distance")
    print("    ✅ Very quiet speakers")
    print("    ✅ Testing maximum microphone range")
    print("\n  Monitor for:")
    print("    ⚠️  Amplified background noise (if beamforming fails)")
    print("    ⚠️  Clipping on loud sounds (check Peak values)")
    print("    ⚠️  AGC pumping (rapid gain changes)")
    print("\n  If you experience issues, use beamforming_aggressive (45dB) instead.\n")
    
    save_config_state('beamforming_ultra', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 2,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 50.0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 3.0,
        'NONSTATNOISEONOFF_SR': 1,
        'GAMMA_NN_SR': 3.0,
        'ECHOONOFF': 0
    })
    return True

def configure_agc1():
    """AGC1 - Mild AGC ONLY (0.05 RMS target)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: AGC1 - Mild AGC (0.05 RMS, 20dB max)")
    print("="*80 + "\n")
    
    print("[1/1] Hardware AGC: ON (mild - 0.05 RMS target)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.05)
    dev.write("AGCMAXGAIN", 20.0)
    dev.write("AGCTIME", 0.2)
    print("         - Target: 0.05 RMS, Max Gain: 20dB, Attack: 0.2s")
    
    # Disable other processing
    dev.write("HPFONOFF", 0)
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ AGC1 Profile Complete")
    print("  Mild amplification only - no filtering\n")
    
    save_config_state('agc1', {
        'HPFONOFF': 0,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.05,
        'AGCMAXGAIN': 20.0,
        'STATNOISEONOFF_SR': 0,
        'GAMMA_NS_SR': 0.0
    })
    return True

def configure_agc2():
    """AGC2 - Moderate AGC ONLY (0.08 RMS target)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: AGC2 - Moderate AGC (0.08 RMS, 30dB max)")
    print("="*80 + "\n")
    
    print("[1/1] Hardware AGC: ON (moderate - 0.08 RMS target)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)
    dev.write("AGCMAXGAIN", 30.0)
    dev.write("AGCTIME", 0.2)
    print("         - Target: 0.08 RMS, Max Gain: 30dB, Attack: 0.2s")
    
    # Disable other processing
    dev.write("HPFONOFF", 0)
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ AGC2 Profile Complete")
    print("  Moderate amplification only - no filtering\n")
    
    save_config_state('agc2', {
        'HPFONOFF': 0,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 30.0,
        'STATNOISEONOFF_SR': 0,
        'GAMMA_NS_SR': 0.0
    })
    return True

def configure_beamforming():
    """Enable optimal beamforming for far-field speech"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 CONFIGURING OPTIMAL BEAMFORMING")
    print("="*80 + "\n")
    
    # Enable adaptive beamforming (tracks speaker direction)
    print("[1/6] Adaptive Beamforming: ON (tracks speaker direction)")
    dev.write("FREEZEONOFF", 0)  # 0 = Adaptive tracking enabled
    
    # Enable high-pass filter (removes low-frequency noise)
    print("[2/6] High-Pass Filter: ON (70 Hz)")
    dev.write("HPFONOFF", 1)
    
    # Enable AGC for consistent levels
    print("[3/6] Hardware AGC: ON (0.08 RMS target)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)
    dev.write("AGCMAXGAIN", 30.0)
    
    # Enable noise suppression to help beamformer
    print("[4/6] Stationary Noise Suppression: ON (gamma=2.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 2.0)
    
    print("[5/6] Non-Stationary Noise Suppression: ON")
    dev.write("NONSTATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NN_SR", 1.5)
    
    # Disable echo cancellation (not needed for voice assistant)
    print("[6/6] Echo Cancellation: OFF")
    dev.write("ECHOONOFF", 0)
    
    # Read current DOA
    try:
        doa = dev.read("DOAANGLE")
        print(f"\n  📍 Current speaker direction: {doa}° (0° = front, 180° = back)")
    except:
        print("\n  ℹ️  Speak to see direction of arrival")
    
    print("\n" + "="*80)
    print("  ✅ BEAMFORMING CONFIGURATION COMPLETE")
    print("="*80)
    print("\n  Features enabled:")
    print("    - Adaptive beamforming (automatically tracks speaker)")
    print("    - Direction of Arrival (DOA) detection")
    print("    - Noise suppression (helps beamformer focus)")
    print("    - Hardware AGC (maintains consistent levels)")
    print("\n  Channel 0 now provides beamformed audio optimized for your voice direction!")
    print("  The beamformer will:")
    print("    • Focus on speech from detected direction")
    print("    • Suppress sounds from other directions")
    print("    • Track speaker movement automatically")
    print("    • Combine 4 mics for better far-field performance\n")
    
    # Save configuration
    config_dict = {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 30.0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 2.0,
        'NONSTATNOISEONOFF_SR': 1,
        'GAMMA_NN_SR': 1.5,
        'ECHOONOFF': 0
    }
    save_config_state('beamforming', config_dict)
    
    return True

def configure_agc3():
    """AGC3 - Aggressive AGC ONLY (0.12 RMS target)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🧪 TEST PROFILE: AGC3 - Aggressive AGC (0.12 RMS, 40dB max)")
    print("="*80 + "\n")
    
    print("[1/1] Hardware AGC: ON (aggressive - 0.12 RMS target)")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.12)
    dev.write("AGCMAXGAIN", 40.0)
    dev.write("AGCTIME", 0.2)
    print("         - Target: 0.12 RMS, Max Gain: 40dB, Attack: 0.2s")
    
    # Disable other processing
    dev.write("HPFONOFF", 0)
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ AGC3 Profile Complete")
    print("  Aggressive amplification only - no filtering\n")
    
    save_config_state('agc3', {
        'HPFONOFF': 0,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.12,
        'AGCMAXGAIN': 40.0,
        'STATNOISEONOFF_SR': 0,
        'GAMMA_NS_SR': 0.0
    })
    return True

def configure_balanced():
    """Balanced - HPF 70Hz + NS gamma=2.0 + AGC (0.08, 30dB)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  ⚖️  BALANCED PROFILE - HPF + NS + AGC")
    print("="*80 + "\n")
    
    print("[1/5] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    dev.write("HPFONOFF", 1)
    
    print("[2/5] Stationary Noise Suppression: MODERATE (gamma=2.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 2.0)
    print("         - Balanced fan noise removal")
    
    print("[3/5] Non-Stationary Noise Suppression: OFF")
    dev.write("NONSTATNOISEONOFF_SR", 0)
    
    print("[4/5] Hardware AGC: ENABLED")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)
    dev.write("AGCMAXGAIN", 30.0)
    dev.write("AGCTIME", 0.2)
    print("         - Target: 0.08 RMS (Whisper sweet spot)")
    print("         - Max Gain: 30dB")
    print("         - Attack Time: 0.2s")
    
    print("[5/5] Other: Beamforming OFF, Echo Cancellation OFF")
    dev.write("FREEZEONOFF", 1)  # Freeze beamforming (disable)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ Balanced Profile Complete")
    print("  Good all-around profile: removes fan noise + auto-gain\n")
    
    save_config_state('balanced', {
        'HPFONOFF': 1,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 2.0,
        'NONSTATNOISEONOFF_SR': 0,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 30.0,
        'AGCTIME': 0.2,
        'FREEZEONOFF': 1,
        'ECHOONOFF': 0
    })
    return True

def configure_balanced_beam():
    """Balanced + Beamforming - HPF 70Hz + NS gamma=2.0 + AGC (0.08, 30dB) + Beamforming"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  ⚖️  BALANCED + BEAMFORMING - Complete Fan Noise Solution")
    print("="*80 + "\n")
    
    print("[1/6] Beamforming: ENABLED (adaptive, tracks voice direction)")
    dev.write("FREEZEONOFF", 0)  # Adaptive beamforming
    print("         - Spatially rejects off-axis fan noise")
    
    print("[2/6] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    dev.write("HPFONOFF", 1)
    
    print("[3/6] Stationary Noise Suppression: MODERATE (gamma=2.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 2.0)
    print("         - Removes residual fan hum after beamforming")
    
    print("[4/6] Non-Stationary Noise Suppression: OFF")
    dev.write("NONSTATNOISEONOFF_SR", 0)
    
    print("[5/6] Hardware AGC: ENABLED")
    dev.write("AGCONOFF", 1)
    dev.write("AGCDESIREDLEVEL", 0.08)  # Optimal for Whisper (far-field friendly)
    dev.write("AGCMAXGAIN", 30.0)       # Good for far-field
    dev.write("AGCTIME", 0.2)
    print("         - Target: 0.08 RMS (Whisper sweet spot)")
    print("         - Max Gain: 30dB (good for far-field)")
    print("         - Attack Time: 0.2s")
    print("         - Software limiter prevents near-field clipping")
    
    print("[6/6] Echo Cancellation: OFF")
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ Balanced + Beamforming Profile Complete")
    print("\n  🎯 COMPLETE FAN NOISE SOLUTION:")
    print("     1️⃣  Beamforming rejects spatial fan noise")
    print("     2️⃣  HPF removes low-frequency rumble")
    print("     3️⃣  NS removes residual stationary hum")
    print("     4️⃣  AGC maintains optimal speech levels")
    print("\n  This is the recommended profile for Jetson NX with fan noise!\n")
    
    save_config_state('balanced_beam', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 2.0,
        'NONSTATNOISEONOFF_SR': 0,
        'AGCONOFF': 1,
        'AGCDESIREDLEVEL': 0.08,
        'AGCMAXGAIN': 30.0,
        'AGCTIME': 0.2,
        'ECHOONOFF': 0
    })
    return True

def configure_bf_ns0():
    """BF_NS0 - Beamforming + HPF 70Hz, NS OFF (baseline)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 FAN NOISE PROFILE: BF_NS0 - Baseline (Beamforming + HPF, NO NS)")
    print("="*80 + "\n")
    
    print("[1/4] Beamforming: ENABLED (adaptive, tracks voice)")
    dev.write("FREEZEONOFF", 0)  # Adaptive beamforming
    
    print("[2/4] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    dev.write("HPFONOFF", 1)
    
    print("[3/4] Noise Suppression: OFF (baseline for comparison)")
    dev.write("STATNOISEONOFF_SR", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    
    print("[4/4] AGC: OFF (manual gain control)")
    dev.write("AGCONOFF", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ BF_NS0 Profile Complete")
    print("  Baseline: Beamforming + HPF only, no noise suppression\n")
    
    save_config_state('bf_ns0', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 0,
        'NONSTATNOISEONOFF_SR': 0,
        'GAMMA_NS_SR': 0.0,
        'ECHOONOFF': 0
    })
    return True

def configure_bf_ns1():
    """BF_NS1 - Beamforming + HPF 70Hz + NS gamma=1.0 (mild)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 FAN NOISE PROFILE: BF_NS1 - Mild Suppression (gamma=1.0)")
    print("="*80 + "\n")
    
    print("[1/4] Beamforming: ENABLED (adaptive, tracks voice)")
    dev.write("FREEZEONOFF", 0)
    
    print("[2/4] High-Pass Filter: 70 Hz")
    dev.write("HPFONOFF", 1)
    
    print("[3/4] Stationary Noise Suppression: MILD (gamma=1.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 1.0)
    print("         - Gentle fan noise removal, preserves speech quality")
    
    print("[4/4] AGC: OFF")
    dev.write("AGCONOFF", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ BF_NS1 Profile Complete")
    print("  Mild noise suppression for light fan noise\n")
    
    save_config_state('bf_ns1', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 1.0,
        'NONSTATNOISEONOFF_SR': 0,
        'ECHOONOFF': 0
    })
    return True

def configure_bf_ns2():
    """BF_NS2 - Beamforming + HPF 70Hz + NS gamma=2.0 (moderate)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 FAN NOISE PROFILE: BF_NS2 - Moderate Suppression (gamma=2.0)")
    print("="*80 + "\n")
    
    print("[1/4] Beamforming: ENABLED (adaptive, tracks voice)")
    dev.write("FREEZEONOFF", 0)
    
    print("[2/4] High-Pass Filter: 70 Hz")
    dev.write("HPFONOFF", 1)
    
    print("[3/4] Stationary Noise Suppression: MODERATE (gamma=2.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 2.0)
    print("         - Balanced fan noise removal, good for most cases")
    
    print("[4/4] AGC: OFF")
    dev.write("AGCONOFF", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ BF_NS2 Profile Complete")
    print("  Moderate noise suppression for typical fan noise\n")
    
    save_config_state('bf_ns2', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 2.0,
        'NONSTATNOISEONOFF_SR': 0,
        'ECHOONOFF': 0
    })
    return True

def configure_bf_ns3():
    """BF_NS3 - Beamforming + HPF 70Hz + NS gamma=3.0 (maximum)"""
    import usb.core
    
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("\n  ❌ ReSpeaker USB device not found")
        return False
    
    dev = Tuning(usb_dev)
    
    print("\n" + "="*80)
    print("  🎯 FAN NOISE PROFILE: BF_NS3 - Maximum Suppression (gamma=3.0)")
    print("="*80 + "\n")
    
    print("[1/4] Beamforming: ENABLED (adaptive, tracks voice)")
    dev.write("FREEZEONOFF", 0)
    
    print("[2/4] High-Pass Filter: 70 Hz")
    dev.write("HPFONOFF", 1)
    
    print("[3/4] Stationary Noise Suppression: MAXIMUM (gamma=3.0)")
    dev.write("STATNOISEONOFF_SR", 1)
    dev.write("GAMMA_NS_SR", 3.0)
    print("         - Aggressive fan noise removal, may slightly affect speech")
    
    print("[4/4] AGC: OFF")
    dev.write("AGCONOFF", 0)
    dev.write("NONSTATNOISEONOFF_SR", 0)
    dev.write("ECHOONOFF", 0)
    
    print("\n  ✅ BF_NS3 Profile Complete")
    print("  Maximum noise suppression for loud fan noise\n")
    
    save_config_state('bf_ns3', {
        'FREEZEONOFF': 0,
        'HPFONOFF': 1,
        'AGCONOFF': 0,
        'STATNOISEONOFF_SR': 1,
        'GAMMA_NS_SR': 3.0,
        'NONSTATNOISEONOFF_SR': 0,
        'ECHOONOFF': 0
    })
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
        "Beamforming": ("FREEZEONOFF", ["Adaptive (Tracking)", "Frozen (Fixed)"]),
        "Direction of Arrival": ("DOAANGLE", None),
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
        
        # Standard presets
        if preset == "far_field":
            success = configure_far_field()
        elif preset == "near_field":
            success = configure_near_field()
        elif preset == "clean":
            success = configure_clean()
        elif preset == "balanced":
            success = configure_balanced()
        elif preset == "balanced_beam" or preset == "bb":
            success = configure_balanced_beam()
        elif preset == "reset":
            success = reset_defaults()
        elif preset == "show":
            success = show_current_settings()
        
        # Beamforming presets
        elif preset == "beamforming" or preset == "beam":
            success = configure_beamforming()
        elif preset == "beamforming_light" or preset == "beam_light":
            success = configure_beamforming_light()
        elif preset == "beamforming_aggressive" or preset == "beam_aggro":
            success = configure_beamforming_aggressive()
        elif preset == "beamforming_ultra" or preset == "beam_ultra":
            success = configure_beamforming_ultra()
        
        # Test profiles (isolated components)
        elif preset == "bf":
            success = configure_bf()
        elif preset == "hpbf":
            success = configure_hpbf()
        elif preset == "hp":
            success = configure_hp()
        elif preset == "hp1":
            success = configure_hp1()
        elif preset == "hp2":
            success = configure_hp2()
        elif preset == "hp3":
            success = configure_hp3()
        elif preset == "agc1":
            success = configure_agc1()
        elif preset == "agc2":
            success = configure_agc2()
        elif preset == "agc3":
            success = configure_agc3()
        
        # Fan noise profiles (Beamforming + HPF + NS, no AGC)
        elif preset == "bf_ns0":
            success = configure_bf_ns0()
        elif preset == "bf_ns1":
            success = configure_bf_ns1()
        elif preset == "bf_ns2":
            success = configure_bf_ns2()
        elif preset == "bf_ns3":
            success = configure_bf_ns3()
        
        else:
            print(f"\n  ❌ Unknown preset: {preset}")
            print(f"\n  📋 STANDARD PRESETS:")
            print(f"    - clean                   : HPF + Optimal AGC (0.08) + Max NS (3.0) - DEFAULT")
            print(f"    - balanced                : HPF 70Hz + NS (2.0) + AGC (0.08, 30dB)")
            print(f"    - balanced_beam (bb)      : Balanced + Beamforming ⭐ RECOMMENDED FOR FAN NOISE")
            print(f"    - beamforming             : Adaptive beamforming (30dB, balanced)")
            print(f"    - beamforming_light       : Beamforming + light NS (30dB, clean env)")
            print(f"    - beamforming_aggressive  : Beamforming + max NS (45dB, far-field)")
            print(f"    - beamforming_ultra       : Beamforming + extreme gain (50dB, max range)")
            print(f"    - far_field               : High AGC + noise suppression (8-16 feet)")
            print(f"    - near_field              : Moderate AGC (1-6 feet)")
            print(f"    - reset                   : Factory defaults (all OFF)")
            print(f"    - show                    : Show current settings")
            print(f"\n  🧪 TEST PROFILES (systematic testing):")
            print(f"    - bf         : Beamforming ONLY (pure test)")
            print(f"    - hpbf       : Beamforming + HPF 70Hz")
            print(f"    - hp         : High-pass filter ONLY (70Hz)")
            print(f"    - hp1        : HPF + NS gamma=1.0 (mild)")
            print(f"    - hp2        : HPF + NS gamma=2.0 (moderate)")
            print(f"    - hp3        : HPF + NS gamma=3.0 (aggressive)")
            print(f"    - agc1       : AGC ONLY (0.05 RMS, 20dB max)")
            print(f"    - agc2       : AGC ONLY (0.08 RMS, 30dB max)")
            print(f"    - agc3       : AGC ONLY (0.12 RMS, 40dB max)")
            print(f"\n  🌬️  FAN NOISE PROFILES (Beamforming + HPF 70Hz + NS, NO AGC):")
            print(f"    - bf_ns0     : Beamforming + HPF, NS OFF (baseline)")
            print(f"    - bf_ns1     : Beamforming + HPF, NS gamma=1.0 (mild)")
            print(f"    - bf_ns2     : Beamforming + HPF, NS gamma=2.0 (moderate) ⭐")
            print(f"    - bf_ns3     : Beamforming + HPF, NS gamma=3.0 (maximum)\n")
            return 1
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

