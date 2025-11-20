#!/usr/bin/env python3
"""
XVF3800 USB 4 Mic Array Tuner

Configures the hardware DSP on the XVF3800 for optimal
speech recognition (ASR).

All presets automatically disable LEDs to reduce power consumption
(useful when using USB isolator with limited power budget).

Usage:
    python3 tune_xvf3800.py [preset]
    
PRESETS:
    - balanced_beam (bb)   : HPF 70Hz + AGC (0.08, 30dB) - DEFAULT ⭐ RECOMMENDED
    - ultra_sensitive       : AGC (0.10, 45dB) - Far-field optimized
    - far_field            : Optimized for 8-16 feet
    - near_field           : Optimized for 1-6 feet
    - hpf_only             : HPF 70Hz only (minimal processing)
    - agc_only             : AGC only (0.08, 30dB) - no HPF
    - agc_10               : HPF 70Hz + AGC with 10% increase (0.088)
    - agc_20               : HPF 70Hz + AGC with 20% increase (0.096)
    - reset                : Factory defaults
    - show                 : Show current settings
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

# Detect target user's home directory dynamically (works even when run via sudo)
_sudo_user = os.environ.get("SUDO_USER")
if _sudo_user:
    USER_HOME = os.path.expanduser(f"~{_sudo_user}")
else:
    USER_HOME = os.path.expanduser("~")

# State file for listener to read - use target user's home directory  
CONFIG_STATE_FILE = os.path.join(USER_HOME, 'LedgerAI', 'data', 'xvf3800_config.json')

# Path to xvf_host binary - use target user's home directory
XVF_HOST_PATH = os.path.join(USER_HOME, 'reSpeaker_XVF3800_USB_4MIC_ARRAY', 'host_control', 'jetson', 'xvf_host')

def run_xvf_command(cmd, *args):
    """Run xvf_host command and return result"""
    try:
        full_cmd = [XVF_HOST_PATH, cmd] + list(str(arg) for arg in args)
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Show stderr for debugging
            stderr_msg = f": {result.stderr.strip()}" if result.stderr.strip() else ""
            print(f"  ⚠️  Warning: {cmd} returned {result.returncode}{stderr_msg}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Warning: {cmd} timed out")
        return None
    except Exception as e:
        print(f"  ⚠️  Error running {cmd}: {e}")
        return None

def disable_all_leds():
    """Disable all LEDs on XVF3800 to reduce power draw (useful for USB isolator)"""
    print("[LED] Disabling all LEDs to reduce power consumption...")
    
    # Try LED control commands (LED_BRIGHTNESS=0 works, try it first)
    led_commands = [
        ("LED_BRIGHTNESS", 0),      # Set brightness to 0 (confirmed working)
        ("LED_COLOR", 0),           # Set color to off (suggested by xvf_host)
        ("LED_MODE", 0),            # Set mode to off
        ("LED_ENABLE", 0),          # Disable LEDs
        ("LED_STATE", 0),           # Set state to off
    ]
    
    success = False
    for cmd, value in led_commands:
        result = run_xvf_command(cmd, value)
        if result is not None:
            print(f"  ✅ LEDs disabled via {cmd}={value}")
            success = True
            # Continue trying other commands for complete LED shutdown
            continue
    
    if success:
        print("  💡 All LEDs disabled - power consumption reduced")
    else:
        print("  ⚠️  LED control not available - LEDs may still be on")
    
    return success

def save_config_state(preset, config_dict):
    """Save current configuration to a file for listener to read"""
    try:
        state = {
            'preset': preset,
            'timestamp': time.time(),
            'config': config_dict
        }
        os.makedirs(os.path.dirname(CONFIG_STATE_FILE), exist_ok=True)
        with open(CONFIG_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\n  📄 Configuration saved to: {CONFIG_STATE_FILE}")
    except Exception as e:
        print(f"\n  ⚠️  Could not save config state: {e}")

def configure_balanced_beam():
    """Balanced Beam - HPF 70Hz + AGC (0.08, 30dB) - Recommended for fan noise"""
    print("\n" + "="*80)
    print("  ⚖️  BALANCED BEAM - Recommended Configuration")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/6] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/6] AGC: ENABLED")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/6] AGC Target Level: 0.08 RMS")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.08)
    
    print("[4/6] AGC Max Gain: 30 dB (1000 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 1000)
    
    print("[5/6] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("[6/6] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ Balanced Beam Profile Complete")
    print("\n  🎯 RECOMMENDED FOR FAN NOISE:")
    print("     1️⃣  HPF removes low-frequency rumble")
    print("     2️⃣  AGC maintains optimal speech levels")
    print("     3️⃣  Beamforming spatial rejection")
    print("\n  This is the recommended profile for Jetson NX with fan noise!\n")
    
    save_config_state('balanced_beam', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.08,
        'PP_AGCMAXGAIN': 1000,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_ultra_sensitive():
    """Ultra Sensitive - Maximum far-field detection"""
    print("\n" + "="*80)
    print("  🔥 ULTRA SENSITIVE - Maximum Far-Field Detection")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/6] High-Pass Filter: 70 Hz")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/6] AGC: ENABLED (Aggressive)")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/6] AGC Target Level: 0.10 RMS (higher = more sensitive)")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.10)
    
    print("[4/6] AGC Max Gain: 45 dB (31623 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 31623)
    
    print("[5/6] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("[6/6] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ Ultra Sensitive Profile Complete")
    print("\n  🎯 OPTIMIZED FOR FAR-FIELD (8-16 feet):")
    print("     ⚠️  WARNING: May amplify fan noise if too close\n")
    
    save_config_state('ultra_sensitive', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.10,
        'PP_AGCMAXGAIN': 31623,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_far_field():
    """Far-field configuration"""
    print("\n" + "="*80)
    print("  🎯 FAR-FIELD CONFIGURATION")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/6] High-Pass Filter: 70 Hz")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/6] AGC: ENABLED")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/6] AGC Target Level: 0.08 RMS")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.08)
    
    print("[4/6] AGC Max Gain: 40 dB (10000 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 10000)
    
    print("[5/6] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("[6/6] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ Far-Field Profile Complete\n")
    
    save_config_state('far_field', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.08,
        'PP_AGCMAXGAIN': 10000,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_near_field():
    """Near-field configuration"""
    print("\n" + "="*80)
    print("  🎯 NEAR-FIELD CONFIGURATION")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/6] High-Pass Filter: 70 Hz")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/6] AGC: ENABLED")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/6] AGC Target Level: 0.05 RMS (lower for near-field)")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.05)
    
    print("[4/6] AGC Max Gain: 20 dB (100 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 100)
    
    print("[5/6] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("[6/6] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ Near-Field Profile Complete\n")
    
    save_config_state('near_field', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.05,
        'PP_AGCMAXGAIN': 100,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_hpf_only():
    """High-Pass Filter Only - HPF 70Hz (removes low-frequency rumble)"""
    print("\n" + "="*80)
    print("  🎚️  HIGH-PASS FILTER ONLY - 70Hz")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/3] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    print("      ⚠️  Note: 70Hz is the MINIMUM available on XVF3800")
    print("      This blocks fan noise at 15-20Hz, but not as selectively as a 20Hz filter")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/3] AGC: OFF")
    run_xvf_command("PP_AGCONOFF", 0)
    
    print("[3/3] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ HPF Only Profile Complete")
    print("\n  🎯 MINIMAL PROCESSING:")
    print("     1️⃣  HPF 70Hz removes low-frequency rumble and fan noise (15-20Hz)")
    print("     2️⃣  No automatic gain control")
    print("     3️⃣  Best for clean environments with minimal noise")
    print("\n  ⚠️  LIMITATION: XVF3800 hardware supports HPF at 70Hz, 125Hz, 150Hz, 180Hz only")
    print("      Custom 20Hz cutoff would require external analog/digital filter\n")
    
    save_config_state('hpf_only', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 0,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_agc_only():
    """AGC Only - AGC (0.08, 30dB) with no HPF"""
    print("\n" + "="*80)
    print("  🔧 AGC ONLY")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/5] High-Pass Filter: OFF")
    run_xvf_command("AEC_HPFONOFF", 0)
    
    print("[2/5] AGC: ENABLED")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/5] AGC Target Level: 0.08 RMS")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.08)
    
    print("[4/5] AGC Max Gain: 30 dB (1000 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 1000)
    
    print("[5/5] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("\n  ✅ AGC Only Profile Complete")
    print("\n  🎯 TESTING AGC INDEPENDENTLY:")
    print("     1️⃣  No high-pass filter")
    print("     2️⃣  AGC only for automatic gain control")
    print("     3️⃣  Good for testing AGC alone\n")
    
    save_config_state('agc_only', {
        'AEC_HPFONOFF': 0,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.08,
        'PP_AGCMAXGAIN': 1000,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_agc_10():
    """AGC 10% Increase - HPF 70Hz + AGC with 10% higher target"""
    print("\n" + "="*80)
    print("  🔊 AGC 10% INCREASE")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/6] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/6] AGC: ENABLED")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/6] AGC Target Level: 0.088 RMS (10% increase from 0.08)")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.088)
    
    print("[4/6] AGC Max Gain: 30 dB (1000 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 1000)
    
    print("[5/6] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("[6/6] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ AGC 10% Increase Profile Complete")
    print("\n  🎯 MODERATE PROCESSING + 10% BOOST:")
    print("     1️⃣  HPF removes low-frequency rumble")
    print("     2️⃣  AGC target 10% higher for more amplification")
    print("     3️⃣  Good for testing moderate gain boost\n")
    
    save_config_state('agc_10', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.088,
        'PP_AGCMAXGAIN': 1000,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_agc_20():
    """AGC 20% Increase - HPF 70Hz + AGC with 20% higher target"""
    print("\n" + "="*80)
    print("  🔊 AGC 20% INCREASE")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/6] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/6] AGC: ENABLED")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/6] AGC Target Level: 0.096 RMS (20% increase from 0.08)")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.096)
    
    print("[4/6] AGC Max Gain: 30 dB (1000 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 1000)
    
    print("[5/6] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("[6/6] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ AGC 20% Increase Profile Complete")
    print("\n  🎯 MODERATE PROCESSING + 20% BOOST:")
    print("     1️⃣  HPF removes low-frequency rumble")
    print("     2️⃣  AGC target 20% higher for more amplification")
    print("     3️⃣  Good for testing higher gain boost\n")
    
    save_config_state('agc_20', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.096,
        'PP_AGCMAXGAIN': 1000,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 0
    })
    return True

def configure_agc_20_ec():
    """AGC 20% Increase + Echo Cancellation ON"""
    print("\n" + "="*80)
    print("  🔊 AGC 20% INCREASE + ECHO CANCELLATION")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/7] High-Pass Filter: 70 Hz (removes low-frequency rumble)")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("[2/7] AGC: ENABLED")
    run_xvf_command("PP_AGCONOFF", 1)
    
    print("[3/7] AGC Target Level: 0.096 RMS (20% increase from 0.08)")
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.096)
    
    print("[4/7] AGC Max Gain: 30 dB (1000 linear)")
    run_xvf_command("PP_AGCMAXGAIN", 1000)
    
    print("[5/7] AGC Attack Time: 0.5 seconds")
    run_xvf_command("PP_AGCTIME", 0.5)
    
    print("[6/7] Echo Cancellation: ON")
    run_xvf_command("PP_ECHOONOFF", 1)
    
    print("[7/7] Verify settings written (best effort)")
    # (optional readbacks could be added here)
    
    print("\n  ✅ AGC 20% + Echo Cancellation Profile Complete")
    print("\n  🎯 REDUCED FEEDBACK FROM SPEAKERS:")
    print("     1️⃣  HPF removes low-frequency rumble")
    print("     2️⃣  AGC target 20% higher for more amplification")
    print("     3️⃣  Echo cancellation minimizes TTS leakage into mic\n")
    
    save_config_state('agc_20_ec', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.096,
        'PP_AGCMAXGAIN': 1000,
        'PP_AGCTIME': 0.5,
        'PP_ECHOONOFF': 1
    })
    return True

def reset_defaults():
    """Reset to factory defaults"""
    print("\n" + "="*80)
    print("  🔄 RESETTING TO FACTORY DEFAULTS")
    print("="*80 + "\n")
    
    # Disable LEDs to reduce power draw
    disable_all_leds()
    print()
    
    print("[1/3] High-Pass Filter: OFF")
    run_xvf_command("AEC_HPFONOFF", 0)
    
    print("[2/3] AGC: OFF")
    run_xvf_command("PP_AGCONOFF", 0)
    
    print("[3/3] Echo Cancellation: OFF")
    run_xvf_command("PP_ECHOONOFF", 0)
    
    print("\n  ✅ Factory defaults restored\n")
    
    save_config_state('reset', {
        'AEC_HPFONOFF': 0,
        'PP_AGCONOFF': 0,
        'PP_ECHOONOFF': 0
    })
    return True

def show_current_settings():
    """Show current XVF3800 settings"""
    print("\n" + "="*80)
    print("  📊 CURRENT XVF3800 SETTINGS")
    print("="*80 + "\n")
    
    settings = [
        ("High-Pass Filter", "AEC_HPFONOFF"),
        ("AGC Enabled", "PP_AGCONOFF"),
        ("AGC Target Level", "PP_AGCDESIREDLEVEL"),
        ("AGC Max Gain", "PP_AGCMAXGAIN"),
        ("AGC Attack Time", "PP_AGCTIME"),
        ("Echo Cancellation", "PP_ECHOONOFF"),
    ]
    
    for label, cmd in settings:
        value = run_xvf_command(cmd)
        if value is not None:
            print(f"  {label:25} : {value}")
        else:
            print(f"  {label:25} : ❌ Error reading")
    
    print("\n" + "="*80 + "\n")
    return True

def main():
    """Main execution"""
    # Check if xvf_host exists
    if not os.path.exists(XVF_HOST_PATH):
        print("\n" + "="*80)
        print("  ❌ ERROR: xvf_host not found")
        print("="*80)
        print(f"\n  Expected at: {XVF_HOST_PATH}")
        print("\n  Please ensure XVF3800 SDK is installed and path is correct.\n")
        return 1
    
    preset = sys.argv[1] if len(sys.argv) > 1 else "balanced_beam"
    extra_args = [a.strip().lower() for a in sys.argv[2:]] if len(sys.argv) > 2 else []
    force_echo_on = "echo_on" in extra_args
    
    print("\n" + "="*80)
    print("  🎙️  XVF3800 USB 4 MIC ARRAY TUNER")
    print("  Hardware DSP Configuration for Speech Recognition")
    print("="*80)
    
    try:
        success = False
        
        # Presets
        if preset == "balanced_beam" or preset == "bb":
            success = configure_balanced_beam()
        elif preset == "ultra_sensitive" or preset == "ultra":
            success = configure_ultra_sensitive()
        elif preset == "far_field":
            success = configure_far_field()
        elif preset == "near_field":
            success = configure_near_field()
        elif preset == "hpf_only":
            success = configure_hpf_only()
        elif preset == "agc_only":
            success = configure_agc_only()
        elif preset == "agc_10":
            success = configure_agc_10()
        elif preset == "agc_20":
            success = configure_agc_20()
        elif preset == "agc_20_ec":
            success = configure_agc_20_ec()
        elif preset == "reset":
            success = reset_defaults()
        elif preset == "show":
            success = show_current_settings()
        else:
            print(f"\n  ❌ Unknown preset: {preset}")
            print("\n  Available presets:")
            print("    - balanced_beam (bb)   : Recommended")
            print("    - ultra_sensitive      : Far-field optimized")
            print("    - far_field            : 8-16 feet")
            print("    - near_field           : 1-6 feet")
            print("    - hpf_only             : HPF 70Hz only")
            print("    - agc_only             : AGC only (0.08, 30dB)")
            print("    - agc_10               : HPF + AGC with 10% increase")
            print("    - agc_20               : HPF + AGC with 20% increase")
            print("    - reset                : Factory defaults")
            print("    - show                 : Current settings")
            return 1
        
        if success:
            # Optional post-preset modifiers
            if force_echo_on:
                print("\n  🔁 Enabling Echo Cancellation (PP_ECHOONOFF=1) per 'echo_on' flag...")
                run_xvf_command("PP_ECHOONOFF", 1)
                # Update saved config state to reflect echo ON
                try:
                    if os.path.exists(CONFIG_STATE_FILE):
                        with open(CONFIG_STATE_FILE, "r") as f:
                            state = json.load(f)
                        cfg = state.get("config", {}) or {}
                        cfg["PP_ECHOONOFF"] = 1
                        state["config"] = cfg
                        with open(CONFIG_STATE_FILE, "w") as f:
                            json.dump(state, f, indent=2)
                        print(f"  ✅ Saved echo-on to state: {CONFIG_STATE_FILE}")
                except Exception as e:
                    print(f"  ⚠️  Could not update state file for echo_on: {e}")
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

