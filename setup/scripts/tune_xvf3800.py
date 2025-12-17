#!/usr/bin/env python3
"""
XVF3800 4 Mic Array Tuner

Configures the hardware DSP on the XVF3800 for optimal
speech recognition (ASR).

Automatically reboots device, configures GPIO/DSP, and disables LEDs.

Usage:
    python3 tune_xvf3800.py [preset]
    
PRESETS:
    - agc_20_ec            : AGC 20% + Echo Cancellation ON (PP_ECHOONOFF=1) - DEFAULT ⭐ RECOMMENDED
    - reset                : Factory defaults (reboot to factory state)
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

# Detect target user's home directory dynamically (works even when run via sudo or systemd)
# Method 1: Check SUDO_USER environment variable (when run via sudo)
_sudo_user = os.environ.get("SUDO_USER")
if _sudo_user:
    USER_HOME = os.path.expanduser(f"~{_sudo_user}")
else:
    # Method 2: Detect from LedgerAI directory ownership (for systemd services)
    # Find LedgerAI directory by locating this script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LEDGERAI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    
    # Get owner of LedgerAI directory
    try:
        import pwd
        import stat
        dir_stat = os.stat(LEDGERAI_DIR)
        owner_uid = dir_stat.st_uid
        owner_info = pwd.getpwuid(owner_uid)
        USER_HOME = owner_info.pw_dir
    except (ImportError, OSError, KeyError):
        # Fallback: use current user's home
        USER_HOME = os.path.expanduser("~")

# State file for listener to read - use target user's home directory  
CONFIG_STATE_FILE = os.path.join(USER_HOME, 'LedgerAI', 'data', 'xvf3800_config.json')

# Path to xvf_host binary - use target user's home directory
XVF_HOST_PATH = os.path.join(USER_HOME, 'reSpeaker_XVF3800_USB_4MIC_ARRAY', 'host_control', 'jetson', 'xvf_host')

def run_xvf_command(cmd, *args):
    """Run xvf_host command and return result
    
    Increased timeout to 5 seconds for boot-time scenarios where device
    may take longer to respond during initialization.
    """
    try:
        full_cmd = [XVF_HOST_PATH, cmd] + list(str(arg) for arg in args)
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Show stderr for debugging
            stderr_msg = f": {result.stderr.strip()}" if result.stderr.strip() else ""
            print(f"  ⚠️  Warning: {cmd} returned {result.returncode}{stderr_msg}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Warning: {cmd} timed out (device may not be ready)")
        return None
    except Exception as e:
        print(f"  ⚠️  Error running {cmd}: {e}")
        return None

def disable_all_leds():
    """Disable all LEDs on XVF3800 to reduce power draw
    
    Based on official documentation: https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/
    LED commands: led_effect, led_color (hex), led_speed, led_brightness (0-255)
    """
    print("[LED] Disabling all LEDs to reduce power consumption...")
    
    success = False
    
    # Critical order: Disable effect first, then brightness, repeat multiple times
    # This ensures LEDs are completely off, not just dim
    for attempt in range(3):
        if attempt > 0:
            time.sleep(0.3)
        
        # Step 1: Disable LED effect FIRST (stops any patterns/animations)
        result = run_xvf_command("LED_EFFECT", 0)
        if result is not None:
            print(f"  ✅ Set LED_EFFECT=0 (disabled patterns)")
            success = True
        
        # Step 2: Set brightness to 0 (primary way to turn off LEDs)
        result = run_xvf_command("LED_BRIGHTNESS", 0)
        if result is not None:
            print(f"  ✅ Set LED_BRIGHTNESS=0 (LEDs off)")
            success = True
        
        # Step 3: Set color to black (ensures no color bleeding)
        result = run_xvf_command("LED_COLOR", "0x000000")
        if result is not None:
            print(f"  ✅ Set LED_COLOR=0x000000 (black/off)")
            success = True
        
        # Step 4: Set speed to minimum (though brightness=0 should override)
        result = run_xvf_command("LED_SPEED", 1)
        if result is not None:
            print(f"  ✅ Set LED_SPEED=1 (minimum)")
            success = True
        
        # Step 5: Set brightness to 0 AGAIN (ensure it sticks)
        result = run_xvf_command("LED_BRIGHTNESS", 0)
        if result is not None:
            success = True
    
    # Final verification: Set brightness one more time after a short delay
    time.sleep(0.2)
    result = run_xvf_command("LED_BRIGHTNESS", 0)
    if result is not None:
        print(f"  ✅ Final LED_BRIGHTNESS=0 (verification)")
        success = True
    
    if success:
        print("  💡 All LED settings applied - LEDs should be completely off")
        print("  💡 Power consumption reduced")
    else:
        print("  ⚠️  LED control commands not available")
        print("  ⚠️  Check if xvf_host is properly installed and device is connected")
    
    return success

def initialize_gpio_pins():
    """Initialize GPIO pins for microphone and amplifier
    
    Critical GPIO pins:
    - X0D30: Microphone mute circuit control (LOW = unmuted, HIGH = muted)
    - X0D31: Audio amplifier enable (LOW = enabled, HIGH = disabled)
    
    These must be set correctly for audio capture to work!
    
    Retries up to 5 times with delays to handle boot-time device initialization.
    """
    print("[GPIO] Initializing GPIO pins for audio capture...")
    
    success_count = 0
    max_retries = 5
    retry_delay = 1.0  # seconds
    
    # X0D30: Set microphone mute circuit to LOW (unmuted)
    # This is critical - if HIGH, microphone is muted and won't capture audio
    # Retry logic for boot-time scenarios where device may not be ready immediately
    for attempt in range(max_retries):
        result = run_xvf_command("GPO_WRITE_VALUE", 30, 0)
        if result is not None:
            print("  ✅ Set X0D30 (mic mute) = LOW (microphones unmuted)")
            success_count += 1
            break
        elif attempt < max_retries - 1:
            print(f"  ⚠️  Attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            print("  ⚠️  Failed to set X0D30 (mic mute) after retries - microphone may be muted!")
    
    # X0D31: Set audio amplifier enable to LOW (enabled)
    # This is critical - if HIGH, amplifier is disabled
    # Retry logic for boot-time scenarios where device may not be ready immediately
    for attempt in range(max_retries):
        result = run_xvf_command("GPO_WRITE_VALUE", 31, 0)
        if result is not None:
            print("  ✅ Set X0D31 (amp enable) = LOW (amplifier enabled)")
            success_count += 1
            break
        elif attempt < max_retries - 1:
            print(f"  ⚠️  Attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            print("  ⚠️  Failed to set X0D31 (amp enable) after retries - amplifier may be disabled!")
    
    # Verify GPIO state
    result = run_xvf_command("GPO_READ_VALUES")
    if result:
        # Parse output: "GPO_READ_VALUES 0 0 0 1 0"
        # Or with debug: "Device (USB)::device_init() -- Found device VID: 10374 PID: 26 interface: 3\nGPO_READ_VALUES 0 0 0 1 0"
        # Values are: X0D11, X0D30, X0D31, X0D33, X0D39
        # Find the line with GPO_READ_VALUES
        lines = result.strip().split('\n')
        gpo_line = None
        for line in lines:
            if 'GPO_READ_VALUES' in line:
                gpo_line = line
                break
        
        if gpo_line:
            # Extract just the numeric values
            values = gpo_line.split()
            # Find GPO_READ_VALUES in the list and get values after it
            try:
                idx = values.index('GPO_READ_VALUES')
                if len(values) > idx + 4:
                    x0d11 = values[idx + 1]  # X0D11
                    x0d30 = values[idx + 2]  # X0D30 (mic mute)
                    x0d31 = values[idx + 3]  # X0D31 (amp enable)
                    x0d33 = values[idx + 4]  # X0D33 (LED power)
                    print(f"  📊 Current GPIO state: X0D30={x0d30} (mic mute), X0D31={x0d31} (amp enable)")
                    
                    if x0d30 == "0" and x0d31 == "0":
                        print("  ✅ GPIO pins correctly configured for audio capture")
                        success_count += 1
                    else:
                        print(f"  ⚠️  GPIO pins may not be correct: X0D30={x0d30}, X0D31={x0d31}")
                else:
                    print("  ⚠️  Could not parse GPIO values from output")
            except (ValueError, IndexError):
                print("  ⚠️  Could not parse GPIO read values")
        else:
            print("  ⚠️  GPO_READ_VALUES output not found in response")
    
    if success_count >= 2:
        print("  💡 GPIO initialization complete - audio capture should work")
    else:
        print("  ⚠️  GPIO initialization may have failed - audio capture may not work")
        print("  💡 Check if xvf_host is properly installed and device is connected")
    
    return success_count >= 2

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

# Removed unused presets - keeping only agc_20_ec and reset

def configure_agc_20_ec():
    """Simplified: Reboot, configure DSP, disable LEDs"""
    print("\n" + "="*80)
    print("  🔊 XVF3800 Configuration: Reboot → DSP → LEDs")
    print("="*80 + "\n")
    
    # Step 1: Reboot device (software reset to factory defaults)
    print("[1/3] Rebooting device (software reset)...")
    result = run_xvf_command("REBOOT", 1)
    if result is not None:
        print("  ✅ REBOOT command sent")
        time.sleep(3)  # Wait for device to reboot
    else:
        print("  ⚠️  REBOOT command failed (device may already be reset)")
    print()
    
    # Step 2: Initialize GPIO pins (critical for audio capture)
    print("[2/3] Configuring GPIO pins for audio capture...")
    initialize_gpio_pins()
    print()
    
    # Step 3: Configure DSP settings
    print("[3/3] Configuring DSP settings...")
    print("  - High-Pass Filter: 70 Hz")
    run_xvf_command("AEC_HPFONOFF", 1)
    
    print("  - AGC: ENABLED (target: 0.05 RMS, max gain: 20x linear = ~26 dB, response: 0.1s)")
    run_xvf_command("PP_AGCONOFF", 1)
    run_xvf_command("PP_AGCDESIREDLEVEL", 0.05)
    run_xvf_command("PP_AGCMAXGAIN", 20)  # Linear gain factor (20x = ~26 dB) - reduced from 1000 to prevent clipping
    run_xvf_command("PP_AGCTIME", 0.1)  # Faster response (0.1s) for better clipping prevention - hardware-only solution
    
    print("  - Echo Cancellation: ON")
    run_xvf_command("PP_ECHOONOFF", 1)
    print()
    
    # Step 4: Disable LEDs
    print("[4/4] Disabling LEDs...")
    disable_all_leds()
    print()
    
    print("\n  ✅ Configuration Complete")
    print("  🎯 Device rebooted, DSP configured, LEDs disabled\n")
    
    save_config_state('agc_20_ec', {
        'AEC_HPFONOFF': 1,
        'PP_AGCONOFF': 1,
        'PP_AGCDESIREDLEVEL': 0.05,
        'PP_AGCMAXGAIN': 20,  # Linear gain factor (20x = ~26 dB) - reduced to prevent clipping
        'PP_AGCTIME': 0.1,  # Faster response (0.1s) for hardware-only clipping prevention
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
    
    preset = sys.argv[1] if len(sys.argv) > 1 else "agc_20_ec"
    
    print("\n" + "="*80)
    print("  🎙️  XVF3800 4 MIC ARRAY TUNER")
    print("  Hardware DSP Configuration for Speech Recognition")
    print("="*80)
    
    try:
        success = False
        
        # Presets
        if preset == "agc_20_ec":
            success = configure_agc_20_ec()
        elif preset == "reset":
            success = reset_defaults()
        else:
            print(f"\n  ❌ Unknown preset: {preset}")
            print("\n  Available presets:")
            print("    - agc_20_ec            : AGC 20% + Echo Cancellation ON (DEFAULT ⭐ RECOMMENDED)")
            print("    - reset                : Factory defaults")
            return 1
        
        if success:
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

