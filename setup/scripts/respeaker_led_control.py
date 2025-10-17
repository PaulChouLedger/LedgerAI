#!/usr/bin/env python3
"""
ReSpeaker LED Control Script

Simple script to control the 12 RGB LEDs on ReSpeaker Mic Array.

Usage:
    sudo python3 setup/scripts/respeaker_led_control.py off
    sudo python3 setup/scripts/respeaker_led_control.py brightness [0-31]
    sudo python3 setup/scripts/respeaker_led_control.py color [red] [green] [blue]
"""

import sys
import time

def turn_off_leds():
    """Turn off all LEDs"""
    try:
        from pixel_ring import pixel_ring
        
        print("Turning off LEDs...")
        
        # Method 1: Set brightness to 0
        pixel_ring.set_brightness(0)
        time.sleep(0.1)
        
        # Method 2: Set mono mode to black
        pixel_ring.mono(0x000000)
        time.sleep(0.1)
        
        # Method 3: Explicit off command
        pixel_ring.off()
        time.sleep(0.1)
        
        print("✅ LEDs turned OFF")
        print("Power savings: ~10mA")
        return True
        
    except ImportError:
        print("❌ pixel_ring library not installed!")
        print("\nInstall with:")
        print("  git clone https://github.com/respeaker/pixel_ring.git")
        print("  cd pixel_ring")
        print("  sudo python setup.py install")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def set_brightness(level):
    """Set LED brightness (0-31)"""
    try:
        from pixel_ring import pixel_ring
        
        level = max(0, min(31, int(level)))
        
        print(f"Setting LED brightness to {level}/31 ({int(level/31*100)}%)...")
        pixel_ring.set_brightness(level)
        
        if level == 0:
            pixel_ring.off()
            print("✅ LEDs OFF")
        else:
            print(f"✅ Brightness set to {level}/31")
        
        return True
        
    except ImportError:
        print("❌ pixel_ring library not installed!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def set_color(red, green, blue):
    """Set all LEDs to a specific color"""
    try:
        from pixel_ring import pixel_ring
        
        r = max(0, min(255, int(red)))
        g = max(0, min(255, int(green)))
        b = max(0, min(255, int(blue)))
        
        color = (r << 16) | (g << 8) | b
        
        print(f"Setting LEDs to RGB({r}, {g}, {b})...")
        pixel_ring.mono(color)
        
        print(f"✅ Color set to #{r:02x}{g:02x}{b:02x}")
        return True
        
    except ImportError:
        print("❌ pixel_ring library not installed!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  sudo python3 setup/scripts/respeaker_led_control.py off")
        print("  sudo python3 setup/scripts/respeaker_led_control.py brightness [0-31]")
        print("  sudo python3 setup/scripts/respeaker_led_control.py color [R] [G] [B]")
        print("\nExamples:")
        print("  sudo python3 setup/scripts/respeaker_led_control.py off")
        print("  sudo python3 setup/scripts/respeaker_led_control.py brightness 5")
        print("  sudo python3 setup/scripts/respeaker_led_control.py color 255 0 0  # Red")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "off":
        success = turn_off_leds()
    elif command == "brightness":
        if len(sys.argv) < 3:
            print("❌ Missing brightness value (0-31)")
            return 1
        success = set_brightness(sys.argv[2])
    elif command == "color":
        if len(sys.argv) < 5:
            print("❌ Missing RGB values")
            print("Usage: sudo python3 respeaker_led_control.py color [R] [G] [B]")
            return 1
        success = set_color(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(f"❌ Unknown command: {command}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

