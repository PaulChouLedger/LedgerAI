#!/usr/bin/env python3
"""
ReSpeaker LED Keeper - Persistent LED Off

Runs in background and continuously ensures LEDs stay off,
even if they revert due to USB interference or power glitches.

Usage:
    sudo python3 setup/scripts/led_keeper.py &
    
To stop:
    killall -9 led_keeper.py
"""

import time
import sys

def keep_leds_off():
    """Continuously keep LEDs off"""
    try:
        from pixel_ring import pixel_ring
    except ImportError:
        print("❌ pixel_ring library not found!")
        print("Install with: sudo pip install pixel-ring")
        return 1
    
    print("🔄 LED Keeper started - keeping LEDs off...")
    print("Press Ctrl+C to stop\n")
    
    iteration = 0
    
    try:
        while True:
            # Set LEDs off every 2 seconds
            pixel_ring.set_brightness(0)
            pixel_ring.mono(0x000000)
            pixel_ring.off()
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"✅ LED Keeper alive - {iteration} iterations")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 LED Keeper stopped")
        pixel_ring.off()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(keep_leds_off())

