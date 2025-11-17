#!/usr/bin/env python3
"""
Test different wake word detection options on Jetson.

This script tests:
1. OpenWakeWord (direct)
2. Mycroft Precise (recommended for Jetson)
3. Vosk (if available)

Run this to see which wake word detection works on your system.
"""

import sys
import numpy as np
import sounddevice as sd

print("="*70)
print("WAKE WORD DETECTION TEST")
print("="*70)
print()

# Test 1: OpenWakeWord
print("[1/3] Testing OpenWakeWord...")
try:
    from openwakeword.model import Model
    oww = Model(wakeword_models=['hey_jarvis'])
    print("  ✅ OpenWakeWord imported successfully")
    print("  📝 Testing with dummy audio...")
    dummy_audio = np.random.randn(1280).astype(np.float32)
    prediction = oww.predict(dummy_audio)
    print(f"  ✅ OpenWakeWord prediction works: {prediction}")
    oww_available = True
except ImportError as e:
    print(f"  ❌ OpenWakeWord not installed: {e}")
    print("  💡 Install: pip install openwakeword")
    oww_available = False
except Exception as e:
    print(f"  ❌ OpenWakeWord error: {e}")
    oww_available = False

print()

# Test 2: Mycroft Precise
print("[2/3] Testing Mycroft Precise...")
try:
    from precise_runner import PreciseEngine, PreciseRunner
    from precise_runner.runner import ListenerEngine
    print("  ✅ Mycroft Precise imported successfully")
    print("  💡 Precise is recommended for Jetson - very reliable!")
    precise_available = True
except ImportError as e:
    print(f"  ❌ Mycroft Precise not installed: {e}")
    print("  💡 Install: pip install precise-runner")
    print("  💡 Models: https://github.com/MycroftAI/precise-data/tree/models")
    precise_available = False
except Exception as e:
    print(f"  ❌ Mycroft Precise error: {e}")
    precise_available = False

print()

# Test 3: Vosk
print("[3/3] Testing Vosk...")
try:
    from vosk import Model as VoskModel, SetLogLevel
    print("  ✅ Vosk imported successfully")
    print("  💡 Vosk has wake word detection built-in")
    vosk_available = True
except ImportError as e:
    print(f"  ❌ Vosk not installed: {e}")
    print("  💡 Install: pip install vosk")
    vosk_available = False
except Exception as e:
    print(f"  ❌ Vosk error: {e}")
    vosk_available = False

print()
print("="*70)
print("SUMMARY")
print("="*70)
if oww_available:
    print("  ✅ OpenWakeWord: Available")
else:
    print("  ❌ OpenWakeWord: Not available")
if precise_available:
    print("  ✅ Mycroft Precise: Available (RECOMMENDED for Jetson)")
else:
    print("  ❌ Mycroft Precise: Not available")
if vosk_available:
    print("  ✅ Vosk: Available")
else:
    print("  ❌ Vosk: Not available")

print()
print("RECOMMENDATION:")
if precise_available:
    print("  🎯 Use Mycroft Precise - it's the most reliable for Jetson!")
    print("  📦 Install: pip install precise-runner")
    print("  📥 Download model: wget https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb")
elif oww_available:
    print("  🎯 Use OpenWakeWord (direct, not Wyoming)")
    print("  📦 Already installed!")
else:
    print("  🎯 Install Mycroft Precise for best results:")
    print("     pip install precise-runner")
print("="*70)

