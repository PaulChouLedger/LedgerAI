#!/usr/bin/env python3
"""
Test script to inspect Chatterbox API and find correct voice cloning parameters
"""
import sys
import os
import inspect

# Add path to import chatterbox
sys.path.insert(0, '/app/chatterbox')

try:
    from chatterbox.tts import ChatterboxTTS
    print("✅ Successfully imported ChatterboxTTS")
except ImportError:
    try:
        from chatterbox import ChatterboxTTS
        print("✅ Successfully imported ChatterboxTTS (alternative path)")
    except ImportError as e:
        print(f"❌ Failed to import ChatterboxTTS: {e}")
        sys.exit(1)

# Check available methods
print("\n" + "=" * 70)
print("  ChatterboxTTS API Inspection")
print("=" * 70)
print()

# Check if from_pretrained exists
if hasattr(ChatterboxTTS, 'from_pretrained'):
    print("✅ ChatterboxTTS.from_pretrained() exists")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Initializing on {device}...")
        tts = ChatterboxTTS.from_pretrained(device=device)
        print(f"   ✅ Initialized successfully")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        sys.exit(1)
else:
    print("❌ ChatterboxTTS.from_pretrained() not found")
    sys.exit(1)

# Inspect generate method
print("\n" + "=" * 70)
print("  generate() Method")
print("=" * 70)
if hasattr(tts, 'generate'):
    sig = inspect.signature(tts.generate)
    params = list(sig.parameters.keys())
    print(f"Parameters: {params}")
    print()
    for param_name in params:
        param = sig.parameters[param_name]
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
        print(f"  - {param_name}{annotation}{default}")
else:
    print("❌ generate() method not found")

# Inspect synthesize method
print("\n" + "=" * 70)
print("  synthesize() Method")
print("=" * 70)
if hasattr(tts, 'synthesize'):
    sig = inspect.signature(tts.synthesize)
    params = list(sig.parameters.keys())
    print(f"Parameters: {params}")
    print()
    for param_name in params:
        param = sig.parameters[param_name]
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
        print(f"  - {param_name}{annotation}{default}")
else:
    print("❌ synthesize() method not found")

# Check for voice cloning related methods
print("\n" + "=" * 70)
print("  Voice Cloning Methods")
print("=" * 70)
voice_methods = [m for m in dir(tts) if 'voice' in m.lower() or 'embedding' in m.lower() or 'prompt' in m.lower() or 'clone' in m.lower()]
if voice_methods:
    for method in voice_methods:
        if not method.startswith('_'):
            print(f"  - {method}")
            if callable(getattr(tts, method)):
                try:
                    sig = inspect.signature(getattr(tts, method))
                    print(f"    Parameters: {list(sig.parameters.keys())}")
                except:
                    pass
else:
    print("  No voice cloning methods found")

# Check all public methods
print("\n" + "=" * 70)
print("  All Public Methods")
print("=" * 70)
all_methods = [m for m in dir(tts) if not m.startswith('_') and callable(getattr(tts, m))]
for method in sorted(all_methods):
    print(f"  - {method}")

print("\n" + "=" * 70)
print("  Test Voice Cloning")
print("=" * 70)

# Test with a sample file
test_file = "/app/voice_samples/audio3.wav"
if os.path.exists(test_file):
    print(f"✅ Test file found: {test_file}")
    print(f"   Testing voice cloning...")
    
    test_text = "Hello, this is a test of voice cloning."
    
    # Try different parameter combinations
    if hasattr(tts, 'generate'):
        print(f"\n🔍 Testing generate() with different parameters...")
        
        # Try audio_prompt_path
        try:
            print(f"   Trying: generate(text, audio_prompt_path='{test_file}')")
            audio = tts.generate(test_text, audio_prompt_path=test_file)
            print(f"   ✅ SUCCESS with audio_prompt_path!")
        except TypeError as e:
            print(f"   ❌ Failed: {e}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
        
        # Try audio_prompt
        try:
            print(f"   Trying: generate(text, audio_prompt='{test_file}')")
            audio = tts.generate(test_text, audio_prompt=test_file)
            print(f"   ✅ SUCCESS with audio_prompt!")
        except TypeError as e:
            print(f"   ❌ Failed: {e}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
        
        # Try voice_embedding
        try:
            print(f"   Trying: generate(text, voice_embedding='{test_file}')")
            audio = tts.generate(test_text, voice_embedding=test_file)
            print(f"   ✅ SUCCESS with voice_embedding!")
        except TypeError as e:
            print(f"   ❌ Failed: {e}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
else:
    print(f"⚠️  Test file not found: {test_file}")
    print(f"   Available files in /app/voice_samples/:")
    if os.path.exists("/app/voice_samples"):
        for f in os.listdir("/app/voice_samples"):
            print(f"     - {f}")

print("\n" + "=" * 70)
