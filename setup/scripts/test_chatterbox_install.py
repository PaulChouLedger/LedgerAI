#!/usr/bin/env python3
"""
Test ChatterboxTTS Installation

This script verifies that ChatterboxTTS is properly installed and can be used.

Usage:
    python setup/scripts/test_chatterbox_install.py
"""

import sys
import os
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

def test_chatterbox_import():
    """Test if ChatterboxTTS can be imported"""
    print("=" * 70)
    print("ChatterboxTTS Installation Test")
    print("=" * 70)
    print()
    
    print("1️⃣ Testing import...")
    try:
        # Try different import paths
        try:
            from chatterbox.tts import ChatterboxTTS
            print("   ✅ Imported from chatterbox.tts")
        except ImportError:
            try:
                from chatterbox import ChatterboxTTS
                print("   ✅ Imported from chatterbox")
            except ImportError:
                print("   ❌ Failed to import ChatterboxTTS")
                print("   💡 Install with: pip install setuptools && pip install chatterbox-tts")
                return False
        
        print("   ✅ ChatterboxTTS import successful!")
        return True
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_chatterbox_initialization():
    """Test if ChatterboxTTS can be initialized"""
    print("\n2️⃣ Testing initialization...")
    try:
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            from chatterbox import ChatterboxTTS
        
        # Try from_pretrained
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   🔧 Initializing with device: {device}")
            model = ChatterboxTTS.from_pretrained(device=device)
            print(f"   ✅ Initialized successfully using from_pretrained()")
            return True, model
        except (AttributeError, TypeError) as e:
            # Fallback to simple constructor
            print(f"   ⚠️  from_pretrained() not available: {e}")
            print("   🔧 Trying simple constructor...")
            model = ChatterboxTTS()
            print(f"   ✅ Initialized successfully using ChatterboxTTS()")
            return True, model
    except Exception as e:
        print(f"   ❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_chatterbox_synthesis(model):
    """Test if ChatterboxTTS can synthesize speech"""
    print("\n3️⃣ Testing speech synthesis...")
    try:
        test_text = "Hello, this is a test of ChatterboxTTS voice synthesis."
        print(f"   📝 Test text: \"{test_text}\"")
        
        # Try to synthesize
        if hasattr(model, 'generate'):
            print("   🔧 Using generate() method...")
            audio = model.generate(test_text)
        elif hasattr(model, 'synthesize'):
            print("   🔧 Using synthesize() method...")
            audio = model.synthesize(test_text)
        else:
            print("   ⚠️  No synthesis method found (generate or synthesize)")
            return False
        
        # Check audio output
        import numpy as np
        if isinstance(audio, np.ndarray):
            duration = len(audio) / 22050  # Assuming 22050 Hz
            print(f"   ✅ Synthesis successful!")
            print(f"   📊 Audio shape: {audio.shape}")
            print(f"   ⏱️  Duration: {duration:.2f} seconds")
            return True
        else:
            print(f"   ✅ Synthesis successful! (returned {type(audio)})")
            return True
            
    except Exception as e:
        print(f"   ❌ Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_cloning(model):
    """Test if voice cloning works (if sample exists)"""
    print("\n4️⃣ Testing voice cloning (if sample available)...")
    
    # Check for voice sample
    sample_paths = [
        workspace_root / "assets" / "voice_samples" / "sample.wav",
        os.getenv("CHATTERBOX_VOICE_SAMPLE", "")
    ]
    
    voice_sample = None
    for path in sample_paths:
        if path and os.path.exists(str(path)):
            voice_sample = str(path)
            break
    
    if not voice_sample:
        print("   ⚠️  No voice sample found (skipping cloning test)")
        print("   💡 Generate one with: python setup/scripts/generate_chatterbox_voice_sample.py")
        return True  # Not an error, just no sample
    
    print(f"   🎭 Found voice sample: {voice_sample}")
    
    try:
        test_text = "This is a test of voice cloning with ChatterboxTTS."
        
        # Try voice cloning
        if hasattr(model, 'generate'):
            import inspect
            sig = inspect.signature(model.generate)
            params = list(sig.parameters.keys())
            
            if 'audio_prompt_path' in params:
                print("   🔧 Using generate() with audio_prompt_path...")
                audio = model.generate(test_text, audio_prompt_path=voice_sample)
                print("   ✅ Voice cloning successful!")
                return True
            else:
                print("   ⚠️  generate() doesn't support audio_prompt_path")
        
        if hasattr(model, 'synthesize'):
            import inspect
            sig = inspect.signature(model.synthesize)
            if 'audio_prompt_path' in sig.parameters:
                print("   🔧 Using synthesize() with audio_prompt_path...")
                audio = model.synthesize(test_text, audio_prompt_path=voice_sample)
                print("   ✅ Voice cloning successful!")
                return True
        
        print("   ⚠️  Voice cloning API not available in this version")
        return True  # Not an error, API may not support it
        
    except Exception as e:
        print(f"   ❌ Voice cloning error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    import platform
    python_version = sys.version_info
    
    print(f"🖥️  Environment: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check Python version compatibility
    if python_version.major == 3 and python_version.minor >= 13:
        print("⚠️  Warning: Python 3.13+ may have compatibility issues with pkuseg")
        print("   💡 Install on Jetson (Python 3.8-3.10) instead")
        print()
    elif python_version.major == 3 and python_version.minor >= 12:
        print("⚠️  Warning: Python 3.12+ may have compatibility issues")
        print("   💡 Install setuptools first: pip install setuptools")
        print()
    
    print()
    
    # Test 1: Import
    if not test_chatterbox_import():
        print("\n" + "=" * 70)
        print("❌ Installation test FAILED at import stage")
        print("=" * 70)
        if python_version.major == 3 and python_version.minor >= 12:
            print("\n💡 Python version compatibility issue detected")
            print("   Install on Jetson (Python 3.8-3.10) or use Python 3.9 locally")
        else:
            print("\n💡 Fix: pip install setuptools && pip install chatterbox-tts")
        sys.exit(1)
    
    # Test 2: Initialization
    success, model = test_chatterbox_initialization()
    if not success:
        print("\n" + "=" * 70)
        print("❌ Installation test FAILED at initialization stage")
        print("=" * 70)
        sys.exit(1)
    
    # Test 3: Synthesis
    if not test_chatterbox_synthesis(model):
        print("\n" + "=" * 70)
        print("❌ Installation test FAILED at synthesis stage")
        print("=" * 70)
        sys.exit(1)
    
    # Test 4: Voice cloning (optional)
    test_voice_cloning(model)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ChatterboxTTS Installation Test PASSED!")
    print("=" * 70)
    print("\n📋 Next steps:")
    print("   1. Generate voice sample: python setup/scripts/generate_chatterbox_voice_sample.py")
    print("   2. Enable ChatterboxTTS in Settings → TTS Engine")
    print("   3. Enable Voice Cloning toggle (if sample exists)")
    print("   4. Test by asking AuraVision a question")
    print()
    
    sys.exit(0)

if __name__ == "__main__":
    main()

