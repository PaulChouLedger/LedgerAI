#!/usr/bin/env python3
"""
Test Porcupine installation and check available keywords
"""

try:
    import pvporcupine
    print("✅ Porcupine imported successfully!")
    
    # Try to get version (may not be available in all versions)
    try:
        version = pvporcupine.__version__
        print(f"   Version: {version}")
    except AttributeError:
        print("   Version: (not available in this build)")
    print()
    
    # Check available keywords
    print("📋 Available built-in keywords:")
    keywords = []
    try:
        if hasattr(pvporcupine, 'KEYWORDS'):
            # KEYWORDS might be a set or dict
            if isinstance(pvporcupine.KEYWORDS, set):
                keywords = sorted(list(pvporcupine.KEYWORDS))
            elif isinstance(pvporcupine.KEYWORDS, dict):
                keywords = sorted(list(pvporcupine.KEYWORDS.keys()))
            else:
                keywords = sorted(list(pvporcupine.KEYWORDS))
        else:
            print("   ⚠️  KEYWORDS attribute not found")
            print("   This may be a source build - you'll need a custom model")
    except Exception as e:
        print(f"   ⚠️  Could not access KEYWORDS: {e}")
        print("   You'll need to use a custom model")
    if keywords:
        for i, keyword in enumerate(keywords[:20], 1):  # Show first 20
            print(f"   {i}. {keyword}")
        if len(keywords) > 20:
            print(f"   ... and {len(keywords) - 20} more")
    else:
        print("   (No built-in keywords found)")
    
    print()
    
    # Check if "hey aura" or similar is available
    aura_keywords = [k for k in keywords if 'aura' in k.lower()]
    if aura_keywords:
        print(f"✅ Found Aura-related keywords: {aura_keywords}")
    else:
        print("⚠️  'hey aura' not found in built-in keywords")
        print("   You'll need to train a custom model at: https://console.picovoice.ai/")
    
except ImportError as e:
    print(f"❌ Failed to import Porcupine: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

