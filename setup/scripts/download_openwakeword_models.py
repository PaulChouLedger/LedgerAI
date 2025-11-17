#!/usr/bin/env python3
"""
Download OpenWakeWord Models

This script downloads the required model files for OpenWakeWord.
Run this if you see "File doesn't exist" errors when using wake word detection.
"""

import sys
import os

def main():
    print("="*70)
    print("OpenWakeWord Model Downloader")
    print("="*70)
    print()
    
    try:
        import openwakeword
        print("✅ OpenWakeWord package found")
    except ImportError:
        print("❌ OpenWakeWord not installed!")
        print("   Install with: pip install openwakeword")
        sys.exit(1)
    
    try:
        from openwakeword.utils import download_models
        print("📥 Downloading models...")
        print("   This may take a few minutes depending on your internet connection.")
        print()
        
        download_models()
        
        # Show where models were saved
        try:
            import openwakeword
            package_path = os.path.dirname(openwakeword.__file__)
            models_dir = os.path.join(package_path, 'resources', 'models')
            print()
            print("Models saved to:")
            print(f"  {models_dir}")
        except:
            pass
        
        print()
        print("="*70)
        print("✅ Models downloaded successfully!")
        print("="*70)
        print()
        print("You can now use wake word detection in Aura.")
        print("Restart Aura if it's currently running.")
        print()
        print("To check model location later, run:")
        print("  python setup/scripts/check_openwakeword_models.py")
        
    except ImportError:
        print("❌ download_models() function not found in this version of OpenWakeWord")
        print("   Try updating: pip install --upgrade openwakeword")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try updating OpenWakeWord: pip install --upgrade openwakeword")
        print("  3. Check OpenWakeWord GitHub for latest instructions:")
        print("     https://github.com/dscripka/openWakeWord")
        sys.exit(1)

if __name__ == "__main__":
    main()

