#!/usr/bin/env python3
"""
Check OpenWakeWord Model Location

This script shows where OpenWakeWord models are stored and lists available models.
"""

import sys
import os

def main():
    print("="*70)
    print("OpenWakeWord Model Location Checker")
    print("="*70)
    print()
    
    try:
        import openwakeword
        print("✅ OpenWakeWord package found")
        
        # Get the package location
        package_path = os.path.dirname(openwakeword.__file__)
        models_dir = os.path.join(package_path, 'resources', 'models')
        
        print()
        print("Package location:")
        print(f"  {package_path}")
        print()
        print("Models directory:")
        print(f"  {models_dir}")
        print()
        
        # Check if models directory exists
        if os.path.exists(models_dir):
            print("✅ Models directory exists")
            print()
            print("Available model files:")
            
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
            if model_files:
                for model_file in sorted(model_files):
                    model_path = os.path.join(models_dir, model_file)
                    size = os.path.getsize(model_path)
                    size_mb = size / (1024 * 1024)
                    print(f"  ✅ {model_file} ({size_mb:.2f} MB)")
            else:
                print("  ❌ No .onnx model files found")
                print()
                print("  Models need to be downloaded. Run:")
                print("    python setup/scripts/download_openwakeword_models.py")
        else:
            print("❌ Models directory does not exist")
            print()
            print("Models need to be downloaded. Run:")
            print("  python setup/scripts/download_openwakeword_models.py")
        
        print()
        print("="*70)
        print("Model Usage")
        print("="*70)
        print()
        print("Pre-trained models are stored in the package directory.")
        print("Custom models can be placed anywhere and referenced by path.")
        print()
        print("To use a custom model, set wake_word_model_path in app_settings.json:")
        print("  ~/LedgerAI/data/app_settings.json")
        print()
        
    except ImportError:
        print("❌ OpenWakeWord not installed!")
        print("   Install with: pip install openwakeword")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

