#!/usr/bin/env python3
"""
Diagnostic script to check transformers and unsloth imports
Run this to diagnose import issues
"""

import sys
import os

print("🔍 Diagnosing import issues...")
print("=" * 60)

# Check Python version
print(f"\n📦 Python version: {sys.version}")
print(f"📦 Python executable: {sys.executable}")

# Check Python path
print("\n📂 Python path:")
for i, p in enumerate(sys.path, 1):
    print(f"  {i}. {p}")

# Check transformers
print("\n🔍 Checking transformers...")
try:
    import transformers
    print(f"  ✅ Transformers imported successfully")
    print(f"  Version: {transformers.__version__}")
    print(f"  Location: {transformers.__file__}")
    
    # Check if top_k_top_p_filtering exists
    try:
        from transformers import top_k_top_p_filtering
        print(f"  ✅ top_k_top_p_filtering is available")
    except ImportError as e:
        print(f"  ❌ top_k_top_p_filtering is NOT available: {e}")
        print(f"  This means transformers version is incompatible!")
        
        # Check version
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse("4.46.0"):
            print(f"  ⚠️  Transformers {transformers.__version__} is >= 4.46.0 (incompatible)")
            print(f"  Need to downgrade to < 4.46.0")
        else:
            print(f"  ⚠️  Version {transformers.__version__} should be compatible, but function missing")
            print(f"  This might be a cache issue")
            
except ImportError as e:
    print(f"  ❌ Could not import transformers: {e}")

# Check for multiple transformers installations
print("\n🔍 Checking for multiple transformers installations...")
import site
user_site = site.getusersitepackages()
site_packages = site.getsitepackages()

transformers_locations = []
for path in [user_site] + site_packages:
    transformers_path = os.path.join(path, "transformers")
    if os.path.exists(transformers_path):
        transformers_locations.append(transformers_path)
        print(f"  Found transformers at: {transformers_path}")

if len(transformers_locations) > 1:
    print(f"  ⚠️  WARNING: Multiple transformers installations found!")
    print(f"  This can cause import conflicts")

# Check unsloth
print("\n🔍 Checking unsloth...")
try:
    from unsloth import FastLanguageModel
    print(f"  ✅ Unsloth imported successfully")
except ImportError as e:
    print(f"  ❌ Could not import unsloth: {e}")
    error_msg = str(e)
    
    if 'unsloth_zoo' in error_msg:
        print(f"  ⚠️  unsloth_zoo is missing")
    elif 'top_k_top_p_filtering' in error_msg:
        print(f"  ⚠️  This is a transformers compatibility issue")
    else:
        print(f"  ⚠️  Unknown error: {error_msg}")

# Check cache files
print("\n🔍 Checking for cached bytecode files...")
cache_dirs = []
for path in [user_site] + site_packages:
    transformers_cache = os.path.join(path, "transformers", "__pycache__")
    if os.path.exists(transformers_cache):
        cache_dirs.append(transformers_cache)
        print(f"  Found cache at: {transformers_cache}")

if cache_dirs:
    print(f"  ⚠️  Cache files found - these might be stale")
    print(f"  Recommendation: Clear cache and retry")

print("\n" + "=" * 60)
print("💡 Recommendations:")
print("  1. If transformers version is wrong, run:")
print("     pip3 install --force-reinstall --no-cache-dir transformers==4.45.2")
print("  2. Clear all Python cache:")
print("     find $(python3 -c 'import site; print(site.getusersitepackages())') -name '*.pyc' -delete")
print("     find $(python3 -c 'import site; print(site.getusersitepackages())') -type d -name __pycache__ -exec rm -r {} +")
print("  3. If multiple transformers found, uninstall all and reinstall:")
print("     pip3 uninstall -y transformers")
print("     pip3 install transformers==4.45.2")

