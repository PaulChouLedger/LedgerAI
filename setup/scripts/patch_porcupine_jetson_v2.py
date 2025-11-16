#!/usr/bin/env python3
"""
Patch Porcupine to support Jetson CPU architectures (Improved version)
Fixes: NotImplementedError: Unsupported CPU: '0xd42' (Jetson Orin)

Usage:
    python3 patch_porcupine_jetson_v2.py
    python3 patch_porcupine_jetson_v2.py --path /path/to/pvporcupine/_util.py
"""

import os
import sys
import re

def patch_porcupine_util(util_file_path):
    """Patch Porcupine's _util.py to support Jetson CPUs"""
    
    if not os.path.exists(util_file_path):
        print(f"❌ File not found: {util_file_path}")
        return False
    
    # Read the file
    with open(util_file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if '0xd42' in content and 'jetson_cpus' in content:
        print("✅ Porcupine already patched for Jetson support")
        return True
    
    # Find the _pv_linux_machine function using regex
    # Look for the pattern: raise NotImplementedError("Unsupported CPU: '%s'." % cpu_part)
    pattern = r'(raise NotImplementedError\("Unsupported CPU: \'%s\'\.\." % cpu_part\))'
    
    if not re.search(pattern, content):
        print("❌ Could not find the NotImplementedError line to patch")
        print("   The Porcupine version may have changed")
        return False
    
    # Create the Jetson support code
    jetson_patch = """    # Jetson CPU support (added by patch)
    jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano
    if cpu_part in jetson_cpus:
        return 'arm64'  # Jetson uses ARM64 architecture
    
    """
    
    # Replace the raise statement with Jetson check + raise
    replacement = jetson_patch + r'\1'
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print("❌ Patch replacement failed - pattern match issue")
        return False
    
    # Create backup
    backup_file = util_file_path + '.backup'
    try:
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_file}")
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")
    
    # Write patched version
    try:
        with open(util_file_path, 'w') as f:
            f.write(new_content)
        print(f"✅ Patched: {util_file_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write patched file: {e}")
        return False

def find_porcupine_util():
    """Find Porcupine _util.py file"""
    possible_paths = [
        os.path.expanduser("~/.local/lib/python3.10/site-packages/pvporcupine/_util.py"),
        os.path.expanduser("~/.local/lib/python3.11/site-packages/pvporcupine/_util.py"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/pvporcupine/_util.py"),
        os.path.expanduser("~/porcupine/binding/python/pvporcupine/_util.py"),
    ]
    
    # Also check site-packages
    try:
        import site
        for site_packages in site.getsitepackages():
            possible_paths.append(os.path.join(site_packages, 'pvporcupine', '_util.py'))
    except:
        pass
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Patch Porcupine for Jetson CPU support')
    parser.add_argument('--path', type=str, help='Manual path to pvporcupine/_util.py file')
    args = parser.parse_args()
    
    print("🔧 Patching Porcupine for Jetson support...")
    
    if args.path:
        util_file = args.path
    else:
        util_file = find_porcupine_util()
        if not util_file:
            print("❌ Porcupine _util.py not found in common locations.")
            print("\n💡 Try specifying the path manually:")
            print("   python3 patch_porcupine_jetson_v2.py --path ~/.local/lib/python3.10/site-packages/pvporcupine/_util.py")
            sys.exit(1)
        print(f"✅ Found Porcupine at: {os.path.dirname(util_file)}")
    
    if patch_porcupine_util(util_file):
        print("\n✅ Patch applied successfully!")
        print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
    else:
        print("\n❌ Patch failed")
        sys.exit(1)

