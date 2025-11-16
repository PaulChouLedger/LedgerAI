#!/usr/bin/env python3
"""
Patch Porcupine to support Jetson CPU architectures
Fixes: NotImplementedError: Unsupported CPU: '0xd42' (Jetson Orin)

Usage:
    python3 patch_porcupine_jetson.py
"""

import os
import sys

def patch_porcupine_util():
    """Patch Porcupine's _util.py to support Jetson CPUs"""
    
    # Find Porcupine installation
    try:
        import pvporcupine
        porcupine_path = os.path.dirname(pvporcupine.__file__)
        util_file = os.path.join(porcupine_path, '_util.py')
    except ImportError:
        print("❌ Porcupine not found. Install it first:")
        print("   pip install pvporcupine")
        return False
    
    if not os.path.exists(util_file):
        print(f"❌ Porcupine util file not found: {util_file}")
        return False
    
    # Read the file
    with open(util_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if '0xd42' in content or 'Jetson' in content:
        print("✅ Porcupine already patched for Jetson support")
        return True
    
    # Find the _pv_linux_machine function
    if '_pv_linux_machine' not in content:
        print("❌ Could not find _pv_linux_machine function")
        return False
    
    # Jetson CPU part numbers:
    # 0xd42 = ARM Cortex-A78AE (Jetson Orin)
    # 0xd49 = ARM Cortex-A78AE (Jetson Orin NX)
    # 0xd0b = ARM Cortex-A57 (Jetson TX1)
    # 0xd07 = ARM Cortex-A57 (Jetson TX2)
    # 0xd08 = ARM Cortex-A72 (Jetson Nano)
    
    # Find the function and add Jetson support
    lines = content.split('\n')
    patched_lines = []
    in_function = False
    patched = False
    
    for i, line in enumerate(lines):
        patched_lines.append(line)
        
        # Look for the function definition
        if '_pv_linux_machine' in line and 'def' in line:
            in_function = True
            continue
        
        # Look for the raise NotImplementedError line
        if in_function and 'raise NotImplementedError' in line and 'Unsupported CPU' in line:
            # Insert Jetson CPU support before the raise
            indent = len(line) - len(line.lstrip())
            jetson_support = f"""{' ' * indent}# Jetson CPU support (added by patch)
{' ' * indent}jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano
{' ' * indent}if cpu_part in jetson_cpus:
{' ' * indent}    return 'arm64'  # Jetson uses ARM64 architecture"""
            
            patched_lines.append(jetson_support)
            patched = True
            in_function = False
    
    if not patched:
        print("❌ Could not find NotImplementedError line to patch")
        print("   The Porcupine version may have changed")
        return False
    
    # Write patched file
    backup_file = util_file + '.backup'
    try:
        # Create backup
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_file}")
        
        # Write patched version
        with open(util_file, 'w') as f:
            f.write('\n'.join(patched_lines))
        print(f"✅ Patched: {util_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to write patched file: {e}")
        return False

if __name__ == '__main__':
    print("🔧 Patching Porcupine for Jetson support...")
    if patch_porcupine_util():
        print("\n✅ Patch applied successfully!")
        print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
    else:
        print("\n❌ Patch failed")
        sys.exit(1)

