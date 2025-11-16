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
    
    # Find Porcupine installation - try multiple locations
    possible_paths = [
        os.path.expanduser("~/.local/lib/python3.10/site-packages/pvporcupine"),
        os.path.expanduser("~/porcupine/binding/python/pvporcupine"),
        os.path.expanduser("~/.local/lib/python3.11/site-packages/pvporcupine"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/pvporcupine"),
    ]
    
    # Also try to find via site-packages
    try:
        import site
        for site_packages in site.getsitepackages():
            possible_paths.append(os.path.join(site_packages, 'pvporcupine'))
    except:
        pass
    
    util_file = None
    for path in possible_paths:
        candidate = os.path.join(path, '_util.py')
        if os.path.exists(candidate):
            util_file = candidate
            print(f"✅ Found Porcupine at: {path}")
            break
    
    if not util_file:
        print("❌ Porcupine _util.py not found in common locations.")
        print("   Please specify the path manually or install Porcupine first.")
        print("\n   Try:")
        print("   - pip install pvporcupine")
        print("   - Or manually edit: ~/.local/lib/python3.10/site-packages/pvporcupine/_util.py")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Patch Porcupine for Jetson CPU support')
    parser.add_argument('--path', type=str, help='Manual path to pvporcupine/_util.py file')
    args = parser.parse_args()
    
    print("🔧 Patching Porcupine for Jetson support...")
    
    # If manual path provided, use it
    if args.path:
        util_file = args.path
        if not os.path.exists(util_file):
            print(f"❌ File not found: {util_file}")
            sys.exit(1)
        
        # Read and patch
        with open(util_file, 'r') as f:
            content = f.read()
        
        if '0xd42' in content or 'Jetson' in content:
            print("✅ Already patched!")
            sys.exit(0)
        
        # Apply patch (same logic as in function)
        lines = content.split('\n')
        patched_lines = []
        in_function = False
        patched = False
        
        for line in lines:
            patched_lines.append(line)
            if '_pv_linux_machine' in line and 'def' in line:
                in_function = True
                continue
            if in_function and 'raise NotImplementedError' in line and 'Unsupported CPU' in line:
                indent = len(line) - len(line.lstrip())
                jetson_support = f"""{' ' * indent}# Jetson CPU support (added by patch)
{' ' * indent}jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano
{' ' * indent}if cpu_part in jetson_cpus:
{' ' * indent}    return 'arm64'  # Jetson uses ARM64 architecture"""
                patched_lines.append(jetson_support)
                patched = True
                in_function = False
        
        if patched:
            backup_file = util_file + '.backup'
            with open(backup_file, 'w') as f:
                f.write(content)
            with open(util_file, 'w') as f:
                f.write('\n'.join(patched_lines))
            print(f"✅ Patched: {util_file}")
            print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
        else:
            print("❌ Could not apply patch - file structure may have changed")
            sys.exit(1)
    else:
        # Auto-detect and patch
        if patch_porcupine_util():
            print("\n✅ Patch applied successfully!")
            print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
        else:
            print("\n❌ Patch failed")
            print("\n💡 Try manual patch:")
            print("   python3 patch_porcupine_jetson.py --path ~/.local/lib/python3.10/site-packages/pvporcupine/_util.py")
            sys.exit(1)

