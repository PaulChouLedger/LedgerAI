#!/usr/bin/env python3
"""
Fix Porcupine platform detection for Jetson
Fixes: NotImplementedError: Unsupported platform
"""

import os
import sys

def fix_porcupine_platform(util_file_path):
    """Fix Porcupine platform detection to support Jetson"""
    
    if not os.path.exists(util_file_path):
        print(f"❌ File not found: {util_file_path}")
        return False
    
    # Read file
    with open(util_file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'Jetson' in content and 'pv_keyword_files_subdir' in content and 'arm64' in content:
        # Check if the fix is actually in the right place
        if 'elif _PV_MACHINE.startswith' in content or 'if _PV_MACHINE.startswith' in content:
            print("✅ Platform detection may already be patched")
            # But let's check if it's correct
    
    # Find pv_keyword_files_subdir function
    lines = content.split('\n')
    subdir_start = None
    subdir_raise = None
    
    for i, line in enumerate(lines):
        if 'def pv_keyword_files_subdir' in line:
            subdir_start = i
        if subdir_start is not None and 'raise NotImplementedError' in line and 'Unsupported platform' in line:
            subdir_raise = i
            break
    
    if subdir_raise is None:
        print("❌ Could not find pv_keyword_files_subdir raise statement")
        return False
    
    # Check what _PV_MACHINE values are checked
    # We need to add support for 'arm64' or 'cortex-a78ae' or similar
    # Look for the pattern that checks _PV_MACHINE
    
    # Find where _PV_MACHINE is checked
    machine_checks = []
    for i in range(subdir_start, subdir_raise):
        if '_PV_MACHINE' in lines[i] and ('==' in lines[i] or 'startswith' in lines[i] or 'in' in lines[i]):
            machine_checks.append((i, lines[i]))
    
    # The function should return a subdirectory name based on _PV_MACHINE
    # We need to add a case for Jetson (which returns 'arm64' or similar)
    
    # Get indentation
    indent = len(lines[subdir_raise]) - len(lines[subdir_raise].lstrip())
    
    # Insert Jetson platform support before the raise
    # Jetson returns 'arm64' or 'cortex-a78ae' + arch_info, so we need to handle that
    platform_patch = [
        ' ' * indent + '# Jetson platform support (added by patch)\n',
        ' ' * indent + "if _PV_MACHINE.startswith('arm64') or _PV_MACHINE.startswith('cortex-a78ae') or '0xd42' in str(_PV_MACHINE):\n",
        ' ' * indent + "    return 'linux'  # Jetson uses Linux ARM64 platform\n",
        '\n'
    ]
    
    # Insert before raise
    new_lines = lines[:subdir_raise] + platform_patch + lines[subdir_raise:]
    
    # Backup
    backup_file = util_file_path + '.backup_platform'
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"✅ Created backup: {backup_file}")
    
    # Write fixed version
    with open(util_file_path, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f"✅ Fixed platform detection: {util_file_path}")
    
    # Show context
    print("\n📋 Patched code (pv_keyword_files_subdir function):")
    start = max(0, subdir_raise - 5)
    end = min(len(new_lines), subdir_raise + len(platform_patch) + 3)
    for i in range(start, end):
        marker = ">>> " if i == subdir_raise else "    "
        print(f"{marker}{i+1:4d}: {new_lines[i]}")
    
    return True

if __name__ == '__main__':
    util_file = os.path.expanduser("~/.local/lib/python3.10/site-packages/pvporcupine/_util.py")
    
    if len(sys.argv) > 1:
        util_file = sys.argv[1]
    
    print("🔧 Fixing Porcupine platform detection for Jetson...")
    print(f"   File: {util_file}\n")
    
    if fix_porcupine_platform(util_file):
        print("\n✅ Platform fix applied!")
        print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
    else:
        print("\n❌ Fix failed")
        print("\n💡 Please show the pv_keyword_files_subdir function:")
        print(f"   sed -n '120,135p' {util_file}")
        sys.exit(1)

