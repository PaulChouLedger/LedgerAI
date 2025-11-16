#!/usr/bin/env python3
"""
Fix Porcupine - remove wrong patches and insert in correct location
The patch needs to be in the FINAL else block, before the last raise NotImplementedError
"""

import os
import sys

def fix_porcupine_correct(util_file_path):
    """Fix Porcupine by removing wrong patches and inserting in correct location"""
    
    if not os.path.exists(util_file_path):
        print(f"❌ File not found: {util_file_path}")
        return False
    
    # Read file
    with open(util_file_path, 'r') as f:
        lines = f.readlines()
    
    # Remove ALL existing Jetson patches (they're in wrong location)
    new_lines = []
    skip_patch = False
    for i, line in enumerate(lines):
        # Skip lines that are part of Jetson patch
        if 'Jetson CPU support' in line or ('jetson_cpus' in line and '=' in line):
            skip_patch = True
            continue
        if skip_patch and ('jetson_cpus' in line or 'if cpu_part in jetson_cpus' in line or "return 'arm64'" in line):
            continue
        if skip_patch and line.strip() == '':
            skip_patch = False
            continue
        if skip_patch and not line.strip():
            skip_patch = False
        if not skip_patch:
            new_lines.append(line)
    
    # Now find the CORRECT location - the final raise NotImplementedError for cpu_part
    correct_location = None
    for i, line in enumerate(new_lines):
        # Look for the final raise that checks cpu_part (not machine)
        if 'raise NotImplementedError' in line and "Unsupported CPU: '%s'." in line and 'cpu_part' in line:
            # Check that we're in the right context - should be after cpu_part is defined
            # Look backwards to see if cpu_part was used
            for j in range(max(0, i-20), i):
                if "cpu_part = " in new_lines[j] or "'0xd08' == cpu_part" in new_lines[j]:
                    correct_location = i
                    break
            if correct_location:
                break
    
    if correct_location is None:
        print("❌ Could not find correct location for patch")
        print("   Looking for: raise NotImplementedError with cpu_part")
        return False
    
    # Get indentation from the raise line
    indent = len(new_lines[correct_location]) - len(new_lines[correct_location].lstrip())
    
    # Insert Jetson patch BEFORE the raise line
    jetson_patch = [
        ' ' * indent + '# Jetson CPU support (added by patch)\n',
        ' ' * indent + "jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano\n",
        ' ' * indent + 'if cpu_part in jetson_cpus:\n',
        ' ' * indent + "    return 'arm64' + arch_info  # Jetson uses ARM64 architecture\n",
        '\n'
    ]
    
    # Insert patch
    final_lines = new_lines[:correct_location] + jetson_patch + new_lines[correct_location:]
    
    # Backup
    backup_file = util_file_path + '.backup_final'
    with open(backup_file, 'w') as f:
        f.writelines(new_lines)
    print(f"✅ Created backup: {backup_file}")
    
    # Write fixed version
    with open(util_file_path, 'w') as f:
        f.writelines(final_lines)
    print(f"✅ Fixed: {util_file_path}")
    print(f"   Inserted Jetson check at line {correct_location + 1} (before final raise)")
    
    # Show context
    print("\n📋 Patched code (final else block):")
    start = max(0, correct_location - 5)
    end = min(len(final_lines), correct_location + len(jetson_patch) + 3)
    for i in range(start, end):
        marker = ">>> " if i == correct_location else "    "
        print(f"{marker}{i+1:4d}: {final_lines[i]}", end='')
    
    return True

if __name__ == '__main__':
    util_file = os.path.expanduser("~/.local/lib/python3.10/site-packages/pvporcupine/_util.py")
    
    if len(sys.argv) > 1:
        util_file = sys.argv[1]
    
    print("🔧 Fixing Porcupine - removing wrong patches and inserting in correct location...")
    print(f"   File: {util_file}\n")
    
    if fix_porcupine_correct(util_file):
        print("\n✅ Fix applied successfully!")
        print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
    else:
        print("\n❌ Fix failed")
        sys.exit(1)

