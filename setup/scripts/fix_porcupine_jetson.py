#!/usr/bin/env python3
"""
Fix Porcupine Jetson patch - ensures patch is in correct location
"""

import os
import sys

def fix_porcupine_util(util_file_path):
    """Fix Porcupine _util.py - ensure Jetson patch is before raise statement"""
    
    if not os.path.exists(util_file_path):
        print(f"❌ File not found: {util_file_path}")
        return False
    
    # Read the file
    with open(util_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the _pv_linux_machine function
    function_start = None
    raise_line = None
    
    for i, line in enumerate(lines):
        if '_pv_linux_machine' in line and 'def' in line:
            function_start = i
        if function_start is not None and 'raise NotImplementedError' in line and 'Unsupported CPU' in line:
            raise_line = i
            break
    
    if raise_line is None:
        print("❌ Could not find raise NotImplementedError line")
        return False
    
    # Check if Jetson patch is already before the raise (within 5 lines)
    has_patch = False
    for i in range(max(function_start or 0, raise_line - 5), raise_line):
        if 'jetson_cpus' in lines[i] or '0xd42' in lines[i]:
            has_patch = True
            break
    
    if has_patch:
        print("⚠️  Patch found but may be in wrong location. Checking...")
        # Check if it's actually being executed
        for i in range(raise_line - 10, raise_line):
            if 'if cpu_part in jetson_cpus' in lines[i]:
                print("✅ Patch found in correct location")
                # But it's still failing, so maybe the logic is wrong
                # Let's check the indentation and structure
                print("\n📋 Current structure around raise line:")
                for j in range(max(0, raise_line - 8), min(len(lines), raise_line + 2)):
                    print(f"{j+1:3d}: {lines[j]}", end='')
                return False
    
    # Remove any existing Jetson patch first
    new_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if 'Jetson CPU support' in line or 'jetson_cpus' in line:
            # Skip this line and next few lines that are part of the patch
            skip_next = True
            continue
        if skip_next and ('jetson_cpus' in line or 'if cpu_part in jetson_cpus' in line or 'return \'arm64\'' in line):
            continue
        if skip_next and line.strip() == '':
            skip_next = False
        if not skip_next:
            new_lines.append(line)
        if skip_next and line.strip() and 'return \'arm64\'' not in line:
            skip_next = False
    
    # Now insert the patch in the correct location
    lines = new_lines
    raise_line = None
    for i, line in enumerate(lines):
        if '_pv_linux_machine' in line and 'def' in line:
            function_start = i
        if function_start is not None and 'raise NotImplementedError' in line and 'Unsupported CPU' in line:
            raise_line = i
            break
    
    if raise_line is None:
        print("❌ Could not find raise line after cleanup")
        return False
    
    # Get indentation from raise line
    indent = len(lines[raise_line]) - len(lines[raise_line].lstrip())
    
    # Insert Jetson patch before raise
    jetson_patch = [
        ' ' * indent + '# Jetson CPU support (added by patch)\n',
        ' ' * indent + "jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano\n",
        ' ' * indent + 'if cpu_part in jetson_cpus:\n',
        ' ' * indent + "    return 'arm64'  # Jetson uses ARM64 architecture\n",
        '\n'
    ]
    
    # Insert before raise line
    new_lines = lines[:raise_line] + jetson_patch + lines[raise_line:]
    
    # Create backup
    backup_file = util_file_path + '.backup2'
    with open(backup_file, 'w') as f:
        f.writelines(lines)
    print(f"✅ Created backup: {backup_file}")
    
    # Write fixed version
    with open(util_file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"✅ Fixed patch in: {util_file_path}")
    
    # Show what we inserted
    print("\n📋 Inserted patch (lines around raise):")
    for i in range(max(0, raise_line - 2), min(len(new_lines), raise_line + len(jetson_patch) + 2)):
        print(f"{i+1:3d}: {new_lines[i]}", end='')
    
    return True

if __name__ == '__main__':
    util_file = os.path.expanduser("~/.local/lib/python3.10/site-packages/pvporcupine/_util.py")
    
    if len(sys.argv) > 1:
        util_file = sys.argv[1]
    
    print("🔧 Fixing Porcupine Jetson patch...")
    print(f"   File: {util_file}\n")
    
    if fix_porcupine_util(util_file):
        print("\n✅ Patch fixed successfully!")
        print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
    else:
        print("\n❌ Fix failed")
        sys.exit(1)

