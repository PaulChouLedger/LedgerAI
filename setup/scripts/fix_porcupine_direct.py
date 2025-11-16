#!/usr/bin/env python3
"""
Direct fix for Porcupine Jetson support - reads file and fixes it properly
"""

import os
import sys

def fix_porcupine_direct(util_file_path):
    """Directly fix the _pv_linux_machine function"""
    
    if not os.path.exists(util_file_path):
        print(f"❌ File not found: {util_file_path}")
        return False
    
    # Read entire file
    with open(util_file_path, 'r') as f:
        content = f.read()
    
    # Find the exact pattern we need to replace
    # Look for: raise NotImplementedError("Unsupported CPU: '%s'." % cpu_part)
    old_pattern = r'(\s+)(raise NotImplementedError\("Unsupported CPU: \'%s\'\.\." % cpu_part\))'
    
    # Replacement with Jetson support
    replacement = r'''\1# Jetson CPU support (added by patch)
\1jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano
\1if cpu_part in jetson_cpus:
\1    return 'arm64'  # Jetson uses ARM64 architecture
\1
\1\2'''
    
    if old_pattern not in content and 'raise NotImplementedError' in content:
        # Try without the exact pattern match - find any raise line
        import re
        # Find lines with raise NotImplementedError and Unsupported CPU
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'raise NotImplementedError' in line and 'Unsupported CPU' in line:
                # Get indentation
                indent = len(line) - len(line.lstrip())
                # Insert Jetson check before this line
                jetson_code = [
                    ' ' * indent + '# Jetson CPU support (added by patch)',
                    ' ' * indent + "jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano",
                    ' ' * indent + 'if cpu_part in jetson_cpus:',
                    ' ' * indent + "    return 'arm64'  # Jetson uses ARM64 architecture",
                    ''
                ]
                # Check if already patched
                if i > 0 and 'jetson_cpus' in lines[i-1]:
                    print("⚠️  Patch already exists but may be in wrong location")
                    # Show context
                    print("\n📋 Current code around line", i+1, ":")
                    for j in range(max(0, i-5), min(len(lines), i+3)):
                        print(f"{j+1:4d}: {lines[j]}")
                    return False
                
                # Insert before this line
                new_lines = lines[:i] + jetson_code + lines[i:]
                new_content = '\n'.join(new_lines)
                
                # Backup
                backup_file = util_file_path + '.backup3'
                with open(backup_file, 'w') as f:
                    f.write(content)
                print(f"✅ Created backup: {backup_file}")
                
                # Write fixed version
                with open(util_file_path, 'w') as f:
                    f.write(new_content)
                print(f"✅ Fixed: {util_file_path}")
                print(f"   Inserted Jetson check before line {i+1}")
                
                # Show what we did
                print("\n📋 Patched code (lines around insertion):")
                for j in range(max(0, i-2), min(len(new_lines), i+len(jetson_code)+2)):
                    print(f"{j+1:4d}: {new_lines[j]}")
                
                return True
        
        print("❌ Could not find raise NotImplementedError line")
        return False
    
    # Try regex replacement
    import re
    new_content = re.sub(old_pattern, replacement, content)
    
    if new_content == content:
        print("❌ Regex replacement failed - trying line-by-line approach")
        return False
    
    # Backup
    backup_file = util_file_path + '.backup3'
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"✅ Created backup: {backup_file}")
    
    # Write fixed version
    with open(util_file_path, 'w') as f:
        f.write(new_content)
    print(f"✅ Fixed: {util_file_path}")
    
    return True

if __name__ == '__main__':
    util_file = os.path.expanduser("~/.local/lib/python3.10/site-packages/pvporcupine/_util.py")
    
    if len(sys.argv) > 1:
        util_file = sys.argv[1]
    
    print("🔧 Direct fix for Porcupine Jetson support...")
    print(f"   File: {util_file}\n")
    
    if fix_porcupine_direct(util_file):
        print("\n✅ Fix applied successfully!")
        print("   Test with: python3 -c 'import pvporcupine; print(\"OK\")'")
    else:
        print("\n❌ Fix failed")
        print("\n💡 Please run this to show the file structure:")
        print(f"   sed -n '45,70p' {util_file}")
        sys.exit(1)

