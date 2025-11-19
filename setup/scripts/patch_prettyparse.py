#!/usr/bin/env python3
"""
Patch prettyparse to add create_parser function if missing
This fixes the ImportError when using mycroft-precise
"""

import os
import sys
import importlib.util

def find_prettyparse_module():
    """Find where prettyparse is installed"""
    try:
        import prettyparse
        return prettyparse.__file__
    except ImportError:
        return None

def patch_prettyparse(prettyparse_path):
    """Add create_parser to prettyparse if missing"""
    print(f"[Patch] Found prettyparse at: {prettyparse_path}")
    
    # Read the current file
    with open(prettyparse_path, 'r') as f:
        content = f.read()
    
    # Check what functions already exist
    has_create_parser = 'def create_parser' in content or 'create_parser =' in content
    has_add_to_parser = 'def add_to_parser' in content or 'add_to_parser =' in content
    
    # Check if there's an old patch that needs updating
    has_old_patch = '# === PATCH:' in content or '# === END PATCH ===' in content
    
    needs_patch = []
    if not has_create_parser:
        needs_patch.append('create_parser')
    if not has_add_to_parser:
        needs_patch.append('add_to_parser')
    
    # If functions exist but there's an old patch, we should update it
    if has_old_patch and (has_create_parser or has_add_to_parser):
        print("[Patch] ⚠️  Old patch detected, will update...")
        # Remove old patch section
        lines = content.split('\n')
        new_lines = []
        in_patch = False
        for line in lines:
            if '# === PATCH:' in line or '# === END PATCH ===' in line:
                in_patch = not in_patch
                continue
            if not in_patch:
                new_lines.append(line)
        content = '\n'.join(new_lines)
        # Write back the cleaned content
        with open(prettyparse_path, 'w') as f:
            f.write(content)
        print("[Patch] ✅ Removed old patch")
        # Re-read content and check what's still needed
        with open(prettyparse_path, 'r') as f:
            content = f.read()
        has_create_parser = 'def create_parser' in content or 'create_parser =' in content
        has_add_to_parser = 'def add_to_parser' in content or 'add_to_parser =' in content
        needs_patch = []
        if not has_create_parser:
            needs_patch.append('create_parser')
        if not has_add_to_parser:
            needs_patch.append('add_to_parser')
    
    if not needs_patch:
        print("[Patch] ✅ All required functions already exist, no patch needed")
        return True
    
    print(f"[Patch] Missing functions: {', '.join(needs_patch)}")
    
    # Check if it's a single file module (not a package)
    if prettyparse_path.endswith('.py'):
        print(f"[Patch] Detected single-file module, adding {', '.join(needs_patch)}...")
        
        # Build patch code with all needed functions
        patch_functions = []
        
        if 'create_parser' in needs_patch:
            patch_functions.append('''
def create_parser(usage_or_description='', **kwargs):
    """
    Create a parser compatible with mycroft-precise expectations.
    This is a compatibility wrapper for the prettyparse module.
    
    Can be called as:
    - create_parser(usage_string)  # First positional arg is usage (most common)
    - create_parser(description=..., usage=...)  # Keyword args
    """
    import argparse
    parser_kwargs = kwargs.copy()
    
    # If first positional arg is provided, treat it as usage
    # (This matches the pattern: create_parser(usage) used by mycroft-precise)
    if usage_or_description:
        # Check if usage is already in kwargs
        if 'usage' not in parser_kwargs:
            parser_kwargs['usage'] = usage_or_description
        elif 'description' not in parser_kwargs:
            # If usage is already set, use first arg as description
            parser_kwargs['description'] = usage_or_description
    
    parser = argparse.ArgumentParser(**parser_kwargs)
    return parser''')
        
        if 'add_to_parser' in needs_patch:
            patch_functions.append('''
def add_to_parser(parser, *args, **kwargs):
    """
    Add arguments to a parser. Compatible with mycroft-precise expectations.
    Usage: add_to_parser(parser, '--flag', help='...') or add_to_parser(parser, '--flag', type=int, default=0)
    """
    # Standard usage: add_to_parser(parser, '--flag', help='...', type=int, default=0)
    # The first arg after parser is the argument name/flag, rest are passed to add_argument
    if args:
        # First arg is the name/flag, rest are additional args for add_argument
        parser.add_argument(*args, **kwargs)
    else:
        # No args provided - this shouldn't happen, but return parser for chaining
        return parser''')
        
        patch_code = '''

# === PATCH: Added functions for mycroft-precise compatibility ===
''' + '\n'.join(patch_functions) + '''
# === END PATCH ===
'''
        
        # Append the patch
        with open(prettyparse_path, 'a') as f:
            f.write(patch_code)
        
        print(f"[Patch] ✅ Added {', '.join(needs_patch)} function(s)")
        return True
    else:
        print("[Patch] ⚠️  prettyparse is a package, not a single file")
        print("[Patch]    Creating __init__.py patch...")
        
        # If it's a package, we need to patch __init__.py
        init_path = os.path.join(prettyparse_path, '__init__.py')
        if os.path.exists(init_path):
            with open(init_path, 'a') as f:
                f.write(patch_code)
            print("[Patch] ✅ Patched __init__.py")
            return True
        else:
            print("[Patch] ❌ Could not find __init__.py")
            return False

def test_import():
    """Test if required functions can be imported"""
    try:
        # Clear any cached imports
        if 'prettyparse' in sys.modules:
            del sys.modules['prettyparse']
        
        from prettyparse import create_parser, add_to_parser
        print("[Test] ✅ create_parser import successful!")
        print("[Test] ✅ add_to_parser import successful!")
        return True
    except ImportError as e:
        print(f"[Test] ❌ Import failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("  Patching prettyparse for mycroft-precise")
    print("=" * 50)
    print()
    
    # Find prettyparse
    prettyparse_path = find_prettyparse_module()
    if not prettyparse_path:
        print("[Error] prettyparse module not found")
        print("        Install it first: pip install prettyparse")
        sys.exit(1)
    
    # Patch it
    if patch_prettyparse(prettyparse_path):
        print()
        print("[Patch] Testing import...")
        if test_import():
            print()
            print("=" * 50)
            print("  ✅ Patch successful!")
            print("=" * 50)
            sys.exit(0)
        else:
            print()
            print("=" * 50)
            print("  ⚠️  Patch applied but import test failed")
            print("=" * 50)
            sys.exit(1)
    else:
        print()
        print("=" * 50)
        print("  ❌ Patch failed")
        print("=" * 50)
        sys.exit(1)

