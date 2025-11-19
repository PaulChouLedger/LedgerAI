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
    
    # Check if create_parser already exists
    if 'def create_parser' in content or 'create_parser =' in content:
        print("[Patch] ✅ create_parser already exists, no patch needed")
        return True
    
    # Check if it's a single file module (not a package)
    if prettyparse_path.endswith('.py'):
        print("[Patch] Detected single-file module, adding create_parser...")
        
        # Add create_parser function
        # This is a simple wrapper that creates an ArgumentParser-like object
        patch_code = '''

# === PATCH: Added create_parser for mycroft-precise compatibility ===
def create_parser(description='', **kwargs):
    """
    Create a parser compatible with mycroft-precise expectations.
    This is a compatibility wrapper for the prettyparse module.
    """
    import argparse
    parser = argparse.ArgumentParser(description=description, **kwargs)
    return parser
# === END PATCH ===
'''
        
        # Append the patch
        with open(prettyparse_path, 'a') as f:
            f.write(patch_code)
        
        print("[Patch] ✅ Added create_parser function")
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
    """Test if create_parser can be imported"""
    try:
        # Clear any cached imports
        if 'prettyparse' in sys.modules:
            del sys.modules['prettyparse']
        
        from prettyparse import create_parser
        print("[Test] ✅ create_parser import successful!")
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

