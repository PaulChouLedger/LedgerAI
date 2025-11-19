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
    
    prettyparse uses a special syntax like: add_to_parser(parser, ':-e', '--epochs', 'int', 10)
    Where the format is: ':-e' means both -e and --epochs, 'int' is type, 10 is default
    
    Also supports: add_to_parser(parser, ':folder', 'str', '.') for positional arguments
    And: add_to_parser(parser, '--tags-folder', 'str', '{folder}/tags') for keyword arguments
    
    Usage patterns:
    - add_to_parser(parser, ':-e', '--epochs', 'int', 10)  # prettyparse format with flags
    - add_to_parser(parser, ':folder', 'str', '.')  # positional argument
    - add_to_parser(parser, '--tags-folder', 'str', '{folder}/tags')  # keyword with default
    - add_to_parser(parser, '--flag', help='...')  # standard argparse format
    """
    import argparse
    
    if not args:
        # No args provided, just return parser
        return parser
    
    # Check if first arg is just a string (usage text or help text)
    # prettyparse allows: add_to_parser(parser, "extra usage text")
    if len(args) == 1 and isinstance(args[0], str) and not args[0].startswith(':') and not args[0].startswith('-'):
        # This is usage/help text, add it to the parser's description or epilog
        usage_text = args[0]
        if hasattr(parser, 'epilog'):
            parser.epilog = (parser.epilog or '') + '\n' + usage_text
        else:
            # Add as description if no description exists
            if not parser.description:
                parser.description = usage_text
        return parser
    
    # Check if first arg uses prettyparse format (starts with ':')
    if isinstance(args[0], str) and args[0].startswith(':'):
        # prettyparse format
        first_arg = args[0]
        
        # Check if it's a flag format (':-e') or positional (':folder')
        if first_arg.startswith(':-'):
            # Flag format: ':-e', '--epochs', 'type', default, ...
            flag_short = first_arg[2:] if len(first_arg) > 2 else None  # Remove ':-' prefix
            flag_long = args[1] if len(args) > 1 and isinstance(args[1], str) and args[1].startswith('--') else None
            arg_type = args[2] if len(args) > 2 and isinstance(args[2], str) else None
            default_val = args[3] if len(args) > 3 else None
            help_text = None
            
            # Find help text (usually a string after default)
            for i in range(4, len(args)):
                if isinstance(args[i], str) and not args[i].startswith('-'):
                    help_text = args[i]
                    break
            
            # Build argument list for argparse
            arg_list = []
            if flag_short:
                arg_list.append('-' + flag_short)
            if flag_long:
                arg_list.append(flag_long)
            
            # Convert type string to actual type
            type_obj = None
            if arg_type:
                if arg_type == 'int':
                    type_obj = int
                elif arg_type == 'float':
                    type_obj = float
                elif arg_type == 'str':
                    type_obj = str
                elif arg_type == 'bool':
                    type_obj = bool
            
            # Build kwargs for add_argument
            add_kwargs = kwargs.copy()
            if type_obj:
                add_kwargs['type'] = type_obj
            if default_val is not None:
                add_kwargs['default'] = default_val
            if help_text:
                add_kwargs['help'] = help_text
            
            # Add the argument
            if arg_list:
                parser.add_argument(*arg_list, **add_kwargs)
            else:
                # Fallback
                parser.add_argument(*args, **kwargs)
        else:
            # Positional argument format: ':folder', 'str', '.'
            arg_name = first_arg[1:] if len(first_arg) > 1 else None  # Remove ':' prefix
            arg_type = args[1] if len(args) > 1 and isinstance(args[1], str) else None
            default_val = args[2] if len(args) > 2 else None
            
            # Convert type
            type_obj = None
            if arg_type:
                if arg_type == 'int':
                    type_obj = int
                elif arg_type == 'float':
                    type_obj = float
                elif arg_type == 'str':
                    type_obj = str
                elif arg_type == 'bool':
                    type_obj = bool
            
            # Build kwargs
            add_kwargs = kwargs.copy()
            if type_obj:
                add_kwargs['type'] = type_obj
            if default_val is not None:
                add_kwargs['default'] = default_val
                add_kwargs['nargs'] = '?'  # Make it optional if default provided
            
            # Add positional argument
            if arg_name:
                parser.add_argument(arg_name, **add_kwargs)
            else:
                parser.add_argument(*args, **kwargs)
    elif isinstance(args[0], str) and args[0].startswith('--'):
        # Standard argparse format: '--flag', type=..., default=...
        # Or prettyparse format without colon: '--tags-folder', 'str', '{folder}/tags'
        flag_name = args[0]
        arg_type = args[1] if len(args) > 1 and isinstance(args[1], str) and not args[1].startswith('-') else None
        default_val = args[2] if len(args) > 2 else None
        
        # Convert type
        type_obj = None
        if arg_type and arg_type in ['int', 'float', 'str', 'bool']:
            if arg_type == 'int':
                type_obj = int
            elif arg_type == 'float':
                type_obj = float
            elif arg_type == 'str':
                type_obj = str
            elif arg_type == 'bool':
                type_obj = bool
        
        # Build kwargs
        add_kwargs = kwargs.copy()
        if type_obj:
            add_kwargs['type'] = type_obj
        if default_val is not None:
            add_kwargs['default'] = default_val
        
        # Add the argument
        parser.add_argument(flag_name, *args[3:], **add_kwargs)
    else:
        # Standard argparse format: add_to_parser(parser, '--flag', help='...')
        parser.add_argument(*args, **kwargs)
    
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

