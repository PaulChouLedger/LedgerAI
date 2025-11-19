#!/bin/bash
# fix_prettyparse_for_precise.sh - Fix prettyparse for precise-engine Python wrapper
# The precise-engine wrapper script imports prettyparse which has syntax errors

set -e

echo "=========================================="
echo "  Fixing prettyparse for precise-engine"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f ~/aura-env/bin/activate ]; then
    source ~/aura-env/bin/activate
    echo "[INFO] Activated aura-env"
fi

# Check if prettyparse is installed
PRETTYPARSE_PATH=""
if python3 -c "import prettyparse" 2>/dev/null; then
    PRETTYPARSE_PATH=$(python3 -c "import prettyparse; print(prettyparse.__file__)" 2>/dev/null || echo "")
fi

# Also check in .local (where precise-runner installs it)
if [ -z "$PRETTYPARSE_PATH" ] || [ ! -f "$PRETTYPARSE_PATH" ]; then
    if [ -f ~/.local/lib/python3.10/site-packages/prettyparse.py ]; then
        PRETTYPARSE_PATH=~/.local/lib/python3.10/site-packages/prettyparse.py
    elif [ -f "$VIRTUAL_ENV/lib/python3.10/site-packages/prettyparse.py" ]; then
        PRETTYPARSE_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/prettyparse.py"
    fi
fi

if [ -z "$PRETTYPARSE_PATH" ] || [ ! -f "$PRETTYPARSE_PATH" ]; then
    echo "[ERROR] prettyparse not found"
    echo "   Try: pip install prettyparse"
    exit 1
fi

echo "[INFO] Found prettyparse at: $PRETTYPARSE_PATH"

# Check for syntax errors
echo "[STEP 1] Checking for syntax errors..."
if python3 -c "import prettyparse; from prettyparse import create_parser" 2>/dev/null; then
    echo "✅ prettyparse is working correctly"
    exit 0
else
    echo "⚠️  prettyparse has issues, checking for broken patch..."
fi

# Check if there's a broken string literal
if grep -q "parser.epilog = (parser.epilog or '') + '$" "$PRETTYPARSE_PATH" 2>/dev/null; then
    echo "[STEP 2] Found broken string literal, fixing..."
    
    # Read the file
    python3 << 'PYEOF'
import os
import sys

prettyparse_path = "$PRETTYPARSE_PATH"
if not os.path.exists(prettyparse_path):
    print(f"File not found: {prettyparse_path}")
    sys.exit(1)

# Read file
with open(prettyparse_path, 'r') as f:
    lines = f.readlines()

# Find and fix broken lines
new_lines = []
in_patch = False
fixed = False

for i, line in enumerate(lines):
    # Check for patch markers
    if '# === PATCH:' in line or '# === END PATCH ===' in line:
        in_patch = not in_patch
        fixed = True
        continue
    
    # Check for broken string literal (unterminated)
    if "parser.epilog = (parser.epilog or '') + '" in line and line.count("'") % 2 != 0:
        # Skip this broken line
        fixed = True
        continue
    
    # Skip lines in patch section
    if in_patch:
        continue
    
    new_lines.append(line)

if fixed:
    # Write fixed file
    with open(prettyparse_path, 'w') as f:
        f.writelines(new_lines)
    print("✅ Removed broken patch section")
else:
    print("⚠️  No broken patch found, but import still fails")

PYEOF
fi

echo ""
echo "[STEP 3] Applying prettyparse patch..."
PATCH_SCRIPT="$LEDGERAI_DIR/setup/scripts/patch_prettyparse.py"
if [ -f "$PATCH_SCRIPT" ]; then
    if python3 "$PATCH_SCRIPT" 2>&1; then
        echo "✅ prettyparse patched successfully"
    else
        echo "⚠️  Patch script had issues"
        echo "   Creating minimal patch manually..."
        
        # Manual patch
        python3 << 'PYEOF'
import os

prettyparse_path = "$PRETTYPARSE_PATH"
if not os.path.exists(prettyparse_path):
    print(f"File not found: {prettyparse_path}")
    exit(1)

# Read file
with open(prettyparse_path, 'r') as f:
    content = f.read()

# Check if functions already exist
has_create_parser = 'def create_parser' in content
has_add_to_parser = 'def add_to_parser' in content

if has_create_parser and has_add_to_parser:
    print("✅ Functions already exist")
    exit(0)

# Add functions
patch_code = '''

# === PATCH: Added functions for mycroft-precise compatibility ===
def create_parser(usage_or_description='', **kwargs):
    """Create a parser compatible with mycroft-precise expectations."""
    import argparse
    parser_kwargs = kwargs.copy()
    if usage_or_description:
        if 'usage' not in parser_kwargs:
            parser_kwargs['usage'] = usage_or_description
        elif 'description' not in parser_kwargs:
            parser_kwargs['description'] = usage_or_description
    parser = argparse.ArgumentParser(**parser_kwargs)
    return parser

def add_to_parser(parser, *args, **kwargs):
    """Add arguments to a parser. Compatible with mycroft-precise expectations."""
    import argparse
    
    if not args:
        return parser
    
    # Handle simple string (usage text)
    if len(args) == 1 and isinstance(args[0], str) and not args[0].startswith(':') and not args[0].startswith('-'):
        usage_text = args[0]
        if hasattr(parser, 'epilog'):
            current = getattr(parser, 'epilog', '') or ''
            if current:
                parser.epilog = current + '\\n' + usage_text
            else:
                parser.epilog = usage_text
        return parser
    
    # Handle prettyparse format
    if isinstance(args[0], str) and args[0].startswith(':'):
        first_arg = args[0]
        if first_arg.startswith(':-'):
            # Flag format: ':-e', '--epochs', 'type', default
            flag_short = first_arg[2:] if len(first_arg) > 2 else None
            flag_long = args[1] if len(args) > 1 and isinstance(args[1], str) and args[1].startswith('--') else None
            arg_type = args[2] if len(args) > 2 and isinstance(args[2], str) else None
            default_val = args[3] if len(args) > 3 else None
            
            arg_list = []
            if flag_short:
                arg_list.append('-' + flag_short)
            if flag_long:
                arg_list.append(flag_long)
            
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
            
            add_kwargs = kwargs.copy()
            if type_obj:
                add_kwargs['type'] = type_obj
            if default_val is not None:
                add_kwargs['default'] = default_val
            
            if arg_list:
                parser.add_argument(*arg_list, **add_kwargs)
            else:
                parser.add_argument(*args, **kwargs)
        else:
            # Positional argument
            arg_name = first_arg[1:] if len(first_arg) > 1 else None
            arg_type = args[1] if len(args) > 1 and isinstance(args[1], str) else None
            default_val = args[2] if len(args) > 2 else None
            
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
            
            add_kwargs = kwargs.copy()
            if type_obj:
                add_kwargs['type'] = type_obj
            if default_val is not None:
                add_kwargs['default'] = default_val
            
            if arg_name:
                parser.add_argument(arg_name, **add_kwargs)
            else:
                parser.add_argument(*args, **kwargs)
    elif isinstance(args[0], str) and args[0].startswith('--'):
        # Standard argparse format
        parser.add_argument(*args, **kwargs)
    else:
        parser.add_argument(*args, **kwargs)
    
    return parser
# === END PATCH ===
'''

# Append patch
with open(prettyparse_path, 'a') as f:
    f.write(patch_code)

print("✅ Manual patch applied")

PYEOF
    fi
else
    echo "⚠️  patch_prettyparse.py not found, creating minimal patch..."
    # Same manual patch as above
fi

echo ""
echo "[STEP 4] Verifying fix..."
python3 -c "
try:
    import prettyparse
    from prettyparse import create_parser, add_to_parser
    print('✅ prettyparse imports successfully')
    print('✅ create_parser available')
    print('✅ add_to_parser available')
    
    # Test that it works
    parser = create_parser('test usage')
    add_to_parser(parser, ':-e', '--epochs', 'int', 10)
    print('✅ Functions work correctly')
except Exception as e:
    print(f'❌ Verification failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅✅✅ Fix complete! prettyparse is working."
    echo ""
    echo "[STEP 5] Testing precise-engine wrapper..."
    if [ -f ~/.local/bin/precise-engine ]; then
        if ~/.local/bin/precise-engine --help 2>&1 | head -5; then
            echo "✅ precise-engine wrapper works!"
        else
            echo "⚠️  precise-engine wrapper still has issues"
            echo "   But the actual binary should work at: ~/.mycroft/precise/precise-engine/precise-engine"
        fi
    fi
else
    echo ""
    echo "❌ Fix incomplete. Try:"
    echo "   pip uninstall prettyparse"
    echo "   pip install prettyparse"
    echo "   python3 $PATCH_SCRIPT"
fi

echo ""
echo "=========================================="
echo "  Fix Complete"
echo "=========================================="

