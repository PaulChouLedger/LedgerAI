#!/usr/bin/env python3
"""
Debug script to see what precise-train actually expects
"""

import sys
import os

# Add the path where precise is installed
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))

try:
    from precise.train_data import TrainData
    from prettyparse import create_parser
    
    print("=" * 60)
    print("Inspecting TrainData.parse_args requirements...")
    print("=" * 60)
    print()
    
    # Try to create a parser and see what happens
    usage = "Train a new model on a dataset"
    parser = create_parser(usage)
    
    print("Created parser with create_parser(usage)")
    print(f"Parser type: {type(parser)}")
    print()
    
    # Try to call parse_args with minimal args
    print("Trying to parse minimal arguments: ['hey-aura.net']")
    try:
        args = parser.parse_args(['hey-aura.net'])
        print("✅ Parsed successfully!")
        print(f"Args namespace: {args}")
        print(f"Args attributes: {dir(args)}")
        print()
        print("Attributes in args:")
        for attr in dir(args):
            if not attr.startswith('_'):
                try:
                    value = getattr(args, attr)
                    print(f"  {attr} = {value}")
                except:
                    pass
    except SystemExit as e:
        print(f"❌ parse_args failed (SystemExit): {e}")
    except Exception as e:
        print(f"❌ parse_args failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Trying TrainData.parse_args...")
    print("=" * 60)
    print()
    
    try:
        # This is what precise-train actually does
        args = TrainData.parse_args(parser)
        print("✅ TrainData.parse_args succeeded!")
        print(f"Args: {args}")
    except Exception as e:
        print(f"❌ TrainData.parse_args failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("This is the error we need to fix!")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying to find precise package...")
    import importlib.util
    for path in [
        os.path.expanduser("~/.local/lib/python3.10/site-packages"),
        os.path.expanduser("~/aura-env/lib/python3.10/site-packages"),
    ]:
        precise_path = os.path.join(path, "precise")
        if os.path.exists(precise_path):
            print(f"Found precise at: {precise_path}")
            break

