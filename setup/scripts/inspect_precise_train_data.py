#!/usr/bin/env python3
"""
Inspect how precise/train_data.py uses add_to_parser
This helps us understand what arguments need to be added
"""

import sys
import os

# Try to find and read the train_data.py file
possible_paths = [
    os.path.expanduser("~/.local/lib/python3.10/site-packages/precise/train_data.py"),
    os.path.expanduser("~/aura-env/lib/python3.10/site-packages/precise/train_data.py"),
    "/usr/local/lib/python3.10/site-packages/precise/train_data.py",
]

train_data_path = None
for path in possible_paths:
    if os.path.exists(path):
        train_data_path = path
        break

if not train_data_path:
    print("Could not find precise/train_data.py")
    print("Trying to import and inspect...")
    try:
        import precise.train_data
        train_data_path = precise.train_data.__file__
        print(f"Found at: {train_data_path}")
    except Exception as e:
        print(f"Error importing: {e}")
        sys.exit(1)

print(f"Reading: {train_data_path}")
print("=" * 60)

with open(train_data_path, 'r') as f:
    content = f.read()
    
    # Find all add_to_parser calls with more context
    import re
    
    # Find lines with add_to_parser
    lines = content.split('\n')
    print("Searching for add_to_parser usage...")
    print()
    
    for i, line in enumerate(lines, 1):
        if 'add_to_parser' in line:
            # Show the line and a few lines before/after for context
            start = max(0, i - 3)
            end = min(len(lines), i + 5)
            print(f"Line {i}:")
            for j in range(start, end):
                marker = ">>> " if j == i - 1 else "    "
                print(f"{marker}{j+1:4d}: {lines[j]}")
            print()
    
    # Also try to find the parse_args method to see what it expects
    print("=" * 60)
    print("Looking for parse_args method and expected arguments...")
    print()
    
    # Find the parse_args method
    parse_args_match = re.search(r'def parse_args\([^)]*\):.*?(?=def |\Z)', content, re.DOTALL)
    if parse_args_match:
        parse_args_code = parse_args_match.group(0)
        print("parse_args method (full):")
        print(parse_args_code)
        print()
        
        # Look for attribute accesses like args.tags_folder
        attr_matches = re.finditer(r'args\.(\w+)', parse_args_code)
        print("Attributes accessed in parse_args:")
        attrs = set()
        for match in attr_matches:
            attrs.add(match.group(1))
        for attr in sorted(attrs):
            print(f"  - {attr}")
        print()
    
    # Find the class definition to see how arguments are added
    print("=" * 60)
    print("Looking for TrainData class and argument definitions...")
    print()
    
    # Find @add_to_parser decorators or add_to_parser calls in class methods
    class_match = re.search(r'class TrainData.*?(?=class |\Z)', content, re.DOTALL)
    if class_match:
        class_code = class_match.group(0)
        print("TrainData class found, searching for argument definitions...")
        
        # Look for @add_to_parser decorators
        decorator_matches = re.finditer(r'@add_to_parser.*?\n\s*def\s+(\w+)', class_code, re.MULTILINE)
        print("\nMethods with @add_to_parser decorator:")
        for match in decorator_matches:
            method_name = match.group(1)
            print(f"  - {method_name}")
        
        # Look for add_to_parser calls in the class
        add_calls = re.finditer(r'add_to_parser\s*\([^)]+\)', class_code)
        print("\nadd_to_parser calls in TrainData class:")
        for i, match in enumerate(add_calls, 1):
            call = match.group(0)
            # Get more context
            start = max(0, match.start() - 100)
            end = min(len(class_code), match.end() + 100)
            context = class_code[start:end]
            print(f"\n{i}. {call}")
            print(f"   Context:\n{context}")

