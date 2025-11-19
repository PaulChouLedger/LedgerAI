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
    
    # Find all add_to_parser calls
    import re
    matches = re.finditer(r'add_to_parser\s*\([^)]+\)', content)
    
    print("Found add_to_parser calls:")
    print()
    for i, match in enumerate(matches, 1):
        call = match.group(0)
        # Get some context
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        context = content[start:end]
        
        print(f"{i}. {call}")
        print(f"   Context: ...{context}...")
        print()

