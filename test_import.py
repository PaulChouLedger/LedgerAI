#!/usr/bin/env python3
"""
Simple test script to verify imports and paths in Ubuntu environment
"""

import sys
import os

print("🔍 Python Environment Debug Information")
print("=" * 50)

print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print(f"\n📁 Current directory contents:")
for item in os.listdir('.'):
    print(f"   - {item}")

print(f"\n📁 llm-container directory contents:")
if os.path.exists('llm-container'):
    for item in os.listdir('llm-container'):
        print(f"   - {item}")
else:
    print("   ❌ llm-container directory not found")

print(f"\n🐍 Python path:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"   {i+1}. {path}")

print(f"\n🔍 Looking for adaptive_diagnostic_engine.py:")
possible_locations = [
    'adaptive_diagnostic_engine.py',
    './adaptive_diagnostic_engine.py', 
    'llm-container/adaptive_diagnostic_engine.py',
    './llm-container/adaptive_diagnostic_engine.py'
]

found = False
for location in possible_locations:
    if os.path.exists(location):
        print(f"   ✅ Found at: {location}")
        found = True
    else:
        print(f"   ❌ Not found at: {location}")

if not found:
    print("   ❌ adaptive_diagnostic_engine.py not found in any expected location")

print(f"\n🧪 Testing import...")
try:
    # Add llm-container to path
    if os.path.exists('llm-container'):
        sys.path.insert(0, os.path.abspath('llm-container'))
        print(f"   ✅ Added llm-container to Python path")
    
    from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
    print(f"   ✅ Successfully imported AdaptiveDiagnosticEngine")
    
    # Try to create an instance
    engine = AdaptiveDiagnosticEngine()
    print(f"   ✅ Successfully created AdaptiveDiagnosticEngine instance")
    
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
except Exception as e:
    print(f"   ❌ Error creating instance: {e}")

print(f"\n" + "=" * 50)
