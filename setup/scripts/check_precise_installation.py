#!/usr/bin/env python3
"""
Check what's available in the precise package installation
"""
import sys
import os

print("Checking precise installation...")
print("=" * 50)

# Check if precise is installed
try:
    import precise
    print(f"✅ precise package found at: {precise.__file__}")
    print(f"   Package path: {os.path.dirname(precise.__file__)}")
    print(f"   Available attributes: {[x for x in dir(precise) if not x.startswith('_')]}")
except ImportError as e:
    print(f"❌ precise package not found: {e}")
    sys.exit(1)

# Check for train module
print("\nChecking for training modules...")
try:
    import precise.train
    print(f"✅ precise.train found at: {precise.train.__file__}")
except ImportError:
    print("❌ precise.train not found")
    try:
        from precise import train
        print(f"✅ from precise import train works")
    except ImportError:
        print("❌ from precise import train also failed")

# Check package structure
print("\nChecking package structure...")
import pkgutil
try:
    modules = [name for _, name, _ in pkgutil.iter_modules(precise.__path__)]
    print(f"   Submodules: {modules}")
except Exception as e:
    print(f"   Could not list modules: {e}")

# Check for entry points (console scripts)
print("\nChecking for console scripts...")
try:
    import pkg_resources
    dist = pkg_resources.get_distribution('precise')
    if dist:
        print(f"   Package: {dist.project_name} {dist.version}")
        entry_points = dist.get_entry_map().get('console_scripts', {})
        if entry_points:
            print(f"   Console scripts: {list(entry_points.keys())}")
        else:
            print("   No console scripts found")
except Exception as e:
    print(f"   Could not check entry points: {e}")

# Check if precise-runner is needed
print("\nChecking precise-runner...")
try:
    import precise_runner
    print(f"✅ precise-runner found at: {precise_runner.__file__}")
except ImportError:
    print("❌ precise-runner not found")
    print("   💡 You may need: pip install precise-runner")

print("\n" + "=" * 50)
print("Summary:")
print("  If precise-train is not available, you may need:")
print("  1. pip install precise-runner  (provides precise-train command)")
print("  2. Or use the training API directly from precise package")

