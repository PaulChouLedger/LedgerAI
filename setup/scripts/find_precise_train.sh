#!/bin/bash
# find_precise_train.sh - Find where precise-train is located
# Usage: ./find_precise_train.sh

echo "Searching for precise-train..."
echo ""

# Check in venv bin
if [ -d ~/aura-env/bin ]; then
    echo "Checking ~/aura-env/bin:"
    ls -la ~/aura-env/bin/ | grep precise || echo "  Not found"
fi

# Check in system PATH
echo ""
echo "Checking system PATH:"
which precise-train 2>/dev/null || echo "  Not in PATH"

# Check if precise-runner provides it
echo ""
echo "Checking precise-runner installation:"
source ~/aura-env/bin/activate 2>/dev/null || true

# Fix platformdirs issue first
pip install --upgrade --force-reinstall platformdirs 2>/dev/null || true

python3 << 'PYEOF' 2>/dev/null || echo "  ❌ Could not check entry points"
try:
    import sys
    # Try importlib.metadata first (newer Python)
    try:
        from importlib.metadata import entry_points, version
        dist_version = version('precise-runner')
        print(f'  ✅ precise-runner installed: {dist_version}')
        scripts = entry_points(group='console_scripts')
        precise_scripts = [ep for ep in scripts if 'precise' in ep.name]
        if precise_scripts:
            print(f'  Console scripts: {[ep.name for ep in precise_scripts]}')
            train_ep = [ep for ep in precise_scripts if 'train' in ep.name]
            if train_ep:
                print(f'  ✅ precise-train entry point found: {train_ep[0].name}')
            else:
                print('  ❌ precise-train not in console_scripts')
        else:
            print('  ⚠️  No precise console scripts found')
    except ImportError:
        # Fallback to pkg_resources
        import pkg_resources
        dist = pkg_resources.get_distribution('precise-runner')
        print(f'  ✅ precise-runner installed: {dist.version}')
        entry_points = dist.get_entry_map().get('console_scripts', {})
        if entry_points:
            print(f'  Console scripts: {list(entry_points.keys())}')
            if 'precise-train' in entry_points:
                print(f'  ✅ precise-train entry point found')
            else:
                print('  ❌ precise-train not in console_scripts')
        else:
            print('  ⚠️  No console scripts found')
except Exception as e:
    print(f'  ❌ Error: {e}')
PYEOF

# Check if it's a Python module
echo ""
echo "Checking Python modules:"
python3 -c "
try:
    import precise_runner.train
    print('  ✅ precise_runner.train module found')
except ImportError:
    try:
        from precise_runner import train
        print('  ✅ from precise_runner import train works')
    except ImportError:
        print('  ❌ precise_runner.train not found')
"

# Check precise package
echo ""
echo "Checking precise package:"
python3 -c "
try:
    import precise
    print(f'  ✅ precise package found')
    import os
    print(f'  Location: {os.path.dirname(precise.__file__)}')
    try:
        import precise.train
        print('  ✅ precise.train module found')
    except ImportError:
        print('  ❌ precise.train not found')
except ImportError:
    print('  ❌ precise package not found')
"

echo ""
echo "If precise-train is not found, try:"
echo "  pip install --upgrade --force-reinstall precise-runner"

