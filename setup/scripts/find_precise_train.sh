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
python3 -c "
import sys
import pkg_resources

try:
    dist = pkg_resources.get_distribution('precise-runner')
    print(f'  ✅ precise-runner installed: {dist.version}')
    entry_points = dist.get_entry_map().get('console_scripts', {})
    if entry_points:
        print(f'  Console scripts: {list(entry_points.keys())}')
        if 'precise-train' in entry_points:
            script_path = entry_points['precise-train'].resolve()
            print(f'  ✅ precise-train entry point: {script_path}')
        else:
            print('  ❌ precise-train not in console_scripts')
    else:
        print('  ⚠️  No console scripts found')
except Exception as e:
    print(f'  ❌ Error: {e}')
"

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

