#!/bin/bash
# fix_precise_train.sh - Fix precise-train command installation
# Usage: ./fix_precise_train.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "=========================================="
echo "  Fixing precise-train Installation"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ~/aura-env ]; then
        print_info "Activating virtual environment..."
        source ~/aura-env/bin/activate
    else
        print_error "Virtual environment not found"
        exit 1
    fi
fi

# Fix platformdirs issue first
print_info "Fixing platformdirs compatibility issue..."
pip install --upgrade --force-reinstall platformdirs 2>/dev/null || true

# Remove corrupted precise file (0 bytes)
if [ -f "$VIRTUAL_ENV/bin/precise" ] && [ ! -s "$VIRTUAL_ENV/bin/precise" ]; then
    print_info "Removing corrupted precise file (0 bytes)..."
    rm -f "$VIRTUAL_ENV/bin/precise"
fi

# Uninstall and reinstall precise-runner to get console scripts
print_info "Reinstalling precise-runner (provides precise-train command)..."
pip uninstall -y precise-runner 2>/dev/null || true

if pip install --upgrade --force-reinstall precise-runner 2>&1 | tee /tmp/precise_runner_reinstall.log; then
    print_success "✅ precise-runner reinstalled"
else
    print_error "❌ precise-runner reinstallation failed"
    exit 1
fi

# Check if precise-train is now available
echo ""
print_info "Verifying installation..."

if command -v precise-train &> /dev/null; then
    print_success "✅ precise-train command found in PATH"
    precise-train --help | head -5
elif [ -f "$VIRTUAL_ENV/bin/precise-train" ] && [ -s "$VIRTUAL_ENV/bin/precise-train" ]; then
    print_success "✅ precise-train found in venv bin"
    "$VIRTUAL_ENV/bin/precise-train" --help | head -5
    print_info "💡 Add to PATH: export PATH=\"$VIRTUAL_ENV/bin:\$PATH\""
else
    print_warning "⚠️  precise-train command still not found"
    print_info "Checking what precise-runner provides..."
    
    # Check entry points
    python3 << 'PYEOF'
try:
    from importlib.metadata import entry_points, version
    scripts = entry_points(group='console_scripts')
    precise_scripts = [ep.name for ep in scripts if 'precise' in ep.name]
    ver = version('precise-runner')
    print(f"  precise-runner version: {ver}")
    if precise_scripts:
        print(f"  Available precise commands: {precise_scripts}")
        if 'precise-train' not in precise_scripts:
            print("  ❌ precise-train not in console_scripts")
            print("  💡 This version of precise-runner may not include training tools")
    else:
        print("  No precise console scripts found")
        print("  💡 precise-runner 0.3.1 may not include precise-train")
except Exception as e:
    print(f"  Error checking: {e}")
PYEOF
    
    print_info ""
    print_info "Note: precise-runner 0.3.1 may not include precise-train command"
    print_info "The training tools might be in the 'precise' package instead"
    print_info ""
    print_info "Checking precise package for training tools..."
    python3 << 'PYEOF' 2>/dev/null || print_info "  Could not check precise package"
try:
    import precise
    import os
    import inspect
    print(f"  ✅ precise package found")
    print(f"  Location: {os.path.dirname(precise.__file__)}")
    
    # List all modules in precise package
    import pkgutil
    modules = [name for _, name, _ in pkgutil.iter_modules(precise.__path__)]
    print(f"  Submodules: {modules}")
    
    # Check for train module
    if 'train' in modules:
        print("  ✅ 'train' module found in precise package")
    else:
        print("  ❌ 'train' module not found")
        
except Exception as e:
    print(f"  Error: {e}")
PYEOF
    
    print_info ""
    print_info "💡 Solution: Install mycroft-precise (contains training tools)"
    print_info "   precise-runner only provides runtime tools, not training"
    print_info ""
    print_info "Installing mycroft-precise (contains precise-train)..."
    if pip install --ignore-installed mycroft-precise 2>&1 | tee /tmp/mycroft_precise_install.log; then
        print_success "✅ mycroft-precise installed"
        
        # Check if precise-train is now available
        if command -v precise-train &> /dev/null; then
            print_success "✅ precise-train command now available!"
            precise-train --help | head -5
        elif [ -f "$VIRTUAL_ENV/bin/precise-train" ] && [ -s "$VIRTUAL_ENV/bin/precise-train" ]; then
            print_success "✅ precise-train found in venv bin"
            "$VIRTUAL_ENV/bin/precise-train" --help | head -5
        else
            print_warning "⚠️  precise-train still not found after mycroft-precise install"
            print_info "   Check logs: cat /tmp/mycroft_precise_install.log"
        fi
    else
        print_error "❌ mycroft-precise installation failed"
        print_info "   Alternative: Training may need to be done via Python API"
        print_info "   Or install from source: pip install git+https://github.com/MycroftAI/mycroft-precise"
    fi
fi

echo ""
print_success "✅ Fix complete!"

