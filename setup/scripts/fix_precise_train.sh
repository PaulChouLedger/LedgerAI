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
    from importlib.metadata import entry_points
    scripts = entry_points(group='console_scripts')
    precise_scripts = [ep for ep in scripts if 'precise' in ep.name]
    if precise_scripts:
        print(f"  Available precise commands: {[ep.name for ep in precise_scripts]}")
    else:
        print("  No precise console scripts found")
        print("  💡 precise-runner may not include precise-train in this version")
except Exception as e:
    print(f"  Error checking: {e}")
PYEOF
    
    print_info ""
    print_info "Alternative: Use Python API directly"
    print_info "  The training tools may be available via Python API"
    print_info "  Run: python3 setup/scripts/check_precise_installation.py"
fi

echo ""
print_success "✅ Fix complete!"

