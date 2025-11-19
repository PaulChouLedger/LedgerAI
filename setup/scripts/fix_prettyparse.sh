#!/bin/bash
# fix_prettyparse.sh - Fix prettyparse import error for precise-train
# Usage: ./fix_prettyparse.sh

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
echo "  Fixing prettyparse Import Error"
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

# Check current prettyparse installation
print_info "Checking current prettyparse installation..."
python3 << 'PYEOF'
try:
    import prettyparse
    print(f"  ✅ prettyparse found: {prettyparse.__file__}")
    print(f"  Checking for create_parser...")
    try:
        from prettyparse import create_parser
        print(f"  ✅ create_parser import successful")
    except ImportError as e:
        print(f"  ❌ create_parser import failed: {e}")
        print(f"  Available attributes: {dir(prettyparse)}")
except ImportError as e:
    print(f"  ❌ prettyparse not found: {e}")
PYEOF

# Uninstall and reinstall prettyparse
print_info "Reinstalling prettyparse..."
pip uninstall -y prettyparse 2>/dev/null || true

# The issue is that prettyparse might be installed as a single file
# We need to install the correct version that has create_parser
# Try installing from MycroftAI's GitHub source (the correct version)
print_info "Installing prettyparse from MycroftAI GitHub source..."
if pip install git+https://github.com/MycroftAI/prettyparse.git 2>&1 | tee /tmp/prettyparse_install.log; then
    print_success "✅ prettyparse installed from GitHub"
else
    print_info "GitHub install failed, trying PyPI..."
    # Fallback to PyPI
    if pip install --upgrade --force-reinstall prettyparse 2>&1 | tee -a /tmp/prettyparse_install.log; then
        print_success "✅ prettyparse reinstalled from PyPI"
    else
        print_error "❌ prettyparse reinstallation failed"
        exit 1
    fi
fi

# Test the import
print_info "Testing prettyparse import..."
if python3 << 'PYEOF'
try:
    from prettyparse import create_parser
    print("  ✅ create_parser import successful!")
    exit(0)
except ImportError as e:
    print(f"  ❌ Import still failing: {e}")
    exit(1)
PYEOF
then
    print_success "✅ prettyparse fixed!"
    
    # Test precise-train
    print_info "Testing precise-train command..."
    if command -v precise-train &> /dev/null; then
        print_success "✅ precise-train command found"
        if precise-train --help &> /dev/null; then
            print_success "✅ precise-train working correctly!"
        else
            print_error "❌ precise-train command failed"
            print_info "   Check error output above"
        fi
    else
        print_info "⚠️  precise-train command not in PATH"
        if [ -f "$VIRTUAL_ENV/bin/precise-train" ]; then
            print_info "   Found in venv: $VIRTUAL_ENV/bin/precise-train"
            if "$VIRTUAL_ENV/bin/precise-train" --help &> /dev/null; then
                print_success "✅ precise-train working correctly!"
            else
                print_error "❌ precise-train command failed"
            fi
        fi
    fi
else
    print_error "❌ prettyparse import still failing"
    print_info "   This may require reinstalling mycroft-precise to get correct dependencies"
    print_info "   Or manually fixing the prettyparse installation"
fi

echo ""
print_success "✅ Fix complete!"

