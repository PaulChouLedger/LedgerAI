#!/bin/bash
# fix_prettyparse_final.sh - Final fix for prettyparse import error
# Usage: ./fix_prettyparse_final.sh

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
echo "  Final Fix: prettyparse Import Error"
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

# Check if prettyparse is installed
print_info "Checking prettyparse installation..."
if ! python3 -c "import prettyparse" 2>/dev/null; then
    print_info "Installing prettyparse from PyPI..."
    pip install prettyparse 2>&1 | tee /tmp/prettyparse_install.log || {
        print_error "Failed to install prettyparse"
        exit 1
    }
fi

# Patch prettyparse to add create_parser
print_info "Patching prettyparse to add create_parser..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if python3 "$SCRIPT_DIR/patch_prettyparse.py" 2>&1 | tee /tmp/prettyparse_patch.log; then
    print_success "✅ prettyparse patched"
else
    print_error "❌ Patching failed"
    print_info "   Check log: cat /tmp/prettyparse_patch.log"
    exit 1
fi

# Test the import
print_info "Testing import..."
if python3 -c "from prettyparse import create_parser; print('✅ Import successful!')" 2>&1; then
    print_success "✅ prettyparse fixed!"
    
    # Test precise-train
    print_info "Testing precise-train..."
    if precise-train --help &> /dev/null 2>&1 || "$VIRTUAL_ENV/bin/precise-train" --help &> /dev/null 2>&1; then
        print_success "✅ precise-train working!"
        echo ""
        print_info "You can now run: precise-train --help"
    else
        print_info "⚠️  precise-train may still have issues"
        print_info "   Try running: precise-train --help"
    fi
else
    print_error "❌ Import still failing"
    print_info "   Check the patch log: cat /tmp/prettyparse_patch.log"
    exit 1
fi

echo ""
print_success "✅ Fix complete!"

