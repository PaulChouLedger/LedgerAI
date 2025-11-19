#!/bin/bash
# fix_prettyparse_simple.sh - Quick fix for prettyparse import error
# Usage: ./fix_prettyparse_simple.sh

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
echo "  Quick Fix: prettyparse Import Error"
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

# Uninstall the broken prettyparse
print_info "Uninstalling broken prettyparse..."
pip uninstall -y prettyparse 2>/dev/null || true

# Install from MycroftAI GitHub (the correct version)
print_info "Installing prettyparse from MycroftAI GitHub source..."
if pip install git+https://github.com/MycroftAI/prettyparse.git 2>&1 | tee /tmp/prettyparse_fix.log; then
    print_success "✅ prettyparse installed from MycroftAI source"
else
    print_error "❌ Installation failed"
    print_info "   Check log: cat /tmp/prettyparse_fix.log"
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
    else
        print_info "⚠️  precise-train may still have issues (check output above)"
    fi
else
    print_error "❌ Import still failing"
    print_info "   Try: pip install --upgrade --force-reinstall git+https://github.com/MycroftAI/prettyparse.git"
fi

echo ""
print_success "✅ Fix complete!"

