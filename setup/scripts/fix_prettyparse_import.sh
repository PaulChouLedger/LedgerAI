#!/bin/bash
# fix_prettyparse_import.sh - Fix prettyparse create_parser import error
# Usage: ./fix_prettyparse_import.sh

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

# The issue is that prettyparse was installed as a single file, not a package
# We need to uninstall it and let mycroft-precise install the correct version
print_info "Uninstalling conflicting prettyparse..."
pip uninstall -y prettyparse 2>/dev/null || true

# Reinstall mycroft-precise to get correct dependencies
print_info "Reinstalling mycroft-precise (will install correct prettyparse)..."
if pip install --upgrade --force-reinstall --no-deps mycroft-precise 2>&1 | tee /tmp/mycroft_precise_reinstall.log; then
    print_info "Installing mycroft-precise dependencies..."
    pip install $(python3 -c "import pkg_resources; dist = pkg_resources.get_distribution('mycroft-precise'); print(' '.join([str(r) for r in dist.requires()]))" 2>/dev/null || echo "") 2>&1 | tee -a /tmp/mycroft_precise_reinstall.log || true
else
    print_error "❌ mycroft-precise reinstallation failed"
    print_info "   Trying alternative: install prettyparse from MycroftAI source..."
    
    # Alternative: install prettyparse from MycroftAI GitHub
    if pip install git+https://github.com/MycroftAI/prettyparse.git 2>&1 | tee /tmp/prettyparse_github.log; then
        print_success "✅ prettyparse installed from MycroftAI source"
    else
        print_error "❌ Both methods failed"
        print_info "   Check logs:"
        print_info "     cat /tmp/mycroft_precise_reinstall.log"
        print_info "     cat /tmp/prettyparse_github.log"
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
    import prettyparse
    print(f"  prettyparse location: {prettyparse.__file__}")
    print(f"  prettyparse attributes: {dir(prettyparse)}")
    exit(1)
PYEOF
then
    print_success "✅ prettyparse fixed!"
    
    # Test precise-train
    print_info "Testing precise-train command..."
    if command -v precise-train &> /dev/null; then
        if precise-train --help &> /dev/null; then
            print_success "✅ precise-train working correctly!"
        else
            print_error "❌ precise-train command failed (check output above)"
        fi
    elif [ -f "$VIRTUAL_ENV/bin/precise-train" ]; then
        if "$VIRTUAL_ENV/bin/precise-train" --help &> /dev/null; then
            print_success "✅ precise-train working correctly!"
        else
            print_error "❌ precise-train command failed (check output above)"
        fi
    else
        print_warning "⚠️  precise-train command not found"
    fi
else
    print_error "❌ prettyparse import still failing"
    print_info "   Manual fix may be required"
    print_info "   Try: pip install git+https://github.com/MycroftAI/prettyparse.git"
fi

echo ""
print_success "✅ Fix complete!"

