#!/bin/bash
# fix_numpy_scipy_compatibility.sh - Fix numpy/scipy binary incompatibility issues
# Usage: ./fix_numpy_scipy_compatibility.sh

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
echo "  Fixing NumPy/SciPy Compatibility"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ~/aura-env ]; then
        print_info "Activating virtual environment..."
        source ~/aura-env/bin/activate
    else
        print_error "Virtual environment not found. Please activate it manually:"
        print_error "  source ~/aura-env/bin/activate"
        exit 1
    fi
fi

print_info "Current numpy version:"
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null || print_error "numpy not installed"

print_info "Current scipy version:"
python3 -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>/dev/null || print_error "scipy not installed"

echo ""
print_info "Step 1: Uninstalling incompatible packages..."
pip uninstall -y numpy scipy kmeans1d precise 2>/dev/null || true

echo ""
print_info "Step 2: Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo ""
print_info "Step 3: Installing compatible numpy/scipy versions..."
# Install numpy first, then scipy (scipy depends on numpy)
if pip install --no-cache-dir numpy; then
    print_success "✅ numpy installed"
else
    print_error "❌ numpy installation failed"
    exit 1
fi

if pip install --no-cache-dir scipy; then
    print_success "✅ scipy installed"
else
    print_error "❌ scipy installation failed"
    exit 1
fi

echo ""
print_info "Step 4: Verifying compatibility..."
if python3 -c "import numpy; import scipy; from scipy import special; print('✅ Compatibility check passed')" 2>/dev/null; then
    print_success "✅ numpy/scipy are compatible"
else
    print_error "❌ Compatibility check failed"
    print_info "   Try: pip install --force-reinstall numpy scipy"
    exit 1
fi

echo ""
print_info "Step 5: Installing precise (kmeans1d may fail, but precise may still work)..."
if pip install precise 2>&1 | tee /tmp/precise_install_fix.log; then
    print_success "✅ precise installed successfully"
else
    print_warning "⚠️  precise installation had issues"
    if grep -q "kmeans1d" /tmp/precise_install_fix.log; then
        print_info "   kmeans1d failed (this is often due to numpy incompatibility)"
        print_info "   Trying to install precise without kmeans1d dependency..."
        # Try installing precise-runner only (doesn't require kmeans1d)
        if pip install precise-runner; then
            print_success "✅ precise-runner installed (training may be limited without precise)"
        fi
    fi
fi

echo ""
print_success "✅ Fix complete!"
echo ""
print_info "Verification:"
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null || print_error "numpy not accessible"
python3 -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>/dev/null || print_error "scipy not accessible"
python3 -c "from precise_runner import PreciseEngine" 2>/dev/null && print_success "✅ precise-runner accessible" || print_warning "⚠️  precise-runner not accessible"

echo ""
print_info "If issues persist, try:"
print_info "  1. pip install --force-reinstall numpy scipy"
print_info "  2. pip install --no-binary :all: precise  # Build from source"
print_info "  3. Or skip precise and use precise-runner only"

