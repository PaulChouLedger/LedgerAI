#!/bin/bash
# install_precise_training.sh - Install precise training tools
# Usage: ./install_precise_training.sh

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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "=========================================="
echo "  Installing Precise Training Tools"
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

# Check current installation
print_info "Checking current installation..."
if command -v precise-train &> /dev/null; then
    print_success "✅ precise-train already available"
    precise-train --help | head -5
    exit 0
fi

if python3 -c "import precise" 2>/dev/null; then
    print_info "✅ precise package is installed"
    print_info "   But precise-train command not found"
    print_info "   Checking if training module is available..."
    if python3 -c "import precise.train" 2>/dev/null; then
        print_success "✅ precise.train module available"
        print_info "   You can use: python3 -m precise.train"
        exit 0
    fi
fi

echo ""
print_info "Installing precise and precise-runner packages..."

# Note: precise-runner provides the precise-train command
# precise provides the training library
# Both may be needed

# Note: precise-runner provides runtime tools, NOT training tools
# Training tools are in mycroft-precise package
print_info "Note: precise-runner provides runtime tools (PreciseEngine)"
print_info "      Training tools are in mycroft-precise package"

# Check if mycroft-precise is installed (provides precise-train command)
if ! command -v precise-train &> /dev/null; then
    print_info "Installing mycroft-precise (provides precise-train command)..."
    if pip install --ignore-installed mycroft-precise 2>&1 | tee /tmp/mycroft_precise_install.log; then
        print_success "✅ mycroft-precise installed"
    else
        print_warning "⚠️  mycroft-precise installation had issues"
        print_info "   Check logs: cat /tmp/mycroft_precise_install.log"
    fi
fi

# Try installing precise with --ignore-installed to avoid conflicts
print_info "Installing precise package (training library)..."
if pip install --ignore-installed precise 2>&1 | tee /tmp/precise_install.log; then
    print_success "✅ precise installed successfully"
    
    # Verify installation - check precise-train from precise-runner first
    if command -v precise-train &> /dev/null; then
        print_success "✅ precise-train command available (from precise-runner)"
        precise-train --help | head -5
    elif python3 -c "import precise_runner.runner" 2>/dev/null && python3 -c "from precise_runner.runner import PreciseTrainer" 2>/dev/null; then
        print_success "✅ PreciseTrainer class available in precise-runner"
        print_info "   Training may be available via Python API"
    elif python3 -c "import precise.train" 2>/dev/null; then
        print_success "✅ precise.train module available"
        print_info "   Use: python3 -m precise.train"
    else
        print_warning "⚠️  precise installed but precise-train command not found"
        print_info "   Checking if precise-runner provides training tools..."
        if python3 -c "import precise_runner" 2>/dev/null; then
            print_info "   ✅ precise-runner is installed"
            print_info "   💡 Try: pip install --upgrade --force-reinstall precise-runner"
            print_info "   Or check: ls ~/aura-env/bin/ | grep precise"
        else
            print_info "   💡 Install precise-runner: pip install precise-runner"
        fi
    fi
else
    print_warning "⚠️  precise installation had issues"
    
    if grep -q "kmeans1d" /tmp/precise_install.log; then
        print_info "   kmeans1d failed - trying to install separately..."
        
        # First ensure numpy/scipy are compatible
        print_info "   Ensuring numpy/scipy compatibility..."
        pip install --ignore-installed --no-cache-dir numpy scipy 2>/dev/null || true
        
        # Try installing kmeans1d separately
        if pip install --ignore-installed --no-cache-dir kmeans1d 2>&1 | tee -a /tmp/precise_install.log; then
            print_success "✅ kmeans1d installed"
            print_info "   Retrying precise installation..."
            if pip install --ignore-installed precise 2>&1 | tee -a /tmp/precise_install.log; then
                print_success "✅ precise installed successfully"
            fi
        else
            print_warning "⚠️  kmeans1d failed - installing precise without it..."
            # Try installing precise without kmeans1d dependency
            if pip install --ignore-installed --no-deps precise 2>&1 | tee -a /tmp/precise_install.log; then
                print_success "✅ precise installed (without kmeans1d)"
                print_warning "   Some features may be limited"
            fi
        fi
    fi
    
    # Final verification
    if command -v precise-train &> /dev/null; then
        print_success "✅ precise-train command available"
    elif python3 -c "import precise.train" 2>/dev/null; then
        print_success "✅ precise.train module available"
        print_info "   Use: python3 -m precise.train"
    else
        print_error "❌ precise-train still not available"
        print_info "   Check logs: cat /tmp/precise_install.log"
        exit 1
    fi
fi

echo ""
print_success "✅ Installation complete!"
echo ""
print_info "Verification:"
if command -v precise-train &> /dev/null; then
    print_success "✅ precise-train command available"
    print_info "   Run: precise-train --help"
elif [ -f "$VIRTUAL_ENV/bin/precise-train" ]; then
    print_info "✅ precise-train found in venv bin (may need PATH update)"
    print_info "   Run: $VIRTUAL_ENV/bin/precise-train --help"
    print_info "   Or: export PATH=\"$VIRTUAL_ENV/bin:\$PATH\""
elif python3 -c "import precise_runner.runner" 2>/dev/null; then
    print_info "✅ precise-runner installed (checking for training tools...)"
    print_info "   Run diagnostic: python3 setup/scripts/check_precise_installation.py"
else
    print_warning "⚠️  precise-train command not found"
    print_info "   Try: pip install --upgrade --force-reinstall precise-runner"
    print_info "   Or check: python3 setup/scripts/check_precise_installation.py"
fi

