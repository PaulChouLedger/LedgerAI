#!/bin/bash
# ============================================================================
# Clean Porcupine Installation (Remove Patched Version)
# ============================================================================
# This script removes any patched Porcupine installations to allow
# a fresh install of picovoice package.
# ============================================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo ""
    echo -e "${BOLD}========================================================================"
    echo "   $1"
    echo "========================================================================"
    echo -e "${NC}"
}

print_step "Cleaning Porcupine Installation"

# Detect if we're in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    print_info "Virtual environment detected: $VIRTUAL_ENV"
    USE_VENV=true
    PIP_CMD="pip"
    PYTHON_CMD="python"
else
    print_info "No virtual environment detected - using system Python"
    USE_VENV=false
    PIP_CMD="pip3"
    PYTHON_CMD="python3"
fi

# Step 1: Uninstall all Porcupine/Picovoice packages
print_step "Step 1: Uninstalling Porcupine/Picovoice Packages"

PACKAGES_TO_REMOVE=("pvporcupine" "picovoice" "picovoicedemo" "pvrhino" "pvrecorder")

for package in "${PACKAGES_TO_REMOVE[@]}"; do
    if $PIP_CMD show "$package" &>/dev/null; then
        print_info "Uninstalling $package..."
        $PIP_CMD uninstall -y "$package" 2>/dev/null || true
        print_success "Removed $package"
    else
        print_info "$package not installed (skipping)"
    fi
done

# Step 2: Find and remove any patched Porcupine files
print_step "Step 2: Removing Patched Files"

# Find Python site-packages directories
if [ "$USE_VENV" = true ]; then
    SITE_PACKAGES="$VIRTUAL_ENV/lib/python3."*/site-packages
else
    # Try to find system site-packages
    SITE_PACKAGES=$($PYTHON_CMD -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null || echo "")
    if [ -z "$SITE_PACKAGES" ]; then
        # Fallback locations
        SITE_PACKAGES="/usr/local/lib/python3.*/dist-packages /usr/lib/python3.*/dist-packages ~/.local/lib/python3.*/site-packages"
    fi
fi

# Remove pvporcupine directories and files
print_info "Searching for pvporcupine installation directories..."

FOUND_DIRS=()
for site_dir in $SITE_PACKAGES; do
    if [ -d "$site_dir/pvporcupine" ]; then
        FOUND_DIRS+=("$site_dir/pvporcupine")
    fi
    if [ -d "$site_dir/picovoice" ]; then
        FOUND_DIRS+=("$site_dir/picovoice")
    fi
done

# Also check user's local site-packages
USER_SITE=$($PYTHON_CMD -c "import site; print(site.getusersitepackages())" 2>/dev/null || echo "")
if [ -n "$USER_SITE" ] && [ -d "$USER_SITE/pvporcupine" ]; then
    FOUND_DIRS+=("$USER_SITE/pvporcupine")
fi
if [ -n "$USER_SITE" ] && [ -d "$USER_SITE/picovoice" ]; then
    FOUND_DIRS+=("$USER_SITE/picovoice")
fi

if [ ${#FOUND_DIRS[@]} -gt 0 ]; then
    for dir in "${FOUND_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            print_info "Removing directory: $dir"
            rm -rf "$dir"
            print_success "Removed $dir"
        fi
    done
else
    print_info "No pvporcupine/picovoice directories found"
fi

# Remove egg-info and dist-info
print_info "Searching for package metadata..."
for site_dir in $SITE_PACKAGES; do
    for pattern in "pvporcupine*.egg-info" "picovoice*.egg-info" "pvporcupine*.dist-info" "picovoice*.dist-info"; do
        for item in $site_dir/$pattern; do
            if [ -e "$item" ]; then
                print_info "Removing metadata: $item"
                rm -rf "$item"
            fi
        done
    done
done

# Step 3: Clear Python cache
print_step "Step 3: Clearing Python Cache"

print_info "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -path "*/pvporcupine/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "__pycache__" -path "*/picovoice/*" -exec rm -rf {} + 2>/dev/null || true

# Step 4: Verify cleanup
print_step "Step 4: Verifying Cleanup"

if $PIP_CMD show pvporcupine &>/dev/null || $PIP_CMD show picovoice &>/dev/null; then
    print_warning "Some packages may still be installed"
    print_info "Try running: $PIP_CMD uninstall -y pvporcupine picovoice picovoicedemo"
else
    print_success "All Porcupine/Picovoice packages removed"
fi

# Check if import still works (it shouldn't)
if $PYTHON_CMD -c "import pvporcupine" 2>/dev/null; then
    print_warning "pvporcupine can still be imported - may need manual cleanup"
    print_info "Check: $PYTHON_CMD -c 'import pvporcupine; print(pvporcupine.__file__)'"
else
    print_success "pvporcupine is no longer importable"
fi

print_step "Cleanup Complete"

print_success "Porcupine installation has been cleaned!"
print_info ""
print_info "Next steps:"
print_info "  1. Install picovoice package:"
if [ "$USE_VENV" = true ]; then
    print_info "     pip install picovoice picovoicedemo"
else
    print_info "     pip3 install picovoice picovoicedemo"
    print_warning "     Or install in virtual environment: source ~/LedgerAI/aura-env/bin/activate"
fi
print_info "  2. Test installation:"
print_info "     python3 -c 'import pvporcupine; print(\"Success\")'"
print_info "  3. Add PORCUPINE_ACCESS_KEY to .env file"
print_info "  4. Restart Aura"

