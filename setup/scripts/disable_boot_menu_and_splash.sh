#!/bin/bash
# Disable ExtLinux Boot Menu and Configure for Silent Boot
# This addresses both the boot menu timeout and kernel parameters

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

EXTLINUX_CONFIG="/boot/extlinux/extlinux.conf"

echo "=========================================="
echo "  Disable Boot Menu & Configure Silent Boot"
echo "=========================================="
echo ""

# Check if ExtLinux config exists
if [ ! -f "$EXTLINUX_CONFIG" ]; then
    print_error "ExtLinux config not found at $EXTLINUX_CONFIG"
    print_info "Searching for alternative locations..."
    
    POSSIBLE_LOCATIONS=(
        "/boot/extlinux/extlinux.conf"
        "/boot/extlinux.conf"
        "/mnt/extlinux/extlinux.conf"
    )
    
    for loc in "${POSSIBLE_LOCATIONS[@]}"; do
        if [ -f "$loc" ]; then
            EXTLINUX_CONFIG="$loc"
            print_info "Found at: $EXTLINUX_CONFIG"
            break
        fi
    done
    
    if [ ! -f "$EXTLINUX_CONFIG" ]; then
        print_error "Could not find ExtLinux configuration file"
        exit 1
    fi
fi

# Backup original
if [ ! -f "${EXTLINUX_CONFIG}.bak" ]; then
    print_info "Creating backup: ${EXTLINUX_CONFIG}.bak"
    sudo cp "$EXTLINUX_CONFIG" "${EXTLINUX_CONFIG}.bak"
    print_success "Backup created"
fi

echo ""
print_info "Current configuration:"
echo "----------------------------------------"
cat "$EXTLINUX_CONFIG"
echo "----------------------------------------"
echo ""

# Step 1: Disable boot menu timeout
print_info "Step 1: Disabling ExtLinux boot menu timeout..."

# Check if TIMEOUT exists
if grep -q "^[[:space:]]*TIMEOUT" "$EXTLINUX_CONFIG"; then
    # Update existing TIMEOUT
    sudo sed -i 's/^[[:space:]]*TIMEOUT.*/TIMEOUT 0/' "$EXTLINUX_CONFIG"
    print_success "Updated TIMEOUT to 0 (boot immediately)"
else
    # Add TIMEOUT at the beginning of the file
    sudo sed -i '1i TIMEOUT 0' "$EXTLINUX_CONFIG"
    print_success "Added TIMEOUT 0 (boot immediately)"
fi

# Step 2: Configure silent boot kernel parameters
print_info "Step 2: Configuring silent boot kernel parameters..."

# Check if APPEND line exists
if grep -q "^[[:space:]]*APPEND" "$EXTLINUX_CONFIG"; then
    # Remove existing quiet/splash/loglevel parameters
    sudo sed -i 's/quiet[^ ]*//g' "$EXTLINUX_CONFIG"
    sudo sed -i 's/splash[^ ]*//g' "$EXTLINUX_CONFIG"
    sudo sed -i 's/loglevel=[^ ]*//g' "$EXTLINUX_CONFIG"
    sudo sed -i 's/mminit_loglevel=[^ ]*/mminit_loglevel=0/g' "$EXTLINUX_CONFIG"
    
    # Add quiet and loglevel=0 after ${cbootargs}
    if grep -q "APPEND.*\${cbootargs}" "$EXTLINUX_CONFIG"; then
        # Add after ${cbootargs}
        sudo sed -i 's/APPEND[[:space:]]*${cbootargs}/APPEND ${cbootargs} quiet loglevel=0/' "$EXTLINUX_CONFIG"
    else
        # Add at the beginning of APPEND line
        sudo sed -i 's/^\([[:space:]]*APPEND\)/\1 quiet loglevel=0/' "$EXTLINUX_CONFIG"
    fi
    
    # Ensure mminit_loglevel=0 is set
    if ! grep -q "mminit_loglevel=0" "$EXTLINUX_CONFIG"; then
        sudo sed -i 's/^\([[:space:]]*APPEND.*\)/\1 mminit_loglevel=0/' "$EXTLINUX_CONFIG"
    fi
    
    print_success "Updated APPEND line with silent boot parameters"
else
    print_warning "APPEND line not found - you may need to add kernel parameters manually"
fi

echo ""
print_info "Modified configuration:"
echo "----------------------------------------"
cat "$EXTLINUX_CONFIG"
echo "----------------------------------------"
echo ""

print_success "Configuration complete!"
echo ""
print_info "Changes made:"
echo "  ✅ TIMEOUT set to 0 (boot menu disabled)"
echo "  ✅ Added 'quiet' parameter (suppress kernel messages)"
echo "  ✅ Added 'loglevel=0' (only emergency messages)"
echo "  ✅ Set 'mminit_loglevel=0' (minimal init messages)"
echo ""
print_warning "⚠️  UEFI Splash Screen Note:"
echo "   The NVIDIA logo splash screen is part of UEFI firmware"
echo "   and appears BEFORE the kernel loads. To disable it, you"
echo "   need to modify the UEFI firmware source code (advanced)."
echo "   See: docs/JETSON_UEFI_BOOT_CONFIGURATION.md"
echo ""
print_info "Backup saved at: ${EXTLINUX_CONFIG}.bak"
echo ""
print_info "To apply changes, reboot:"
echo "  sudo reboot"
echo ""
