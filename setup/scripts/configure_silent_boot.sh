#!/bin/bash
# Silent Boot Configuration Script for Jetson Orin NX
# Detects bootloader and configures silent boot accordingly

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLON='\033[1;33m'
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

echo "=========================================="
echo "  Silent Boot Configuration for Jetson"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ] && [ ! -f /proc/device-tree/model ]; then
    print_warning "This script is designed for Jetson devices"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect bootloader
print_info "Detecting bootloader..."

BOOTLOADER="unknown"
GRUB_CONFIG="/etc/default/grub"
EXT_LINUX_CONFIG="/boot/extlinux/extlinux.conf"
UBOOT_CONFIG="/boot/extlinux/extlinux.conf"

# Check for GRUB
if [ -f "$GRUB_CONFIG" ] || command -v update-grub &> /dev/null; then
    BOOTLOADER="grub"
    print_success "Detected: GRUB bootloader"
elif [ -f "$EXT_LINUX_CONFIG" ]; then
    BOOTLOADER="extlinux"
    print_success "Detected: ExtLinux bootloader (common on Jetson)"
elif [ -d "/boot/extlinux" ]; then
    BOOTLOADER="extlinux"
    print_success "Detected: ExtLinux bootloader (common on Jetson)"
else
    print_warning "Could not detect bootloader automatically"
    print_info "Checking common locations..."
    
    # Check for extlinux in different locations
    if [ -f "/boot/extlinux/extlinux.conf" ]; then
        BOOTLOADER="extlinux"
        EXT_LINUX_CONFIG="/boot/extlinux/extlinux.conf"
    elif [ -f "/boot/extlinux.conf" ]; then
        BOOTLOADER="extlinux"
        EXT_LINUX_CONFIG="/boot/extlinux.conf"
    elif [ -f "/mnt/extlinux/extlinux.conf" ]; then
        BOOTLOADER="extlinux"
        EXT_LINUX_CONFIG="/mnt/extlinux/extlinux.conf"
    fi
fi

echo ""
print_info "Bootloader: $BOOTLOADER"
echo ""

# Show current kernel parameters
print_info "Current kernel command line:"
if [ -f /proc/cmdline ]; then
    cat /proc/cmdline | sed 's/ /\n  /g'
    echo ""
else
    print_warning "Could not read /proc/cmdline"
fi

echo ""
echo "=========================================="
echo "  Configuration Options"
echo "=========================================="
echo ""
echo "1. Quiet boot (suppress most messages)"
echo "2. Silent boot (suppress all messages)"
echo "3. Disable splash screen only"
echo "4. Custom configuration"
echo "5. Show current configuration"
echo "6. Exit"
echo ""
read -p "Select option (1-6): " OPTION

case $OPTION in
    1)
        KERNEL_PARAMS="quiet splash"
        ;;
    2)
        KERNEL_PARAMS="quiet loglevel=0"
        ;;
    3)
        KERNEL_PARAMS="quiet"
        ;;
    4)
        read -p "Enter kernel parameters: " KERNEL_PARAMS
        ;;
    5)
        echo ""
        print_info "Current configuration:"
        if [ "$BOOTLOADER" = "grub" ] && [ -f "$GRUB_CONFIG" ]; then
            echo "GRUB Configuration:"
            cat "$GRUB_CONFIG" | grep -E "^GRUB_CMDLINE" || echo "  (no GRUB_CMDLINE found)"
        elif [ "$BOOTLOADER" = "extlinux" ] && [ -f "$EXT_LINUX_CONFIG" ]; then
            echo "ExtLinux Configuration:"
            grep -A 5 "APPEND" "$EXT_LINUX_CONFIG" || echo "  (no APPEND found)"
        fi
        echo ""
        echo "Current kernel parameters:"
        cat /proc/cmdline
        echo ""
        exit 0
        ;;
    6)
        exit 0
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

echo ""
print_info "Selected kernel parameters: $KERNEL_PARAMS"
echo ""

# Configure based on bootloader
if [ "$BOOTLOADER" = "grub" ]; then
    print_info "Configuring GRUB..."
    
    # Check if file exists
    if [ ! -f "$GRUB_CONFIG" ]; then
        print_warning "GRUB config file doesn't exist, creating it..."
        sudo mkdir -p /etc/default
        sudo touch "$GRUB_CONFIG"
    fi
    
    # Backup original
    if [ ! -f "${GRUB_CONFIG}.bak" ]; then
        print_info "Creating backup: ${GRUB_CONFIG}.bak"
        sudo cp "$GRUB_CONFIG" "${GRUB_CONFIG}.bak"
    fi
    
    # Read current config
    CURRENT_CONFIG=$(cat "$GRUB_CONFIG" 2>/dev/null || echo "")
    
    # Check if GRUB_CMDLINE_LINUX_DEFAULT exists
    if echo "$CURRENT_CONFIG" | grep -q "^GRUB_CMDLINE_LINUX_DEFAULT="; then
        # Update existing line
        print_info "Updating existing GRUB_CMDLINE_LINUX_DEFAULT"
        sudo sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"$KERNEL_PARAMS\"|" "$GRUB_CONFIG"
    else
        # Add new line
        print_info "Adding GRUB_CMDLINE_LINUX_DEFAULT"
        echo "GRUB_CMDLINE_LINUX_DEFAULT=\"$KERNEL_PARAMS\"" | sudo tee -a "$GRUB_CONFIG" > /dev/null
    fi
    
    # Also set GRUB_CMDLINE_LINUX if it doesn't exist
    if ! echo "$CURRENT_CONFIG" | grep -q "^GRUB_CMDLINE_LINUX="; then
        echo "GRUB_CMDLINE_LINUX=\"$KERNEL_PARAMS\"" | sudo tee -a "$GRUB_CONFIG" > /dev/null
    fi
    
    print_info "Updating GRUB..."
    if command -v update-grub &> /dev/null; then
        sudo update-grub
        print_success "GRUB updated successfully"
    else
        print_warning "update-grub not found, trying grub-mkconfig..."
        if command -v grub-mkconfig &> /dev/null; then
            sudo grub-mkconfig -o /boot/grub/grub.cfg
            print_success "GRUB configuration generated"
        else
            print_error "Could not find update-grub or grub-mkconfig"
            print_info "You may need to manually run: sudo update-grub"
        fi
    fi
    
elif [ "$BOOTLOADER" = "extlinux" ]; then
    print_info "Configuring ExtLinux..."
    
    # Check if file exists
    if [ ! -f "$EXT_LINUX_CONFIG" ]; then
        print_error "ExtLinux config file not found at: $EXT_LINUX_CONFIG"
        print_info "Searching for extlinux.conf..."
        
        # Try to find it
        POSSIBLE_LOCATIONS=(
            "/boot/extlinux/extlinux.conf"
            "/boot/extlinux.conf"
            "/mnt/extlinux/extlinux.conf"
            "/boot/EFI/BOOT/extlinux.conf"
        )
        
        for loc in "${POSSIBLE_LOCATIONS[@]}"; do
            if [ -f "$loc" ]; then
                EXT_LINUX_CONFIG="$loc"
                print_info "Found ExtLinux config at: $EXT_LINUX_CONFIG"
                break
            fi
        done
        
        if [ ! -f "$EXT_LINUX_CONFIG" ]; then
            print_error "Could not find ExtLinux configuration file"
            print_info "You may need to configure boot parameters manually"
            exit 1
        fi
    fi
    
    # Backup original
    if [ ! -f "${EXT_LINUX_CONFIG}.bak" ]; then
        print_info "Creating backup: ${EXT_LINUX_CONFIG}.bak"
        sudo cp "$EXT_LINUX_CONFIG" "${EXT_LINUX_CONFIG}.bak"
    fi
    
    # Read current config
    CURRENT_CONFIG=$(cat "$EXT_LINUX_CONFIG" 2>/dev/null || echo "")
    
    # Find APPEND line and update it
    if echo "$CURRENT_CONFIG" | grep -q "^[[:space:]]*APPEND"; then
        print_info "Updating existing APPEND line"
        # Remove existing quiet/splash/loglevel parameters first
        sudo sed -i "s|^\([[:space:]]*APPEND.*\)quiet[^ ]*|\1|g" "$EXT_LINUX_CONFIG"
        sudo sed -i "s|^\([[:space:]]*APPEND.*\)splash[^ ]*|\1|g" "$EXT_LINUX_CONFIG"
        sudo sed -i "s|^\([[:space:]]*APPEND.*\)loglevel=[^ ]*|\1|g" "$EXT_LINUX_CONFIG"
        # Add new parameters
        sudo sed -i "s|^\([[:space:]]*APPEND.*\)|\1 $KERNEL_PARAMS|" "$EXT_LINUX_CONFIG"
    else
        print_warning "Could not find APPEND line in ExtLinux config"
        print_info "You may need to manually add: APPEND ... $KERNEL_PARAMS"
        print_info "Config file location: $EXT_LINUX_CONFIG"
    fi
    
    print_success "ExtLinux configuration updated"
    print_info "Note: ExtLinux changes take effect on next boot"
    
else
    print_error "Unknown or unsupported bootloader: $BOOTLOADER"
    print_info "Manual configuration may be required"
    print_info "Kernel parameters to add: $KERNEL_PARAMS"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Configuration Complete"
echo "=========================================="
echo ""
print_success "Boot configuration updated"
echo ""
print_info "Changes will take effect on next reboot"
echo ""
print_info "To apply changes now, run: sudo reboot"
echo ""
print_info "To verify current kernel parameters: cat /proc/cmdline"
echo ""
print_info "Backup saved at:"
if [ "$BOOTLOADER" = "grub" ]; then
    echo "  ${GRUB_CONFIG}.bak"
elif [ "$BOOTLOADER" = "extlinux" ]; then
    echo "  ${EXT_LINUX_CONFIG}.bak"
fi
echo ""
