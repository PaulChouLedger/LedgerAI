#!/bin/bash
# Restore Jetson to Fresh State
# Removes all Aura installations and configurations
# Usage: bash restore_jetson_fresh.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Jetson Fresh Restore Script"
echo "=========================================="
echo ""
echo -e "${RED}⚠️  WARNING: This will remove all Aura installations!${NC}"
echo ""
echo "This script will:"
echo "  - Remove systemd services (xvf3800-tuning, aura, disable-keyboard-monitor, jetson-maxn-power)"
echo "  - Remove virtual environment (~/aura-env)"
echo "  - Remove cloned repositories (jetson-containers, reSpeaker_XVF3800_USB_4MIC_ARRAY)"
echo "  - Remove .asoundrc file"
echo "  - Remove git hooks"
echo "  - Remove Polkit rules"
echo "  - Remove X11 setup script"
echo "  - Remove user from groups (docker, audio, nm-authed)"
echo "  - Restore GDM3 config (automatic login)"
echo "  - Optionally remove .env and data directories"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Detect user
AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"
LEDGERAI_DIR="$AURA_HOME/LedgerAI"
VENV_DIR="$AURA_HOME/aura-env"
JETSON_CONTAINERS_DIR="$AURA_HOME/jetson-containers"
XVF3800_REPO_DIR="$AURA_HOME/reSpeaker_XVF3800_USB_4MIC_ARRAY"

echo ""
echo "=========================================="
echo "  Removing Systemd Services"
echo "=========================================="
echo ""

# Stop and disable services
SERVICES=("aura.service" "xvf3800-tuning.service" "disable-keyboard-monitor.service" "jetson-maxn-power.service")

for service in "${SERVICES[@]}"; do
    if systemctl list-unit-files | grep -q "$service"; then
        echo "Stopping and disabling $service..."
        sudo systemctl stop "$service" 2>/dev/null || true
        sudo systemctl disable "$service" 2>/dev/null || true
        echo "✅ $service stopped and disabled"
    fi
done

# Remove service files
for service in "${SERVICES[@]}"; do
    SERVICE_FILE="/etc/systemd/system/$service"
    if [ -f "$SERVICE_FILE" ] || [ -L "$SERVICE_FILE" ]; then
        echo "Removing $SERVICE_FILE..."
        sudo rm -f "$SERVICE_FILE"
        echo "✅ Removed $service"
    fi
done

# Reload systemd
sudo systemctl daemon-reload
echo "✅ Systemd reloaded"
echo ""

echo "=========================================="
echo "  Removing Virtual Environment"
echo "=========================================="
echo ""

if [ -d "$VENV_DIR" ]; then
    echo "Removing virtual environment: $VENV_DIR"
    rm -rf "$VENV_DIR"
    echo "✅ Virtual environment removed"
    
    # Remove auto-activation from .bashrc
    if grep -q "source $VENV_DIR/bin/activate" "$AURA_HOME/.bashrc" 2>/dev/null; then
        sed -i "\|source $VENV_DIR/bin/activate|d" "$AURA_HOME/.bashrc"
        sed -i "\|# Auto-activate Aura virtual environment|d" "$AURA_HOME/.bashrc"
        echo "✅ Removed venv auto-activation from .bashrc"
    fi
else
    echo "Virtual environment not found (already removed?)"
fi
echo ""

echo "=========================================="
echo "  Removing Cloned Repositories"
echo "=========================================="
echo ""

if [ -d "$JETSON_CONTAINERS_DIR" ]; then
    echo "Removing jetson-containers: $JETSON_CONTAINERS_DIR"
    rm -rf "$JETSON_CONTAINERS_DIR"
    echo "✅ jetson-containers removed"
    
    # Remove from .bashrc
    if grep -q "jetson-containers" "$AURA_HOME/.bashrc" 2>/dev/null; then
        sed -i "\|jetson-containers|d" "$AURA_HOME/.bashrc"
        sed -i "\|# jetson-containers|d" "$AURA_HOME/.bashrc"
        echo "✅ Removed jetson-containers from PATH in .bashrc"
    fi
else
    echo "jetson-containers not found"
fi

if [ -d "$XVF3800_REPO_DIR" ]; then
    echo "Removing ReSpeaker XVF3800 repository: $XVF3800_REPO_DIR"
    rm -rf "$XVF3800_REPO_DIR"
    echo "✅ ReSpeaker XVF3800 repository removed"
else
    echo "ReSpeaker XVF3800 repository not found"
fi
echo ""

echo "=========================================="
echo "  Removing Configuration Files"
echo "=========================================="
echo ""

# Remove .asoundrc
if [ -f "$AURA_HOME/.asoundrc" ]; then
    echo "Removing .asoundrc..."
    rm -f "$AURA_HOME/.asoundrc"
    echo "✅ .asoundrc removed"
else
    echo ".asoundrc not found"
fi

# Remove X11 setup script
if [ -f "$AURA_HOME/.x11_setup.sh" ]; then
    echo "Removing X11 setup script..."
    rm -f "$AURA_HOME/.x11_setup.sh"
    echo "✅ X11 setup script removed"
else
    echo "X11 setup script not found"
fi

# Remove git hooks
if [ -d "$LEDGERAI_DIR/.git/hooks" ]; then
    echo "Removing git hooks..."
    rm -f "$LEDGERAI_DIR/.git/hooks/post-merge"
    rm -f "$LEDGERAI_DIR/.git/hooks/pre-commit"
    echo "✅ Git hooks removed"
else
    echo "Git hooks directory not found"
fi
echo ""

echo "=========================================="
echo "  Removing Polkit Rules"
echo "=========================================="
echo ""

POLKIT_RULES=(
    "/etc/polkit-1/rules.d/10-allow-nmcli.rules"
    "/etc/polkit-1/rules.d/50-allow-nmcli-wifi.rules"
    "/etc/polkit-1/rules.d/50-allow-shutdown.rules"
    "/etc/polkit-1/localauthority/50-local.d/10-allow-nmcli.pkla"
    "/etc/polkit-1/localauthority/50-local.d/50-allow-nmcli-wifi.pkla"
    "/etc/polkit-1/localauthority/50-local.d/50-allow-shutdown.pkla"
)

for rule in "${POLKIT_RULES[@]}"; do
    if [ -f "$rule" ]; then
        echo "Removing $rule..."
        sudo rm -f "$rule"
        echo "✅ Removed $(basename $rule)"
    fi
done

# Restart polkit if running
if systemctl list-unit-files | grep -q "polkit"; then
    sudo systemctl restart polkit 2>/dev/null || sudo systemctl restart polkit.service 2>/dev/null || true
    echo "✅ Polkit restarted"
fi
echo ""

echo "=========================================="
echo "  Removing User from Groups"
echo "=========================================="
echo ""

GROUPS=("docker" "audio" "nm-authed")

for group in "${GROUPS[@]}"; do
    if groups "$AURA_USER" 2>/dev/null | grep -q "\b$group\b"; then
        echo "Removing $AURA_USER from $group group..."
        sudo gpasswd -d "$AURA_USER" "$group" 2>/dev/null || true
        echo "✅ Removed from $group group"
        echo "   Note: You may need to logout/login for changes to take effect"
    else
        echo "$AURA_USER not in $group group"
    fi
done
echo ""

echo "=========================================="
echo "  Restoring GDM3 Configuration"
echo "=========================================="
echo ""

GDM_CONFIG="/etc/gdm3/custom.conf"
if [ -f "$GDM_CONFIG" ]; then
    # Check if automatic login was configured
    if grep -q "AutomaticLogin=$AURA_USER" "$GDM_CONFIG" 2>/dev/null; then
        echo "Restoring GDM3 config (removing automatic login)..."
        
        # Remove automatic login lines
        sudo sed -i "/AutomaticLogin=$AURA_USER/d" "$GDM_CONFIG"
        sudo sed -i "/AutomaticLoginEnable=true/d" "$GDM_CONFIG"
        
        echo "✅ GDM3 automatic login removed"
    else
        echo "GDM3 automatic login not configured"
    fi
else
    echo "GDM3 config file not found"
fi
echo ""

echo "=========================================="
echo "  Optional: Remove .env and Data"
echo "=========================================="
echo ""

read -p "Remove .env file? (yes/no): " REMOVE_ENV
if [ "$REMOVE_ENV" = "yes" ]; then
    if [ -f "$LEDGERAI_DIR/.env" ]; then
        echo "Removing .env file..."
        rm -f "$LEDGERAI_DIR/.env"
        echo "✅ .env file removed"
    else
        echo ".env file not found"
    fi
fi

read -p "Remove data directories? (yes/no): " REMOVE_DATA
if [ "$REMOVE_DATA" = "yes" ]; then
    DATA_DIRS=(
        "$LEDGERAI_DIR/data/input"
        "$LEDGERAI_DIR/data/parsed"
        "$LEDGERAI_DIR/data/embeddings"
        "$LEDGERAI_DIR/data/models"
        "$LEDGERAI_DIR/shared/input_audio"
        "$LEDGERAI_DIR/shared/output_audio"
    )
    
    for dir in "${DATA_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo "Removing $dir..."
            rm -rf "$dir"
            echo "✅ Removed $(basename $dir)"
        fi
    done
fi
echo ""

echo "=========================================="
echo "  Checking for Remaining Aura Files"
echo "=========================================="
echo ""

REMAINING_FILES=(
    "$VENV_DIR"
    "$JETSON_CONTAINERS_DIR"
    "$XVF3800_REPO_DIR"
    "$AURA_HOME/.asoundrc"
    "$AURA_HOME/.x11_setup.sh"
)

REMAINING_COUNT=0
for file in "${REMAINING_FILES[@]}"; do
    if [ -e "$file" ]; then
        echo "⚠️  Still exists: $file"
        REMAINING_COUNT=$((REMAINING_COUNT + 1))
    fi
done

if [ $REMAINING_COUNT -eq 0 ]; then
    echo "✅ No remaining Aura files found"
else
    echo "⚠️  $REMAINING_COUNT file(s) still exist (may need manual removal)"
fi
echo ""

echo "=========================================="
echo "  Restore Complete!"
echo "=========================================="
echo ""
echo "✅ All Aura installations have been removed"
echo ""
echo "Next steps:"
echo "  1. Logout and login again (or reboot) to apply group changes"
echo "  2. If you want to reinstall: bash $LEDGERAI_DIR/setup/scripts/install_aura_bootable.sh"
echo ""
echo "Note: LedgerAI repository itself was NOT removed"
echo "      (You can still access code at: $LEDGERAI_DIR)"
echo ""

