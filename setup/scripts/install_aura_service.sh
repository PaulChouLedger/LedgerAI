#!/bin/bash
#
# Install Aura Systemd Service
#
# This script installs the Aura systemd service for automatic startup on boot.
#
# Usage:
#   sudo bash setup/scripts/install_aura_service.sh
#

set -e

echo ""
echo "================================================================================"
echo "  🔧 AURA SYSTEMD SERVICE INSTALLER"
echo "================================================================================"
echo ""
echo "  This will install Aura as a systemd service that:"
echo "  • Starts automatically on boot"
echo "  • Restarts automatically if it crashes"
echo "  • Manages Docker containers"
echo ""
echo "================================================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "  ❌ ERROR: Please run with sudo"
    echo ""
    echo "  Usage:"
    echo "    sudo bash setup/scripts/install_aura_service.sh"
    echo ""
    exit 1
fi

# Get user running the script (should be aura)
SCRIPT_USER=${SUDO_USER:-$USER}
AURA_HOME="/home/$SCRIPT_USER"
SERVICE_FILE="/etc/systemd/system/aura.service"
SOURCE_SERVICE="$(dirname "$0")/aura.service"

# Check if aura user exists
if [ ! -d "$AURA_HOME" ]; then
    echo "  ❌ ERROR: User home directory not found: $AURA_HOME"
    echo ""
    echo "  Please create the user first:"
    echo "    sudo useradd -m -s /bin/bash aura"
    echo ""
    exit 1
fi

# Check if LedgerAI directory exists
if [ ! -d "$AURA_HOME/LedgerAI" ]; then
    echo "  ❌ ERROR: LedgerAI directory not found: $AURA_HOME/LedgerAI"
    echo ""
    echo "  Please clone the repository first:"
    echo "    cd ~"
    echo "    git clone https://github.com/PaulChouLedger/LedgerAI.git"
    echo ""
    exit 1
fi

echo "[1/5] Checking prerequisites..."
if [ ! -d "$AURA_HOME/LedgerAI/aura-env" ]; then
    echo "     ⚠️  Virtual environment not found at $AURA_HOME/LedgerAI/aura-env"
    echo "     💡 Run: python3 -m venv $AURA_HOME/LedgerAI/aura-env"
    echo ""
    read -p "Continue anyway? (y/n): " answer
    if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
        exit 1
    fi
fi

if ! command -v docker &> /dev/null; then
    echo "     ⚠️  Docker not found"
    echo "     💡 Install Docker first: curl -fsSL https://get.docker.com | sudo sh"
    echo ""
    read -p "Continue anyway? (y/n): " answer
    if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
        exit 1
    fi
fi
echo "     ✅ Prerequisites check complete"

echo "[2/5] Updating service file with correct paths..."
# Create temporary service file with correct paths
sed "s|/home/aura|$AURA_HOME|g" "$SOURCE_SERVICE" > "${SERVICE_FILE}.tmp"
echo "     ✅ Service file configured for $AURA_HOME"

echo "[3/5] Copying service file..."
mv "${SERVICE_FILE}.tmp" "$SERVICE_FILE"
echo "     ✅ Copied to $SERVICE_FILE"

echo "[4/5] Reloading systemd..."
systemctl daemon-reload
echo "     ✅ Systemd reloaded"

echo "[5/5] Enabling service (auto-start on boot)..."
systemctl enable aura.service
echo "     ✅ Service enabled"

echo ""
echo "================================================================================"
echo "  ✅ INSTALLATION COMPLETE"
echo "================================================================================"
echo ""
echo "  Aura service installed successfully!"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl start aura          # Start service now"
echo "    sudo systemctl stop aura           # Stop service"
echo "    sudo systemctl restart aura        # Restart service"
echo "    sudo systemctl status aura         # Check service status"
echo "    sudo journalctl -u aura -f         # View live logs"
echo ""
echo "  The service will start automatically on boot."
echo ""
read -p "Start the service now? (y/n): " answer
if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
    systemctl start aura.service
    echo ""
    echo "  ✅ Service started"
    echo ""
    echo "  Check status:"
    echo "    sudo systemctl status aura.service"
    echo ""
fi

