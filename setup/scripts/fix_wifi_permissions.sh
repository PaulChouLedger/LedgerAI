#!/bin/bash
# Fix WiFi permissions for NetworkManager
# This script creates polkit rules to allow WiFi connection without password prompts

set -e

# Get the current user (or use the first argument)
AURA_USER="${1:-${SUDO_USER:-$USER}}"

if [ -z "$AURA_USER" ] || [ "$AURA_USER" = "root" ]; then
    echo "❌ Error: Cannot determine user. Please run as: sudo $0 <username>"
    echo "   Example: sudo $0 ledger"
    exit 1
fi

echo "🔧 Fixing WiFi permissions for user: $AURA_USER"

# Ensure polkit is installed
if ! dpkg -s policykit-1 >/dev/null 2>&1; then
    echo "📦 Installing policykit-1..."
    sudo apt install -y policykit-1 || true
fi

# Determine Polkit rules directory
POLKIT_DIR="/etc/polkit-1/rules.d"
if [ ! -d "$POLKIT_DIR" ]; then
    sudo mkdir -p "$POLKIT_DIR" 2>/dev/null || true
fi

if [ ! -d "$POLKIT_DIR" ]; then
    POLKIT_DIR="/usr/share/polkit-1/rules.d"
    sudo mkdir -p "$POLKIT_DIR" 2>/dev/null || true
fi

POLKIT_RULE="$POLKIT_DIR/50-allow-nmcli-wifi.rules"

echo "📝 Creating polkit rule at $POLKIT_RULE..."

# Create a comprehensive rule that allows the user to use NetworkManager
sudo tee "$POLKIT_RULE" >/dev/null <<EORULE
// Allow NetworkManager WiFi operations for $AURA_USER
// This rule allows WiFi scanning, connecting, and disconnecting without password prompts

polkit.addRule(function(action, subject) {
  // Allow all NetworkManager actions for the specific user
  if (action.id.indexOf("org.freedesktop.NetworkManager") === 0) {
    // Allow for the specific user
    if (subject.user == "$AURA_USER") {
      return polkit.Result.YES;
    }
    // Also allow for users in the nm-authed group (if it exists)
    if (subject.isInGroup("nm-authed")) {
      return polkit.Result.YES;
    }
  }
  
  // Specifically allow WiFi operations
  if (action.id == "org.freedesktop.NetworkManager.wifi.scan" ||
      action.id == "org.freedesktop.NetworkManager.settings.modify.system" ||
      action.id == "org.freedesktop.NetworkManager.settings.modify.own" ||
      action.id == "org.freedesktop.NetworkManager.network-control") {
    if (subject.user == "$AURA_USER") {
      return polkit.Result.YES;
    }
  }
});
EORULE

sudo chmod 0644 "$POLKIT_RULE" 2>/dev/null || true

echo "✅ Polkit rule created"

# Also create a .pkla file as fallback for older systems
PKLA_DIR="/etc/polkit-1/localauthority/50-local.d"
sudo mkdir -p "$PKLA_DIR" 2>/dev/null || true
PKLA_FILE="$PKLA_DIR/50-allow-nmcli-wifi.pkla"

echo "📝 Creating fallback .pkla rule at $PKLA_FILE..."

sudo tee "$PKLA_FILE" >/dev/null <<EOPKLA
[Allow NetworkManager WiFi for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes

[Allow NetworkManager WiFi for nm-authed group]
Identity=unix-group:nm-authed
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOPKLA

sudo chmod 0644 "$PKLA_FILE" 2>/dev/null || true

echo "✅ Fallback .pkla rule created"

# Restart polkit to apply rules immediately
echo "🔄 Restarting polkit service..."
if systemctl list-unit-files | grep -q "polkit"; then
    sudo systemctl restart polkit 2>/dev/null || sudo systemctl restart polkit.service 2>/dev/null || true
    echo "✅ Polkit service restarted"
else
    echo "⚠️  Polkit service not found (may not be needed)"
fi

echo ""
echo "✅ WiFi permissions fixed!"
echo ""
echo "📋 Summary:"
echo "   - User: $AURA_USER"
echo "   - Polkit rule: $POLKIT_RULE"
echo "   - Fallback rule: $PKLA_FILE"
echo ""
echo "💡 Note: If WiFi connection still fails, try:"
echo "   1. Logout and login again (or reboot)"
echo "   2. Check if user is in nm-authed group: groups $AURA_USER"
echo "   3. Test manually: nmcli device wifi connect <SSID> password <PASSWORD>"
echo ""

