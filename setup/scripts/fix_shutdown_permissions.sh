#!/bin/bash
# Fix shutdown permissions for passwordless system shutdown
# This script creates polkit rules to allow shutdown without password prompts

set -e

# Get the current user (or use the first argument)
AURA_USER="${1:-${SUDO_USER:-$USER}}"

if [ -z "$AURA_USER" ] || [ "$AURA_USER" = "root" ]; then
    echo "❌ Error: Cannot determine user. Please run as: sudo $0 <username>"
    echo "   Example: sudo $0 ledger"
    exit 1
fi

echo "🔧 Fixing shutdown permissions for user: $AURA_USER"

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

POLKIT_RULE="$POLKIT_DIR/50-allow-shutdown.rules"

echo "📝 Creating polkit rule at $POLKIT_RULE..."

# Create a comprehensive rule that allows the user to shutdown/reboot
sudo tee "$POLKIT_RULE" >/dev/null <<EORULE
// Allow system shutdown/reboot for $AURA_USER
// This rule allows passwordless shutdown and reboot operations

polkit.addRule(function(action, subject) {
  // Allow system power management actions for the specific user
  if (action.id == "org.freedesktop.login1.power-off" ||
      action.id == "org.freedesktop.login1.power-off-multiple-sessions" ||
      action.id == "org.freedesktop.login1.reboot" ||
      action.id == "org.freedesktop.login1.reboot-multiple-sessions" ||
      action.id == "org.freedesktop.login1.hibernate" ||
      action.id == "org.freedesktop.login1.suspend" ||
      action.id == "org.freedesktop.login1.set-wall-message") {
    if (subject.user == "$AURA_USER") {
      polkit.log("action=" + action.id + " subject=" + subject.user);
      return polkit.Result.YES;
    }
  }
  
  // Also allow systemctl poweroff/reboot commands
  if (action.id.indexOf("org.freedesktop.systemd1") === 0) {
    if (subject.user == "$AURA_USER") {
      polkit.log("action=" + action.id + " subject=" + subject.user);
      return polkit.Result.YES;
    }
  }
  
  // Allow polkit exec for systemctl commands
  if (action.id == "org.freedesktop.policykit.exec") {
    if (subject.user == "$AURA_USER") {
      // Check if it's a systemctl poweroff/reboot command
      if (action.lookup("command") && 
          (action.lookup("command").indexOf("systemctl") >= 0 && 
           (action.lookup("command").indexOf("poweroff") >= 0 || 
            action.lookup("command").indexOf("reboot") >= 0))) {
        polkit.log("action=" + action.id + " subject=" + subject.user);
        return polkit.Result.YES;
      }
    }
  }
});
EORULE

sudo chmod 0644 "$POLKIT_RULE" 2>/dev/null || true

echo "✅ Polkit rule created"

# Also create a .pkla file as fallback for older systems
PKLA_DIR="/etc/polkit-1/localauthority/50-local.d"
sudo mkdir -p "$PKLA_DIR" 2>/dev/null || true
PKLA_FILE="$PKLA_DIR/50-allow-shutdown.pkla"

echo "📝 Creating fallback .pkla rule at $PKLA_FILE..."

sudo tee "$PKLA_FILE" >/dev/null <<EOPKLA
[Allow Shutdown for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.login1.power-off;org.freedesktop.login1.power-off-multiple-sessions;org.freedesktop.login1.reboot;org.freedesktop.login1.reboot-multiple-sessions;org.freedesktop.login1.hibernate;org.freedesktop.login1.suspend;org.freedesktop.login1.set-wall-message
ResultAny=yes
ResultInactive=yes
ResultActive=yes

[Allow systemctl for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.systemd1.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes

[Allow polkit exec for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.policykit.exec
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
echo "✅ Shutdown permissions fixed!"
echo ""
echo "📋 Summary:"
echo "   - User: $AURA_USER"
echo "   - Polkit rule: $POLKIT_RULE"
echo "   - Fallback rule: $PKLA_FILE"
echo ""
echo "💡 The shutdown button in Settings will now work without password prompts."
echo "   Test it by holding the shutdown button for 3 seconds."
echo ""

