#!/bin/bash
# setup_ble_advertising.sh — One-time setup to enable BLE GATT advertising for AuraConnect.
#
# Run this ONCE on each puck as root:
#   sudo bash setup/scripts/setup_ble_advertising.sh
#
# What it does:
#   1. Enables --experimental in bluetoothd (required for LE advertising)
#   2. Adds passwordless sudo for btmgmt (so the app can enable LE without prompts)
#   3. Installs dbus-next Python package if missing
#   4. Restarts bluetooth service

set -e

echo "[1/4] Enabling experimental features in bluetoothd..."
BT_SERVICE="/lib/systemd/system/bluetooth.service"
if ! grep -q '\-\-experimental' "$BT_SERVICE"; then
    sed -i 's|ExecStart=/usr/libexec/bluetoothd|ExecStart=/usr/libexec/bluetoothd --experimental|' "$BT_SERVICE" 2>/dev/null ||
    sed -i 's|ExecStart=/usr/lib/bluetooth/bluetoothd|ExecStart=/usr/lib/bluetooth/bluetoothd --experimental|' "$BT_SERVICE"
    echo "  Added --experimental flag"
else
    echo "  Already has --experimental"
fi

echo "[2/4] Adding passwordless sudo for btmgmt..."
SUDOERS_FILE="/etc/sudoers.d/aura-btmgmt"
cat > "$SUDOERS_FILE" <<'SUDOERS'
# Allow the ledger user to run btmgmt without password (for BLE advertising)
ledger ALL=(ALL) NOPASSWD: /usr/bin/btmgmt
SUDOERS
chmod 440 "$SUDOERS_FILE"
echo "  Created $SUDOERS_FILE"

echo "[3/4] Installing dbus-next if missing..."
sudo -u ledger bash -c 'source ~/aura-env/bin/activate && pip install dbus-next 2>/dev/null || echo "  already installed"'

echo "[4/4] Restarting bluetooth service..."
systemctl daemon-reload
systemctl restart bluetooth
sleep 2

# Verify
if systemctl is-active --quiet bluetooth; then
    echo ""
    echo "Done! Bluetooth is running with experimental features."
    echo "AuraConnect should now be able to register advertisements."
    echo ""
    # Quick test
    btmgmt le on && echo "LE advertising: enabled" || echo "WARNING: btmgmt le on failed"
else
    echo "ERROR: bluetooth service failed to start"
    systemctl status bluetooth --no-pager
    exit 1
fi
