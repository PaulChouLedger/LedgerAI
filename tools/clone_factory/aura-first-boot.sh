#!/bin/bash
# Aura clone first-boot identity reset.
#
# Runs once on a freshly-cloned Jetson, regenerates per-host identity so
# the new puck doesn't collide with its sibling on the network, then
# disables itself. Logs to /var/log/aura-first-boot.log.
#
# Installed by tools/clone_factory/inject_first_boot.sh.

set -e
LOG=/var/log/aura-first-boot.log
exec > >(tee -a "$LOG") 2>&1
echo "[first-boot] $(date -Iseconds) start"

# 1. Hostname — derive a stable suffix from the wired NIC's MAC so each
#    cloned puck gets a unique, deterministic name.
NIC=$(ip -o link | awk -F': ' '$2 !~ /^lo$/ {print $2; exit}')
MAC=$(cat "/sys/class/net/${NIC}/address" 2>/dev/null | tr -d ':' | tail -c 7)
NEW_HOST="puck-${MAC:-$(date +%s | tail -c 6)}"
hostnamectl set-hostname "$NEW_HOST"
echo "[first-boot] hostname=$NEW_HOST"

# 2. machine-id — must be unique per machine; many services key on it.
rm -f /etc/machine-id /var/lib/dbus/machine-id
systemd-machine-id-setup
ln -sf /etc/machine-id /var/lib/dbus/machine-id
echo "[first-boot] machine-id regenerated"

# 3. SSH host keys — prevents StrictHostKeyChecking collisions across pucks.
rm -f /etc/ssh/ssh_host_*
ssh-keygen -A
systemctl restart sshd 2>/dev/null || systemctl restart ssh 2>/dev/null || true
echo "[first-boot] ssh host keys regenerated"

# 4. Tailscale — clear the cached node identity so this puck registers as
#    a new node on next 'tailscale up'. Operator must manually run
#    'sudo tailscale up' after first boot to claim the puck.
if [ -f /var/lib/tailscale/tailscaled.state ]; then
    systemctl stop tailscaled 2>/dev/null || true
    rm -f /var/lib/tailscale/tailscaled.state
    systemctl start tailscaled 2>/dev/null || true
    echo "[first-boot] tailscale state cleared (run 'sudo tailscale up' to re-auth)"
fi

# 5. Aura-specific per-puck state.
#    Keep voicelines (10k WAV pool is hardware-agnostic) and the LLM
#    cache. Drop only the bits that *should* be fresh on a new puck:
#    app_settings (color scheme, picked at first run), and voice
#    profiles (re-enroll the household's voices on the new device).
rm -f /home/ledger/Aura4/data/app_settings.json
rm -rf /home/ledger/Aura4/data/voice_profiles
echo "[first-boot] aura per-puck state cleared (voicelines + memory preserved)"

echo "[first-boot] $(date -Iseconds) done — disabling self"
systemctl disable aura-first-boot.service
