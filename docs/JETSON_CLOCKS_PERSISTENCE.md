# Jetson Clocks Persistence

## Quick Answer

**No**, `sudo jetson_clocks` does **NOT** persist across reboots by default. You need to run it again after each reboot, or set up a systemd service to run it automatically.

## Your Setup

The LedgerAI install script (`install_aura_bootable.sh`) already creates a systemd service to make `jetson_clocks` persistent:

**Service:** `/etc/systemd/system/jetson-maxn-power.service`

This service:
1. Sets power mode to MAXN (`nvpmodel -m 0`)
2. Runs `jetson_clocks` to set maximum clocks
3. Runs automatically on every boot

## Check Status

```bash
# Check if the service exists and is enabled
sudo systemctl status jetson-maxn-power.service

# Check if it's enabled to run on boot
sudo systemctl is-enabled jetson-maxn-power.service
```

## Enable Manually (if needed)

If the service exists but isn't enabled:

```bash
sudo systemctl enable jetson-maxn-power.service
sudo systemctl start jetson-maxn-power.service
```

## Create Manually (if service doesn't exist)

If the service wasn't created during installation, create it:

```bash
sudo nano /etc/systemd/system/jetson-maxn-power.service
```

Add this content:

```ini
[Unit]
Description=Set Jetson to MAXN Power Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvpmodel -m 0
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable jetson-maxn-power.service
sudo systemctl start jetson-maxn-power.service
```

## Verify It's Working

After enabling, verify the clocks are set:

```bash
# Check current clocks (requires sudo)
sudo tegrastats

# Or check power mode
sudo nvpmodel -q
```

## One-Time Run (Non-Persistent)

If you just want to run it once without persistence:

```bash
sudo jetson_clocks
```

This will set max clocks until the next reboot.

