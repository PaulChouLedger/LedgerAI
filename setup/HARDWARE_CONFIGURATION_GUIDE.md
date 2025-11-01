# Hardware Configuration Guide

## Overview

Hardware configuration for microphone arrays is managed separately from the listener to avoid permission issues and ensure consistent settings.

## Architecture

### Separation of Concerns

| Component | Purpose | Permissions | When Runs |
|-----------|---------|-------------|-----------|
| **`tune_xvf3800.py`** | Configure XVF3800 USB 4 Mic Array DSP | User permissions | On boot via systemd service |
| **`listener.py`** | Audio capture, VAD, transcription | User permissions | Manually or via service |

## XVF3800 USB 4 Mic Array Configuration

### Removing Old ReSpeaker Service (If Installed)

If you have the old `respeaker-tuning.service` installed, remove it first:

```bash
# Stop and disable old service
sudo systemctl stop respeaker-tuning.service
sudo systemctl disable respeaker-tuning.service

# Remove service file
sudo rm /etc/systemd/system/respeaker-tuning.service

# Reload systemd
sudo systemctl daemon-reload
```

### Option 1: Boot Service (Recommended)

Configure XVF3800 automatically on system boot:

**1. Install service file:**
```bash
sudo cp /home/aura/LedgerAI/setup/scripts/xvf3800-tuning.service /etc/systemd/system/
sudo systemctl daemon-reload
```

**2. Enable and start:**
```bash
sudo systemctl enable xvf3800-tuning.service
sudo systemctl start xvf3800-tuning.service
```

**3. Verify:**
```bash
systemctl status xvf3800-tuning.service
```

**4. Check logs:**
```bash
journalctl -u xvf3800-tuning.service -n 50
```

### Option 2: Manual Configuration

Run configuration script manually before starting listener:

```bash
# Change to home directory (xvf_host is installed there)
cd ~/LedgerAI

# Run tuning script
python3 setup/scripts/tune_xvf3800.py balanced_beam

# Then start listener as normal user
cd aura-control
python3 listener.py
```

**Available Presets:**
- `balanced_beam` - HPF 70Hz + AGC (0.08, 30dB) ⭐ RECOMMENDED
- `ultra_sensitive` - AGC (0.10, 45dB) - Far-field optimized
- `far_field` - Optimized for 8-16 feet
- `near_field` - Optimized for 1-6 feet
- `hpf_only` - HPF 70Hz only (minimal processing)
- `reset` - Factory defaults
- `show` - Current settings

### Option 3: No Configuration (Factory Defaults)

Disable boot service:
```bash
sudo systemctl stop xvf3800-tuning.service
sudo systemctl disable xvf3800-tuning.service
```

## Troubleshooting

### Problem: xvf_host not found

**Solution:**
```bash
# Check if xvf_host exists
ls ~/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host

# If not found, XVF3800 SDK needs to be installed
```

### Problem: Boot service not working

**Check service status:**
```bash
systemctl status xvf3800-tuning.service
```

**View logs:**
```bash
journalctl -u xvf3800-tuning.service -n 50
```

**Common issues:**
- File not found → Check path in ExecStart
- Permission denied → Check xvf_host permissions
- USB device not found → Check USB connection

## Summary

- **Hardware config**: Use `tune_xvf3800.py` (manual or boot service)
- **Boot service**: Recommended for consistent settings
- **Monitoring**: Check service status with `systemctl status`

This separation keeps the listener simple and avoids permission issues while giving you full control over hardware configuration when needed.

