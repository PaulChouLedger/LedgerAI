# Hardware Setup Revert Guide

## Overview

This guide explains how to revert all hardware modifications made to the system for the ReSpeaker 4 Mic Array.

## What Was Modified

### 1. **USB Permissions (udev rules)**
- **File:** `/etc/udev/rules.d/99-respeaker.rules`
- **Purpose:** Allow non-root access to ReSpeaker USB device
- **Content:** Rules to set device permissions for vendor ID 0x2886

### 2. **Systemd Service (Auto-start on boot)**
- **Possible files:**
  - `/etc/systemd/system/aura-listener.service`
  - `/etc/systemd/system/ledgerai.service`
  - `/etc/systemd/system/aura.service`
- **Purpose:** Automatically start listener on system boot

### 3. **Hardware DSP Configuration (Temporary)**
- **What:** AGC, HPF, noise suppression settings on ReSpeaker chip
- **Location:** In-memory on the ReSpeaker's DSP
- **Persistence:** Resets on device power cycle or system reboot

### 4. **User Group Membership**
- **Groups:** `plugdev`, `audio`
- **Purpose:** Grant user permission to access USB devices
- **Note:** May be needed for other devices, not removed by default

## Quick Revert (Automated)

### Option 1: Run the Revert Script

```bash
cd /home/aura/LedgerAI
sudo bash scripts/revert_hardware_setup.sh
```

This will:
- ✅ Remove udev rules
- ✅ Stop and disable systemd services
- ✅ Reload udev and systemd
- ✅ Provide reboot instructions

### Option 2: Reboot (Hardware settings only)

```bash
sudo reboot
```

This will reset:
- ✅ ReSpeaker DSP settings (AGC, HPF, etc.)
- ❌ Will NOT remove udev rules or systemd services

## Manual Revert Steps

### Step 1: Remove udev Rules

```bash
# Check if rule exists
ls -la /etc/udev/rules.d/99-respeaker.rules

# Remove the rule
sudo rm /etc/udev/rules.d/99-respeaker.rules

# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Verify:**
```bash
# Should show "No such file"
ls -la /etc/udev/rules.d/99-respeaker.rules
```

### Step 2: Remove Systemd Service

```bash
# Check what services exist
ls -la /etc/systemd/system/ | grep -E 'aura|ledger'

# For each service found:
sudo systemctl stop aura-listener.service      # Stop if running
sudo systemctl disable aura-listener.service   # Disable auto-start
sudo rm /etc/systemd/system/aura-listener.service

# Reload systemd
sudo systemctl daemon-reload
```

**Verify:**
```bash
# Should show "not-found" or "inactive"
systemctl status aura-listener.service
```

### Step 3: Reset ReSpeaker Hardware

```bash
# Option A: Reboot (recommended)
sudo reboot

# Option B: Unplug and replug USB device
# 1. Unplug ReSpeaker from USB
# 2. Wait 5 seconds
# 3. Replug ReSpeaker
```

### Step 4: (Optional) Remove User from Groups

**⚠️ Warning:** Only do this if you're sure other devices don't need it!

```bash
# Check current groups
groups $USER

# Remove from plugdev (if present)
sudo deluser $USER plugdev

# Logout and login for changes to take effect
```

## Verification

After reverting, verify the changes:

### 1. Check udev Rules

```bash
ls -la /etc/udev/rules.d/ | grep respeaker
# Should return nothing
```

### 2. Check Systemd Services

```bash
systemctl list-units --type=service | grep -E 'aura|ledger'
# Should return nothing
```

### 3. Check ReSpeaker Access

```bash
# Try to access ReSpeaker without sudo
python3 -c "import usb.core; dev = usb.core.find(idVendor=0x2886); print('Found' if dev else 'Not found')"

# After revert, this may show "Permission denied" (expected)
```

### 4. Test Audio Capture

```bash
# Without udev rules, may need sudo
sudo python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

## What Happens After Revert

### With udev rules removed:
- ❌ Cannot access ReSpeaker USB control without sudo
- ❌ Cannot configure AGC/HPF/DSP without sudo
- ✅ Can still record audio (audio driver doesn't need special permissions)

### With systemd service removed:
- ❌ Listener won't auto-start on boot
- ✅ Must manually start: `cd aura-control && python3 listener.py`

### With hardware settings reset:
- ✅ ReSpeaker returns to factory defaults
- ✅ No AGC, no HPF, no noise suppression
- ✅ Clean slate for testing

## Re-applying Setup (If Needed Later)

If you want to re-apply the setup:

### 1. Create udev Rule Again

```bash
sudo nano /etc/udev/rules.d/99-respeaker.rules
```

Add:
```
# ReSpeaker 4 Mic Array
SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666", GROUP="plugdev"
```

Reload:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 2. Create Systemd Service Again

```bash
sudo nano /etc/systemd/system/aura-listener.service
```

Add service definition, then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable aura-listener.service
sudo systemctl start aura-listener.service
```

## Troubleshooting

### "Permission denied" when accessing ReSpeaker

**After revert:**
- ✅ This is expected! udev rules were removed
- **Solution:** Use sudo, or re-add udev rules

### Listener won't start on boot

**After revert:**
- ✅ This is expected! systemd service was removed
- **Solution:** Manually start, or re-add service

### Audio still sounds distorted

**Possible causes:**
1. **Hardware settings not reset**
   - Solution: Reboot or replug USB

2. **Audio source is too loud**
   - Solution: Speak more quietly or move back from mic

3. **Whisper model issue**
   - Solution: Check container logs, rebuild container

### Script fails with "Permission denied"

**Problem:** Script needs root access
```bash
sudo bash scripts/revert_hardware_setup.sh
```

## Files Modified by This Project

### System Files (require sudo to modify):
- `/etc/udev/rules.d/99-respeaker.rules` - USB permissions
- `/etc/systemd/system/aura-listener.service` - Auto-start service

### User Files (no sudo needed):
- `~/LedgerAI/` - Project directory
- `~/.cache/` - Python package cache
- `~/usb_4_mic_array/` - ReSpeaker Python library

### Docker:
- Docker containers (persist until manually removed)
- Docker volumes (persist data)

## Complete Clean Slate

For a completely clean system:

```bash
# 1. Revert hardware modifications
sudo bash scripts/revert_hardware_setup.sh

# 2. Stop and remove Docker containers
docker-compose down -v

# 3. Remove project directory (⚠️ CAUTION: This deletes everything!)
# cd ~
# rm -rf LedgerAI

# 4. Reboot
sudo reboot
```

## Summary

| What | Where | How to Revert |
|------|-------|---------------|
| **udev rules** | `/etc/udev/rules.d/99-respeaker.rules` | Delete file, reload udev |
| **systemd service** | `/etc/systemd/system/*.service` | Stop, disable, delete, reload |
| **Hardware DSP** | ReSpeaker chip memory | Reboot or replug USB |
| **User groups** | `/etc/group` | Optional: deluser |
| **Docker** | Docker daemon | `docker-compose down` |

**Quickest revert:** Run `sudo bash scripts/revert_hardware_setup.sh` then reboot.

