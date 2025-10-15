# Hardware Configuration Guide

## Overview

Hardware configuration for the ReSpeaker 4 Mic Array is managed separately from the listener to avoid permission issues and ensure consistent settings.

## Architecture

### Separation of Concerns

| Component | Purpose | Permissions | When Runs |
|-----------|---------|-------------|-----------|
| **`tune_respeaker.py`** | Configure hardware DSP (AGC, HPF, etc.) | Requires root/sudo | On boot via systemd service |
| **`listener.py`** | Audio capture, VAD, transcription | User permissions | Manually or via service |

### Why Separate?

1. **Permission Management**
   - Hardware configuration requires USB access (may need root)
   - Listener can run as normal user after hardware is configured

2. **Consistency**
   - Hardware configured once on boot
   - Settings persist for entire session
   - No mid-session configuration changes

3. **Simplicity**
   - Listener focuses on audio processing
   - Hardware setup is separate concern
   - Easier to debug issues

## Hardware Configuration

### Option 1: Boot Service (Recommended)

Configure hardware automatically on system boot:

**1. Edit configuration in `scripts/tune_respeaker.py`:**
```python
# Choose a preset: 'default', 'far_field', 'near_field', 'raw'
# See tune_respeaker.py for details
```

**2. Create systemd service:**
```bash
sudo nano /etc/systemd/system/respeaker-tuning.service
```

Add:
```ini
[Unit]
Description=ReSpeaker Hardware DSP Configuration
After=multi-user.target
Wants=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=root
ExecStart=/usr/bin/python3 /home/aura/LedgerAI/scripts/tune_respeaker.py far_field
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**3. Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable respeaker-tuning.service
sudo systemctl start respeaker-tuning.service
```

**4. Verify:**
```bash
systemctl status respeaker-tuning.service
```

### Option 2: Manual Configuration

Run configuration script manually before starting listener:

```bash
# Run as root if needed
sudo python3 scripts/tune_respeaker.py far_field

# Then start listener as normal user
cd aura-control
python3 listener.py
```

### Option 3: No Configuration (Factory Defaults)

Disable boot service and don't run configuration:

```bash
# Disable service
sudo systemctl stop respeaker-tuning.service
sudo systemctl disable respeaker-tuning.service

# Reboot to reset hardware
sudo reboot

# Listener will show factory default settings
```

## Listener Behavior

### Read-Only Hardware Monitoring

The listener **reads and displays** hardware configuration but **does not modify** it:

**On startup:**
```
[Hardware] 📖 Reading current configuration...

======================================================================
[Hardware] 📋 CURRENT ReSPEAKER CONFIGURATION:
======================================================================
  AGC:                    ✅ ENABLED
    Target Level:         0.12
    Max Gain:             30.0 dB
  High-Pass Filter:       ✅ ENABLED
  Stationary Noise Supp:  ❌ DISABLED
  Non-Stat Noise Supp:    ❌ DISABLED
  Echo Cancellation:      ❌ DISABLED
======================================================================
```

This shows you:
- What hardware settings are active
- Whether boot service configured the device
- If factory defaults are being used

### Software AGC (Optional)

The listener can apply **software AGC** after audio capture:

**Enable in `listener.py`:**
```python
USE_SOFTWARE_AGC = True
SOFTWARE_AGC_TARGET = 0.1
```

**When to use:**
- Hardware AGC is disabled
- Need different gain than hardware provides
- Testing different configurations

## Configuration Presets

### Available in `tune_respeaker.py`:

#### 1. **`default`** - Balanced
- AGC: Enabled (target=0.12, max_gain=30dB)
- HPF: Enabled
- Noise Suppression: Disabled
- **Use for:** General purpose, balanced settings

#### 2. **`far_field`** - Distance/Quiet Speech
- AGC: Enabled (target=0.15, max_gain=40dB)
- HPF: Enabled  
- Stationary Noise: Enabled (gamma=3.0)
- **Use for:** Speaking from distance, quiet environment

#### 3. **`near_field`** - Close/Loud Speech
- AGC: Enabled (target=0.05, max_gain=15dB)
- HPF: Enabled
- Noise Suppression: Disabled
- **Use for:** Speaking close to mic, loud environment

#### 4. **`raw`** - No Processing
- AGC: Disabled
- HPF: Enabled (only)
- All other processing: Disabled
- **Use for:** Debugging, testing, clean audio

## Troubleshooting

### Problem: Audio is clipping (distorted)

**Symptoms:**
- Whisper hallucinations/repetitions
- Peak audio values = 1.0
- Distorted transcriptions

**Solutions:**
1. **Check current hardware config:**
   ```bash
   python3 listener.py  # Look at config display
   ```

2. **Try lower AGC preset:**
   ```bash
   sudo python3 scripts/tune_respeaker.py near_field
   ```

3. **Or disable AGC completely:**
   ```bash
   sudo python3 scripts/tune_respeaker.py raw
   ```

### Problem: Audio too quiet

**Symptoms:**
- RMS < 0.01
- "RMS too low" messages
- No transcription

**Solutions:**
1. **Enable higher AGC:**
   ```bash
   sudo python3 scripts/tune_respeaker.py far_field
   ```

2. **Or use software AGC:**
   Edit `listener.py`:
   ```python
   USE_SOFTWARE_AGC = True
   ```

### Problem: Don't know what hardware config is active

**Solution:**
```bash
# Start listener to see config
python3 aura-control/listener.py

# Look for this output:
# [Hardware] 📋 CURRENT RESPEAKER CONFIGURATION:
```

### Problem: Boot service not working

**Check service status:**
```bash
systemctl status respeaker-tuning.service
```

**View logs:**
```bash
journalctl -u respeaker-tuning.service -n 50
```

**Common issues:**
- Permission denied → Service needs `User=root`
- File not found → Check path in ExecStart
- USB device not found → Check udev rules

## Best Practices

### 1. Use Boot Service for Production

**Advantages:**
- ✅ Consistent settings every boot
- ✅ No manual configuration needed
- ✅ Separate from listener process

**Setup:**
```bash
# Configure once
sudo systemctl enable respeaker-tuning.service

# Forget about it - works automatically
```

### 2. Test Manually First

Before setting up boot service:

```bash
# Test different presets
sudo python3 scripts/tune_respeaker.py raw
python3 scripts/find_optimal_rms.py

sudo python3 scripts/tune_respeaker.py near_field  
python3 scripts/find_optimal_rms.py

sudo python3 scripts/tune_respeaker.py far_field
python3 scripts/find_optimal_rms.py

# Find what works, then set up boot service with that preset
```

### 3. Monitor Hardware Config

Always check hardware config in listener startup:

```bash
python3 aura-control/listener.py

# Verify settings match expectations
# If not, check boot service or run tune_respeaker.py
```

### 4. Keep Software AGC Disabled

Unless you have a specific need:

```python
USE_SOFTWARE_AGC = False  # Default
```

**Only enable if:**
- Testing different gain levels
- Hardware AGC insufficient
- Debugging audio issues

## Example Workflows

### Workflow 1: Set Up Fresh System

```bash
# 1. Test raw audio
sudo python3 scripts/tune_respeaker.py raw
python3 aura-control/listener.py
# Test transcription quality

# 2. If too quiet, try default preset
sudo python3 scripts/tune_respeaker.py default
python3 aura-control/listener.py
# Test again

# 3. Once satisfied, set up boot service
sudo systemctl enable respeaker-tuning.service
sudo reboot

# 4. Verify on boot
python3 aura-control/listener.py
# Check hardware config display
```

### Workflow 2: Debug Clipping Issues

```bash
# 1. Check current hardware settings
python3 aura-control/listener.py
# Look at AGC settings

# 2. If AGC is high, disable it
sudo python3 scripts/tune_respeaker.py raw
python3 aura-control/listener.py
# Test if clipping stops

# 3. If raw is too quiet, use low AGC
sudo python3 scripts/tune_respeaker.py near_field
python3 aura-control/listener.py
# Should be balanced
```

### Workflow 3: Revert All Hardware Modifications

```bash
# Use the revert script
sudo bash scripts/revert_hardware_setup.sh

# Reboot to reset hardware
sudo reboot

# Hardware will have factory defaults
python3 aura-control/listener.py
# Check config - should show all disabled/defaults
```

## Summary

- **Hardware config**: Use `tune_respeaker.py` (manual or boot service)
- **Listener**: Reads and displays config only (no modifications)
- **Boot service**: Recommended for consistent settings
- **Software AGC**: Optional, for special cases only
- **Monitoring**: Listener shows current hardware state on startup

This separation keeps the listener simple and avoids permission issues while giving you full control over hardware configuration when needed.

