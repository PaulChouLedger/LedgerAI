# ExtLinux Silent Boot Configuration Example

## Your Current Configuration

```
APPEND ${cbootargs} root=PARTUUID=3f576c6f-e47d-41d3-a26f-cbb44dd4d1a5 rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0 video=efifb:off console=...
```

## Parameters Explained

- `mminit_loglevel=4` - **This controls boot message verbosity** (4 = very verbose)
- `console=ttyTCU0,115200` - Serial console output
- `fbcon=map:0 video=efifb:off` - Already disabling some framebuffer
- `quiet` - **NOT present** (needed to suppress messages)
- `loglevel=0` - **NOT present** (needed for silent boot)

## Modified Configuration for Silent Boot

### Option 1: Minimal Messages (Recommended)

```
APPEND ${cbootargs} quiet root=PARTUUID=3f576c6f-e47d-41d3-a26f-cbb44dd4d1a5 rw rootwait rootfstype=ext4 mminit_loglevel=0 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0 video=efifb:off
```

**Changes:**
- Added `quiet` - suppresses most kernel messages
- Changed `mminit_loglevel=4` → `mminit_loglevel=0` - minimal init messages

### Option 2: Completely Silent Boot

```
APPEND ${cbootargs} quiet loglevel=0 root=PARTUUID=3f576c6f-e47d-41d3-a26f-cbb44dd4d1a5 rw rootwait rootfstype=ext4 mminit_loglevel=0 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0 video=efifb:off
```

**Changes:**
- Added `quiet` - suppresses most kernel messages
- Added `loglevel=0` - only emergency kernel messages
- Changed `mminit_loglevel=4` → `mminit_loglevel=0` - minimal init messages

### Option 3: Silent + No Console Output

If you want to completely disable console output:

```
APPEND ${cbootargs} quiet loglevel=0 root=PARTUUID=3f576c6f-e47d-41d3-a26f-cbb44dd4d1a5 rw rootwait rootfstype=ext4 mminit_loglevel=0 firmware_class.path=/etc/firmware fbcon=map:0 video=efifb:off console=tty1
```

**Changes:**
- Added `quiet` and `loglevel=0`
- Changed `mminit_loglevel=4` → `mminit_loglevel=0`
- Changed `console=ttyTCU0,115200` → `console=tty1` (redirects to virtual console instead of serial)

## How to Apply

1. **Edit the ExtLinux config:**
   ```bash
   sudo nano /boot/extlinux/extlinux.conf
   ```

2. **Find the APPEND line** and replace it with one of the options above

3. **Save and exit** (Ctrl+X, then Y, then Enter)

4. **Reboot to test:**
   ```bash
   sudo reboot
   ```

## Log Level Reference

**mminit_loglevel:**
- `0` - Minimal messages (recommended for silent boot)
- `1` - Errors only
- `2` - Warnings and errors
- `3` - Info, warnings, errors
- `4` - Verbose (your current setting)

**loglevel (kernel):**
- `0` - Emergency only
- `1` - Critical
- `2` - Error
- `3` - Warning (default)
- `4` - Notice
- `5` - Info
- `6` - Debug
- `7` - All messages

## Verification

After rebooting, check current kernel parameters:
```bash
cat /proc/cmdline
```

You should see `quiet` and `loglevel=0` (or `mminit_loglevel=0`) in the output.

## Reverting Changes

If you need to revert:
```bash
sudo nano /boot/extlinux/extlinux.conf
```

Change back to:
```
APPEND ${cbootargs} root=PARTUUID=3f576c6f-e47d-41d3-a26f-cbb44dd4d1a5 rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0 video=efifb:off
```

Or restore from backup if you created one:
```bash
sudo cp /boot/extlinux/extlinux.conf.bak /boot/extlinux/extlinux.conf
```
