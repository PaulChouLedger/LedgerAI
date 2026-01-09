# Quick Reference: Disabling Boot Screens on Jetson Orin NX

## Understanding What Appears During Boot

```
┌─────────────────────────────────────────┐
│ 1. UEFI Splash Screen (NVIDIA Logo)     │ ← UEFI Firmware (BEFORE kernel)
│    - Shows during hardware init         │   Requires UEFI source modification
│    - Cannot be disabled via ExtLinux    │
├─────────────────────────────────────────┤
│ 2. ExtLinux Boot Menu                   │ ← ExtLinux Bootloader
│    - Shows boot options                 │   Can disable with TIMEOUT 0
│    - Can be disabled                    │
├─────────────────────────────────────────┤
│ 3. Kernel Boot Messages                │ ← Linux Kernel
│    - Hardware initialization messages  │   Can disable with kernel parameters
│    - Can be suppressed                  │
├─────────────────────────────────────────┤
│ 4. Systemd Boot Messages                │ ← Systemd Init System
│    - Service startup messages           │   Can suppress with loglevel
│    - Can be suppressed                  │
└─────────────────────────────────────────┘
```

## What You Can Disable (Easy)

### ✅ ExtLinux Boot Menu
**File:** `/boot/extlinux/extlinux.conf`

Add or modify:
```
TIMEOUT 0
```

**Script:**
```bash
bash ~/LedgerAI/setup/scripts/disable_boot_menu_and_splash.sh
```

### ✅ Kernel Boot Messages
**File:** `/boot/extlinux/extlinux.conf`

Modify APPEND line:
```
APPEND ${cbootargs} quiet loglevel=0 mminit_loglevel=0 ...
```

**Script:**
```bash
bash ~/LedgerAI/setup/scripts/disable_boot_menu_and_splash.sh
```

### ✅ Systemd Boot Messages
Already handled by `loglevel=0` in kernel parameters.

## What Requires UEFI Modification (Advanced)

### ❌ UEFI Splash Screen (NVIDIA Logo)

**Why it can't be disabled via ExtLinux:**
- Appears **BEFORE** the kernel loads
- Part of UEFI firmware, not Linux
- Kernel parameters have no effect on it

**To disable (requires UEFI source modification):**

**Simplest Method:**
1. Get UEFI source code from NVIDIA (edk2-nvidia)
2. Edit `edk2-nvidia/Platform/NVIDIA/Jetson/Jetson.defconfig`
3. Add line: `# CONFIG_LOGO is not set`
4. Rebuild UEFI firmware
5. Flash new firmware to device

**Alternative:**
- Use **miniUEFI** which already has the logo disabled
- Or modify `DisplaySystemAndHotkeyInformation()` function in source code

**⚠️ WARNING:** This can brick your device if done incorrectly!

## Quick Solution Script

Run this to disable everything that can be disabled without UEFI modification:

```bash
cd ~/LedgerAI
bash setup/scripts/disable_boot_menu_and_splash.sh
```

This will:
- ✅ Disable ExtLinux boot menu (TIMEOUT 0)
- ✅ Suppress kernel boot messages (quiet loglevel=0)
- ✅ Suppress init messages (mminit_loglevel=0)
- ❌ **Cannot** disable UEFI splash screen (requires UEFI modification)

## Expected Results

**After running the script:**
- ✅ No ExtLinux boot menu (boots immediately)
- ✅ No kernel boot messages
- ✅ No systemd boot messages
- ❌ **UEFI splash screen will still appear** (NVIDIA logo during early boot)

**The UEFI splash screen will still show because it's part of the firmware and appears before Linux even starts.**

## If You Must Disable UEFI Splash Screen

See detailed instructions in:
- `docs/JETSON_UEFI_BOOT_CONFIGURATION.md` - Method 1: Modify UEFI Source Code

**Requirements:**
- UEFI source code
- EDK2 build environment
- Knowledge of firmware flashing
- Recovery method ready (USB recovery mode)

## Summary

| Component | Can Disable? | Method | Difficulty |
|-----------|-------------|--------|------------|
| ExtLinux Boot Menu | ✅ Yes | `TIMEOUT 0` | Easy |
| Kernel Messages | ✅ Yes | `quiet loglevel=0` | Easy |
| Init Messages | ✅ Yes | `mminit_loglevel=0` | Easy |
| UEFI Splash Screen | ❌ No (via ExtLinux) | UEFI source modification | Advanced |

**Bottom line:** You can disable the boot menu and all messages, but the UEFI splash screen requires firmware modification.
