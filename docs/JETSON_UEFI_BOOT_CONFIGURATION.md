# Jetson Orin NX UEFI Boot Configuration Guide

## ⚠️ Important: Jetson Bootloader Note

**Jetson devices typically use ExtLinux, NOT GRUB!**

If `/etc/default/grub` doesn't exist or is blank, your Jetson is likely using **ExtLinux**. The configuration file is usually at:
- `/boot/extlinux/extlinux.conf`

**Quick check:**
```bash
# Check for ExtLinux (Jetson default)
ls /boot/extlinux/extlinux.conf

# Check for GRUB
ls /etc/default/grub
```

**Recommended:** Use the automated script that auto-detects your bootloader:
```bash
bash ~/LedgerAI/setup/scripts/configure_silent_boot.sh
```

## Understanding UEFI Components

### What is UEFI?

**UEFI (Unified Extensible Firmware Interface)** is a modern replacement for the legacy BIOS firmware interface. On Jetson Orin NX, UEFI handles:

1. **Hardware initialization** - CPU, memory, storage, display
2. **Boot device selection** - Choosing which device to boot from
3. **Boot splash screen** - The NVIDIA logo displayed during boot
4. **Boot messages** - Text output during hardware initialization
5. **Boot menu access** - Hotkeys (ESC, F11) to enter setup

### UEFI Components on Jetson Orin NX

```
┌─────────────────────────────────────────┐
│         UEFI Firmware (Flash)           │
├─────────────────────────────────────────┤
│ 1. Boot Manager                         │
│    - Device selection                   │
│    - Boot order                         │
│                                         │
│ 2. Display Driver                       │
│    - Splash screen rendering           │
│    - Text console output               │
│                                         │
│ 3. Platform Initialization              │
│    - Hardware setup                    │
│    - Boot messages                     │
│                                         │
│ 4. Boot Services                        │
│    - Handoff to OS loader              │
│    - Memory management                 │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Linux Kernel (GRUB/EFI)            │
├─────────────────────────────────────────┤
│ - Kernel boot messages                  │
│ - Framebuffer console                   │
│ - System initialization                 │
└─────────────────────────────────────────┘
```

### Key UEFI Functions (Source Code)

The UEFI firmware source code contains these critical functions:

1. **`DisplaySystemAndHotkeyInformation()`**
   - Displays the boot splash screen (NVIDIA logo)
   - Shows hotkey information (ESC, F11)
   - Location: UEFI source code

2. **`PlatformRegisterOptionsAndKeys()`**
   - Registers hotkeys for UEFI menu access
   - Controls ESC and F11 key handling

3. **Boot Message Functions**
   - Various `Print()` functions output boot messages
   - Hardware initialization status messages

## Disabling Boot Splash Screen

### Method 1: Disable Logo in UEFI Configuration (Simplest Method)

**Prerequisites:**
- UEFI source code from NVIDIA (edk2-nvidia)
- UEFI build environment (EDK2)
- Ability to flash new firmware

**Steps:**

1. **Navigate to the Jetson configuration file:**
   ```bash
   cd edk2-nvidia/Platform/NVIDIA/Jetson
   ```

2. **Edit Jetson.defconfig:**
   ```bash
   nano Jetson.defconfig
   ```

3. **Add the following line to disable the logo:**
   ```
   # CONFIG_LOGO is not set
   ```

4. **Rebuild UEFI firmware:**
   ```bash
   # In UEFI build directory
   make -j$(nproc)
   ```

5. **Flash the new firmware:**
   ```bash
   # WARNING: This can brick your device if done incorrectly!
   # Follow NVIDIA's official flashing procedures
   sudo ./flash.sh jetson-orin-nx-devkit internal
   ```

### Method 1b: Use miniUEFI (Recommended for Minimal Boot)

**miniUEFI** is a minimal UEFI configuration that already has the logo disabled and boots faster. This is the recommended approach if you want a minimal boot experience.

**Building miniUEFI:**
```bash
cd nvidia-uefi
edk2_docker edk2-nvidia/Platform/NVIDIA/JetsonMinimal/build.sh
```

The generated binaries (`uefi_JetsonMinimal_DEBUG.bin` and `uefi_JetsonMinimal_RELEASE.bin`) will be in the `images` directory.

**Flashing miniUEFI:**
```bash
# Replace the UEFI binary
cp images/uefi_JetsonMinimal_RELEASE.bin Linux_for_Tegra/bootloader/uefi_jetson.bin

# Flash the device
sudo ADDITIONAL_DTB_OVERLAY="BootOrderEmmc.dtbo" ./flash.sh jetson-agx-orin-devkit internal
```

**miniUEFI Features:**
- Minimal configuration (smaller, faster boot)
- Logo/splash screen disabled
- Optimized for eMMC boot
- Security via encrypted load targets
- UEFI Secure boot support

**Note:** miniUEFI is currently available for Jetson AGX Orin series. Check NVIDIA documentation for other platforms.

**Reference:** [NVIDIA UEFI Documentation - miniUEFI Support](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#miniuefi-support)

**📖 Detailed Setup Guide:** See [miniUEFI Setup Guide](MINIUEFI_SETUP_GUIDE.md) for complete step-by-step instructions.

### Method 1c: Customize Logo Files (Replace with Custom Logo)

If you want to replace the NVIDIA logo with your own custom logo:

1. **Locate logo files:**
   ```bash
   cd edk2-nvidia/Silicon/NVIDIA/Assets
   ```
   
   Logo files:
   - `nvidiagray480.bmp` - 480p logo
   - `nvidiagray720.bmp` - 720p logo
   - `nvidiagray1080.bmp` - 1080p logo

2. **Replace with your custom BMP files:**
   - Use same resolutions (480p, 720p, 1080p)
   - Ensure file format is BMP
   - Keep same filenames or update `NVIDIA.fvmain.fdf.inc`

3. **Update logo configuration (if filenames changed):**
   ```bash
   nano Platform/NVIDIA/NVIDIA.fvmain.fdf.inc
   ```
   
   Update the section:
   ```
   FILE FREEFORM = gNVIDIAPlatformLogoGuid {
     SECTION RAW = Silicon/NVIDIA/Assets/nvidiagray480.bmp
     SECTION RAW = Silicon/NVIDIA/Assets/nvidiagray720.bmp
     SECTION RAW = Silicon/NVIDIA/Assets/nvidiagray1080.bmp
   }
   ```

4. **Important:** Ensure the UEFI binary with logo files doesn't exceed the UEFI partition size (3.5MB). Check partition XML files in `./bootloader/generic/cfg/` directory.

5. **Rebuild and flash as above**

**Reference:** [NVIDIA UEFI Documentation - Customized Logo](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#customized-logo)

### Method 1d: Modify UEFI Source Code (Alternative Method)

If you prefer to modify the source code directly:

1. **Locate the splash screen function:**
   ```c
   // In UEFI source code, find:
   DisplaySystemAndHotkeyInformation()
   ```

2. **Comment out or remove the call:**
   ```c
   // Original:
   DisplaySystemAndHotkeyInformation();
   
   // Modified:
   // DisplaySystemAndHotkeyInformation();  // Disabled splash screen
   ```

3. **Rebuild and flash as above**

### Method 2: Kernel-Level Suppression (Easier, Less Permanent)

**Important:** Jetson devices typically use **ExtLinux** instead of GRUB. Check which bootloader you have first:

```bash
# Check for ExtLinux (common on Jetson)
ls /boot/extlinux/extlinux.conf

# Check for GRUB
ls /etc/default/grub
```

#### Option A: Using ExtLinux (Jetson Default)

1. **Edit ExtLinux configuration:**
   ```bash
   sudo nano /boot/extlinux/extlinux.conf
   ```

2. **Find the APPEND line and modify it:**
   ```bash
   # Original might look like:
   APPEND ${cbootargs} quiet root=/dev/mmcblk0p1 rw
   
   # Modified for silent boot:
   APPEND ${cbootargs} quiet loglevel=0 root=/dev/mmcblk0p1 rw
   
   # To disable splash screen:
   APPEND ${cbootargs} quiet loglevel=0 root=/dev/mmcblk0p1 rw
   ```

3. **Reboot to test:**
   ```bash
   sudo reboot
   ```

#### Option B: Using GRUB (if installed)

1. **Modify GRUB configuration:**
   ```bash
   sudo nano /etc/default/grub
   ```

2. **Add/Modify these lines:**
   ```bash
   # Disable boot messages
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
   
   # For complete silence (no messages at all):
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash loglevel=0"
   
   # To disable framebuffer console:
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash loglevel=0 vga=normal"
   ```

3. **Update GRUB:**
   ```bash
   sudo update-grub
   ```

4. **Reboot to test:**
   ```bash
   sudo reboot
   ```

#### Option C: Automated Script (Recommended)

Use the provided script that auto-detects your bootloader:

```bash
cd ~/LedgerAI
bash setup/scripts/configure_silent_boot.sh
```

This script will:
- Detect whether you're using GRUB or ExtLinux
- Show current kernel parameters
- Allow you to choose configuration options
- Automatically update the correct configuration file

## Disabling Boot Messages

### Method 1: Kernel Command Line Parameters

**For ExtLinux (Jetson default):**

```bash
sudo nano /boot/extlinux/extlinux.conf
```

Find the `APPEND` line and add parameters:
```bash
APPEND ${cbootargs} quiet loglevel=0 root=/dev/mmcblk0p1 rw
```

**For GRUB (if installed):**

```bash
sudo nano /etc/default/grub
```

Add kernel parameters:
```bash
# Minimal output
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash loglevel=0"

# Or completely silent
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash loglevel=0 console=tty1"
```

**Available log levels:**
- `loglevel=0` - Only emergency messages
- `loglevel=1` - Only critical messages
- `loglevel=2` - Only error messages
- `loglevel=3` - Warnings and above (default)

**Update and reboot:**
```bash
# For ExtLinux - changes take effect on next boot
sudo reboot

# For GRUB
sudo update-grub
sudo reboot
```

### Method 2: Disable Framebuffer Console

**Modify kernel configuration:**
```bash
sudo nano /etc/default/grub
```

**Add:**
```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash loglevel=0 vga=normal"
```

**Or disable framebuffer entirely:**
```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash loglevel=0 nomodeset"
```

### Method 3: Modify Kernel Configuration (Advanced)

If you're building a custom kernel:

1. **Disable framebuffer console:**
   ```bash
   # In kernel config (.config file):
   # CONFIG_FRAMEBUFFER_CONSOLE=n
   ```

2. **Reduce console verbosity:**
   ```bash
   # CONFIG_CONSOLE_LOGLEVEL_DEFAULT=0
   ```

## Disabling ExtLinux Boot Menu

The ExtLinux boot menu can be disabled by setting the timeout to 0:

```bash
sudo nano /boot/extlinux/extlinux.conf
```

Find or add the `TIMEOUT` line:
```
TIMEOUT 0
```

**Note:** The timeout value is in tenths of a second:
- `TIMEOUT 0` = Boot immediately (no menu)
- `TIMEOUT 30` = 3 second delay
- `TIMEOUT 100` = 10 second delay

**Or use the automated script:**
```bash
bash ~/LedgerAI/setup/scripts/disable_boot_menu_and_splash.sh
```

This script will:
- Set `TIMEOUT 0` to disable boot menu
- Configure silent boot kernel parameters
- Set `mminit_loglevel=0` for minimal init messages

## Disabling UEFI Boot Menu Hotkeys

To prevent users from accessing UEFI setup during boot:

**In UEFI source code:**
```c
// Find PlatformRegisterOptionsAndKeys()
// Comment out hotkey registrations:

// Original:
RegisterHotkey(SCAN_ESC, NULL, EnterSetupMenu);
RegisterHotkey(SCAN_F11, NULL, BootDeviceMenu);

// Modified:
// RegisterHotkey(SCAN_ESC, NULL, EnterSetupMenu);  // Disabled
// RegisterHotkey(SCAN_F11, NULL, BootDeviceMenu);   // Disabled
```

## ⚠️ Important: UEFI Splash Screen Limitation

**The UEFI splash screen (NVIDIA logo) appears BEFORE the kernel loads.**

This means:
- ❌ Kernel parameters (`quiet`, `loglevel=0`) **cannot** disable the UEFI splash screen
- ❌ ExtLinux configuration **cannot** disable the UEFI splash screen
- ✅ Only UEFI firmware modification can disable it

**To disable UEFI splash screen, you must:**
1. Modify the UEFI configuration (add `# CONFIG_LOGO is not set` to `Jetson.defconfig`)
2. Rebuild the UEFI firmware
3. Flash the new firmware to your device

**Simplest method:** Add `# CONFIG_LOGO is not set` to `edk2-nvidia/Platform/NVIDIA/Jetson/Jetson.defconfig`

**Alternative:** Use miniUEFI which already has the logo disabled.

**This is an advanced operation that can brick your device if done incorrectly!**

See "Method 1: Disable Logo in UEFI Configuration" section above for details.

## Quick Reference: Kernel Parameters

| Parameter | Effect |
|-----------|--------|
| `quiet` | Suppresses most kernel messages |
| `splash` | Shows splash screen (remove to disable) |
| `loglevel=0` | Only emergency messages |
| `loglevel=1` | Only critical messages |
| `loglevel=2` | Only errors |
| `loglevel=3` | Warnings and above (default) |
| `vga=normal` | Disable framebuffer |
| `nomodeset` | Disable kernel mode setting |
| `console=tty1` | Redirect console to tty1 only |

## Complete Silent Boot Configuration

For a completely silent boot (no splash, no messages):

**For ExtLinux (Jetson default):**
```bash
sudo nano /boot/extlinux/extlinux.conf

# Modify APPEND line to:
APPEND ${cbootargs} quiet loglevel=0 root=/dev/mmcblk0p1 rw

# Save and reboot
sudo reboot
```

**For GRUB (if installed):**
```bash
# Edit GRUB
sudo nano /etc/default/grub

# Set to:
GRUB_CMDLINE_LINUX_DEFAULT="quiet loglevel=0 vga=normal"
GRUB_CMDLINE_LINUX="quiet loglevel=0 vga=normal"

# Also disable GRUB menu timeout (optional):
GRUB_TIMEOUT=0
GRUB_HIDDEN_TIMEOUT=0
GRUB_HIDDEN_TIMEOUT_QUIET=true

# Update GRUB
sudo update-grub
```

**Or use the automated script:**
```bash
bash ~/LedgerAI/setup/scripts/configure_silent_boot.sh
# Select option 2 (Silent boot)
```

## Verifying Changes

**Check current kernel parameters:**
```bash
cat /proc/cmdline
```

**Check boot messages:**
```bash
# View last boot messages
dmesg | head -50

# Check systemd boot messages
journalctl -b | head -50
```

**Test boot sequence:**
```bash
# Reboot and observe
sudo reboot
```

## Important Warnings

⚠️ **UEFI Firmware Modification:**
- **CAN BRICK YOUR DEVICE** if done incorrectly
- Always backup original firmware
- Test on development device first
- Follow NVIDIA's official procedures
- Have recovery method ready (USB recovery mode)

⚠️ **Kernel Parameter Changes:**
- Some parameters can prevent boot
- Test in controlled environment first
- Keep backup of working configuration
- Can always boot from recovery USB

⚠️ **Silent Boot:**
- Makes debugging difficult
- May hide important error messages
- Consider keeping minimal logging for production

## Recovery Procedures

**If system won't boot after changes:**

1. **Boot from USB recovery image:**
   - Use NVIDIA's recovery image
   - Flash original firmware/kernel

2. **Access GRUB menu:**
   - Hold Shift during boot
   - Select recovery mode
   - Edit kernel parameters manually

3. **Emergency shell:**
   - Boot to initramfs shell
   - Mount root filesystem
   - Edit `/etc/default/grub` manually

## Additional Resources

- **[NVIDIA UEFI Adaptation Guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#sd-bootloader-uefi)** - Official NVIDIA documentation
- **[NVIDIA/edk2-nvidia](https://github.com/NVIDIA/edk2-nvidia)** - UEFI source code repository
- **EDK2 UEFI Development Kit Documentation**
- **NVIDIA Developer Forums** - Jetson section
- **UEFI Specification** (uefi.org)

## Summary

**For Boot Splash Screen:**
- **Easiest (Kernel level):** Remove `splash` from kernel parameters (doesn't affect UEFI logo)
- **Permanent (UEFI level):** Add `# CONFIG_LOGO is not set` to `Jetson.defconfig` and rebuild UEFI
- **Alternative:** Use miniUEFI which already has logo disabled

**For Boot Messages:**
- **Easiest:** Add `quiet loglevel=0` to ExtLinux/GRUB
- **Complete:** Disable framebuffer console

**For Boot Menu:**
- **ExtLinux:** Set `TIMEOUT 0` in `/boot/extlinux/extlinux.conf`
- **GRUB:** Set `GRUB_TIMEOUT=0` in `/etc/default/grub`

**Recommended Approach:**
1. Start with ExtLinux/GRUB parameter changes (safe, reversible) - disables boot menu and messages
2. Test thoroughly
3. If you need to disable UEFI splash screen, add `# CONFIG_LOGO is not set` to `Jetson.defconfig` and rebuild UEFI
4. Or use miniUEFI for a minimal boot experience
