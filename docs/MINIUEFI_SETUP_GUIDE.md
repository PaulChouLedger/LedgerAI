# miniUEFI Setup Guide for Jetson Orin NX

**Official NVIDIA Documentation:** [UEFI Adaptation Guide - miniUEFI Support](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#miniuefi-support)

## What is miniUEFI?

**miniUEFI** is a minimal UEFI configuration for Jetson devices that:
- ✅ **Logo/splash screen already disabled** (no NVIDIA logo)
- ✅ **Faster boot time** (minimal configuration)
- ✅ **Smaller binary size**
- ✅ **Optimized for eMMC boot**
- ✅ **Security via encrypted load targets**
- ✅ **UEFI Secure boot support**

**Note:** Currently available for **Jetson AGX Orin series**. Check NVIDIA documentation for Jetson Orin NX support.

## Prerequisites

1. **UEFI build environment** (EDK2 Docker container)
2. **NVIDIA UEFI source code** (edk2-nvidia)
3. **Jetson Linux BSP** (Linux_for_Tegra directory)
4. **Device in recovery mode** for flashing

## Step-by-Step Setup

### Step 1: Get UEFI Source Code

```bash
# Clone NVIDIA UEFI repository
git clone https://github.com/NVIDIA/edk2-nvidia.git
cd edk2-nvidia
```

### Step 2: Set Up Build Environment

Ensure you have the EDK2 Docker build environment set up. NVIDIA provides Docker containers for building UEFI.

```bash
# Check if edk2_docker is available
which edk2_docker

# If not available, follow NVIDIA's build environment setup
# Refer to: Building UEFI for NVIDIA Platforms
```

### Step 3: Build miniUEFI

```bash
# Navigate to nvidia-uefi directory (or your build directory)
cd nvidia-uefi

# Build miniUEFI using the JetsonMinimal build script
edk2_docker edk2-nvidia/Platform/NVIDIA/JetsonMinimal/build.sh
```

**Build Output:**
- `images/uefi_JetsonMinimal_DEBUG.bin` - Debug version
- `images/uefi_JetsonMinimal_RELEASE.bin` - Release version (use this for production)

### Step 4: Prepare for Flashing

```bash
# Navigate to your Jetson Linux BSP directory
cd /path/to/Linux_for_Tegra

# Backup original UEFI binary (recommended)
cp bootloader/uefi_jetson.bin bootloader/uefi_jetson.bin.backup

# Copy miniUEFI binary to replace the standard UEFI
cp /path/to/images/uefi_JetsonMinimal_RELEASE.bin bootloader/uefi_jetson.bin
```

### Step 5: Put Device in Recovery Mode

1. **Power off** your Jetson device
2. **Press and hold** the Recovery button
3. **Press and release** the Reset button (while holding Recovery)
4. **Release** the Recovery button after 2 seconds
5. Device should now be in recovery mode

**Verify recovery mode:**
```bash
# On your host machine, check if device is detected
lsusb | grep -i nvidia
```

You should see something like:
```
Bus 001 Device 005: ID 0955:7f21 NVIDIA Corp.
```

### Step 6: Flash miniUEFI to Device

```bash
# Navigate to Linux_for_Tegra directory
cd /path/to/Linux_for_Tegra

# Flash with eMMC boot order (recommended for miniUEFI)
sudo ADDITIONAL_DTB_OVERLAY="BootOrderEmmc.dtbo" ./flash.sh jetson-agx-orin-devkit internal

# For Jetson Orin NX (if supported):
# sudo ADDITIONAL_DTB_OVERLAY="BootOrderEmmc.dtbo" ./flash.sh jetson-orin-nx-devkit internal
```

**Alternative boot order options:**
- `BootOrderEmmc.dtbo` - eMMC (recommended)
- `BootOrderNvme.dtbo` - NVMe
- `BootOrderUsb.dtbo` - USB
- `BootOrderSd.dtbo` - SD card

### Step 7: Verify miniUEFI

After flashing completes:

1. **Reboot the device**
2. **Observe boot sequence:**
   - ✅ No NVIDIA logo splash screen should appear
   - ✅ Boot should be faster
   - ✅ System should boot directly to ExtLinux/kernel

3. **Check UEFI version (optional):**
   ```bash
   # Boot into UEFI menu (press ESC during boot)
   # Check system firmware version displayed
   ```

## Troubleshooting

### Problem: Build fails

**Solution:**
- Ensure EDK2 Docker environment is properly set up
- Check NVIDIA documentation for build requirements
- Verify you have the correct branch/tag of edk2-nvidia

### Problem: Device won't boot after flashing

**Solution:**
1. **Recovery mode:**
   - Put device back in recovery mode
   - Flash original UEFI binary:
     ```bash
     cp bootloader/uefi_jetson.bin.backup bootloader/uefi_jetson.bin
     sudo ./flash.sh jetson-agx-orin-devkit internal
     ```

2. **Check boot order:**
   - Ensure boot device is properly configured
   - Try different boot order DTBO files

### Problem: miniUEFI not available for my device

**Solution:**
- Check NVIDIA documentation for device support
- For Jetson Orin NX, you may need to:
  1. Use the `# CONFIG_LOGO is not set` method instead
  2. Wait for NVIDIA to add support
  3. Build custom minimal UEFI configuration

### Problem: Boot is slower than expected

**Solution:**
- Ensure you're using the RELEASE binary (not DEBUG)
- Check that boot order is optimized (eMMC recommended)
- Verify ExtLinux configuration is optimized

## Comparison: Standard UEFI vs miniUEFI

| Feature | Standard UEFI | miniUEFI |
|---------|---------------|----------|
| Logo/Splash Screen | ✅ Enabled | ❌ Disabled |
| Boot Time | Normal | ⚡ Faster |
| Binary Size | Larger | 📦 Smaller |
| Features | Full | Minimal |
| Security | Standard | Encrypted load targets |
| Persistent Variables | ✅ Supported | ❌ Not supported |
| UEFI Shell | ✅ Available | ❌ Disabled |
| Best For | Development | Production |

## Reverting to Standard UEFI

If you need to revert to standard UEFI:

```bash
# Restore backup
cp bootloader/uefi_jetson.bin.backup bootloader/uefi_jetson.bin

# Or re-flash with standard BSP
# Download fresh Jetson Linux BSP and flash normally
sudo ./flash.sh jetson-agx-orin-devkit internal
```

## Additional Configuration

### Custom Boot Order

After flashing miniUEFI, you can customize boot order in the UEFI menu:
1. Press **ESCAPE** during boot
2. Navigate to **Boot Maintenance Manager**
3. Select **Boot Options** → **Change Boot Order**

### Boot Mode Selection

miniUEFI uses L4tLauncher by default. Boot modes:
- **ExtLinux** - Normal kernel boot (default)
- **GRUB** - If GRUB is installed
- **Recovery** - Recovery kernel

## Summary

**Quick Setup:**
```bash
# 1. Build miniUEFI
cd nvidia-uefi
edk2_docker edk2-nvidia/Platform/NVIDIA/JetsonMinimal/build.sh

# 2. Replace UEFI binary
cp images/uefi_JetsonMinimal_RELEASE.bin Linux_for_Tegra/bootloader/uefi_jetson.bin

# 3. Flash device
sudo ADDITIONAL_DTB_OVERLAY="BootOrderEmmc.dtbo" ./flash.sh jetson-agx-orin-devkit internal
```

**Benefits:**
- ✅ No splash screen
- ✅ Faster boot
- ✅ Smaller binary
- ✅ Production-ready

**References:**
- [NVIDIA UEFI Documentation](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#sd-bootloader-uefi)
- [NVIDIA/edk2-nvidia GitHub](https://github.com/NVIDIA/edk2-nvidia)
