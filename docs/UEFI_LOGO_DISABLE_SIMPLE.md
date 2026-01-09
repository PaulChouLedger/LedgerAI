# Simple Method to Disable UEFI Logo on Jetson Orin NX

**Official NVIDIA Documentation:** [UEFI Adaptation Guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#sd-bootloader-uefi)

## Quick Method

The simplest way to disable the UEFI splash screen (NVIDIA logo) is to add a single line to the UEFI configuration file.

## Steps

### 1. Get UEFI Source Code

Clone or download the NVIDIA UEFI source code (edk2-nvidia):
```bash
git clone https://github.com/NVIDIA/edk2-nvidia.git
cd edk2-nvidia
```

### 2. Edit Jetson Configuration

Navigate to the Jetson platform configuration:
```bash
cd Platform/NVIDIA/Jetson
nano Jetson.defconfig
```

### 3. Add Logo Disable Line

Add this line to the file:
```
# CONFIG_LOGO is not set
```

**Example:**
```bash
# Before:
CONFIG_SOME_OPTION=y
CONFIG_ANOTHER_OPTION=y

# After:
CONFIG_SOME_OPTION=y
CONFIG_ANOTHER_OPTION=y
# CONFIG_LOGO is not set
```

### 4. Rebuild UEFI Firmware

```bash
# Return to build directory
cd ../../..

# Build UEFI firmware
make -j$(nproc)
```

### 5. Flash New Firmware

**⚠️ WARNING: This can brick your device if done incorrectly!**

Follow NVIDIA's official flashing procedures:
```bash
sudo ./flash.sh jetson-orin-nx-devkit internal
```

## Alternative: Use miniUEFI (Recommended)

**miniUEFI** is a minimal UEFI implementation that already has the logo disabled and boots faster. This is the recommended approach.

**Building miniUEFI:**
```bash
cd nvidia-uefi
edk2_docker edk2-nvidia/Platform/NVIDIA/JetsonMinimal/build.sh
```

**Flashing miniUEFI:**
```bash
# Replace UEFI binary
cp images/uefi_JetsonMinimal_RELEASE.bin Linux_for_Tegra/bootloader/uefi_jetson.bin

# Flash device
sudo ADDITIONAL_DTB_OVERLAY="BootOrderEmmc.dtbo" ./flash.sh jetson-agx-orin-devkit internal
```

**miniUEFI Features:**
- Logo/splash screen already disabled
- Minimal configuration (smaller, faster boot)
- Optimized for eMMC boot
- Security via encrypted load targets

**Reference:** [NVIDIA UEFI Documentation - miniUEFI Support](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#miniuefi-support)

**Note:** miniUEFI is currently available for Jetson AGX Orin series. Check NVIDIA documentation for other platforms.

## Verification

After flashing:
1. Reboot your device
2. The NVIDIA logo splash screen should no longer appear
3. Boot should proceed directly to ExtLinux/kernel

## Recovery

If something goes wrong:
1. Use USB recovery mode
2. Flash original firmware from NVIDIA
3. Follow NVIDIA's recovery procedures

## Summary

**Method 1: Configuration File (Simplest)**
- Add `# CONFIG_LOGO is not set` to `edk2-nvidia/Platform/NVIDIA/Jetson/Jetson.defconfig`

**Method 2: Use miniUEFI (Recommended)**
- Build miniUEFI which already has logo disabled
- Faster boot, minimal configuration

**Method 3: Replace Logo Files**
- Replace BMP files in `edk2-nvidia/Silicon/NVIDIA/Assets/`
- Update `Platform/NVIDIA/NVIDIA.fvmain.fdf.inc` if needed

**Official Documentation:**
- [NVIDIA UEFI Adaptation Guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Bootloader/UEFI.html#sd-bootloader-uefi)
