# Fixing GDRCopy Build Error in Jetson Containers

## Problem

When building jetson-containers images (specifically `unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard`), you may encounter this error:

```
install: cannot stat '/usr/lib/dkms/common.postinst': No such file or directory
make[1]: *** [Makefile:20: install] Error 1
```

This occurs because GDRCopy requires `dkms` (Dynamic Kernel Module Support) to build, but it's not installed in the Docker container.

## Solutions

### Solution 1: Disable GDRCopy (Recommended if not needed)

GDRCopy is typically only needed for high-performance GPU-to-GPU communication in multi-GPU setups. For most single-GPU Jetson use cases, it's not required.

**Option A: Build without GDRCopy**

When building the unsloth container, disable GDRCopy:

```bash
cd ~/jetson-containers
jetson-containers build --tag unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard \
  --build-arg WITH_GDRCOPY=0 \
  unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard
```

**Option B: Modify the build script**

Edit the jetson-containers build configuration to exclude GDRCopy by default.

### Solution 2: Install dkms in the Base Image (If GDRCopy is required)

If you actually need GDRCopy, you need to patch the jetson-containers build process to install `dkms` before building GDRCopy.

1. **Locate the GDRCopy build script:**

```bash
cd ~/jetson-containers
find . -name "*gdrcopy*" -type f
```

2. **Find the Dockerfile or build script that needs modification:**

The issue is in the cudastack Dockerfile. You'll need to modify:
- `packages/cuda/cudastack/Dockerfile` - to install dkms before GDRCopy
- Or `packages/cuda/cudastack/build/build_gdrcopy.sh` - to install dkms as part of the build

3. **Patch the build script:**

Add dkms installation before GDRCopy build. Create a patch file or modify directly:

```bash
# Edit the build script
cd ~/jetson-containers/packages/cuda/cudastack/build
# Or edit the install script
cd ~/jetson-containers/packages/cuda/cudastack/install
```

Add this before the GDRCopy build:

```bash
# Install dkms if not present
if ! command -v dkms &> /dev/null; then
    apt-get update && apt-get install -y dkms
fi
```

### Solution 3: Quick Fix - Patch jetson-containers Repository

Create a patch script to fix the issue:

```bash
#!/bin/bash
# fix_gdrcopy_dkms.sh

cd ~/jetson-containers

# Find and patch the GDRCopy build script
GDRCOPY_BUILD_SCRIPT=$(find . -path "*/build/build_gdrcopy.sh" -o -path "*/install/install_gdrcopy.sh" | head -1)

if [ -f "$GDRCOPY_BUILD_SCRIPT" ]; then
    echo "Patching $GDRCOPY_BUILD_SCRIPT"
    
    # Create backup
    cp "$GDRCOPY_BUILD_SCRIPT" "${GDRCOPY_BUILD_SCRIPT}.bak"
    
    # Add dkms installation at the beginning of the script
    sed -i '1a\apt-get update && apt-get install -y dkms || true' "$GDRCOPY_BUILD_SCRIPT"
    
    echo "Patch applied. Backup saved to ${GDRCOPY_BUILD_SCRIPT}.bak"
else
    echo "GDRCopy build script not found. Checking Dockerfile..."
    
    # Try to patch the Dockerfile instead
    DOCKERFILE=$(find . -path "*/cudastack/Dockerfile" | head -1)
    if [ -f "$DOCKERFILE" ]; then
        echo "Patching $DOCKERFILE"
        cp "$DOCKERFILE" "${DOCKERFILE}.bak"
        
        # Add dkms installation before GDRCopy build
        sed -i '/WITH_GDRCOPY.*1/i\RUN apt-get update \&\& apt-get install -y dkms || true' "$DOCKERFILE"
        
        echo "Patch applied to Dockerfile. Backup saved to ${DOCKERFILE}.bak"
    else
        echo "Could not find GDRCopy build files. Manual intervention required."
        exit 1
    fi
fi
```

Save this as `fix_gdrcopy_dkms.sh`, make it executable, and run it:

```bash
chmod +x fix_gdrcopy_dkms.sh
./fix_gdrcopy_dkms.sh
```

### Solution 4: Use Pre-built Image (Easiest)

If available, use a pre-built image instead of building from source:

```bash
docker pull dustynv/unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard
```

## Recommended Approach

For most LedgerAI/Aura use cases on Jetson:

1. **Disable GDRCopy** - It's not needed for single-GPU inference
2. **Build with `WITH_GDRCOPY=0`** flag
3. **Or use pre-built images** if available

## Verification

After applying a fix, rebuild:

```bash
cd ~/jetson-containers
jetson-containers build --tag unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard \
  unsloth:r36.4.tegra-aarch64-cu126-22.04-cudastack_standard
```

## Additional Notes

- GDRCopy is primarily used for GPU Direct RDMA in multi-GPU clusters
- For single Jetson device inference, it's typically unnecessary
- The dkms package is large and adds build time
- Consider whether you actually need the cudastack variant or if a simpler base image works

