# Building Chatterbox Container with Jetson-Containers

## Problem

When building the chatterbox container using jetson-containers, the build fails during PyTorch compilation from source:

```
ninja: build stopped: subcommand failed.
The command '/bin/sh -c /tmp/pytorch/install.sh || /tmp/pytorch/build.sh' returned a non-zero code: 1
```

This happens because jetson-containers tries to build PyTorch from source, which:
- Takes 2+ hours on Jetson devices
- Requires significant memory (often fails on devices with <16GB RAM)
- Can fail during NCCL compilation
- Is unnecessary since pre-built PyTorch images are available

## Solution: Use Pre-built PyTorch Image

The `Dockerfile` already uses a pre-built PyTorch image (`dustynv/pytorch:2.6-r36.4.0-cu128-24.04`), so you should **build directly with Docker** instead of using jetson-containers' PyTorch build process.

### Option 1: Use Standard Docker Build (Recommended)

```bash
cd chatterbox-container
./build.sh
```

Or manually:

```bash
cd chatterbox-container
docker build --network=host --shm-size=8g -t chatterbox-tts:latest .
```

### Option 2: Use Jetson-Containers with Pre-built Base

If you must use jetson-containers, you can modify the build to use a pre-built PyTorch base:

1. First, pull the pre-built PyTorch image:
```bash
docker pull dustynv/pytorch:2.6-r36.4.0-cu128-24.04
```

2. Tag it for jetson-containers:
```bash
docker tag dustynv/pytorch:2.6-r36.4.0-cu128-24.04 \
  chatterbox-tts:r36.4.tegra-aarch64-cu126-22.04-torch
```

3. Then build your container on top of it:
```bash
cd chatterbox-container
docker build --network=host --shm-size=8g \
  --build-arg BASE_IMAGE=chatterbox-tts:r36.4.tegra-aarch64-cu126-22.04-torch \
  -t chatterbox-tts:latest .
```

### Option 3: Skip PyTorch Build in Jetson-Containers

If you're using jetson-containers to build, you can skip the PyTorch build step by:

1. Using `FORCE_BUILD=off` to use pre-built images when available
2. Or building the container directly without going through jetson-containers' PyTorch build process

## Why This Works

The `Dockerfile` uses `FROM dustynv/pytorch:2.6-r36.4.0-cu128-24.04`, which is a pre-built image that already has:
- PyTorch 2.6 with CUDA 12.8 support
- All necessary CUDA libraries
- Optimized for Jetson devices

Building from source is only necessary if you need:
- A custom PyTorch build with specific flags
- A version not available as a pre-built image
- Debugging PyTorch itself

For most use cases, the pre-built image is sufficient and much faster.

## Troubleshooting

### Build Still Fails

1. **Check disk space**: Container build needs ~10-15GB free space
   ```bash
   df -h
   ```

2. **Increase shared memory**: Already set to 8GB in build script, but you can increase:
   ```bash
   docker build --shm-size=16g ...
   ```

3. **Skip model download**: Models can be downloaded at runtime:
   ```bash
   SKIP_MODEL_DOWNLOAD=1 ./build.sh
   ```

4. **Check Docker logs**: If build fails, check the last few lines:
   ```bash
   docker build ... 2>&1 | tail -50
   ```

### Memory Issues

If you encounter OOM (Out of Memory) errors:

1. **Increase swap space** on Jetson:
   ```bash
   sudo systemctl disable nvzramconfig
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Build with fewer parallel jobs** (if building from source):
   ```bash
   export MAX_JOBS=2
   ```

### Network Issues

If model download fails:

1. **Skip model download during build**:
   ```bash
   SKIP_MODEL_DOWNLOAD=1 ./build.sh
   ```

2. Models will download automatically on first use (requires internet)

## Quick Start

```bash
# Build the container
cd chatterbox-container
./build.sh

# Run the container
docker run -d --name chatterbox-tts --runtime=nvidia --network=host \
  -v $(pwd)/../assets/voice_samples:/app/voice_samples \
  -v $(pwd)/../data/voice_cache:/app/voice_cache \
  chatterbox-tts:latest

# Test it
curl http://localhost:11437/health
```

## Summary

**Don't use jetson-containers to build PyTorch from source** - use the pre-built image instead. The `Dockerfile` is already configured correctly; just build it with standard Docker.
