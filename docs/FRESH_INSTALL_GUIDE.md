# Aura Fresh Installation Guide

Complete step-by-step guide for installing Aura on a fresh system (new drive, clean OS installation).

## Quick Start (For Experienced Users - Jetson Orin NX)

```bash
# 1. Install Jetson dependencies (recommended for Jetson devices)
cd ~
git clone https://github.com/PaulChouLedger/LedgerAI.git
cd LedgerAI
bash setup/install_jetson.sh
# This installs git, Docker, and Jetson Container Tools automatically

# 2. Set up Python environment
python3 -m venv aura-env
source aura-env/bin/activate
pip install -r aura-control/requirements/requirements.txt

# 3. Configure environment
cp .env.example .env
./aura_config.sh  # Configure API keys

# 4. Build Docker containers (requires Jetson Container Tools from step 1)
cd setup && docker compose build

# 5. Install systemd service
sudo bash setup/scripts/install_aura_service.sh

# 6. Start Aura
sudo systemctl start aura.service
```

**Estimated time:** 30-60 minutes (depending on internet speed and system performance)

**Note**: For Jetson Orin NX 16GB, use `install_jetson.sh` instead of `install_dependencies.sh` for Jetson-specific optimizations.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Setup](#system-setup)
3. [Repository Setup](#repository-setup)
4. [Environment Configuration (.env)](#environment-configuration-env)
5. [Docker Setup](#docker-setup)
6. [Systemd Services](#systemd-services)
7. [Hardware Configuration](#hardware-configuration)
8. [Testing & Verification](#testing--verification)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Primary Platform: NVIDIA Jetson Orin NX 16GB**
- **System**: NVIDIA Jetson Orin NX with 16GB RAM
- **Storage**: Minimum 32GB (64GB+ recommended for models and data)
- **USB microphone array**: XVF3800 USB 4 Mic Array (recommended)
- **Display**: Circular touch display (1080x1080 recommended)
- **Network**: WiFi or Ethernet connection

**Note**: This guide is optimized for Jetson Orin NX 16GB, but can be adapted for other Jetson devices (AGX Xavier, Xavier NX, Orin AGX) or Linux systems with GPU.

### Software Requirements

- **JetPack 6.4** (r36.4) - NVIDIA Jetson SDK
  - Ubuntu 24.04 base
  - CUDA 12.8+
  - cuDNN 9.x
  - TensorRT 10.x
- **Python 3.8+**
- **Docker & Docker Compose** (with NVIDIA Container Toolkit)
- **Git**
- **NVIDIA Container Toolkit** (for GPU access in containers)

---

## System Setup

### 1. Create User Account

```bash
# Create aura user (if not exists)
sudo useradd -m -s /bin/bash aura
sudo usermod -aG docker,sudo,audio,video aura

# Switch to aura user
sudo su - aura
```

### 2. Install Base System Dependencies

**Option A: Use Jetson-Specific Installer (Recommended)**

```bash
cd ~/LedgerAI
bash setup/install_jetson.sh
```

This script handles Jetson-specific dependencies including:
- CUDA libraries
- JetPack dependencies
- Python packages optimized for Jetson
- Audio libraries
- Display tools

**Option B: Manual Installation**

```bash
# Update package list
sudo apt-get update

# Install essential packages (including git)
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    nano \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    sox \
    libasound2-dev \
    pulseaudio \
    unclutter \
    xdotool \
    wmctrl \
    x11-xserver-utils \
    network-manager \
    polkit \
    python3-pyqt5
```

### 2.1. Install Jetson Container Tools (Required for Base Images)

The Docker containers use base images from `dusty-nv/jetson-containers`. Install the container tools:

```bash
# Clone the Jetson containers repository
cd ~
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers

# Install container tools
bash install.sh

# This installs:
# - NVIDIA Container Toolkit configuration
# - Jetson-optimized Docker runtime
# - Base image utilities (needed for dustynv/* images)
```

**Note**: This step is required before building Aura containers, as they depend on `dustynv/llama_cpp` and `dustynv/faster-whisper` base images.

### 3. Install Docker & Docker Compose (Jetson)

**For Jetson devices, use `docker.io` instead of Docker CE:**

```bash
# Install Docker (Jetson uses docker.io, not docker-ce)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version

# Log out and back in for group changes to take effect
# (Or run: newgrp docker)
```

**Install Jetson Container Tools (Required):**

```bash
# Install the container tools (provides dustynv/* base images)
cd ~
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

**Alternative: Use Jetson-specific installation script (Recommended)**

```bash
# Run Jetson-specific installer
cd ~/LedgerAI
bash setup/install_jetson.sh
```

This script handles Jetson-specific dependencies and configurations, including:
- Git installation
- JetPack 6.4 compatibility checks
- NVIDIA Container Toolkit verification
- Jetson containers installation (dusty-nv/jetson-containers)
- ARM64/aarch64 optimizations

---

## Repository Setup

### 1. Clone Repository

```bash
# Navigate to home directory
cd ~

# Clone repository (with or without GitHub token)
git clone https://github.com/PaulChouLedger/LedgerAI.git
cd LedgerAI
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv aura-env

# Activate virtual environment
source aura-env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r aura-control/requirements/requirements.txt
```

### 3. Create Required Directories

```bash
# Create data directories
mkdir -p data/input data/parsed data/embeddings data/learning
mkdir -p shared

# Set permissions
chmod -R 755 data
```

---

## Environment Configuration (.env)

### 1. Create .env File

```bash
# Copy example file
cp .env.example .env

# Or create from scratch
touch .env
```

### 2. Configure Essential Settings

Use the interactive configuration script:

```bash
# Make script executable
chmod +x aura_config.sh

# Run configuration
./aura_config.sh
```

### 3. Minimum Required .env Settings

Your `.env` file must have at minimum:

```bash
# ============================================
# REQUIRED SETTINGS
# ============================================

# ElevenLabs API Key (REQUIRED for TTS)
ELEVENLABS_API_KEY=sk_your_api_key_here
ELEVENLABS_VOICE_ID=default

# LLM Model Paths
SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SIMPLE_N_CTX=2048
SIMPLE_CHAT_FORMAT=llama-3.2

# LLM Configuration
LLM_TEMPERATURE_SIMPLE=0.7
LLM_TOP_P=0.9
LLM_TOP_K=40
LLM_REPEAT_PENALTY=1.1
LLM_NUM_PREDICT=512

# LLM Mode
USE_MEDICAL_MODE=true

# RAG Configuration
RAG_MODE=CPU
RAG_THRESHOLD=0.3
RAG_TOP_K=5
RAG_USE_PHONETIC_MATCHING=true

# Medical Guidelines (only GI curated)
ENABLED_MEDICAL_CATEGORIES=GI

# ============================================
# OPTIONAL SETTINGS
# ============================================

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here

# GitHub OTA Updates (optional)
GITHUB_TOKEN=ghp_your_token_here

# EHR Integration (keep disabled for now)
EHR_INTEGRATION_ENABLED=false

# Debug Settings
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### 4. Verify Configuration

```bash
# View all settings
./aura_config.sh show

# Should show:
# ✅ ElevenLabs API Key configured
# ✅ Medical Mode enabled
# ✅ RAG Mode: CPU
```

---

## Docker Setup

**⚠️ IMPORTANT: These containers use Jetson-specific base images**

All containers are built from NVIDIA Jetson-optimized base images:
- **LLM Containers**: `dustynv/llama_cpp:b5283-r36.4-cu128-24.04` (JetPack 6.4, CUDA 12.8, ARM64)
- **Whisper Container**: `dustynv/faster-whisper:r36.4.0-cu128-24.04` (JetPack 6.4, CUDA 12.8, ARM64)
- **RAG Container**: `faiss_lite:r36.4.tegra-aarch64-cu129-22.04fi` (JetPack 6.4, CUDA 12.9, ARM64)

**Requirements:**
- NVIDIA Jetson Orin NX with **JetPack 6.4** (r36.4)
- Docker with NVIDIA Container Toolkit configured
- **Jetson Container Tools** installed (from `dusty-nv/jetson-containers`)
- CUDA 12.8+ support
- Git (for cloning repositories)

**Note**: The base images (`dustynv/*`) are provided by the [jetson-containers](https://github.com/dusty-nv/jetson-containers) project. You must install the container tools before building Aura containers.

### 1. Verify JetPack Version

```bash
# Check JetPack version (should be 6.4 / r36.4)
cat /etc/nv_tegra_release

# Should show something like:
# R36 (release), REVISION: 6.4, GCID: 12345678, BOARD: t186ref, EABI: aarch64, DATE: ...
```

### 2. Verify Jetson Container Tools Installation

```bash
# Check if jetson-containers is installed
ls -la ~/jetson-containers

# If not installed, install it:
cd ~
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# This installs the container tools needed for dustynv/* base images
```

### 3. Build Docker Images

```bash
# Navigate to setup directory
cd ~/LedgerAI/setup

# Build all containers (this will pull Jetson-specific base images)
docker compose build

# This will build:
# - whisper-container (Jetson-optimized faster-whisper)
# - llm-medical-container (Jetson-optimized llama.cpp, if USE_MEDICAL_MODE=true)
# - llm-generic-container (Jetson-optimized llama.cpp, if USE_MEDICAL_MODE=false)
# - rag-container (Jetson-optimized FAISS, only if RAG_MODE=GPU)
```

**Note**: First build may take 20-40 minutes as it downloads:
- Base images (~2-5GB each) from `dustynv/*` (requires jetson-containers)
- Python packages
- Pre-downloaded models (sentence-transformers)

**If build fails with "base image not found":**
- Ensure Jetson Container Tools are installed (see Step 2 above)
- Verify you're on JetPack 6.4: `cat /etc/nv_tegra_release`

### 4. Verify Docker Images

```bash
# List built images
docker images | grep -E "aura|llm|whisper|rag|dustynv"

# Should see:
# - setup-whisper:latest (based on dustynv/faster-whisper)
# - setup-llm-medical:latest (based on dustynv/llama_cpp)
# - setup-llm-generic:latest (based on dustynv/llama_cpp)
# - Base images: dustynv/llama_cpp, dustynv/faster-whisper

# Verify base images are Jetson-specific
docker images | grep dustynv
# Should show ARM64/aarch64 images with r36.4 tags
```

### 5. Verify NVIDIA Container Toolkit

```bash
# Check if NVIDIA runtime is available
docker info | grep -i nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Should show GPU information (Jetson Orin NX)
```

**Note**: The Jetson Container Tools installation (Step 2) configures NVIDIA Container Toolkit automatically.

### 6. Test Container Startup

```bash
# Start containers manually
cd ~/LedgerAI/setup
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

---

## Systemd Services

### 1. Install Aura Service

**Option A: Automated Installation (Recommended)**

```bash
# Use the installation script
sudo bash ~/LedgerAI/setup/scripts/install_aura_service.sh
```

This script will:
- Check prerequisites
- Copy service file to `/etc/systemd/system/aura.service`
- Configure paths automatically
- Enable service for auto-start on boot

**Option B: Manual Installation**

```bash
# Copy service file
sudo cp ~/LedgerAI/setup/scripts/aura.service /etc/systemd/system/

# Edit paths if different from default
sudo nano /etc/systemd/system/aura.service
# Update /home/aura paths if your user is different
```

### 2. Enable and Start Service

If using automated installation, the script will prompt you to start the service. Otherwise:

```bash
# Reload systemd (if manual installation)
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable aura.service

# Start service now
sudo systemctl start aura.service

# Check status
sudo systemctl status aura.service

# View logs
sudo journalctl -u aura.service -f
```

### 3. XVF3800 Hardware Tuning Service (If Using XVF3800 USB 4-Mic Array)

**Option A: Automated Installation (via install_jetson.sh)**

If you ran `bash setup/install_jetson.sh` and the ReSpeaker was detected, the service was automatically installed. You can verify:

```bash
sudo systemctl status xvf3800-tuning.service
```

**Option B: Manual Installation**

```bash
# Copy service file
sudo cp ~/LedgerAI/setup/scripts/xvf3800-tuning.service /etc/systemd/system/

# Edit paths if different from default (if user is not 'aura')
sudo nano /etc/systemd/system/xvf3800-tuning.service
# Update User=aura and /home/aura/LedgerAI paths if needed

# Reload systemd
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
sudo systemctl enable xvf3800-tuning.service

# Start service now
sudo systemctl start xvf3800-tuning.service

# Verify
sudo systemctl status xvf3800-tuning.service
```

**Configure Preset (Change Tuning Profile):**

The service defaults to `agc_20` preset. To change the preset:

```bash
# 1. Edit the service file
sudo nano /etc/systemd/system/xvf3800-tuning.service

# 2. Find the ExecStart line and change "agc_20" to your desired preset:
#    ExecStart=/usr/bin/python3 /home/aura/LedgerAI/setup/scripts/tune_xvf3800.py agc_20
#                                                                    ^^^^^^^^
#                                                                    Change this

# 3. Reload systemd
sudo systemctl daemon-reload

# 4. Restart service to apply new preset
sudo systemctl restart xvf3800-tuning.service
```

**Available Presets:**

| Preset | Description | Use Case |
|--------|-------------|----------|
| `agc_20` | HPF 70Hz + AGC with 20% increase (0.096) | **DEFAULT** ⭐ Recommended for most cases |
| `agc_10` | HPF 70Hz + AGC with 10% increase (0.088) | Moderate gain boost |
| `balanced_beam` | HPF 70Hz + AGC (0.08, 30dB) | Balanced processing |
| `agc_only` | AGC only (0.08, 30dB) - no HPF | No high-pass filtering |
| `hpf_only` | HPF 70Hz only | Minimal processing |
| `ultra_sensitive` | AGC (0.10, 45dB) | Far-field optimized |
| `far_field` | Optimized for 8-16 feet | Distant speakers |
| `near_field` | Optimized for 1-6 feet | Close speakers |
| `reset` | Factory defaults | Restore original settings |

**Useful Commands:**

```bash
# Check service status
sudo systemctl status xvf3800-tuning.service

# View service logs
sudo journalctl -u xvf3800-tuning.service -f

# Manually re-tune (without restarting service)
python3 ~/LedgerAI/setup/scripts/tune_xvf3800.py agc_20

# Disable auto-tune (service won't run on boot)
sudo systemctl disable xvf3800-tuning.service

# Re-enable auto-tune
sudo systemctl enable xvf3800-tuning.service
```

### 4. Configure WiFi Permissions (for Settings Dialog)

```bash
# Create polkit rule for WiFi scanning
sudo mkdir -p /etc/polkit-1/rules.d

sudo bash -c 'cat > /etc/polkit-1/rules.d/50-org.freedesktop.NetworkManager.wifi.rules << EOF
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.NetworkManager.wifi.scan" && subject.user == "aura") {
        return polkit.Result.YES;
    }
});
EOF'

# Restart polkit
sudo systemctl restart polkit
```

---

## Hardware Configuration

### 1. Display Configuration

Aura automatically configures display on startup, but you can verify:

```bash
# Test display wake
xset dpms force on

# Test cursor hiding
unclutter -idle 0.1 -root
```

### 2. Audio Configuration

```bash
# Check audio devices
arecord -l

# Test microphone
arecord -d 3 -f cd test.wav
aplay test.wav

# Configure XVF3800 (if applicable)
cd ~/LedgerAI
python3 setup/scripts/tune_xvf3800.py agc_20
```

### 3. USB Permissions (for XVF3800)

```bash
# Create udev rules
sudo nano /etc/udev/rules.d/99-xvf3800.rules
```

Add:
```
SUBSYSTEM=="usb", ATTRS{idVendor}=="0d8c", ATTRS{idProduct}=="013c", MODE="0666", GROUP="audio"
```

Then:
```bash
# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## Testing & Verification

### 1. Manual Test (Without Service)

```bash
# Activate virtual environment
source ~/LedgerAI/aura-env/bin/activate

# Start Aura manually
cd ~/LedgerAI/aura-control/core
python3 main.py
```

Expected output:
```
[Aura] 🚀 Starting Aura...
[Aura] 🖥️  Configuring display...
[Aura] ⌨️  Disabling Ubuntu on-screen keyboard...
[Aura] ✅ Ubuntu keyboard monitor started
[Aura] 🌀 Launching Aura...
[Aura] ✅ GUI ready - starting services
[Aura] 🚀 Starting containers...
[Aura] ✅ Core services started successfully!
```

### 2. Test GUI

- GUI should appear on display
- Circular buttons should be visible
- Aura eye should be visible

### 3. Test Voice Input

- Click microphone button
- Speak a test phrase
- Should see transcription appear

### 4. Test Web Upload

```bash
# Find your IP address
hostname -I

# Access from browser
http://YOUR_IP:5000
```

### 5. Verify Containers (Jetson-Specific)

```bash
# Check running containers
docker ps

# Should show:
# - setup-whisper-1 (Jetson-optimized faster-whisper)
# - setup-llm-medical-1 (Jetson-optimized llama.cpp) or setup-llm-generic-1

# Verify containers are using Jetson GPU
docker exec setup-llm-medical-1 nvidia-smi
# Should show GPU usage (Jetson Orin NX)

# Check base images are Jetson-specific
docker images | grep dustynv
# Should show: dustynv/llama_cpp, dustynv/faster-whisper with r36.4 tags
```

### 6. Test Health Endpoints

```bash
# Whisper health
curl http://localhost:11433/health

# LLM health
curl http://localhost:11434/health

# LLM generic health (if using)
curl http://localhost:11436/health
```

---

## Post-Installation Configuration

### 1. Configure GitHub OTA Updates (Optional)

```bash
# Generate GitHub Personal Access Token
# https://github.com/settings/tokens

# Configure via script
./aura_config.sh
# Choose option 10 (Configure GitHub OTA updates)
```

### 2. Configure Medical Categories

```bash
# Configure which guideline categories to use
./aura_config.sh
# Choose option 5 (Configure Medical Categories)
# Currently only GI is curated, so use "GI"
```

### 3. Configure RAG Mode

```bash
# Choose GPU or CPU RAG
./aura_config.sh
# Choose option 7 (Configure RAG search)
# Option 1 (Toggle RAG mode)
# Recommend: CPU for simpler setup, GPU for better performance
```

---

## Troubleshooting

### Issue: Docker Containers Won't Start

```bash
# Check Docker daemon
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check logs
docker compose logs

# Verify NVIDIA Container Toolkit (Jetson-specific)
docker info | grep -i nvidia
# Should show: Runtimes: nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Jetson-specific:**
- Ensure JetPack 6.4 (r36.4) is installed: `cat /etc/nv_tegra_release`
- Base images (`dustynv/*:r36.4-*`) require matching JetPack version
- If base image pull fails, verify your JetPack version matches the Dockerfile tags

### Issue: Permission Denied Errors

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in
exit
# (login again)

# Verify
groups
# Should include "docker"

# Jetson-specific: Verify NVIDIA Container Toolkit permissions
ls -la /usr/bin/nvidia-container-runtime
# Should be executable
```

### Issue: Audio Not Working

```bash
# Check audio devices
arecord -l

# Test microphone
arecord -d 3 test.wav

# Check permissions
ls -l /dev/snd/

# Add user to audio group
sudo usermod -aG audio $USER
```

### Issue: Base Image Pull Failures (Jetson)

```bash
# Verify JetPack version matches base image requirements
cat /etc/nv_tegra_release
# Should show: R36 (release), REVISION: 6.4

# Check if Jetson Container Tools are installed
ls -la ~/jetson-containers
# If missing, install:
cd ~
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# Check Dockerfile base images
grep "FROM" llm-medical-container/Dockerfile
grep "FROM" whisper-container/Dockerfile

# Base images must match your JetPack version:
# - dustynv/llama_cpp:b5283-r36.4-cu128-24.04 (requires JetPack 6.4)
# - dustynv/faster-whisper:r36.4.0-cu128-24.04 (requires JetPack 6.4)

# If JetPack version differs, update Dockerfiles to match your version
```

**Common cause**: Missing Jetson Container Tools. The `dustynv/*` base images require the container tools from `dusty-nv/jetson-containers`.

### Issue: GPU Not Accessible in Containers (Jetson)

```bash
# Verify NVIDIA Container Toolkit
docker info | grep -i nvidia

# Install if missing (JetPack 6.4 includes this)
# Check: /usr/bin/nvidia-container-runtime

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Should show Jetson Orin NX GPU information
```

### Issue: Display Not Appearing

```bash
# Check DISPLAY variable
echo $DISPLAY

# Set if needed
export DISPLAY=:0

# Test X11
xset q
```

### Issue: Model Files Not Found

```bash
# Check model paths in .env
grep MODEL_PATH .env

# Verify files exist
ls -lh /models/

# Update paths if needed
./aura_config.sh
# Choose option 6 (Configure LLM models)
```

### Issue: Service Won't Start

```bash
# Check service status
sudo systemctl status aura.service

# View logs
sudo journalctl -u aura.service -n 50

# Check for errors
sudo journalctl -u aura.service -p err
```

### Issue: Containers Fail to Start

```bash
# Check Docker logs
docker compose logs

# Check container status
docker compose ps -a

# Restart containers
docker compose down
docker compose up -d
```

---

## Quick Start Checklist

After installation, verify these items:

- [ ] `.env` file exists and has `ELEVENLABS_API_KEY` set
- [ ] Docker images built successfully
- [ ] Containers start without errors
- [ ] Systemd service enabled and running
- [ ] GUI appears on display
- [ ] Microphone input working
- [ ] Web upload server accessible
- [ ] Voice transcription working
- [ ] TTS (text-to-speech) working

---

## Maintenance Commands

### Update Aura

```bash
cd ~/LedgerAI
git pull

# Restart service
sudo systemctl restart aura.service
```

### View Logs

```bash
# Aura service logs
sudo journalctl -u aura.service -f

# Container logs
docker compose logs -f

# Specific container
docker logs aura-llm-medical -f
```

### Restart Services

```bash
# Restart Aura
sudo systemctl restart aura.service

# Restart containers
cd ~/LedgerAI/setup
docker compose restart

# Restart specific container
docker compose restart llm-medical
```

### Stop Services

```bash
# Stop Aura service
sudo systemctl stop aura.service

# Stop all containers
cd ~/LedgerAI/setup
docker compose down
```

---

## Next Steps

After successful installation:

1. **Test all features** - Voice input, transcription, TTS, GUI
2. **Configure settings** - Use `aura_config.sh` to fine-tune
3. **Upload documents** - Add files to `data/input/` for RAG
4. **Test medical mode** - Try symptom assessment
5. **Set up autostart** - Service should start on boot automatically

---

## Support

For issues or questions:
- Check logs: `sudo journalctl -u aura.service -n 100`
- Check container logs: `docker compose logs`
- Review configuration: `./aura_config.sh show`

---

## Summary

This guide covers:
- ✅ Complete system setup
- ✅ Environment configuration
- ✅ Docker container setup
- ✅ Systemd service configuration
- ✅ Hardware configuration
- ✅ Testing and verification

Your Aura system should now be fully configured and ready to use!

