# LedgerAI Installation Guide

This guide will help you set up LedgerAI on a fresh system (new NVMe drive, clean OS installation, etc.).

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url> LedgerAI
cd LedgerAI
```

### 2. Run the Automated Setup Script

The setup script will install all dependencies automatically:

```bash
bash setup_new_system.sh
```

This script will:
- ✅ Detect your OS (Ubuntu/Debian or macOS)
- ✅ Install system dependencies (Python, audio libraries, Qt, etc.)
- ✅ Install Docker and Docker Compose
- ✅ Create a Python virtual environment
- ✅ Install all Python packages
- ✅ Build Docker containers (Whisper, LLM, RAG)
- ✅ Set up ReSpeaker hardware support (Linux only)
- ✅ Create required directories
- ✅ Generate environment configuration files

**Estimated time:** 10-30 minutes (depending on internet speed)

### 3. Configure Environment Variables

Edit the `.env` file to add your API keys:

```bash
nano llm-container/.env
```

Required configuration:
```env
# ElevenLabs API Key (for Text-to-Speech)
ELEVENLABS_API_KEY=your_api_key_here

# Optional: Telegram Bot Token
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

### 4. Activate Virtual Environment

```bash
source ~/ledgerai-venv/bin/activate
```

### 5. Start the System

Start Docker containers:
```bash
docker compose up -d
```

Run the main application:
```bash
python3 aura-control/main.py
```

---

## Manual Installation (Alternative)

If you prefer manual installation or need to customize:

### System Requirements

**Linux (Ubuntu/Debian):**
- Python 3.8+
- Docker & Docker Compose
- PortAudio, ALSA, PulseAudio
- Qt5 development libraries

**macOS:**
- Python 3.8+
- Docker Desktop
- Homebrew
- PortAudio

### Step-by-Step Manual Setup

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    portaudio19-dev libsndfile1 ffmpeg sox \
    libasound2-dev pulseaudio \
    qt5-default python3-pyqt5 \
    docker.io docker-compose
```

**macOS:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 portaudio ffmpeg sox

# Install Docker Desktop manually from:
# https://www.docker.com/products/docker-desktop
```

#### 2. Create Python Virtual Environment

```bash
python3 -m venv ~/ledgerai-venv
source ~/ledgerai-venv/bin/activate
pip install --upgrade pip
```

#### 3. Install Python Dependencies

```bash
# Install from requirements file
pip install -r aura-control/requirements.txt

# Or install PyTorch separately for your platform:
# macOS:
pip install torch torchvision torchaudio

# Linux (CPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Build Docker Containers

```bash
docker compose build
```

#### 5. Setup Hardware (Linux only - ReSpeaker)

For ReSpeaker 4-Mic Array support:

```bash
# USB permissions
sudo bash scripts/setup_usb_permissions.sh

# Auto-tune service (optional)
sudo bash scripts/install_auto_tune.sh
```

#### 6. Create Required Directories

```bash
mkdir -p shared/{input_audio,output_audio}
mkdir -p data/{embeddings,parsed,input}
mkdir -p rag-container/cache
```

---

## Usage

### Starting the System

1. **Activate virtual environment:**
   ```bash
   source ~/ledgerai-venv/bin/activate
   ```

2. **Start Docker containers:**
   ```bash
   docker compose up -d
   ```

3. **Run the main application:**
   ```bash
   python3 aura-control/main.py
   ```

### Stopping the System

```bash
# Stop main application: Ctrl+C

# Stop Docker containers:
docker compose down
```

### Viewing Logs

```bash
# Docker container logs
docker compose logs -f

# Individual container logs
docker compose logs -f whisper
docker compose logs -f llm
docker compose logs -f rag
```

---

## Components Overview

### Docker Containers

1. **whisper-container** (Port 5051)
   - Speech-to-text using faster-whisper
   - Model: distil-small.en

2. **llm-container** (Port 11434)
   - Language model inference
   - Natural language generation

3. **rag-container** (Port 11435)
   - Retrieval Augmented Generation
   - Document embeddings and search

### Aura Control (Main Application)

- **main.py** - Main orchestrator
- **listener.py** - Audio input and VAD
- **speaker.py** - Text-to-speech output
- **aura_gui.py** - Visual interface
- **telegram_bot.py** - Telegram integration (optional)
- **web_upload_server.py** - Web-based file upload

---

## RAG Document Ingestion

To add documents for the RAG system:

1. **Place documents in `data/input/`:**
   ```bash
   cp your_document.pdf data/input/
   ```

2. **Rebuild embeddings:**
   ```bash
   # Using host script
   python3 scripts/rebuild_embeddings_host.py
   
   # Or rebuild RAG container
   bash scripts/rebuild_rag.sh
   ```

See [AUTO_INGEST_GUIDE.md](AUTO_INGEST_GUIDE.md) for more details.

---

## Troubleshooting

### Docker Issues

```bash
# Rebuild containers
docker compose build --no-cache

# Check container status
docker compose ps

# View logs
docker compose logs -f
```

### Python Dependencies

```bash
# Reinstall dependencies
pip install --force-reinstall -r aura-control/requirements.txt
```

### Audio Issues (Linux)

```bash
# Test audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Reset PulseAudio
pulseaudio --kill && pulseaudio --start
```

### ReSpeaker Issues (Linux)

```bash
# Check USB connection
lsusb | grep 2886:0018

# Check permissions
ls -la /dev/bus/usb/

# Restart auto-tune service
sudo systemctl restart respeaker-tuning
sudo journalctl -u respeaker-tuning
```

---

## Environment Variables Reference

### llm-container/.env

```env
# Required
ELEVENLABS_API_KEY=sk-xxx...        # ElevenLabs TTS API key

# Optional
TELEGRAM_BOT_TOKEN=xxx...           # Telegram bot token
PYTHONUNBUFFERED=1                  # Python logging
```

---

## Additional Documentation

- **[AUTO_INGEST_GUIDE.md](AUTO_INGEST_GUIDE.md)** - RAG document ingestion
- **[CIRCULAR_BORDER_SYSTEM.md](CIRCULAR_BORDER_SYSTEM.md)** - GUI visual system
- **[DYNAMIC_RAG_IMPROVEMENTS.md](DYNAMIC_RAG_IMPROVEMENTS.md)** - RAG enhancements
- **[RAG_PHONETIC_MATCHING.md](RAG_PHONETIC_MATCHING.md)** - Phonetic search
- **[WHISPER_NAME_GUIDANCE.md](WHISPER_NAME_GUIDANCE.md)** - Speech recognition config
- **[SINGLE_CHANNEL_FIRMWARE_SUCCESS.md](SINGLE_CHANNEL_FIRMWARE_SUCCESS.md)** - Hardware config

---

## Platform-Specific Notes

### macOS
- GUI requires X11 or native display server
- Docker Desktop must be running before starting containers
- Audio device access may require permissions in System Preferences

### Linux (Ubuntu/Debian)
- User must be in `docker` and `plugdev` groups
- May need to log out/in for group changes
- PulseAudio or ALSA required for audio
- ReSpeaker requires udev rules for non-root access

### Raspberry Pi / ARM
- Use ARM-compatible Docker images
- May need to adjust PyTorch installation for ARM architecture
- Consider lighter models for resource constraints

---

## Updating the System

### Pull Latest Changes

```bash
git pull origin main
```

### Rebuild Containers

```bash
docker compose down
docker compose build
docker compose up -d
```

### Update Python Dependencies

```bash
source ~/ledgerai-venv/bin/activate
pip install --upgrade -r aura-control/requirements.txt
```

---

## Support

For issues and questions:
- Check existing documentation files
- Review Docker container logs
- Verify environment configuration
- Test components individually

---

## Quick Reference Commands

```bash
# Setup
bash setup_new_system.sh                    # Full automated setup

# Environment
source ~/ledgerai-venv/bin/activate         # Activate venv

# Docker
docker compose up -d                        # Start containers
docker compose down                         # Stop containers
docker compose logs -f                      # View logs
docker compose build --no-cache             # Rebuild

# Application
python3 aura-control/main.py                # Run main app

# Hardware (Linux)
sudo bash scripts/setup_usb_permissions.sh  # USB permissions
sudo bash scripts/install_auto_tune.sh      # Auto-tune service
sudo systemctl status respeaker-tuning      # Check service

# RAG
python3 scripts/rebuild_embeddings_host.py  # Rebuild embeddings
```

