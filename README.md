# LedgerAI

AI-powered voice assistant with speech recognition, natural language processing, and retrieval-augmented generation (RAG).

## 🚀 Quick Start

### New Installation (Fresh NVMe/System)

Run the automated setup script:

```bash
bash setup_new_system.sh
```

This will install all dependencies, build Docker containers, and configure your system.

**See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed instructions.**

### Running the Application

```bash
# 1. Activate virtual environment
source ~/ledgerai-venv/bin/activate

# 2. Start Docker containers
docker compose up -d

# 3. Run the main application
python3 aura-control/main.py
```

## 📋 System Requirements

### Supported Platforms
- Ubuntu 20.04+ / Debian 11+
- macOS 11+
- Raspberry Pi OS (ARM)

### Hardware
- 8GB+ RAM recommended
- Microphone (ReSpeaker 4-Mic Array supported on Linux)
- Audio output device

### Software
- Python 3.8+
- Docker & Docker Compose
- Audio libraries (PortAudio, ALSA/PulseAudio on Linux)
- Qt5 (for GUI)

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Aura Control (GUI)                      │
│  main.py • listener.py • speaker.py • aura_gui.py          │
└─────────┬───────────────────────────────────┬───────────────┘
          │                                   │
          ↓                                   ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Whisper (STT)  │  │   LLM Engine    │  │   RAG System    │
│   Port 5051     │  │   Port 11434    │  │   Port 11435    │
│                 │  │                 │  │                 │
│  faster-whisper │  │  NLG/Inference  │  │  Embeddings     │
│  distil-small   │  │                 │  │  Doc Search     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Docker Containers

1. **whisper-container-faster** - Speech-to-text recognition
2. **llm-container** - Language model inference
3. **rag-container** - Retrieval augmented generation

### Aura Control Features

- **Voice Activity Detection (VAD)** - Automatic speech detection
- **Real-time Transcription** - Continuous speech recognition
- **Natural Language Processing** - Intent understanding
- **Text-to-Speech (TTS)** - ElevenLabs integration
- **Document RAG** - Search and reference uploaded documents
- **Visual Interface** - Modern Qt5 GUI
- **Web Upload** - Browser-based file upload
- **Telegram Bot** - Remote interaction (optional)

## 📚 Documentation

- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Complete setup instructions
- **[AUTO_INGEST_GUIDE.md](AUTO_INGEST_GUIDE.md)** - RAG document ingestion
- **[CIRCULAR_BORDER_SYSTEM.md](CIRCULAR_BORDER_SYSTEM.md)** - GUI visual system
- **[DYNAMIC_RAG_IMPROVEMENTS.md](DYNAMIC_RAG_IMPROVEMENTS.md)** - RAG enhancements
- **[RAG_PHONETIC_MATCHING.md](RAG_PHONETIC_MATCHING.md)** - Phonetic search features
- **[WHISPER_NAME_GUIDANCE.md](WHISPER_NAME_GUIDANCE.md)** - Speech recognition config
- **[SINGLE_CHANNEL_FIRMWARE_SUCCESS.md](SINGLE_CHANNEL_FIRMWARE_SUCCESS.md)** - Hardware setup

## ⚙️ Configuration

### Environment Variables

Create `llm-container/.env`:

```env
# Required
ELEVENLABS_API_KEY=your_api_key_here

# Optional
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

### RAG Documents

Add documents to `data/input/` and rebuild embeddings:

```bash
cp your_document.pdf data/input/
python3 scripts/rebuild_embeddings_host.py
```

## 🔧 Common Commands

```bash
# Docker Management
docker compose up -d              # Start all containers
docker compose down               # Stop all containers
docker compose logs -f            # View logs
docker compose build              # Rebuild containers

# Python Environment
source ~/ledgerai-venv/bin/activate    # Activate venv
pip install -r aura-control/requirements.txt  # Install deps

# Application
python3 aura-control/main.py      # Run main application

# Hardware (Linux only)
sudo bash scripts/setup_usb_permissions.sh    # USB access
sudo bash scripts/install_auto_tune.sh        # Auto-tune service
```

## 🐛 Troubleshooting

### Docker Issues
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Audio Issues
```bash
# Test audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Linux: Reset PulseAudio
pulseaudio --kill && pulseaudio --start
```

### Python Dependencies
```bash
pip install --force-reinstall -r aura-control/requirements.txt
```

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for more troubleshooting.

## 📁 Project Structure

```
LedgerAI/
├── aura-control/              # Main application
│   ├── main.py               # Application orchestrator
│   ├── listener.py           # Audio input & VAD
│   ├── speaker.py            # Text-to-speech
│   ├── aura_gui.py           # Visual interface
│   ├── telegram_bot.py       # Telegram integration
│   └── requirements.txt      # Python dependencies
│
├── whisper-container-faster/ # Speech recognition
│   ├── whisper_engine.py
│   └── Dockerfile
│
├── llm-container/            # Language model
│   ├── llm_inference.py
│   ├── nlg.py
│   └── Dockerfile
│
├── rag-container/            # RAG system
│   ├── rag.py
│   ├── ingest.py
│   └── Dockerfile
│
├── data/                     # Data storage
│   ├── embeddings/          # Vector embeddings
│   ├── input/               # Documents for RAG
│   └── parsed/              # Processed documents
│
├── scripts/                  # Utility scripts
│   ├── rebuild_embeddings_host.py
│   ├── setup_usb_permissions.sh
│   └── install_auto_tune.sh
│
├── docker-compose.yml        # Container orchestration
├── setup_new_system.sh       # Automated installer
└── INSTALLATION_GUIDE.md     # Setup documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license here]

## 🆘 Support

For issues and questions:
1. Check the documentation files
2. Review Docker container logs: `docker compose logs -f`
3. Verify environment configuration
4. Test components individually

---

**Made with ❤️ for seamless voice interaction**

