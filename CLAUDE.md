# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**LedgerAI / Aura** is a voice-first conversational AI assistant designed for NVIDIA Jetson hardware. The system uses a Seeed XVF3800 4-mic array for far-field capture, runs inference locally on GPU, and presents a circular PyQt5 GUI. Everything runs on-device — no cloud inference.

## Running the System

```bash
# Start main application (from workspace root)
python3 aura-control/core/main.py

# Start all Docker containers
cd setup && docker-compose up

# Configure API keys / credentials
./aura_config.sh

# Manage a specific config section
./aura_config.sh tts
./aura_config.sh wake
```

Containers are managed via `setup/docker-compose.yml`. After changing container code, rebuild with:
```bash
cd setup && docker-compose build <service-name>
docker-compose restart <service-name>
```

## Architecture

### Container Services

| Service | Port | Description |
|---------|------|-------------|
| `whisper` | 5000 | faster-whisper speech-to-text (GPU) |
| `llm-medical` | 11434 | Qwen2.5 LLM for medical conversations |
| `llm-generic` | 11434 | Qwen2.5 LLM for general conversations |
| `memory` | 11438 | Conversation memory with FAISS semantic search |
| `chatterbox-tts` | varies | Chatterbox voice-cloning TTS (GPU) |
| `rag` | 11435 | Optional GPU RAG container (disabled by default) |

All containers use `network_mode: host` and share `../shared:/shared` volume.

### Main Application (`aura-control/`)

The entry point is `aura-control/core/main.py`. It orchestrates:

- **`core/listener.py`** — Continuous audio capture from XVF3800 via sounddevice, Silero VAD, optional OpenWakeWord wake-word gating, and HTTP POST to Whisper container for transcription.
- **`core/speaker.py`** — TTS playback using either ChatterboxTTS (local GPU) or ElevenLabs API; plays audio via `aplay` (ALSA). Lazy-loads TTS engines.
- **`core/state.py`** — Global singleton for playback state, settings (LLM mode, TTS engine, wake word config). Persists to `data/app_settings.json`. Import this everywhere instead of sharing raw globals.
- **`core/memory_integration.py`** — Bridge to the memory container.
- **`core/openwakeword_wake_word.py`** — OpenWakeWord detector factory.
- **`gui/aura_gui.py`** — PyQt5 circular GUI with animated states.
- **`tts/chatterbox_tts.py`** — ChatterboxTTS wrapper used by `speaker.py`.

### Shared Module (`shared/`)

`shared/llm_base.py` contains `BaseLLMContainer` — the base class for both LLM containers. It handles model loading, llama.cpp inference, health checks, and sentence tagging (`<sentence_start>` / `<sentence_end>`) for TTS chunking.

`shared/medical_terms.json` is the single source of truth for medical vocabulary (used by Whisper and LLM containers). Edit only here; restart containers to apply — no rebuild needed.

`shared/rag/` contains the modular RAG client used inside LLM containers. It supports two modes:
- **CPU mode** (default): In-process FAISS with no external service
- **GPU mode**: HTTP calls to the `rag` container at port 11435

### LLM Containers (`llm-container/`, `llm-medical-container/`)

Both run Flask REST APIs via `container_rest.py` and extend `BaseLLMContainer`. The generic container loads two models:
- **Base model** (`Qwen2.5-1.5B-Instruct.Q4_K_M_base.gguf`) — conversational, no RAG
- **CoT model** (`Qwen2.5-1.5B-Instruct.Q4_K_M-rag-cot.gguf`) — lazy-loaded for RAG queries with Chain-of-Thought reasoning

Model selection is automatic based on whether RAG context is present.

Key LLM endpoints:
- `POST /chat-tts` — Streaming for voice/TTS (returns sentence-tagged SSE stream)
- `POST /chat-tg` — Non-streaming for Telegram
- `GET /health` — Health check

### Settings Priority

Settings cascade as follows:
1. `data/app_settings.json` (GUI selections, persisted by `state.py`)
2. Environment variables in `.env` (API keys; loaded via `aura_config.sh`)
3. Hardcoded defaults in `container_rest.py` or `state.py`

LLM/RAG parameters are hardcoded at the top of each `container_rest.py` as module-level constants — edit there for tuning.

### Audio Pipeline

```
XVF3800 USB mic → sounddevice (listener.py)
  → Silero VAD (speech detection)
  → [OpenWakeWord gate, if enabled]
  → Advanced spectral filters (ZCR, flatness, centroid, RMS)
  → WAV bytes → HTTP POST → Whisper container (:5000)
  → Transcript text → LLM container (:11434) /chat-tts
  → Streamed sentence tokens → speaker.py
  → ChatterboxTTS or ElevenLabs → aplay (ALSA)
```

## Configuration Files

- `.env` — API keys (`ELEVENLABS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `GITHUB_TOKEN`, `NHS_CLIENT_ID`/`NHS_CLIENT_SECRET`)
- `data/app_settings.json` — Runtime settings (LLM mode, TTS engine, wake word on/off)
- `data/xvf3800_config.json` — Microphone DSP/tuning parameters
- `setup/docker-compose.yml` — Container definitions

## Key Conventions

- **TTS engine toggle**: `state.get_tts_engine()` returns `"chatterbox"` or `"elevenlabs"`. `speaker.py` branches on this.
- **LLM mode toggle**: `state.get_llm_mode()` returns `"medical"` or `"generic"`. `main.py` routes to the correct container port.
- **RAG mode**: Set via `RAG_MODE` env var (`CPU` or `GPU`). Defaults to CPU (in-process FAISS).
- **Whisper model default**: Read dynamically from `whisper-container/container_rest.py`'s `MODEL_NAME` variable at startup — do not duplicate it in `state.py`.
- **Backup files**: The repo contains many `.bak.*` files. Ignore them; they are not part of the active codebase.
- **`shared/` rule**: Any resource used by 2+ containers belongs in `shared/` and is mounted at `/shared` inside each container.
