# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**LedgerAI / Aura** is a voice-first conversational AI assistant designed for NVIDIA Jetson hardware. The system uses a Seeed XVF3800 4-mic array for far-field capture, runs inference locally on GPU, and presents a circular PyQt5 GUI. Everything runs on-device — no cloud inference.

## Running the System

```bash
# Start main application (launched by boot/start_aura.sh)
cd aura && python3 -u aura.py

# Start Docker containers
cd setup && docker-compose up

# Configure API keys / credentials
./aura_config.sh
```

Containers are managed via `setup/docker-compose.yml`. After changing container code, rebuild with:
```bash
cd setup && docker-compose build <service-name>
docker-compose restart <service-name>
```

## Directory Structure

```
aura/                   # Main application (entry point: aura.py)
  core/                 # bus, state, config, updater
  voice/                # listener, speaker, llm_client, wake, intents
  gui/                  # PyQt5 circular GUI, complications, renderer
  boot/                 # orchestrator, enrollment, power_manager
  services/             # health, perpetual, memlog, diaglog

containers/
  llm/                  # Qwen2.5 LLM (runs native via run_llm_native.sh, not Docker)
  whisper/              # faster-whisper speech-to-text (Docker, port 5000)
  memory/               # Conversation memory + FAISS (Docker, port 11438)
  chatterbox/           # Chatterbox voice-cloning TTS (Docker)

shared/                 # Code shared across containers (mounted at /shared)
data/                   # Runtime state, settings, voice profiles, briefings
assets/                 # Media: boot prompts, voice samples, thinking fillers, logos
voices/                 # Voice model files (Piper ONNX, XTTS refs)
setup/                  # docker-compose.yml, hardware scripts
boot/                   # start_aura.sh (systemd entry point)
tests/                  # QA and benchmark scripts
```

## Architecture

### Container Services

| Service | Port | Description |
|---------|------|-------------|
| `whisper` | 5000 | faster-whisper speech-to-text (GPU) |
| `memory` | 11438 | Conversation memory with FAISS semantic search |
| `chatterbox-tts` | varies | Chatterbox voice-cloning TTS (GPU) |

The LLM runs natively (not in Docker) via `run_llm_native.sh` on port 11434.
All containers use `network_mode: host` and share `../shared:/shared` volume.

### Main Application (`aura/`)

The entry point is `aura/aura.py`. It orchestrates:

- **`voice/listener.py`** — Continuous audio capture from XVF3800 via sounddevice, Silero VAD, optional OpenWakeWord wake-word gating, and HTTP POST to Whisper container for transcription.
- **`voice/speaker.py`** — TTS playback using Piper (local) or ElevenLabs API; plays audio via `aplay` (ALSA).
- **`voice/llm_client.py`** — HTTP client to the LLM container (streaming + non-streaming).
- **`core/state.py`** — Global singleton for playback state, settings. Persists to `data/app_settings.json`.
- **`core/config.py`** — All paths, constants, color schemes. Single source of truth.
- **`gui/window.py`** — PyQt5 circular GUI with animated states.
- **`services/perpetual.py`** — Background rumination engine (daily briefs, proactive questions).

### Shared Module (`shared/`)

`shared/llm_base.py` contains `BaseLLMContainer` — the base class for LLM containers. It handles model loading, llama.cpp inference, health checks, and sentence tagging for TTS chunking.

`shared/rag/` contains the modular RAG client (CPU mode: in-process FAISS, no external service).

### LLM Container (`containers/llm/`)

Runs natively via `run_llm_native.sh`. Flask REST API via `container_rest.py`, extends `BaseLLMContainer`. Uses Qwen2.5-3B-Instruct-Q4_K_M.

Key endpoints:
- `POST /chat-tts` — Streaming for voice/TTS (returns sentence-tagged SSE stream)
- `POST /chat-tg` — Non-streaming for Telegram
- `GET /health` — Health check

### Settings Priority

1. `data/app_settings.json` (GUI selections, persisted by `state.py`)
2. Environment variables in `.env` (API keys; loaded via `aura_config.sh`)
3. Hardcoded defaults in `container_rest.py` or `state.py`

LLM/RAG parameters are hardcoded at the top of `containers/llm/container_rest.py` as module-level constants.

### Audio Pipeline

```
XVF3800 USB mic → sounddevice (listener.py)
  → Silero VAD (speech detection)
  → [OpenWakeWord gate, if enabled]
  → Advanced spectral filters (ZCR, flatness, centroid, RMS)
  → WAV bytes → HTTP POST → Whisper container (:5000)
  → Transcript text → LLM container (:11434) /chat-tts
  → Streamed sentence tokens → speaker.py
  → Piper TTS or ElevenLabs → aplay (ALSA)
```

## Configuration Files

- `.env` — API keys (`ELEVENLABS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `GITHUB_TOKEN`)
- `data/app_settings.json` — Runtime settings (TTS engine, wake word on/off)
- `setup/docker-compose.yml` — Container definitions

## Key Conventions

- **RAG mode**: Set via `RAG_MODE` env var (`CPU` or `GPU`). Defaults to CPU (in-process FAISS).
- **Whisper model default**: Read dynamically from `containers/whisper/container_rest.py`'s `MODEL_NAME` variable.
- **`shared/` rule**: Any resource used by 2+ containers belongs in `shared/` and is mounted at `/shared` inside each container.
