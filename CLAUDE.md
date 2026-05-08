# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**LedgerAI / Aura** is a voice-first conversational AI assistant designed for NVIDIA Jetson hardware. The system uses a Seeed XVF3800 4-mic array for far-field capture, runs inference locally on GPU, and presents a circular PyQt5 GUI. Everything runs on-device natively — no cloud, no containers.

## Running the System

On a puck, everything is launched by systemd at boot via `aura4.service` (which runs `boot/start_aura.sh`). For dev:

```bash
# Manual start (boot script does this automatically on the puck)
cd aura && python3 -u aura.py

# Memory + RAG ingest are separate processes; aura.py starts memory itself.
# RAG ingest runs as its own systemd unit:
sudo systemctl status aura-ingest.service

# Configure API keys / credentials
./aura_config.sh
```

All services run natively via Python virtualenv (`~/aura-env`). No containers.

**Persistent systemd units on each puck:**
- `aura4.service` — runs `boot/start_aura.sh` → `aura.py` (Whisper + LLM in-process, plus child memory process)
- `aura-ingest.service` — standalone RAG auto-ingest watcher (decoupled from aura.py so the FAISS index stays in sync even if aura.py is down)

## Directory Structure

```
aura/                   # Main application (entry point: aura.py)
  core/                 # bus, state, config, updater
  voice/                # listener, speaker, llm_client, wake, intents
  gui/                  # PyQt5 circular GUI, complications, renderer
  boot/                 # orchestrator, enrollment, power_manager
  services/             # health, perpetual, memlog, diaglog, ingest_watcher
  ble_server.py         # On-demand BLE GATT server for AuraConnect file transfers

containers/
  llm/                  # Qwen2.5 model + RAG modules (loaded in-process by aura.py)
  whisper/              # faster-whisper STT (loaded in-process by aura.py)
  memory/               # Conversation memory + FAISS (separate REST proc, port 11438)

shared/                 # Code shared across services (symlinked to /shared)
data/                   # Runtime state, settings, voice profiles, briefings
assets/                 # Media: boot prompts, voice samples, thinking fillers, logos
voices/                 # Voice model files (Piper ONNX)
setup/                  # Hardware scripts
boot/                   # start_aura.sh (systemd entry point)
tests/                  # QA and benchmark scripts
```

## Architecture

### Services

| Component | How it runs | Port | Notes |
|---|---|---|---|
| LLM (Qwen2.5-7B Q4_K_M) | **In-process** inside `aura.py` via llama.cpp | — | No HTTP endpoint; called directly from Python |
| Whisper STT (faster-whisper) | **In-process** inside `aura.py` (GPU, int8) | — | No HTTP endpoint |
| Memory + FAISS | Separate process, child of `aura.py` | 11438 | REST API for conversation memory & semantic search |
| RAG auto-ingest watcher | `aura-ingest.service` (systemd, independent) | — | Watches `data/input/`, embeds new files into `data/embeddings/faiss_index.bin` |
| BLE GATT server (`ble_server.py`) | On-demand subprocess from watchface AuraConnect page | — | Receives files from Mac AuraConnect app, drops them in `data/input/` for ingest |

`aura.py` shares the CUDA context across LLM + Whisper (unified memory on Jetson). Symlinks at `/shared`, `/app`, `/models` point into the repo.

> **Heads-up for older docs:** earlier revisions of this file described the LLM and Whisper as separate REST services (`run_llm_native.sh` on :11434, `run_whisper_native.sh` on :5000). That architecture was retired; both are now in-process. The boot log line `[health] Memory service managed by start_aura.sh; Whisper+LLM in-process` is the canonical confirmation.

### Main Application (`aura/`)

The entry point is `aura/aura.py`. It orchestrates:

- **`voice/listener.py`** — Continuous audio capture from XVF3800 via sounddevice, Silero VAD, optional OpenWakeWord wake-word gating, and direct in-process call to Whisper for transcription.
- **`voice/speaker.py`** — TTS playback using Piper; plays audio via `aplay` (ALSA).
- **`voice/llm_client.py`** — Wrapper around the in-process LLM (streaming + non-streaming).
- **`core/state.py`** — Global singleton for playback state, settings. Persists to `data/app_settings.json`.
- **`core/config.py`** — All paths, constants, color schemes. Single source of truth.
- **`gui/window.py`** — PyQt5 circular GUI with animated states.
- **`services/perpetual.py`** — Background rumination engine (daily briefs, proactive questions).
- **`services/ingest_watcher.py`** — Standalone RAG auto-ingest daemon (run by `aura-ingest.service`, not by `aura.py`).
- **`ble_server.py`** — Standalone BLE GATT server, launched on-demand by the AuraConnect watchface page.

### Shared Module (`shared/`)

`shared/llm_base.py` contains `BaseLLMContainer` — the base class for LLM services. It handles model loading, llama.cpp inference, health checks, and sentence tagging for TTS chunking.

`shared/rag/` contains the modular RAG client (CPU mode: in-process FAISS, no external service).

### LLM (`containers/llm/`)

The LLM module lives at `containers/llm/`, but on the puck it is **loaded in-process** by `aura.py` (no Flask server, no port 11434). `container_rest.py` is still used as the inference module — its routes and helpers are imported and called directly from Python rather than reached over HTTP. Model: Qwen2.5-7B-Instruct-Q4_K_M.

If you need to expose it as a REST service for testing on a non-puck machine, the legacy `run_llm_native.sh` will still work — but on the pucks themselves nothing listens on 11434.

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
  → WAV bytes → in-process Whisper (faster-whisper, GPU int8)
  → Transcript text → in-process LLM (Qwen2.5-7B, llama.cpp)
  → Streamed sentence tokens → speaker.py
  → Piper TTS → aplay (ALSA)
```

### File-transfer / RAG ingest pipeline

```
Mac AuraConnect app
  → BLE GATT (write-with-response, MTU-capped 180 B chunks, SHA256 verify)
  → ble_server.py on the puck (on-demand from watchface)
  → data/input/<file>
  → aura-ingest.service (watchdog → CPU sentence-transformers → FAISS)
  → data/embeddings/{faiss_index.bin, metadata.pkl}
  → in-process LLM RAG retrieval on next query
```

## Configuration Files

- `.env` — API keys (`TELEGRAM_BOT_TOKEN`, `GITHUB_TOKEN`)
- `data/app_settings.json` — Runtime settings (TTS engine, wake word on/off)

## Key Conventions

- **RAG mode**: Set via `RAG_MODE` env var (`CPU` or `GPU`). Defaults to CPU (in-process FAISS).
- **Whisper model default**: Read dynamically from `containers/whisper/container_rest.py`'s `MODEL_NAME` variable.
- **`shared/` rule**: Any resource used by 2+ services belongs in `shared/` and is symlinked at `/shared`.
- **No containers**: Everything runs natively on Jetson via Python virtualenv. No Docker anywhere.
