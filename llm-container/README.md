# Aura Generic Conversational Container

## Overview

Generic conversational container for Aura providing general conversation and RAG-powered knowledge queries. This is the non-medical alternative to `llm-medical-container`.

## Features

- **General conversation** - Friendly, helpful responses
- **RAG integration** - Document Q&A powered by FAISS semantic search
- **Fast responses** - Lightweight LLM model (Nemotron-Mini-4B or Llama-3.2-1B)
- **Streaming support** - Real-time response streaming for TTS
- **Session management** - Per-user conversation context

## Architecture

```
User Input
  ↓
Generic Conversational Handler
  ↓
┌─────────────────────┬─────────────────┐
│ RAG Search          │ Direct LLM      │
│ (if RAG enabled)    │ (fallback)      │
│                     │                 │
│ - Document Q&A      │ - Conversation  │
│ - Context-aware     │ - General chat  │
└─────────────────────┴─────────────────┘
  ↓
Response (streaming or non-streaming)
```

## Configuration

Set in `.env` via `aura_config.sh`:

- `USE_MEDICAL_MODE=false` - Enable generic mode
- `SIMPLE_MODEL_PATH` - Model to use
- `SIMPLE_CHAT_FORMAT` - Chat template format
- `RAG_ENABLED` - Enable/disable RAG

## Toggle Between Modes

```bash
# Switch to generic mode
./aura_config.sh mode off

# Switch to medical mode
./aura_config.sh mode on

# Restart containers
docker-compose restart
```

## Endpoints

- `POST /chat-tg` - Non-streaming (Telegram)
- `POST /chat-tts` - Streaming (TTS/Voice)
- `GET /health` - Health check

## Differences from Medical Container

| Feature | Medical Container | Generic Container |
|---------|------------------|-------------------|
| Symptom Assessment | ✅ | ❌ |
| Adaptive Diagnostics | ✅ | ❌ |
| OLDCARTS Questioning | ✅ | ❌ |
| Medical Guidelines | ✅ (144 conditions) | ❌ |
| Generic Conversation | ✅ | ✅ |
| RAG Q&A | ✅ | ✅ |
| File Size | ~100MB | ~50MB |

