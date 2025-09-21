# Aura Voice Assistant - Deployment Guide

## Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for aura-control)
- ElevenLabs API key

## Quick Start

### 1. Setup Environment

```bash
cp env.template .env
# Edit .env with your ELEVEN_API_KEY and ELEVEN_VOICE_ID
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Run Aura Control

```bash
cd aura-control
uv sync
uv run python main.py
```

## Verify Deployment

```bash
docker-compose ps
curl http://localhost:5000/health  # Whisper
curl http://localhost:5001/health  # LLM
curl http://localhost:5002/health  # TTS
curl http://localhost:5003/health  # RAPIDS
```

## Environment Variables

### Required
- `ELEVEN_API_KEY`: Your ElevenLabs API key
- `ELEVEN_VOICE_ID`: Your ElevenLabs voice ID

### Optional (with defaults)
- `CUDA_VISIBLE_DEVICES`: GPU device (default: `0`)
- `WHISPER_MODEL`: Whisper model (default: `base.en`)
- `MODEL_PATH`: LLM model path (default: `/models/qwen2.5-1.5b-instruct-q4_0.gguf`)

### RAG Configuration
- `CHUNK_SIZE`: Text chunk size (default: `512`)
- `EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_BATCH_SIZE`: Batch size for embeddings (default: `32`)
- `TOP_K`: Number of chunks to retrieve (default: `3`)
- `SCORE_THRESHOLD`: Similarity threshold (default: `0.7`)
- `LLM_URL`: LLM container URL (default: `http://localhost:11434`)
- `LLM_TIMEOUT`: LLM request timeout (default: `30`)
- `INPUT_DIR`: Document input directory (default: `/shared/input`)
- `INDEX_PATH`: Vector index storage path (default: `/shared/vector_index`)
- `DEBUG`: Enable debug logging (default: `false`)

## Troubleshooting

**Services not starting:**
```bash
docker-compose logs <service-name>
```

**GPU issues:**
Set `CUDA_VISIBLE_DEVICES=-1` in `.env` to disable GPU
