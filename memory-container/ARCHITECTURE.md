# Memory Container Architecture

## Component Overview

The Memory Container is designed as a **lightweight, focused service** that leverages existing containers rather than duplicating functionality.

## Architecture Decisions

### 1. Whisper Integration: HTTP API (Not Local)

**Question:** Should memory container use the same image as whisper container?

**Answer:** No - Memory container calls Whisper container via HTTP API.

**Why:**
- ✅ **Resource Efficiency**: Avoids duplicating GPU resources and Whisper model
- ✅ **Separation of Concerns**: Whisper container handles transcription, Memory handles storage/analysis
- ✅ **Reusability**: Reuses existing optimized Whisper service
- ✅ **Scalability**: Can scale containers independently

**How it works:**
```python
# background_listener.py calls Whisper via HTTP
response = requests.post(
    f"{self.whisper_service_url}/transcribe",  # http://localhost:5000/transcribe
    files={"audio": ("speech.wav", wav_io, "audio/wav")},
    timeout=10
)
```

**Dependencies:**
- Memory container needs: `requests` (for HTTP calls)
- Memory container does NOT need: `faster-whisper`, `torch`, GPU drivers

### 2. Sentence Transformers: Required (For Embeddings)

**Question:** Why is Sentence Transformer loaded in memory container?

**Answer:** For generating embeddings, which is completely separate from transcription.

**Purpose:**
- **Transcription** (Whisper): Audio → Text
- **Embeddings** (Sentence Transformers): Text → Vector (for semantic search)

**Why needed:**
- Memory container needs to vectorize conversations for semantic search
- FAISS requires vector embeddings to perform similarity search
- This is the core functionality of the memory system

**Model:** `all-distilroberta-v1` (lightweight, fast, good quality)

### 3. FAISS-CPU: Required (For Semantic Search)

**Question:** Is faiss-cpu required as a dependency?

**Answer:** Yes, absolutely required.

**Why:**
- Memory container stores conversations as vector embeddings
- FAISS provides fast similarity search across stored conversations
- This enables the proactive suggestion system (finding similar past conversations)

**Alternative:** Could use FAISS-GPU, but:
- CPU version is sufficient for typical conversation volumes
- Avoids GPU dependency (memory container can run on CPU)
- Simpler deployment

## Dependency Breakdown

### Required Dependencies

| Package | Purpose | Why Needed |
|---------|---------|------------|
| `sentence-transformers` | Generate embeddings | Core functionality - vectorizes conversations |
| `faiss-cpu` | Semantic search | Core functionality - searches stored conversations |
| `flask` | REST API | Exposes memory operations via HTTP |
| `requests` | HTTP client | Calls Whisper container and LLM container |
| `sounddevice` | Audio capture | Background listener needs to capture audio |
| `soundfile` | Audio I/O | Process audio files for Whisper API calls |
| `numpy` | Numerical operations | Required by sentence-transformers and faiss |

### NOT Required

| Package | Why NOT Needed |
|---------|----------------|
| `faster-whisper` | Calls Whisper container via HTTP instead |
| `torch` / `cuda` | Whisper runs in separate container |
| `faiss-gpu` | Using CPU version (sufficient, simpler) |

## Data Flow

```
┌─────────────────────┐
│  Audio Input        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      HTTP API      ┌──────────────────┐
│ Background Listener │ ──────────────────> │ Whisper Container│
│ (Memory Container)   │                    │ (Transcription)  │
└──────────┬──────────┘                    └─────────┬────────┘
           │                                         │
           │ Text                                    │ Text
           ▼                                         ▼
┌─────────────────────┐                    ┌──────────────────┐
│ Memory Manager      │                    │ (Returns text)    │
│                     │                    └──────────────────┘
│ - Sentence Trans.   │
│   (Embeddings)      │
│                     │
│ - FAISS Index       │
│   (Search)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Proactive Analyzer  │
│ (LLM Integration)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ TTS Output          │
│ (Suggestions)       │
└─────────────────────┘
```

## Optimization Opportunities

### Current Design (Recommended)
- ✅ Memory container calls Whisper via HTTP
- ✅ Uses CPU-based FAISS
- ✅ Lightweight sentence transformer model

### Alternative: Local Whisper (Not Recommended)
If we wanted memory container to run Whisper locally:
- ❌ Would need `faster-whisper` dependency
- ❌ Would need GPU access
- ❌ Would duplicate Whisper model in memory
- ❌ More complex deployment
- ✅ Slightly lower latency (no HTTP overhead)

**Verdict:** Current design is better - HTTP overhead is minimal compared to transcription time.

## Resource Usage

### Memory Container
- **CPU**: Moderate (embeddings, FAISS search)
- **Memory**: ~500MB-1GB (sentence transformer model + FAISS index)
- **GPU**: Not needed (uses CPU FAISS, calls Whisper container)

### Whisper Container (Separate)
- **GPU**: Required (faster-whisper)
- **Memory**: ~2-4GB (Whisper model)

## Summary

1. **Whisper**: Memory container calls Whisper container via HTTP ✅
2. **Sentence Transformers**: Required for embeddings ✅
3. **FAISS-CPU**: Required for semantic search ✅

The current architecture is optimal for:
- Resource efficiency
- Separation of concerns
- Scalability
- Maintainability

