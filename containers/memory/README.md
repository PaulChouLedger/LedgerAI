# Memory Container - Proactive AI Brain Component

The Memory Container serves as the "brain" or logical thinking component of Aura, working in the background to continuously analyze conversations and provide proactive suggestions.

## Overview

The Memory Container:
1. **Continuously transcribes audio** - Even when wake word is not active, it listens and transcribes all conversations
2. **Vectorizes and stores conversations** - All transcriptions are vectorized and stored in a RAG system for semantic search
3. **Runs real-time analysis** - Compares current conversations with stored data to identify patterns and insights
4. **Provides proactive suggestions** - When useful insights are found, Aura proactively suggests ideas to help users solve problems

## Architecture

### Components

1. **MemoryManager** (`memory_manager.py`)
   - Handles vectorization of conversations using sentence transformers
   - Stores conversations in FAISS index for fast semantic search
   - Manages conversation metadata and timestamps

2. **ProactiveAnalyzer** (`proactive_analyzer.py`)
   - Analyzes current conversations against stored memory
   - Uses LLM to generate contextual suggestions
   - Implements cooldown and deduplication to avoid repetitive suggestions

3. **BackgroundListener** (`background_listener.py`)
   - Continuously listens to audio input
   - Uses VAD (Voice Activity Detection) to identify speech segments
   - Transcribes audio via Whisper service
   - Works independently of wake word system

4. **REST API** (`container_rest.py`)
   - Provides HTTP endpoints for memory operations
   - Manages container lifecycle (start/stop listener)
   - Handles conversation storage and search

## API Endpoints

### Health & Status
- `GET /health` - Health check with memory statistics
- `GET /stats` - Get detailed memory statistics

### Listener Control
- `POST /start` - Start background listener
- `POST /stop` - Stop background listener

### Memory Operations
- `POST /store` - Manually store a conversation
  ```json
  {
    "text": "conversation text",
    "source": "wake_word",
    "metadata": {}
  }
  ```

- `POST /search` - Search for similar conversations
  ```json
  {
    "query": "search query",
    "k": 5,
    "threshold": 0.5
  }
  ```

- `GET /recent?hours=24&limit=50` - Get recent conversations

- `POST /analyze` - Manually trigger analysis
  ```json
  {
    "text": "conversation to analyze"
  }
  ```

## Configuration

Environment variables:
- `WHISPER_SERVICE_URL` - URL of Whisper transcription service (default: `http://localhost:5000`)
- `LLM_SERVICE_URL` - URL of LLM service (default: `http://localhost:11434`)
- `MEMORY_DIR` - Directory for storing memory data (default: `/app/data/memory`)
- `AUDIO_DEVICE_NAME` - Audio device name (default: `reSpeaker`)
- `PORT` - REST API port (default: `11438`)

## Usage

### Starting the Container

The container is included in `docker-compose.yml` and starts automatically with other services.

### Starting Background Listener

```bash
curl -X POST http://localhost:11438/start
```

### Storing Conversations

The main listener can forward transcriptions to memory container:

```python
import requests

# After transcription in main listener
requests.post(
    "http://localhost:11438/store",
    json={
        "text": transcribed_text,
        "source": "wake_word",
        "metadata": {"duration": audio_duration}
    }
)
```

### Manual Analysis

```bash
curl -X POST http://localhost:11438/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I am having trouble with X"}'
```

## Integration with Main Pipeline

The memory container works **independently** of the main SST->TTS pipeline:

1. **Wake word triggers main pipeline**: SST → LLM → TTS
2. **Memory container runs in background**: Continuously transcribes and analyzes
3. **Proactive suggestions**: When insights are found, memory container sends suggestions to TTS

### Integration Options

**Option 1: Background Listener (Recommended)**
- Memory container runs its own background listener
- Continuously transcribes all audio
- No changes needed to main listener

**Option 2: Transcription Forwarding**
- Main listener forwards all transcriptions to memory container
- Memory container analyzes forwarded transcriptions
- Disable background listener to avoid conflicts

## Proactive Suggestions

When the analyzer finds useful insights, it generates suggestions like:

> "Excuse me, have you thought about trying X? Based on your previous experience with Y, I think X might be beneficial to your case."

Suggestions are:
- Generated using LLM analysis of current vs. stored conversations
- Filtered for relevance (similarity threshold)
- Deduplicated to avoid repetition
- Rate-limited (cooldown period)

## Data Storage

Memory data is stored in `/app/data/memory/`:
- `conversations.jsonl` - All conversations (JSONL format)
- `embeddings.npy` - Vector embeddings
- `metadata.pkl` - Conversation metadata
- `memory_index.faiss` - FAISS search index

## Performance Considerations

- **Background listener** uses VAD to minimize unnecessary transcriptions
- **FAISS index** is rebuilt periodically (every 10 conversations) for optimal performance
- **Embeddings** are cached and persisted to disk
- **Analysis** runs asynchronously to avoid blocking

## Troubleshooting

### Audio Device Conflicts
If the background listener conflicts with main listener:
1. Disable background listener: `POST /stop`
2. Use transcription forwarding from main listener instead

### Memory Growth
Memory grows over time. To manage:
- Old conversations are still searchable but don't affect recent analysis
- Consider periodic cleanup of very old conversations if needed

### Suggestion Quality
Adjust in `proactive_analyzer.py`:
- `similarity_threshold` - Minimum similarity for relevant matches
- `suggestion_cooldown` - Time between suggestions
- `min_conversations_for_analysis` - Minimum conversations needed

