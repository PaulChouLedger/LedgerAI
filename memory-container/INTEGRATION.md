# Memory Container Integration Guide

## Quick Start

1. **Build and start the container:**
   ```bash
   cd setup
   docker-compose up -d memory
   ```

2. **Start the background listener:**
   ```bash
   curl -X POST http://localhost:11438/start
   ```

3. **Verify it's working:**
   ```bash
   curl http://localhost:11438/health
   ```

## How It Works

### Architecture Flow

```
┌─────────────────┐
│  Audio Input    │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────────┐
│  Main Listener  │  │ Background       │
│  (Wake Word)    │  │ Listener         │
└────────┬────────┘  └────────┬─────────┘
         │                   │
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌──────────────────┐
│  Whisper        │  │  Whisper         │
│  (Wake Word)    │  │  (Background)    │
└────────┬────────┘  └────────┬─────────┘
         │                   │
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌──────────────────┐
│  LLM → TTS      │  │  Memory          │
│  (Main Pipeline)│  │  Container        │
└─────────────────┘  └────────┬─────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Vectorize   │
                        │  & Store     │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Analyze     │
                        │  & Compare   │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Generate    │
                        │  Suggestion  │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  TTS         │
                        │  (Proactive) │
                        └──────────────┘
```

### Two Integration Modes

#### Mode 1: Background Listener (Recommended)
- Memory container runs its own continuous audio listener
- Transcribes all audio independently
- No changes needed to main listener
- **Use when:** You want complete independence

#### Mode 2: Transcription Forwarding
- Main listener forwards all transcriptions to memory container
- Memory container analyzes forwarded transcriptions
- Background listener disabled
- **Use when:** Audio device conflicts occur

### Enabling Transcription Forwarding

The main listener automatically forwards transcriptions when:
- `MEMORY_ENABLED=true` (environment variable)
- `MEMORY_CONTAINER_URL` is set (default: `http://localhost:11438`)
- Memory container is running

To disable forwarding:
```bash
export MEMORY_ENABLED=false
```

## Proactive Suggestions

### How Suggestions Work

1. **Continuous Transcription**: All audio is transcribed (wake word or not)
2. **Vectorization**: Conversations are converted to embeddings
3. **Storage**: Stored in FAISS index for fast semantic search
4. **Analysis**: Current conversation compared with stored conversations
5. **Pattern Detection**: LLM analyzes patterns and generates insights
6. **Suggestion**: If useful insight found, suggestion is generated
7. **TTS**: Suggestion is spoken via TTS system

### Suggestion Examples

- "Excuse me, have you thought about trying X? Based on your previous experience with Y, I think X might be beneficial."
- "I just thought of something - in your conversation about Z, you mentioned A. Have you considered B as a potential solution?"

### Configuration

Adjust suggestion behavior in `proactive_analyzer.py`:
- `similarity_threshold`: Minimum similarity for relevant matches (default: 0.65)
- `suggestion_cooldown`: Time between suggestions in seconds (default: 60)
- `min_conversations_for_analysis`: Minimum conversations needed (default: 3)

## Troubleshooting

### Audio Device Conflicts

If background listener conflicts with main listener:

1. **Stop background listener:**
   ```bash
   curl -X POST http://localhost:11438/stop
   ```

2. **Use transcription forwarding instead:**
   - Ensure `MEMORY_ENABLED=true`
   - Main listener will forward transcriptions automatically

### Memory Container Not Responding

1. **Check if container is running:**
   ```bash
   docker ps | grep memory
   ```

2. **Check logs:**
   ```bash
   docker logs memory-container
   ```

3. **Verify port:**
   ```bash
   curl http://localhost:11438/health
   ```

### Suggestions Not Appearing

1. **Check if analyzer is working:**
   ```bash
   curl -X POST http://localhost:11438/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "test conversation"}'
   ```

2. **Check memory stats:**
   ```bash
   curl http://localhost:11438/stats
   ```

3. **Verify minimum conversations:**
   - Need at least 3 conversations for analysis
   - Check with `/recent` endpoint

### TTS Integration Issues

If suggestions aren't being spoken:

1. **Check shared file method:**
   - Memory container writes to `/shared/memory_suggestion.txt`
   - Main process should read this file periodically

2. **Add REST endpoint to speaker:**
   - Create endpoint in speaker module
   - Memory container can call it directly

3. **Use speaker module directly:**
   - If memory container runs in same process, import speaker module
   - Call `enqueue_tts_chunk()` directly

## Performance Tips

1. **Background listener uses VAD** to minimize unnecessary transcriptions
2. **FAISS index is rebuilt periodically** (every 10 conversations) for optimal performance
3. **Embeddings are cached** and persisted to disk
4. **Analysis runs asynchronously** to avoid blocking

## Data Management

### Memory Location
- `/app/data/memory/` (inside container)
- `/data/memory/` (on host, if mounted)

### Files
- `conversations.jsonl` - All conversations
- `embeddings.npy` - Vector embeddings
- `metadata.pkl` - Conversation metadata
- `memory_index.faiss` - FAISS search index

### Backup
Memory data is automatically persisted. To backup:
```bash
cp -r /data/memory /backup/memory-$(date +%Y%m%d)
```

### Cleanup
Old conversations remain searchable. To clean up very old data:
- Manually edit `conversations.jsonl` (remove old entries)
- Rebuild index: Restart container (auto-rebuilds)

