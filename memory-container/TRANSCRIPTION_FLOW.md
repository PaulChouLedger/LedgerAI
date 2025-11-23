# Memory Container Transcription Flow

## How Memory Container Receives Conversations

### Two Methods (Both Active):

1. **Background Listener (Always Running)** ✅
   - **Continuously transcribes ALL audio** (with or without wake word)
   - Starts automatically when memory container starts
   - Stores all conversations for analysis
   - This is the **primary method** for continuous memory

2. **Wake Word Forwarding (Additional)** ✅
   - Main listener also forwards transcriptions after wake word
   - Provides redundancy and ensures wake word conversations are captured
   - Works in parallel with background listener

## Current Setup: Always Listening

**The memory container is ALWAYS listening and transcribing:**

```
Background Listener (Always Running)
    ↓
Continuously Transcribes ALL Audio
    ↓
Transcription: "How do I treat pneumonia?"
    ↓
Memory Container Receives & Stores
    ↓
Vectorization & Analysis
    ↓
(If wake word was used) → Aura responds with TTS
(If no wake word) → Conversation stored silently
```

**Wake Word Purpose:**
- **Wake word = Aura responds with TTS** to the current conversation
- **No wake word = Conversation still stored**, but Aura doesn't speak back
- **Proactive suggestions** from analysis are separate and can be spoken anytime

## How to See Debug Logs

### 1. Enable Debug Logging

**Option A: Set in docker-compose.yml** (Recommended)

```yaml
# In setup/docker-compose.yml
memory:
  environment:
    - LOG_LEVEL=DEBUG  # Change from INFO to DEBUG
```

Then restart:
```bash
cd ~/LedgerAI/setup
docker compose restart memory
```

**Option B: Set at runtime**

```bash
# Stop container
docker stop memory-container

# Start with debug
docker run -e LOG_LEVEL=DEBUG ...
```

### 2. Watch Logs in Real-Time

```bash
# Watch all memory container logs
docker logs -f memory-container

# Filter for key events
docker logs -f memory-container 2>&1 | grep -E "📥|📤|💾|🔢|🔍|✅|💡"

# Watch both main Aura and memory container
docker logs -f memory-container & docker logs -f setup-llm-generic-1
```

### 3. Watch Main Aura Logs (Forwarding)

In your main Aura console, you should see:

```
[Memory] 📤 Forwarding transcription to memory container (source: wake_word)
[Memory] ✅ Forwarded to memory container (ID: b2d16e6bd5d20062, 0.023s)
```

## Complete Debug Flow Example

### Step 1: Wake Word Triggered

**Main Aura Console:**
```
[Listener] 🎤 Wake word detected!
[Whisper] 📝 Transcribed: 'How do I treat pneumonia?'
[Memory] 📤 Forwarding transcription to memory container (source: wake_word)
[Memory] ✅ Forwarded to memory container (ID: b2d16e6bd5d20062, 0.023s)
[LLM] ✅ Prompt to LLM: How do I treat pneumonia?
```

**Memory Container Logs (with DEBUG):**
```
[memory-container] [INFO] 📥 Received conversation to store (source: wake_word)
[memory-container] [DEBUG] 📝 Text: 'How do I treat pneumonia?...'
[memory-container] [DEBUG] 💾 Storing conversation in memory manager...
```

### Step 2: Storage & Vectorization

**Memory Container Logs:**
```
[MemoryManager] [DEBUG] 📝 Storing conversation (source: wake_word, length: 32 chars)
[MemoryManager] [DEBUG] 📚 Added to conversations list (total: 1)
[MemoryManager] [DEBUG] 💾 Saved conversation to disk
[MemoryManager] [DEBUG] 🔢 Generating embedding for conversation...
[MemoryManager] [DEBUG] ✅ Embedding generated (shape: (768,))
[MemoryManager] [DEBUG] 📊 Added to embeddings array (total: 1)
[MemoryManager] [DEBUG] ✅ Added embedding to FAISS index (total vectors: 1)
[MemoryManager] [INFO] ✅ Stored conversation: 'How do I treat pneumonia?...' (ID: b2d16e6bd5d20062, source: wake_word)
[memory-container] [INFO] ✅ Conversation stored (ID: b2d16e6bd5d20062)
```

### Step 3: Analysis (if 5+ conversations)

**Memory Container Logs:**
```
[memory-container] [DEBUG] 🔍 Analyzing stored conversation for suggestions...
[ProactiveAnalyzer] [DEBUG] 🔍 Starting analysis for: 'How do I treat pneumonia?...'
[ProactiveAnalyzer] [DEBUG] 📊 Total conversations: 5, proceeding with analysis
[ProactiveAnalyzer] [DEBUG] 🔎 Searching for similar conversations (threshold: 0.65)...
[MemoryManager] [DEBUG] 🔍 Searching for similar conversations (query: '...', k=5, threshold: 0.65)
[MemoryManager] [DEBUG] 🔢 Generating query embedding...
[MemoryManager] [DEBUG] ✅ Query embedding normalized
[MemoryManager] [DEBUG] 🔎 Searching FAISS index (5 vectors)...
[MemoryManager] [DEBUG] 📊 Search returned 2 candidates above threshold
[ProactiveAnalyzer] [DEBUG] 💡 Generating suggestion based on similar conversations...
[ProactiveAnalyzer] [INFO] 💡 Suggestion generated: 'Based on previous conversations about pneumonia...'
[memory-container] [INFO] 💡 Suggestion sent to TTS
```

## Quick Debug Commands

### Check if Memory Container is Receiving Data

```bash
# Watch for incoming requests
docker logs -f memory-container 2>&1 | grep "📥 Received"

# Check recent stored conversations
curl http://localhost:11438/recent?hours=1&limit=5

# Check stats
curl http://localhost:11438/stats
```

### Verify Forwarding is Working

**In main Aura console, look for:**
```
[Memory] 📤 Forwarding transcription to memory container
[Memory] ✅ Forwarded to memory container
```

**If you DON'T see these:**
1. Check memory is enabled: Settings → AI Model Settings → Memory Container (should be ON)
2. Check memory container is running: `docker ps | grep memory`
3. Check health: `curl http://localhost:11438/health`

### Check Storage & Vectorization

```bash
# View stored conversations
curl http://localhost:11438/recent?limit=10

# Check embedding count
curl http://localhost:11438/stats | jq '.total_embeddings'

# View FAISS index info
docker exec memory-container ls -lh /app/data/memory/
```

## Background Listener (Always Running)

The background listener **starts automatically** when the memory container starts:

```bash
# Check if running
curl http://localhost:11438/health | jq '.listener_enabled'

# Stop background listener (if needed)
curl -X POST http://localhost:11438/stop

# Restart background listener
curl -X POST http://localhost:11438/start
```

**Note:** Background listener may conflict with main listener if using same audio device. If conflicts occur, the background listener will log warnings but continue attempting to transcribe.

## Summary

✅ **Memory container is ALWAYS listening** - background listener runs continuously
✅ **All conversations are transcribed and stored** - with or without wake word
✅ **Wake word only controls TTS response** - not whether conversation is stored
✅ **Proactive suggestions** from analysis are separate from wake word TTS responses
✅ **Enable DEBUG logging** to see detailed flow: transcription → storage → vectorization → analysis
✅ **Watch logs** with: `docker logs -f memory-container`

**Key Distinction:**
- **Wake Word = Aura speaks back** to current conversation (TTS response)
- **No Wake Word = Conversation still stored**, but Aura doesn't speak
- **Proactive Suggestions = Separate** - generated from analysis, can be spoken anytime

