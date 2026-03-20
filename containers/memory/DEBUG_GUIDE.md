# Debug Guide - Tracking Information Flow

This guide shows you how to track information as it flows through the memory container system.

## Enabling Debug Logging

### Environment Variable

Set `LOG_LEVEL=DEBUG` to see detailed information flow:

```bash
# In docker-compose.yml
environment:
  - LOG_LEVEL=DEBUG

# Or when running container
docker run -e LOG_LEVEL=DEBUG ...
```

### Log Levels

- **INFO** (default): Shows key operations and results
- **DEBUG**: Shows detailed step-by-step information flow
- **WARNING**: Shows only warnings and errors

## Console Output Guide

### Information Flow Icons

The console uses emojis/icons to track different operations:

| Icon | Meaning |
|------|---------|
| 📥 | Receiving data |
| 📤 | Sending data |
| 📝 | Text/transcription |
| 💾 | Storing data |
| 🔢 | Generating embeddings |
| 🔍 | Searching |
| 🧠 | LLM processing |
| 💡 | Suggestion generated |
| ✅ | Success |
| ⚠️ | Warning |
| ❌ | Error |
| ⏳ | Waiting/cooldown |
| ℹ️ | Information |

## Example Console Output

### 1. Wake Word Triggered

```
[2024-01-15 10:30:45] [Memory] [INFO] 📤 Forwarding transcription to memory container (source: wake_word)
[2024-01-15 10:30:45] [Memory] [DEBUG] 📝 Text: 'I'm having trouble with my project...'
[2024-01-15 10:30:45] [Memory] [INFO] ✅ Forwarded to memory container (ID: a1b2c3d4, 0.023s)
```

### 2. Memory Container Receives

```
[2024-01-15 10:30:45] [memory-container] [INFO] 📥 Received conversation to store (source: wake_word)
[2024-01-15 10:30:45] [memory-container] [DEBUG] 📝 Text: 'I'm having trouble with my project...'
[2024-01-15 10:30:45] [memory-container] [DEBUG] 💾 Storing conversation in memory manager...
```

### 3. Storing Conversation

```
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 📝 Storing conversation (source: wake_word, length: 45 chars)
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 📚 Added to conversations list (total: 12)
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 💾 Saved conversation to disk
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 🔢 Generating embedding for conversation...
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] ✅ Embedding generated (shape: (768,))
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 📊 Added to embeddings array (total: 12)
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] ✅ Added embedding to FAISS index (total vectors: 12)
[2024-01-15 10:30:45] [MemoryManager] [INFO] ✅ Stored conversation: 'I'm having trouble with my project...' (ID: a1b2c3d4, source: wake_word)
```

### 4. Analysis Triggered

```
[2024-01-15 10:30:45] [memory-container] [DEBUG] 🔍 Analyzing conversation for suggestions...
[2024-01-15 10:30:45] [ProactiveAnalyzer] [DEBUG] 🔍 Starting analysis for: 'I'm having trouble with my project...'
[2024-01-15 10:30:45] [ProactiveAnalyzer] [DEBUG] 📊 Total conversations: 12, proceeding with analysis
[2024-01-15 10:30:45] [ProactiveAnalyzer] [DEBUG] 🔎 Searching for similar conversations (threshold: 0.65)...
```

### 5. Similarity Search

```
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 🔍 Searching for similar conversations (query: 'I'm having trouble with my project...', k=5, threshold=0.65)
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 🔢 Generating query embedding...
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] ✅ Query embedding normalized
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 🔎 Searching FAISS index (12 vectors)...
[2024-01-15 10:30:45] [MemoryManager] [DEBUG] 📊 Search returned 5 candidates
[2024-01-15 10:30:45] [MemoryManager] [DEBUG]   [1] Score: 0.782 >= 0.65 ✅ - 'I had issues with my project last week...'
[2024-01-15 10:30:45] [MemoryManager] [DEBUG]   [2] Score: 0.701 >= 0.65 ✅ - 'Project management is challenging...'
[2024-01-15 10:30:45] [MemoryManager] [DEBUG]   [3] Score: 0.523 < 0.65 ❌ - Skipped
[2024-01-15 10:30:45] [MemoryManager] [INFO] ✅ Found 2 similar conversations (threshold: 0.65)
```

### 6. Suggestion Generation

```
[2024-01-15 10:30:45] [ProactiveAnalyzer] [INFO] ✅ Found 2 similar conversations
[2024-01-15 10:30:45] [ProactiveAnalyzer] [DEBUG] 📚 Retrieved 5 recent conversations for context
[2024-01-15 10:30:45] [ProactiveAnalyzer] [DEBUG] 🧠 Generating suggestion using LLM...
[2024-01-15 10:30:46] [ProactiveAnalyzer] [INFO] 💡 Suggestion generated successfully
[2024-01-15 10:30:46] [memory-container] [INFO] 💡 Generated suggestion: 'Excuse me, have you thought about trying X? Based on your previous experience with Y...'
```

### 7. Suggestion Sent to TTS

```
[2024-01-15 10:30:46] [memory-container] [INFO] ✅ Suggestion sent to TTS via speaker module
```

## Complete Flow Example

Here's a complete example showing the full flow:

```
[10:30:45] [Listener] [INFO] Wake word detected
[10:30:45] [Listener] [INFO] Transcribed: "I'm having trouble with my project"
[10:30:45] [Memory] [INFO] 📤 Forwarding transcription to memory container (source: wake_word)
[10:30:45] [Memory] [INFO] ✅ Forwarded to memory container (ID: a1b2c3d4, 0.023s)
[10:30:45] [memory-container] [INFO] 📥 Received conversation to store (source: wake_word)
[10:30:45] [MemoryManager] [INFO] ✅ Stored conversation: 'I'm having trouble with my project...' (ID: a1b2c3d4)
[10:30:45] [memory-container] [DEBUG] 🔍 Analyzing conversation for suggestions...
[10:30:45] [MemoryManager] [INFO] ✅ Found 2 similar conversations (threshold: 0.65)
[10:30:45] [ProactiveAnalyzer] [INFO] 💡 Suggestion generated successfully
[10:30:46] [memory-container] [INFO] 💡 Generated suggestion: 'Excuse me, have you thought about...'
[10:30:46] [memory-container] [INFO] ✅ Suggestion sent to TTS via speaker module
[10:30:46] [Speaker] [INFO] Speaking: "Excuse me, have you thought about..."
```

## Tracking Specific Operations

### Track Storage Operations

Look for:
- `📝 Storing conversation`
- `💾 Saved conversation to disk`
- `🔢 Generating embedding`
- `✅ Stored conversation`

### Track Search Operations

Look for:
- `🔍 Searching for similar conversations`
- `🔎 Searching FAISS index`
- `📊 Search returned X candidates`
- `✅ Found X similar conversations`

### Track Analysis Operations

Look for:
- `🔍 Starting analysis`
- `📊 Total conversations: X`
- `🧠 Generating suggestion using LLM`
- `💡 Suggestion generated`

### Track Forwarding Operations

Look for:
- `📤 Forwarding transcription`
- `✅ Forwarded to memory container`
- `⏱️ Memory container request timeout`

## Debugging Tips

### 1. Check if Memory Container is Running

```bash
curl http://localhost:11438/health
```

Expected output:
```json
{
  "status": "healthy",
  "listener_enabled": true,
  "memory_stats": {
    "total_conversations": 12,
    "total_embeddings": 12
  }
}
```

### 2. Check Memory Stats

```bash
curl http://localhost:11438/stats
```

### 3. View Recent Conversations

```bash
curl http://localhost:11438/recent?hours=24&limit=10
```

### 4. Test Manual Storage

```bash
curl -X POST http://localhost:11438/store \
  -H "Content-Type: application/json" \
  -d '{"text": "test conversation", "source": "manual"}'
```

Watch console for:
- Storage confirmation
- Embedding generation
- Analysis trigger

### 5. Test Search

```bash
curl -X POST http://localhost:11438/search \
  -H "Content-Type: application/json" \
  -d '{"query": "project trouble", "k": 5, "threshold": 0.5}'
```

## Common Issues

### No Logs Appearing

1. Check log level: `export LOG_LEVEL=DEBUG`
2. Check if logger is configured correctly
3. Verify container is running: `docker ps | grep memory`

### Memory Container Not Receiving Data

Look for:
- `📤 Forwarding transcription` (should appear)
- `📥 Received conversation` (should appear)
- If missing, check network connectivity

### Suggestions Not Generated

Look for:
- `⏳ Cooldown active` - Wait for cooldown to expire
- `ℹ️ Not enough conversations` - Need at least 3 conversations
- `ℹ️ No similar conversations found` - No matches above threshold

### Slow Performance

Look for timing information:
- `✅ Forwarded to memory container (ID: x, 0.023s)` - Should be < 0.1s
- `⏱️ Memory container request timeout` - Indicates slow response

## Filtering Logs

### View Only Memory Container Logs

```bash
docker logs memory-container 2>&1 | grep -E "\[Memory|\[memory-container|\[MemoryManager|\[ProactiveAnalyzer"
```

### View Only Storage Operations

```bash
docker logs memory-container 2>&1 | grep -E "Stored|Storing|💾|📝"
```

### View Only Search Operations

```bash
docker logs memory-container 2>&1 | grep -E "Search|🔍|🔎"
```

### View Only Suggestions

```bash
docker logs memory-container 2>&1 | grep -E "Suggestion|💡"
```

## Real-Time Monitoring

### Watch Logs in Real-Time

```bash
docker logs -f memory-container
```

### Watch with Filter

```bash
docker logs -f memory-container 2>&1 | grep --line-buffered -E "📥|📤|💾|🔍|💡|✅"
```

## Summary

The console output provides a complete trace of information flow:

1. **📤 Forwarding** - Main listener sends to memory container
2. **📥 Receiving** - Memory container receives data
3. **💾 Storing** - Conversation stored and vectorized
4. **🔍 Analyzing** - Analysis triggered
5. **🔎 Searching** - Similar conversations found
6. **💡 Suggesting** - Suggestion generated
7. **✅ Complete** - Operation successful

Enable `LOG_LEVEL=DEBUG` for the most detailed information flow tracking.

