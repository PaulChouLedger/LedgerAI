# Wake Word Flow with Memory Container

## Overview

When wake word is triggered, the transcription is forwarded to **BOTH**:
1. **LLM Container** → TTS (for immediate response)
2. **Memory Container** → Storage & Vectorization (for proactive suggestions)

This happens in **parallel** to ensure Aura responds quickly via TTS while also storing the conversation for future analysis.

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Wake Word Detected                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Transcribe    │
                    │  (Whisper)     │
                    └───────┬────────┘
                            │
                    Text: "User query"
                            │
                            ├─────────────────────┐
                            │                     │
                            ▼                     ▼
                    ┌───────────────┐    ┌──────────────────┐
                    │  LLM Container│    │ Memory Container │
                    │  (Main Path)   │    │  (Parallel Path) │
                    └───────┬────────┘    └────────┬─────────┘
                            │                     │
                            │                     │
                            ▼                     ▼
                    ┌───────────────┐    ┌──────────────────┐
                    │  Generate     │    │  Store &         │
                    │  Response     │    │  Vectorize       │
                    └───────┬────────┘    └────────┬─────────┘
                            │                     │
                            │                     │
                            ▼                     ▼
                    ┌───────────────┐    ┌──────────────────┐
                    │  TTS          │    │  Analyze &       │
                    │  (Speak)      │    │  Compare        │
                    └───────────────┘    └────────┬─────────┘
                                                   │
                                                   │ (Later)
                                                   ▼
                                          ┌──────────────────┐
                                          │  Proactive       │
                                          │  Suggestion      │
                                          │  (If insight)     │
                                          └──────────────────┘
```

## Implementation Details

### Code Location: `listener.py`

```python
def send_to_llm(text):
    # ... context management ...
    
    # Forward to memory container (NON-BLOCKING, parallel)
    if MEMORY_AVAILABLE:
        threading.Thread(
            target=forward_to_memory,
            args=(text, "wake_word"),
            daemon=True
        ).start()
    
    # Send to LLM for TTS response (MAIN PATH, must not be blocked)
    speak_llm_response(text)
```

### Key Points

1. **Parallel Execution**: Memory forwarding happens in a separate thread
2. **Non-Blocking**: Memory container call doesn't delay TTS response
3. **Timeout Protection**: Memory forwarding has 2-second timeout
4. **Graceful Degradation**: If memory container is unavailable, main pipeline continues

## Two Paths Explained

### Path 1: Main Pipeline (LLM → TTS)
- **Purpose**: Immediate response to user
- **Flow**: Transcription → LLM → TTS → Speak
- **Priority**: High (user expects quick response)
- **Blocking**: Yes (sequential, must complete)

### Path 2: Memory Pipeline (Storage → Analysis)
- **Purpose**: Store conversation for future analysis
- **Flow**: Transcription → Memory Container → Store → Vectorize → Analyze
- **Priority**: Low (background operation)
- **Blocking**: No (runs in parallel thread)

## Benefits

1. **Fast Response**: TTS response is not delayed by memory operations
2. **Complete Storage**: All wake word conversations are stored
3. **Proactive Suggestions**: Memory container can generate suggestions later
4. **Resilient**: If memory container fails, main pipeline continues

## Example Flow

**User says:** "Hey Aura, I'm having trouble with my project"

1. **Wake word detected** → "Hey Aura"
2. **Transcription** → "I'm having trouble with my project"
3. **Parallel execution:**
   - **Path 1 (Main)**: LLM generates response → TTS speaks → "I can help you with that. What specific issue are you facing?"
   - **Path 2 (Memory)**: Stores "I'm having trouble with my project" → Vectorizes → Analyzes
4. **Later**: If memory container finds similar past conversations, it may suggest: "Excuse me, based on your previous project issues, have you tried X?"

## Configuration

### Enable/Disable Memory Forwarding

```bash
# Enable (default)
export MEMORY_ENABLED=true

# Disable
export MEMORY_ENABLED=false
```

### Memory Container URL

```bash
export MEMORY_CONTAINER_URL=http://localhost:11438
```

## Troubleshooting

### Memory forwarding not working

1. **Check if memory container is running:**
   ```bash
   curl http://localhost:11438/health
   ```

2. **Check logs:**
   ```bash
   # Main listener logs
   tail -f /path/to/aura-control/core/listener.log
   
   # Memory container logs
   docker logs memory-container
   ```

3. **Verify environment variable:**
   ```bash
   echo $MEMORY_ENABLED
   echo $MEMORY_CONTAINER_URL
   ```

### TTS response delayed

- Memory forwarding should NOT delay TTS
- If delayed, check if memory container is blocking (should timeout in 2 seconds)
- Verify threading is working correctly

## Summary

✅ **Wake word transcriptions go to BOTH:**
- LLM Container (for TTS response)
- Memory Container (for storage/vectorization)

✅ **Executes in parallel:**
- Main path (LLM → TTS) is not blocked
- Memory path runs in background thread

✅ **Resilient:**
- If memory container fails, main pipeline continues
- Timeout protection prevents blocking

