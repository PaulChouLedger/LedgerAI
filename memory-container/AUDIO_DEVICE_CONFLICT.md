# Audio Device Conflict Resolution

## The Problem

Both the **main listener** (aura-control) and the **background listener** (memory-container) try to access the same audio device. Only one process can access an audio device at a time, causing conflicts.

## Current Solution: Wake Word Forwarding (Default)

**Background listener is DISABLED by default** to avoid conflicts. Instead:

1. **Main listener** handles all audio (wake word detection + transcription)
2. **Main listener** forwards transcriptions to memory container via `/store` API
3. **Memory container** receives and stores all conversations
4. **No audio device conflict** ✅

## How It Works

```
Main Listener (aura-control)
    ↓
Handles ALL Audio
    ↓
Wake Word Detection
    ↓
Transcription (Whisper)
    ↓
[Parallel] Forward to Memory Container (/store API)
    ↓
Memory Container Stores & Analyzes
```

## Enabling Background Listener (Advanced)

If you want the background listener to run (e.g., for continuous transcription without wake word), you need to:

### Option 1: Disable Main Listener (Not Recommended)
- Disable wake word detection in main listener
- Enable background listener in memory container
- **Downside**: No wake word detection

### Option 2: Use Different Audio Devices
- Main listener uses one device
- Background listener uses a different device
- **Requires**: Multiple audio input devices

### Option 3: Share Audio Stream (Complex)
- Main listener captures audio
- Shares audio stream with memory container
- **Requires**: Custom audio sharing implementation

## Configuration

### Default (Recommended)
```yaml
# docker-compose.yml
environment:
  - ENABLE_BACKGROUND_LISTENER=false  # Default
```

### Enable Background Listener
```yaml
# docker-compose.yml
environment:
  - ENABLE_BACKGROUND_LISTENER=true  # May conflict with main listener
```

Or set environment variable:
```bash
export ENABLE_BACKGROUND_LISTENER=true
docker compose restart memory
```

## Current Status

✅ **Wake word forwarding works** - All wake word conversations are stored
✅ **No audio conflicts** - Background listener disabled by default
✅ **Memory container receives data** - Via `/store` API
✅ **All conversations stored** - Wake word transcriptions are forwarded

## Recommendation

**Keep background listener disabled** (default). Wake word forwarding provides:
- ✅ All conversations stored
- ✅ No audio device conflicts
- ✅ Reliable operation
- ✅ Works with existing setup

The background listener is only needed if you want continuous transcription **without** wake word detection, which is not the typical use case.

