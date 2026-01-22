# TTS (Text-to-Speech) Component

## Overview

The TTS component converts text responses from the LLM into natural-sounding speech using **ElevenLabs API**. It handles streaming responses, sentence batching, and audio playback via ALSA.

## Architecture

### Core Module
- **Location**: `aura-control/core/speaker.py`
- **API**: ElevenLabs Cloud API
- **Format**: PCM 22050 Hz mono
- **Playback**: ALSA `aplay` with automatic format conversion

## Core Functions

### 1. LLM Response Streaming

**Function**: `speak_llm_response(prompt, context="")`

**Process**:
1. **HTTP Request**: POST to `http://localhost:11434/chat-tts`
   - Sends prompt and context
   - Requests streaming response
   - Timeout: 20 seconds

2. **Token Processing**:
   - Receives tokens via Server-Sent Events (SSE)
   - Handles sentence control tags:
     - `<sentence_start>`: Start buffering new sentence
     - `<sentence_end>`: Send buffered sentence to TTS
     - `<pause>`: Pause marker (future use)

3. **Sentence Buffering**:
   - Accumulates tokens between tags
   - Joins tokens into complete sentences
   - Sends to TTS queue when `<sentence_end>` received

**Sentence Tag Handling**:
```python
if token == '<sentence_start>':
    sentence_buffer = []
    in_sentence = True
elif token == '<sentence_end>':
    if sentence_buffer:
        clean_text = ''.join(sentence_buffer).strip()
        enqueue_tts_chunk(clean_text)
    sentence_buffer = []
    in_sentence = False
else:
    # Accumulate tokens
    sentence_buffer.append(token)
```

### 2. Sentence Batching

**Function**: `enqueue_tts_chunk(text)`

**Purpose**: Reduces API calls by batching multiple sentences

**Configuration**:
```python
TTS_BATCH_ENABLED = True
TTS_BATCH_MAX_WORDS = 50        # Max words per batch
TTS_BATCH_MIN_WORDS = 3         # Min words for first batch
TTS_BATCH_MAX_CHUNKS = 2        # Max chunks per batch
TTS_BATCH_TIMEOUT = 0.02        # Timeout (20ms) for batching
```

**Low-Latency Start**:
- **First Batch**: Flushes immediately when `TTS_BATCH_MIN_WORDS` reached
- **Subsequent Batches**: Wait for threshold or timeout
- **Single Chunks**: Flushed immediately (no delay)

**Batching Logic**:
1. **Accumulate**: Add chunks to batch buffer
2. **Check Thresholds**:
   - If total words ≥ `TTS_BATCH_MAX_WORDS` → flush immediately
   - If total chunks ≥ `TTS_BATCH_MAX_CHUNKS` → flush immediately
   - If single chunk → flush immediately (low latency)
3. **Timeout**: Start timer if batch not flushed
4. **Flush**: Join chunks with spaces, send to queue

### 3. Initials Merging

**Function**: `check_for_initials_merge(text)`

**Purpose**: Handles cases like "J.K." followed by "Rowling"

**Logic**:
1. **Detect Initials**: Pattern: `[A-Z]\.(?:[A-Z]\.)*`
2. **Store Pending**: If text ends with initials, store for next chunk
3. **Merge**: If next chunk is a capitalized single word → merge
4. **Enqueue**: Send merged text to TTS

**Example**:
- Chunk 1: "written by J.K."
- Chunk 2: "Rowling."
- Result: "written by J.K. Rowling."

### 4. SSML Wrapping

**Function**: `ssml_wrap(text)`

**Purpose**: Adds SSML markup for better speech quality

**Features**:
- **Breaks**: Commas/semicolons → 300ms pause
- **Sentence Pauses**: Periods/exclamation → 600ms pause
- **Emphasis**: Important words (`really`, `important`, `urgent`)
- **Emotion Detection**: Analyzes text for emotion
- **Prosody**: Rate and pitch control

**SSML Structure**:
```xml
<speak>
  <voice emotion='excited'>
    <prosody rate='100%' pitch='100%'>
      {text with breaks and emphasis}
    </prosody>
  </voice>
</speak>
```

**Emotion Detection**:
- "awesome", "great", "excited" → `excited`
- "sorry", "unfortunately" → `disappointed`
- Default → `neutral`

### 5. Audio Playback

**Function**: `tts_playback_thread(text, tts_start_time)`

**Process**:
1. **API Request**:
   - Calls ElevenLabs `text_to_speech.convert()`
   - Format: PCM 22050 Hz mono
   - Voice settings:
     - Stability: 0.5
     - Similarity boost: 0.0
     - Style: 0.0
     - Speaker boost: False
     - Streaming latency optimization: True

2. **Audio Stream Processing**:
   - Receives PCM audio chunks from API
   - Writes directly to ALSA `aplay` process
   - Uses `plughw:` device for automatic format conversion

3. **Device Selection**:
   - **Primary**: UACDemoV1.0 (if detected)
   - **Fallback**: Any USB Audio device
   - **Default**: `plug:default` with format conversion

4. **Format Conversion**:
   - ALSA `plug` plugin handles:
     - Sample rate conversion (22050 Hz → device rate)
     - Channel conversion (mono → stereo if needed)
   - No manual conversion needed

**Playback Setup**:
```bash
aplay -D plughw:{card},0 -f S16_LE -r 22050 -c 1
```

### 6. Volume Control

**Function**: `set_volume()`

**Configuration**:
- **Source**: `.env` file → `TTS_VOLUME` (default: 100%)
- **Range**: 0-100%

**Methods** (tried in order):
1. **PulseAudio/PipeWire**:
   - Uses `pactl set-sink-volume`
   - Modern systems (Ubuntu 22.04+)

2. **ALSA**:
   - Uses `amixer sset`
   - Controls: PCM, Speaker, Master
   - Fallback for older systems

## State Management

### Playing State

**Functions**: `set_playing(True/False)`, `is_playing()`

**Purpose**: Prevents transcription during TTS playback

**Integration**:
- Listener checks `is_playing()` before processing audio
- Blocks wake word detection during playback
- Streams stopped during TTS

### Playback Lock

**Thread Safety**:
```python
playback_lock = threading.Lock()

with playback_lock:
    set_playing(True)
    # ... TTS playback ...
    set_playing(False)
```

**Purpose**: Prevents concurrent TTS playback

## Queue System

### Sentence Queue

**Implementation**: `queue.Queue()`

**Threading**:
- **Producer**: `enqueue_tts_chunk()` (called from LLM response handler)
- **Consumer**: `playback_loop()` (background thread)
- **Worker**: `tts_playback_thread()` (per-sentence thread)

**Flow**:
1. LLM response → sentence chunks → queue
2. Background thread consumes queue
3. Each sentence spawns playback thread
4. ALSA playback happens in parallel

## Latency Measurement

### TTS Latency Tracking

**Measurement Points**:
1. **LLM Request Start**: `_llm_request_start_time`
2. **First Audio Chunk**: Received from ElevenLabs API
3. **Calculation**: `time.time() - tts_start_time`

**Output**:
```
⏱️ TTS latency: 1.23s
```

**Optimization**:
- Streaming response (tokens as generated)
- Low-latency first batch (min 3 words)
- Immediate single chunks
- Sentence batching (reduces API calls, not latency)

## Error Handling

### ElevenLabs API Errors

**API Key Validation**:
- Checks `.env` file for `ELEVENLABS_API_KEY`
- Raises RuntimeError if missing
- Provides setup instructions

**Network Errors**:
- HTTP timeout: 20 seconds
- Connection errors logged
- Continues operation (error logged)

### ALSA Playback Errors

**Device Detection**:
- Auto-detects output device on startup
- Falls back gracefully if device not found
- Uses default device if specific device unavailable

**Process Errors**:
- Checks if `aplay` process is alive before writing
- Handles broken pipe errors
- Logs errors without crashing

## Token Usage Tracking

### Integration with Wallet

**Function**: Records TTS generation usage

**Calculation**:
- Approximate speech duration: `words / 2.5` (150 words/min)
- Records usage with `multiplier=speech_duration_seconds`

**Purpose**: Tracks API usage for billing/monitoring

## Code Locations

- **Main Module**: `aura-control/core/speaker.py`
- **State Management**: `aura-control/core/state.py`
- **LLM Integration**: `aura-control/core/listener.py` (calls `speak_llm_response`)

## Dependencies

- `elevenlabs`: ElevenLabs Python SDK
- `python-dotenv`: Environment variable loading
- `numpy`: Audio processing (frequency analysis)
- `subprocess`: ALSA `aplay` execution

## Configuration

### Environment Variables

**`.env` file**:
```bash
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=default  # Optional
TTS_VOLUME=100              # 0-100%
```

### Hardcoded Settings

**Audio**:
- PCM Sample Rate: 22050 Hz
- Format: PCM 22050 (mono)
- Target RMS: 0.18 (for normalization, if used)

**Batching**:
- Max Words: 50
- Min Words: 3
- Max Chunks: 2
- Timeout: 0.02s (20ms)

**SSML**:
- Rate: 100%
- Pitch: 100%
- Break times: 300ms (commas), 600ms (sentences)

