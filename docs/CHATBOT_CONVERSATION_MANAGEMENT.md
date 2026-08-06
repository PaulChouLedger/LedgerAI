# Chat Bot / Conversation Management Component

## Overview

The chat bot component orchestrates the entire conversation flow, from speech input to text output. It integrates SST, LLM, RAG, and TTS components into a cohesive conversational AI system.

## Architecture

### Main Entry Point
- **Location**: `aura-control/core/main.py`
- **Purpose**: Orchestrates all components
- **Flow**: GUI → Welcome Setup → Containers → Listener

### Conversation Flow

```
User Speech
    ↓
Hardware DSP (XVF3800)
    ↓
Wake Word Detection (optional)
    ↓
VAD (Voice Activity Detection)
    ↓
SST (Whisper Transcription)
    ↓
Chat Bot Logic
    ↓
LLM (Response Generation)
    ↓
TTS (Speech Synthesis)
```

## Core Components

### 1. Listener Module

**Location**: `aura-control/core/listener.py`

**Responsibilities**:
- Audio capture via PortAudio
- Wake word detection (OpenWakeWord)
- VAD (Silero VAD)
- Speech detection and filtering
- Audio normalization
- Transcription triggering

**Main Loop** (`listen()` function):

1. **Wake Word Detection** (if enabled):
   - Listens for "hey aura" or custom wake word
   - Blocks transcription until detected
   - Visual feedback (solid red LED)
   - Waits 0.3s after detection before VAD

2. **Voice Activity Detection**:
   - Silero VAD model detects speech
   - Threshold: `VAD_START_THRESHOLD = 0.25`
   - Silence detection: `VAD_SILENCE_THRESHOLD = 0.15`
   - Timeout: 0.2s of silence ends recording

3. **Advanced Speech Filtering**:
   - Multi-feature analysis
   - RMS energy checks
   - Zero crossing rate
   - Spectral features
   - Filters noise bursts

4. **Audio Normalization**:
   - Target RMS: 0.22 (optimal for Whisper)
   - Soft limiting (prevents clipping)
   - Sends to Whisper container

5. **Transcription**:
   - HTTP POST to `http://localhost:5000/transcribe`
   - Receives text result
   - Strips wake word if present
   - Sends to LLM via `send_to_llm()`

### 2. Speaker Module Integration

**Function**: `speak_llm_response(text)`

**Called From**: `listener.py` → `send_to_llm()`

**Process**:
1. Receives transcribed text
2. Calls LLM container (`/chat-tts` endpoint)
3. Streams response tokens
4. Sends sentences to TTS queue
5. TTS plays audio via ElevenLabs

### 3. Conversation Memory

**Location**: `llm-container/conversation_manager.py`

**Status**: ⚠️ **Implemented but NOT currently used** in active listener flow

**Implementation Status**:
- ✅ **Code exists**: Fully implemented in `conversation_manager.py`
- ✅ **Initialized**: Objects created in `llm-container/container_rest.py`:
  ```python
  conversation_memory = ConversationMemoryIndex(...)
  conversation_orchestrator = ConversationOrchestrator(...)
  ```
- ✅ **Endpoint exists**: `POST /voice/transcript` endpoint functional
- ❌ **NOT called**: Current listener (`listener.py`) does NOT call this endpoint

**Current Flow** (listener.py):
1. Transcribes audio via `/transcribe` (Whisper container)
2. Directly calls `/chat-tts` (LLM) via `speak_llm_response()`
3. **Bypasses** `/voice/transcript` endpoint entirely
4. **Memory context always None**: `handle_conversation()` receives `memory_context=None`

**Designed Purpose** (not currently active):
The conversation memory system was designed for **passive listening mode**:
- Continuous transcript ingestion via `/voice/transcript`
- Keyword-triggered activation (e.g., "hey aura")
- Long-term memory storage with FAISS indexing
- Activation window (15 seconds after keyword)
- Only responds when activation keywords detected
- Stores all transcripts for semantic search

**Implementation Details** (if enabled):
- **Semantic Indexing**: FAISS IndexFlatIP for cosine similarity
- **Embeddings**: Uses RAG client's embedding function
- **Activation Keywords**: `["hey aura"]` triggers memory retrieval (line 64)
- **Activation Window**: 15 seconds after keyword (line 65)
- **Top-K Retrieval**: 3 most relevant memories (line 70)
- **Min Score**: 0.35 similarity threshold (line 71)
- **Storage**: Pickle file with vectors and metadata
- **Persist Frequency**: Every 10 entries (line 68)
- **Max Entries**: 5000 (line 69)

**Storage** (if used):
- Directory: `data/learning/conversation_memory/`
- File: `conversation_memory.pkl`
- Format: Pickle file with embeddings, metadata, and vectors
- Persistence: Auto-saved every 10 entries, flushed on exit

**How It Would Work** (if enabled):
1. Listener sends transcripts to `/voice/transcript` endpoint
2. `ConversationOrchestrator.process_chunk()` receives transcript
3. Indexes transcript in FAISS memory index
4. Checks for activation keywords in transcript
5. If keyword detected, opens 15-second activation window
6. Retrieves relevant memories (top-K similar conversations)
7. Passes memory context to `handle_conversation()`
8. LLM uses memory context + current prompt for response

**Current Behavior**:
- Conversation memory system **exists** but is **not called**
- Listener uses direct transcription → LLM flow
- No passive listening mode active
- Memory index remains empty (no transcripts indexed)
- Memory context always None in LLM prompts

**Code Evidence**:
```python
# In llm-container/container_rest.py (lines 374-391)
conversation_memory = ConversationMemoryIndex(...)  # ✅ Initialized
conversation_orchestrator = ConversationOrchestrator(...)  # ✅ Created

# In handle_conversation() (line 190)
def handle_conversation(prompt, session_id, memory_context=None, stream=False):
    # memory_context is always None in current flow

# In listener.py (line 1161)
send_to_llm(text)  # Direct call, doesn't use /voice/transcript
```

### 4. Session Management

**Medical LLM Sessions**:

**State Storage**:
- Location: `/app/data/sessions/{session_id}.json`
- Persisted after each interaction
- Auto-cleanup after 2 hours inactivity

**State Structure**:
```json
{
  "mode": "triage" | "clinician",
  "condition": "chest_pain",
  "step_index": 3,
  "answers": ["severe", "crushing", "yes"],
  "flags": {"emergency": true},
  "user_name": "Rafael",
  "conversation_history": [...],
  "findings": {...},
  "differential_diagnoses": [...]
}
```

**Session Lifecycle**:
1. **Creation**: `get_or_create_session(session_id)`
2. **Update**: State saved after each LLM interaction
3. **Reset**: `reset_session(session_id)` clears state
4. **Cleanup**: Background thread removes inactive sessions

### 5. Passive Listening Mode

**Status**: ⚠️ **Not currently active** - Endpoint exists but not used by listener

**Endpoint**: `POST /voice/transcript` (exists in `llm-container/container_rest.py`)

**Designed Purpose**: Continuous transcription without immediate response

**Designed Flow** (not currently implemented):
1. SST sends transcripts continuously to `/voice/transcript`
2. Conversation orchestrator indexes transcripts in FAISS
3. Checks for activation keywords (`["hey aura"]`)
4. Only responds when keywords detected (within 15s window)
5. Uses conversation memory for context
6. Stores all transcripts for long-term memory

**Current Reality**:
- Endpoint exists and is functional
- Code is initialized (`conversation_orchestrator` created)
- **But**: `listener.py` does NOT call this endpoint
- Listener uses direct flow: Whisper → LLM

**To Enable Passive Listening**:
Would require changes to `listener.py`:
```python
# Instead of:
text = transcribe(mono)
send_to_llm(text)

# Would need:
response = requests.post(
    "http://localhost:11434/voice/transcript",
    json={
        "text": text,
        "session_id": "voice_session",
        "is_final": True
    }
)
if response.json().get("response"):
    # Use response from conversation orchestrator
    speak_llm_response(response.json()["response"])
```

**Use Case** (if enabled): Background listening, only responds to specific triggers

## Wake Word Detection

### OpenWakeWord Integration

**Location**: `aura-control/core/openwakeword_wake_word.py`

**Framework**: OpenWakeWord (open-source, ARM64-compatible)

**Default Model**: `hey_orah` (customizable)

**Custom Models**:
- Location: `data/models/wake_words/`
- Format: `.onnx` files
- Auto-discovery: Handles model variations (e.g., `hey_orah-2.onnx`)

### Configuration

**Settings** (`state.py`):
- **Enabled**: Via Settings dialog → AI Model Settings
- **Default**: Disabled (can be toggled)
- **Persistent**: State saved in `app_settings.json`
- **No Restart Required**: Can be toggled without restarting system

**Model Selection**:
- **Default**: `hey_orah`
- **Custom Models**: Place `.onnx` files in `data/models/wake_words/`
- **Auto-Detection**: Finds variations like `hey_orah-2.onnx`
- **Fallback**: Uses built-in models if custom not found

**Threshold Configuration**:
- **Default**: 0.01 (very sensitive)
- **Location**: `openwakeword_wake_word.py` → `DEFAULT_THRESHOLD`
- **Range**: 0.0-1.0 (higher = less sensitive)
- **Tuning**: Adjust based on false positive/negative rate

### Initialization Flow

**Process** (`listener.py` → `listen()` function):

1. **Settings Check**:
   ```python
   from state import get_wake_word_enabled
   wake_word_setting_enabled = get_wake_word_enabled()
   ```

2. **Transcription Blocking**:
   - If enabled, blocks transcription immediately
   - Prevents transcription until detector ready
   - Logs: "🔒 Blocking transcription until wake word detector is ready"

3. **Detector Initialization**:
   ```python
   wake_word_detector = create_openwakeword_detector()
   ```
   - Tries to load custom model first
   - Falls back to built-in models if not found
   - Downloads models on first use if missing
   - Uses ONNX framework (better ARM64 support)

4. **Initialization Success**:
   - Unblocks transcription
   - Sets `wake_word_enabled = True`
   - Logs: "✅ Wake word detector ready"

5. **Initialization Failure**:
   - Transcription remains BLOCKED
   - Logs error messages
   - Provides setup instructions
   - User must fix or disable wake word

### Detection Loop

**Location**: `listener.py` → `listen()` → Wake word detection loop

**Audio Processing**:
- **Same Pipeline**: Uses same audio processing as VAD
- **Shared Function**: `read_audio_frame()` for consistency
- **Normalization**: Same target RMS (0.22) as VAD
- **Channel Extraction**: Channel 0 (beamformed output)

**Frame Processing**:
- **Input**: float32 audio (normalized to [-1, 1])
- **Frame Size**: 1280 samples (80ms at 16kHz) required by OpenWakeWord
- **Buffering**: Accumulates frames until 1280 samples available
- **Conversion**: float32 → int16 for OpenWakeWord

**Detection Process**:
```python
# Buffer audio until we have enough samples
self.audio_buffer = np.concatenate([self.audio_buffer, audio])

if len(self.audio_buffer) >= 1280:
    frame_audio = self.audio_buffer[:1280]
    audio_int16 = (frame_audio * 32767).astype(np.int16)
    prediction = self.engine.predict(audio_int16)
    confidence = prediction[self.model_name]['score']
    detected = confidence >= self.threshold
```

**Confidence Calculation**:
- **Score Range**: 0.0-1.0
- **Threshold**: Default 0.01 (configurable)
- **Detection**: `confidence >= threshold`
- **Logging**: Logged when confidence > threshold/10 or confidence > 0.001

### After Wake Word Detection

**State Changes**:
1. **Listening Active**: Sets `listening_active = True`
2. **Visual Feedback**: Solid red LED (not pulsating yet)
3. **Wait Period**: Waits 0.3s before VAD activation
4. **VAD Reset**: Resets VAD state for clean start
5. **Transition**: Proceeds to VAD loop

**VAD Activation**:
- After wake word, VAD loop begins
- Listens for speech immediately
- No additional wake word needed during session
- Returns to wake word loop after transcription completes

**Visual Feedback Flow**:
```
Wake Word Detected → Solid Red LED
    ↓
Speech Detected → Pulsating Red LED
    ↓
Transcription Complete → Reset to Wake Word Loop
```

### Echo Prevention

**TTS Echo Blocking**:

**Critical Check**:
- `is_playing()` checked **BEFORE** reading audio
- Prevents wake word processing during TTS
- Same check in both wake word and VAD loops

**Implementation**:
```python
# In read_audio_frame() - shared by wake word and VAD
if is_playing():
    return None, None, None, True  # Skip processing, continue loop
```

**Stream Management**:
1. **During TTS**:
   - Stream stopped: `stream.stop()`
   - Mic muted: No audio captured
   - Processing skipped: Wake word and VAD both skip

2. **After TTS Ends**:
   - Stream restarted: `stream.start()`
   - Buffer flushed: 5 frames discarded
   - Detector buffer cleared: `wake_word_detector.clear_buffer()`
   - VAD state reset: `model_vad.reset_states()`

3. **Buffer Clearing**:
   ```python
   # Flush stream buffer
   for i in range(5):
       stream.read(FRAME_SIZE)  # Discard
   
   # Clear OpenWakeWord buffer
   if wake_word_detector:
       wake_word_detector.clear_buffer()
   ```

**Timing**:
- Echo decay pause: Handled by stream flush (5 frames = ~160ms)
- No explicit delay: Stream flush provides sufficient pause
- Clean state: Ensures no stale audio triggers wake word

### Wake Word Removal

**Post-Transcription Processing**:

**Location**: `listener.py` after transcription completes

**Process**:
```python
if wake_word_enabled:
    text_lower = text.lower().strip()
    wake_phrases = [
        "hey aura", "hey aura,", "hey aura.",
        "aura", "aura,", "aura.",
        "hey jarvis", "hey jarvis,", "hey jarvis."
    ]
    for phrase in wake_phrases:
        if text_lower.startswith(phrase):
            text = text[len(phrase):].strip().lstrip(",.")
            print(f"[Wake Word] 🧹 Removed wake word from transcription")
            break
```

**Purpose**: Prevents wake word from appearing in user query to LLM

### Performance Optimization

**Model Selection**:
- **ONNX Framework**: Better ARM64/Jetson support
- **Lightweight**: Low CPU usage (~5-10% on Jetson NX)
- **Efficient Buffering**: Only processes when enough samples

**Debug Logging**:
- **First 10 Frames**: Logged for initialization verification
- **Every 100 Frames**: Heartbeat to confirm still listening
- **High Confidence**: Logged when confidence > threshold/10
- **Detection**: Detailed log when wake word detected

**Logging Format**:
```
[Wake Word] 🔴 QUIET Confidence: 0.000123 (threshold: 0.010000, 1.2%) - Frame 42
[Wake Word] 🟡 ACTIVITY Confidence: 0.004567 (threshold: 0.010000, 45.7%) - Frame 123
[Wake Word] 🟢 DETECTED! Confidence: 0.012345 (threshold: 0.010000, 123.5%) - Frame 456
[Wake Word] ✅ Wake word detected! (confidence: 0.012345)
```

### Troubleshooting

**Wake Word Not Detecting**:
1. **Check Initialization**: Look for "✅ OpenWakeWord initialized" in logs
2. **Verify Model**: Check `data/models/wake_words/` for custom models
3. **Check Threshold**: Lower threshold for more sensitivity (default: 0.01)
4. **Audio Levels**: Verify audio normalization (RMS ~0.22)
5. **Device Selection**: Ensure correct microphone device selected

**False Positives**:
1. **Increase Threshold**: Higher threshold = less sensitive
2. **Enable Echo Cancellation**: Prevents TTS echo triggers
3. **Check TTS Blocking**: Verify `is_playing()` checks working
4. **Buffer Clearing**: Ensure buffer cleared after TTS

**Model Not Loading**:
1. **Check File Exists**: Verify `.onnx` file in `data/models/wake_words/`
2. **Verify Format**: Must be valid ONNX file
3. **Check Downloads**: Models download automatically on first use
4. **Fallback**: System uses built-in models if custom fails
5. **Permissions**: Ensure read permissions on model directory

**Transcription Blocked**:
1. **Check Settings**: Verify wake word enabled in Settings dialog
2. **Check Detector**: Look for initialization errors in logs
3. **Disable Wake Word**: Temporarily disable to unblock transcription
4. **Restart System**: Restart if detector failed to initialize

## Transcription Blocking

### Blocking Mechanisms

**Global Flag**: `_transcription_blocked`

**Reasons for Blocking**:
1. **Dialog Open**: GUI dialog blocks transcription
2. **Microphone Button**: User manually mutes mic
3. **Wake Word Waiting**: Wake word enabled but not detected
4. **Initialization**: System startup

**Functions**:
- `block_transcription(reason)`: Block transcription
- `unblock_transcription()`: Unblock transcription
- `is_transcription_blocked()`: Check status
- `toggle_transcription()`: Toggle mic mute

**Integration**:
- `read_audio_frame()` checks blocking before processing
- Visual feedback in GUI (muted state)

## Audio Processing Pipeline

### Hardware DSP (XVF3800)

**Location**: `setup/scripts/tune_xvf3800.py`

**Hardware Features**:
- **Beamforming**: Multi-microphone spatial filtering (4-mic array)
- **AGC (Automatic Gain Control)**: Maintains optimal speech levels
- **HPF (High-Pass Filter)**: Removes low-frequency rumble (<70Hz)
- **Echo Cancellation**: Cancels feedback from speakers (optional)

**Configuration Presets**:
- **`balanced_beam`** ⭐ RECOMMENDED: HPF 70Hz + AGC (0.05 RMS, 30dB) - Best for fan noise
- **`agc_20_ec`** (Default): HPF 70Hz + AGC (0.096 RMS) + Echo Cancellation
- **`ultra_sensitive`**: AGC (0.10 RMS, 45dB) - Far-field optimized
- **`far_field`**: Optimized for 8-16 feet distance
- **`near_field`**: Optimized for 1-6 feet distance

**Configuration Process**:
1. **Boot Service**: Automatically configures on system startup via systemd service
2. **Manual Configuration**: Run `sudo python3 setup/scripts/tune_xvf3800.py [preset]`
3. **State Storage**: Configuration saved to `data/xvf3800_config.json`
4. **Display**: Configuration displayed in console on listener startup

**Configuration Parameters**:
- **AGC Target**: `PP_AGCDESIREDLEVEL` (0.22 = recommended)
- **AGC Max Gain**: `PP_AGCMAXGAIN` (1000 = 30dB)
- **HPF Setting**: `AEC_HPFONOFF` (1 = 70Hz recommended)
- **Echo Cancellation**: `PP_ECHOONOFF` (1 = ON, 0 = OFF)

**Detailed Documentation**: See [SST Component - Microphone Tuning](SST_SPEECH_TO_TEXT.md#microphone-tuning-xvf3800)

**Output**: Processed mono channel audio (beamformed, filtered, normalized)

### Software Processing

**1. Audio Normalization**:
- Target RMS: 0.22 (optimal for Whisper)
- Calculates gain factor
- Applies gain with soft clipping

**2. Soft Limiting**:
- Prevents clipping from near-field speech
- Tanh-based soft knee compression
- Threshold: 0.95 peak level

**3. Advanced Filtering**:
- Multi-feature speech validation
- Filters noise bursts
- Validates speech characteristics

## State Synchronization

### GUI State Updates

**States Tracked**:
- `_listening_ready`: System ready for transcription
- `_transcribing`: User currently speaking
- `_wake_word_detected`: Wake word detected (solid red)
- `_tts_playing`: TTS currently playing
- `_setup_complete`: Initial setup complete
- `_microphone_muted`: Mic manually muted

**GUI Functions**:
- `set_listening_ready(True)`
- `set_transcribing(True/False)`
- `set_wake_word_detected(True/False)`
- `set_tts_playing(True/False)`
- `set_setup_complete()`

**Visual Feedback**:
- **Idle**: White pulsing border
- **Wake Word Detected**: Solid red border
- **Transcribing**: Pulsating red border (speed based on audio frequency)
- **TTS Playing**: Border animations

## Error Handling

### Container Failures

**Health Checks**:
- Whisper: `http://localhost:5000/health`
- LLM: `http://localhost:11434/health`
- RAG (GPU): `http://localhost:11435/health`

**Graceful Degradation**:
- Continues operation if health check fails
- Logs errors for debugging
- User notified via GUI

### Audio Stream Errors

**PortAudio Errors**:
- Stream invalid errors handled
- Attempts to restart stream
- Falls back gracefully

**Device Not Found**:
- Retry logic with exponential backoff
- System diagnostics provided
- Suggestions for fixing

## Code Locations

- **Main Orchestration**: `aura-control/core/main.py`
- **Listener**: `aura-control/core/listener.py`
- **Speaker Integration**: `aura-control/core/speaker.py`
- **Wake Word**: `aura-control/core/openwakeword_wake_word.py`
- **Conversation Memory**: `llm-container/conversation_manager.py`
- **State Management**: `aura-control/core/state.py`

## Dependencies

- `sounddevice`: PortAudio wrapper
- `torch`: PyTorch for VAD
- `openwakeword`: Wake word detection
- `requests`: HTTP client for containers
- `numpy`: Audio processing
- `scipy`: Signal processing


---

## 2026-08-06 — she went quiet in Area31, and the logs could not say why

Reported as "why did she not respond to Cody". Two causes, and only the
second is a defect. Recorded together because the first one is the answer
people will reach for and the second is the one that actually cost the room
two hours of silence.

### 1. Cody's two messages were SCORED OUT (working as designed)

`RESPOND_THRESHOLD = 0.30` (`config.py:55`).

    00:45:46  "Can you see reactions to your messages"
              score 0.15 — conversational turn, rapid-fire (5 in last 10)
    00:47:49  "@phasic17 have you worked out getting her to react..."
              score 0.00 — addressed to someone else, msg cooldown (1/2),
                           rapid-fire (4 in last 10)

`W_RAPID_FIRE_PENALTY = -0.35` dominated both. The first is worth a second
look: he was asking HER, quickly, whether she notices things, and the
penalty that silenced her exists to stop her interrupting a fast exchange
between other people. The penalty does not distinguish "this room is busy"
from "this person is talking to me quickly". If the threshold gets retuned,
that is the case to tune against.

### 2. The room was HARD-MUTED by a false positive, and it logged nothing

`/home/paul/LedgerAI/data/telegram/muted_chats.json` held
`{"-1003025733750": 1786002373}` — Area31, muted 01:46:13 → 03:46:13.

The trigger was Paul DESCRIBING THE ALGORITHM:

    01:46:12 [IN] The Real Paul Chou: yes, it's dynamically adaptive. she
             will jump in if she feels it crosses a certain "value add"
             threshold but also doesn't breach the "i already spoke too
             much recently" threshold...
    01:46:13 WARNING RETRACT: too_much complaint from The Real Paul Chou
             in -1003025733750 — deleted 1 msgs, paused 2h pending review

`feedback.py:52` is a bare substring match:

    (re.compile(r"\btoo\s+much\b", re.I), "too_much"),

and `record_implicit()` (`feedback.py:278-302`) has **no guards at all**:
it does not check whether the message replies to her, does not check
whether she has spoken recently, does not handle quotation or negation,
and has **no owner exemption despite `OWNER_USER_IDS` existing**. Any
message in an allowed group containing "too much" / "relax" / "chill out" /
"tone it down" deletes up to three of her messages and mutes the room for
`RETRACT_PAUSE_S` (7200s).

**TALKING ABOUT THE FEATURE IS ENOUGH TO TRIGGER THE FEATURE.** That is the
part worth remembering: demoing her means describing her, so anyone showing
her off will walk into this. It fired twice in one evening — once
legitimately ("lol ok aura tone it down", 20:47:08) and once on a sentence
that was praising her.

**The silence was unexplainable from the logs.** `_is_muted()` returns at
`bot.py:1285`, BEFORE scoring — so a muted room produces no `[SKIP]`, no
`[RESPOND]`, no line of any kind. The investigation that found this had to
read a JSON state file to discover why a healthy process was saying
nothing. A gate that can decline to act must leave a mark (PRINCIPLES §1).

### Fixes, in order of value

1. **Owner exemption** on `record_implicit()`. `OWNER_USER_IDS` is right
   there and unused by this path.
2. **Require the complaint to be ABOUT her** — a reply to one of her
   messages, or at minimum her having spoken in the last N turns.
3. **Quote/description detection** — the trigger phrase inside quotation
   marks is somebody discussing the rule, not invoking it.
4. **Log the mute.** One line per dropped message, or one line per mute
   with its expiry, at minimum.

### Two operational notes found on the way

- **The mute is held in memory.** `_muted_chats` is loaded once at startup
  (`bot.py:619`), so editing `muted_chats.json` does NOT release a running
  bot. `/aurastart` in the group is the clean release (`bot.py:669-678`).
- **The unit does not match the process.** `aura-telegram.service` is
  `disabled; inactive (dead)` and points at `/home/paul/.venvs/aura/bin/
  python`; the live bot is a manually launched, detached `/usr/bin/python3`
  (PPID 1). `systemctl --user start aura-telegram` would put a SECOND
  poller on the same token and 409 them both. Reconcile before any reboot.

### What was done

Nothing was changed, restarted, or unmuted. At the owner's instruction the
bot was made to post the explanation to Area31 itself — sent over the Bot
API directly, which bypasses the in-process mute — because an explanation
of her silence means more in her own voice than in a status report.
Message 15477.

## 2026-08-06, later — the complaint detector gets a target, not a vocabulary

Implements the four fixes listed in the section above. `feedback.py`
`record_implicit()` and its one caller (`bot.py:1391`); tests in
`tests/test_complaint_guard.py`.

### The shape of the answer, which is not the shape the fix list assumed

The fix list said "require the complaint to be ABOUT her — a reply to one of
her messages, or at minimum her having spoken in the last N turns", and that
is what was built first. It passed thirteen hand-written tests. It was
wrong, and the thing that showed it was wrong was **replaying the detector
over `data/telegram/dm_history.jsonl`** — 1058 real recorded inbound
messages. Seven of them fired the old detector. Every one of the seven is
now a fixture in the test file, the whole population rather than a
selection. Two of them:

    Do you know, is too much farting good for my health? 😁
    And he likes taco bell too much

Both would have deleted three of her messages and muted the chat for two
hours. Neither is a complaint about anything. And in a DM the first rule
drafted — a reply, or her name — is useless in both directions: **nobody
says "aura" in a one-to-one chat**, so it silenced every genuine complaint
in the sample while the farting one still passed on "Do you know".

The discriminator that survives the replay is **second person in the same
clause as the trigger phrase**, plus imperatives that open a message. A
complaint has to be aimed at somebody, and she is only a candidate if it is
aimed at a "you". "he likes taco bell too much" is third person. "Do you
know, is too much farting good for my health" has a *you* and a *too much*
in different clauses — hence the clause split, which is crude, and is the
entire difference between that message being learned from and it gagging
the room.

### The three outcomes

Two outcomes were the bug. The gate now has three (PRINCIPLES §2), because
the costs are wildly asymmetric (§8): a missed complaint costs one unwanted
message; a false one costs two hours of silence with her own messages
deleted out from under the room.

| | what it takes | what happens |
|---|---|---|
| IGNORED | quoted, or she has not spoken and is not named | nothing, but a log line |
| RECORDED | she spoke lately, or is named/replied to | queued, learned from, nothing deleted |
| ACTIONABLE | that, **and** aimed at a "you" | may delete and mute, as before |

Measured on the 1058: **7 old fires → 4 actionable, 3 recorded.** The four
are the four that were real ("Stop being evasive", "you hallucinated",
"you are acting like…", "Stop that over shit"). One genuine complaint is
conceded as a downgrade — Cody's "if i am asking too much. Your responses
so far come off as passive aggressive", where the trigger phrase describes
his own asking and not her. It is recorded and learned from; it just does
not delete anything. **Downgraded is not dropped**, and a test asserts that
all seven are still recorded.

### The guards, in the order they run

1. **Quoted → ignored.** `"…"`, `“…”`, `«…»`, `` `…` ``. Apostrophes are
   deliberately not quote characters, or "don't" swallows the sentence.
   This one rule alone kills the Area31 message, whoever sends it.
2. **Not pointed at her and she has not spoken in 5 messages → ignored.**
3. **Owner, not pointed at her → ignored.** `OWNER_USER_IDS` existed and
   this path never used it. He describes her for a living, and describing
   her means saying the trigger phrases out loud. When he means it he
   replies to her or says her name — both tested.
4. **Not aimed at a "you" → recorded, never actioned.**

### And the mute now says so

`_is_muted()` returned silently and `handle_message` returns on it *before*
scoring, so a muted room produced no line of any kind — which is how two
hours of silence became unexplainable from the logs. It now logs `[MUTED]`
with the minutes remaining and the wall-clock expiry, throttled to one line
per chat per 5 minutes, and a line when the mute lapses. `RETRACT` logs the
expiry time and the triggering text, so the next false positive identifies
itself in the log instead of in a JSON file.

### Not verified

**Nothing was restarted, so none of this is live.** The running bot is PID
221981, a detached `/usr/bin/python3` started 18:28 on 2026-08-05, and it
holds the old code plus `_muted_chats` in memory. Restarting it is not free
— see the unit/process mismatch in the section above, which will 409 two
pollers against one token if `systemctl` is used to do it.

Area31's mute from the incident was still live while this was written
(expiry 03:46:13 EDT) with `muted_chats.json` reading `{}` — the in-memory
copy exactly as documented. It lapses on its own; `/aurastart` in the group
releases it now.

## 2026-08-06, later still — she asks; she does not decide

Owner, same evening, after reading the section above: *"remove the mute
forever, it's unnecessary unless it gets bad, in which case have the TG bot
telegram DM me directly for approval to stop talking."*

So the guards in the previous section are now the *first* of two gates, and
the second one is a human. Nothing in the complaint path mutes or deletes
anything any more. An actionable complaint sends him a DM — who said what,
what she had said, and four buttons:

    [ Quiet 2h ] [ Quiet 24h ]
    [ Delete my last 3 + quiet 2h ]
    [ Leave it — keep talking ]

**She keeps talking while she waits.** There is no pending state and no
provisional silence; if he never presses anything, nothing happens. That is
the deliberate direction — the previous default treated two hours of
enforced silence as free, and the incident above is what it actually cost.

One ask per chat per 30 minutes (`_QUIET_ASK_COOLDOWN_S`), because a room
that is genuinely annoyed produces the same complaint several times and it
is the same question every time. Only `OWNER_USER_IDS` can press the
buttons. If the DM cannot be delivered she carries on and logs that she
could not ask — it must not be possible to believe she asked.

### The invariant, and how it is held

Two places can mute a chat and both are him deciding directly:
`on_quiet_decision` (the button) and `cmd_aurastop` (`/aurastop`).

`tests/test_complaint_guard.py::test_only_the_owner_can_mute` walks
`bot.py`'s AST and fails if any other function assigns `_muted_chats`. It is
structural rather than behavioural on purpose — importing `bot.py` needs a
live token and the whole stack, and the thing worth protecting is not "this
complaint does not mute" but "**nothing** mutes on its own authority". A
future session re-adding an automatic pause trips it without having to
reproduce a complaint. `test_the_owner_can_still_mute` is its other half:
the leash came off, the hand did not.

`RETRACT_ON_COMPLAINT` is dead. It is left in `config.py` with its date and
a SUPERSEDED note rather than deleted, because the flag name still appears
in the 2026-07-31 write-up and a reader who greps for it should find out
what happened to it rather than nothing.

### Not verified

Still nothing restarted. PID 221981 holds the pre-fix code, and the
unit/process mismatch in the first section still applies — `systemctl` would
put a second poller on the same token. The approval flow is committed and
tested and is NOT live until that restart happens.
