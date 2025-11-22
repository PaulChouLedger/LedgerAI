# SST (Speech-to-Text) Component

## Overview

The SST component uses **faster-whisper** to convert spoken audio into text. It runs in a Docker container and provides a REST API endpoint for transcription services.

## Architecture

### Container Setup
- **Location**: `whisper-container/`
- **Image**: `aura-whisper:latest`
- **Port**: `5000`
- **Model**: Configurable (default: `distil-small.en`)
- **Framework**: Faster-Whisper (optimized PyTorch implementation)

## Core Functions

### 1. Audio Transcription (`/transcribe` endpoint)

**File**: `whisper-container/container_rest.py`

**Function Flow**:
1. **Audio Reception**: Receives audio file via HTTP POST
2. **Preprocessing**:
   - Converts stereo to mono if needed
   - Resamples to 16kHz (Whisper requirement)
   - Converts to float32 format
3. **Model Transcription**:
   - Uses configurable beam size (default: 10)
   - Temperature: 0.0 (deterministic)
   - Patience: 1.0
   - Length penalty: 1.0
   - Optional initial prompt for context (medical terms)
4. **Result Processing**:
   - Joins all segments into single text
   - Returns JSON with text and timing statistics

**Key Parameters**:
```python
BEAM_SIZE = 10          # Higher = better accuracy, slower
TEMPERATURE = 0.0       # Deterministic output
PATIENCE = 1.0          # Wait time for better results
LENGTH_PENALTY = 1.0    # Prevents cutting off words
```

### 2. Audio Preprocessing

**Function**: `preprocess_audio(path, target_sr=16000)`

**Process**:
1. Loads audio file using `soundfile`
2. Converts multi-channel to mono (average across channels)
3. Resamples to target sample rate (16kHz) using `scipy.signal.resample_poly`
4. Returns float32 array

### 3. Medical Vocabulary Management

**Features**:
- Loads medical terms from `/shared/medical_terms.json`
- Builds initial prompt with medical terminology
- Supports adding new terms via `/add_medical_term` endpoint
- Prioritizes organ systems: cardiovascular, respiratory, gastrointestinal, endocrine, neurological

**Function**: `load_medical_terms()`
- Samples 5 terms from each priority system
- Limits to ~40 terms to keep prompt reasonable
- Creates context-aware prompt for transcription

### 4. Model Loading

**Initialization**:
- Model loaded on container startup
- GPU required (no CPU fallback)
- Uses `int8_float16` compute type for efficiency
- Caches models in `/root/.cache/huggingface/hub`

**Model Options**:
- `distil-small.en`: Fast, lower accuracy
- `small.en`: Better accuracy
- `medium.en`: Much better for names
- `large-v3-turbo`: Best accuracy, higher latency
- `distil-large-v3`: Excellent accuracy, low latency (recommended)

### 5. Timing Statistics

**Tracking**:
- Request to completion time
- File processing time
- Audio preprocessing time
- Model transcription time
- Audio duration
- Efficiency (real-time factor)

**Health Endpoint**: `/health`
- Returns timing statistics
- Shows model information
- Provides performance metrics

## Integration with Listener

### Audio Flow from Hardware

1. **Hardware DSP**: XVF3800 processes audio (beamforming, AGC, HPF)
2. **PortAudio Capture**: `listener.py` captures audio stream
3. **VAD Detection**: Silero VAD detects speech segments
4. **Audio Normalization**: Normalized to optimal RMS (0.12) for Whisper
5. **Soft Limiting**: Prevents clipping from near-field speech
6. **HTTP Request**: Audio sent to `http://localhost:5000/transcribe`

### Audio Normalization Logic

**Function**: `normalize_audio_for_whisper(audio_data, target_rms=0.12)`

**Process**:
1. Calculates current RMS
2. Computes gain factor: `target_rms / current_rms`
3. Applies gain to audio
4. Soft clips to prevent distortion (-0.95 to 0.95)

### Advanced Filter Integration

**Speech Detection**:
- VAD provides initial speech detection
- Advanced filter validates speech using multi-feature analysis:
  - RMS energy thresholds
  - Peak amplitude thresholds
  - Zero crossing rate
  - Spectral centroid
  - Spectral flatness
  - Speech band energy ratio

**Only validated speech is sent to Whisper**, reducing false transcriptions.

## Microphone Tuning (XVF3800)

### Hardware DSP Configuration

The XVF3800 USB 4-Mic Array includes hardware DSP processing that occurs before audio reaches the software layer.

**Location**: `setup/scripts/tune_xvf3800.py`

**Hardware Features**:
- **Beamforming**: Multi-microphone spatial filtering
- **AGC (Automatic Gain Control)**: Maintains optimal speech levels
- **HPF (High-Pass Filter)**: Removes low-frequency rumble
- **Echo Cancellation**: Cancels feedback from speakers (optional)

### Configuration Presets

**Recommended Presets**:

1. **`balanced_beam`** ⭐ RECOMMENDED
   - HPF: 70 Hz (removes fan noise)
   - AGC: Enabled (target: 0.08 RMS, max gain: 30dB)
   - AGC Time: 0.5 seconds
   - Echo Cancellation: OFF
   - **Best for**: Jetson NX with fan noise

2. **`agc_20_ec`** (Default for systemd service)
   - HPF: 70 Hz
   - AGC: Enabled (target: 0.096 RMS, 20% boost)
   - Echo Cancellation: ON
   - **Best for**: TTS echo prevention

3. **`ultra_sensitive`**
   - AGC: Enabled (target: 0.10 RMS, max gain: 45dB)
   - **Best for**: Far-field speech (8-16 feet)

4. **`far_field`**
   - Optimized for 8-16 feet distance
   - Maximum AGC gain

5. **`near_field`**
   - Optimized for 1-6 feet distance
   - Lower AGC gain

### Configuration Commands

**Manual Configuration**:
```bash
# Change to LedgerAI directory
cd ~/LedgerAI

# Run tuning script with preset
sudo python3 setup/scripts/tune_xvf3800.py balanced_beam

# Show current settings
sudo python3 setup/scripts/tune_xvf3800.py show

# Reset to factory defaults
sudo python3 setup/scripts/tune_xvf3800.py reset
```

**Boot Service Configuration**:
```bash
# Service file location
/etc/systemd/system/xvf3800-tuning.service

# Change preset in ExecStart line
ExecStart=/usr/bin/python3 /path/to/LedgerAI/setup/scripts/tune_xvf3800.py balanced_beam

# Reload after changes
sudo systemctl daemon-reload
sudo systemctl restart xvf3800-tuning.service
```

### Configuration Parameters

**AGC (Automatic Gain Control)**:
- `PP_AGCONOFF`: Enable/disable AGC (0=OFF, 1=ON)
- `PP_AGCDESIREDLEVEL`: Target RMS level (0.08 = recommended)
- `PP_AGCMAXGAIN`: Maximum gain in linear units (1000 = 30dB)
- `PP_AGCTIME`: Attack time in seconds (0.5 = fast response)

**HPF (High-Pass Filter)**:
- `AEC_HPFONOFF`: HPF setting (0=OFF, 1=70Hz, 2=125Hz, 3=150Hz, 4=180Hz)
- **70Hz recommended** for fan noise rejection

**Echo Cancellation**:
- `PP_ECHOONOFF`: Enable/disable (0=OFF, 1=ON)
- **ON recommended** if TTS plays through same device

**LED Control**:
- `LED_BRIGHTNESS`: 0-255 (0 = off)
- `LED_COLOR`: Hex color (0x000000 = black/off)
- All presets disable LEDs to reduce power consumption

### Configuration State

**Storage**: `data/xvf3800_config.json`

**Structure**:
```json
{
  "preset": "balanced_beam",
  "timestamp": 1234567890.123,
  "config": {
    "AEC_HPFONOFF": 1,
    "PP_AGCONOFF": 1,
    "PP_AGCDESIREDLEVEL": 0.08,
    "PP_AGCMAXGAIN": 1000,
    "PP_AGCTIME": 0.5,
    "PP_ECHOONOFF": 0
  }
}
```

**Loading**:
- Listener reads config on startup
- Displays configuration in console
- No permissions needed (read-only)

### Audio Processing Pipeline

**Hardware Processing** (XVF3800 DSP):
1. **Microphone Array**: 4 microphones capture audio
2. **Beamforming**: Spatial filtering for directional pickup
3. **HPF**: Removes low-frequency rumble (<70Hz)
4. **AGC**: Adjusts gain to target RMS level
5. **Echo Cancellation**: Removes speaker feedback (if enabled)

**Software Processing** (after hardware):
1. **PortAudio Capture**: Receives processed audio from hardware
2. **Channel Extraction**: Selects channel 0 (beamformed output)
3. **Normalization**: Adjusts to optimal RMS (0.12) for Whisper
4. **Soft Limiting**: Prevents clipping from near-field speech
5. **Advanced Filter**: Validates speech characteristics

### Tuning for Different Environments

**Jetson NX (Fan Noise)**:
- Use `balanced_beam` preset
- HPF 70Hz removes fan rumble
- AGC maintains speech levels

**Quiet Environment**:
- Use `ultra_sensitive` preset
- Higher AGC gain for far-field speech
- No HPF needed (low noise floor)

**TTS Echo Issues**:
- Use `agc_20_ec` preset
- Enable echo cancellation
- Reduces false wake word triggers

**Near-Field Use**:
- Use `near_field` preset
- Lower AGC gain prevents clipping
- Optimal for 1-6 feet distance

## Wake Word Integration

### OpenWakeWord Implementation

**Location**: `aura-control/core/openwakeword_wake_word.py`

**Framework**: OpenWakeWord (open-source, ARM64-compatible)

**Default Model**: `hey_orah` (can be customized)

**Custom Models**:
- Location: `data/models/wake_words/`
- Format: `.onnx` files
- Auto-discovery: Finds variations like `hey_orah-2.onnx`

### Wake Word Detection Flow

**Initialization**:
1. **Settings Check**: Reads wake word enabled state from `state.py`
2. **Model Loading**:
   - Tries custom model from `data/models/wake_words/`
   - Falls back to built-in models if not found
   - Downloads models on first use if missing
3. **Transcription Blocking**: Blocks transcription until detector ready
4. **Unblocking**: Unblocks transcription once detector initialized

**Detection Loop**:
1. **Audio Processing**:
   - Uses same audio pipeline as VAD
   - Frame size: 1280 samples (80ms at 16kHz)
   - Buffering: Accumulates frames until enough samples
   - Normalization: Same as main transcription (target RMS: 0.12)

2. **Model Prediction**:
   - Input: int16 audio array (1280 samples)
   - Output: Confidence score (0.0-1.0)
   - Threshold: Default 0.01 (configurable)
   - Framework: ONNX (better ARM64 support)

3. **Detection Logic**:
   ```python
   confidence = model.predict(audio_frame)
   detected = confidence >= threshold
   ```

4. **After Detection**:
   - Sets `listening_active = True`
   - Visual feedback: Solid red LED
   - Waits 0.3s before VAD activation
   - Resets VAD state for clean start
   - Proceeds to VAD loop

### Configuration

**Settings** (`state.py`):
- Wake word enabled: Toggle in Settings dialog
- Default: Disabled
- Can be toggled without restart

**Threshold Tuning**:
- Default: 0.01 (very sensitive)
- Higher = less sensitive (fewer false positives)
- Lower = more sensitive (more false positives)
- Tuned via `DEFAULT_THRESHOLD` constant

**Model Selection**:
- Default: `hey_orah`
- Custom models: Place `.onnx` files in `data/models/wake_words/`
- Auto-detects model variations (e.g., `hey_orah-2.onnx`)

### Echo Prevention

**TTS Echo Blocking**:
- **Check Before Processing**: `is_playing()` checked before audio read
- **Skip Processing**: Audio processing skipped entirely during TTS
- **Stream Management**: Stream stopped during TTS playback

**Buffer Management**:
1. **Stream Flush**: Buffer flushed after TTS ends (5 frames discarded)
2. **OpenWakeWord Buffer Clear**: Detector buffer cleared via `clear_buffer()`
3. **VAD Reset**: VAD state reset after flush
4. **Clean State**: Ensures no stale audio processed

**Implementation**:
```python
# In read_audio_frame()
if is_playing():
    return None, None, None, True  # Skip processing, continue loop

# After TTS ends
stream.flush()  # Discard accumulated frames
wake_word_detector.clear_buffer()  # Clear detector buffer
model_vad.reset_states()  # Reset VAD state
```

### Wake Word Removal from Transcription

**Post-Processing**:
- Strips wake word from transcription if present
- Patterns: "hey aura", "hey aura,", "hey aura.", etc.
- Prevents wake word appearing in user query

**Function**: In `listener.py` after transcription
```python
if wake_word_enabled:
    wake_phrases = ["hey aura", "hey aura,", "hey aura."]
    for phrase in wake_phrases:
        if text_lower.startswith(phrase):
            text = text[len(phrase):].strip().lstrip(",.")
            break
```

### Audio Synchronization

**Shared Audio Pipeline**:
- **Same Source**: Both wake word and VAD use same audio stream
- **Same Processing**: Same normalization, same channel extraction
- **Synchronized State**: Stream management synchronized between both

**Frame Processing**:
- Wake word: 1280 samples (80ms) per frame
- VAD: 512 samples (32ms) per frame
- Buffering: Wake word buffers until 1280 samples available

### Performance Optimization

**Model Selection**:
- ONNX framework: Better ARM64 support
- Lightweight models: Low CPU usage
- Efficient buffering: Only processes when enough samples

**Debug Logging**:
- First 10 frames: Logged for initialization verification
- Every 100 frames: Heartbeat to confirm still listening
- High confidence: Logged when confidence > threshold/10

### Troubleshooting

**Wake Word Not Detecting**:
- Check model is loaded: Look for "✅ OpenWakeWord initialized" in logs
- Verify threshold: Lower threshold for more sensitivity
- Check audio normalization: Ensure RMS ~0.12
- Verify device selection: Correct microphone device

**False Positives**:
- Increase threshold: Higher = less sensitive
- Enable echo cancellation: Prevents TTS echo triggers
- Check TTS echo blocking: Verify `is_playing()` checks

**Model Not Loading**:
- Check model file exists: `data/models/wake_words/`
- Verify ONNX format: Must be valid ONNX file
- Check download: Models download on first use
- Fallback: Uses built-in models if custom fails

## Error Handling

### GPU Errors
- **cuDNN Errors**: Attempts GPU memory clear and retry
- **CUDA Failures**: Raises RuntimeError (GPU required)

### Audio Issues
- **Silent Audio**: Detects and warns about quiet audio
- **File Errors**: Returns error JSON with message
- **Timeout**: 10-second timeout on HTTP requests

## Performance Optimization

### Model Selection Trade-offs
- **BEAM_SIZE=5**: ~1x latency, good accuracy
- **BEAM_SIZE=10**: ~2x latency, better accuracy (current)
- **BEAM_SIZE=20**: ~4x latency, best accuracy

### Warmup
- Whisper model warmed up on startup with dummy audio
- Triggers PyTorch JIT compilation
- Reduces first-transcription latency

## Configuration

### Environment Variables
- `WHISPER_MODEL`: Model name (default: `distil-small.en`)

### Medical Terms
- Location: `/shared/medical_terms.json`
- Format: JSON with categories (cardiovascular, respiratory, etc.)
- Auto-loaded on container start

## API Endpoints

### `POST /transcribe`
- **Input**: Audio file (WAV format)
- **Optional**: `initial_prompt` (for context)
- **Output**: JSON with `text` and `timing` information

### `GET /health`
- **Output**: JSON with status, model info, and timing statistics

### `POST /add_medical_term`
- **Input**: JSON with `term` and optional `category`
- **Output**: Success/error response

### `GET /medical_terms`
- **Output**: All medical terms organized by category

## Code Locations

- **Main Service**: `whisper-container/container_rest.py`
- **Audio Processing**: `listener.py` (audio capture and normalization)
- **VAD Integration**: `listener.py` (Silero VAD)
- **Wake Word**: `openwakeword_wake_word.py`

## Dependencies

- `faster-whisper`: Optimized Whisper implementation
- `torch`: PyTorch for GPU acceleration
- `soundfile`: Audio file I/O
- `scipy`: Signal processing (resampling)
- `numpy`: Array operations

