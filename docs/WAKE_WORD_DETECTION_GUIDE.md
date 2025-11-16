# Wake Word Detection Guide

## 📋 Table of Contents
1. [What is Wake Word Detection?](#what-is-wake-word-detection)
2. [Current Architecture vs. Wake Word Architecture](#current-vs-wake-word-architecture)
3. [Wake Word Detection Technologies](#wake-word-detection-technologies)
4. [Integration Architecture](#integration-architecture)
5. [Implementation Examples](#implementation-examples)
6. [Trade-offs and Considerations](#trade-offs-and-considerations)

---

## What is Wake Word Detection?

**Wake word detection** is a specialized form of voice activity detection that identifies specific phrases (like "Hey Aura", "Alexa", "Hey Siri") before activating the main speech processing pipeline.

### Key Differences from VAD:

| **VAD (Current)** | **Wake Word Detection** |
|-------------------|------------------------|
| Detects **any** speech | Detects **specific** phrases |
| Processes everything | Only processes after wake word |
| Lower latency | Slightly higher latency (~100-300ms) |
| No privacy filtering | Privacy: ignores non-wake-word speech |
| Higher false positive rate | Lower false positive rate |

### Why Use Wake Words?

✅ **Privacy**: Device only processes audio after wake word  
✅ **Battery Efficiency**: Less CPU usage (no transcription until activated)  
✅ **False Positive Reduction**: Fewer accidental activations  
✅ **User Control**: Clear indication of when device is "listening"  
✅ **Multi-User**: Can differentiate users with different wake words  

---

## Current Architecture vs. Wake Word Architecture

### Current Architecture (Always-Listening VAD)

```
Microphone (XVF3800)
    ↓
Hardware DSP (Beamforming)
    ↓
Silero VAD → Detects ANY speech (VAD_START_THRESHOLD = 0.25)
    ↓
Advanced Multi-Feature Filter (filters noise)
    ↓
Whisper Transcription (ALWAYS runs)
    ↓
LLM Processing (ALWAYS processes)
```

**Flow in `listener.py`:**
```python
# Line 496-533: Wait for speech
while True:
    audio_block, _ = stream.read(FRAME_SIZE)
    vad_prob = model_vad(audio_block).item()
    
    if vad_prob > VAD_START_THRESHOLD:  # ANY speech triggers
        # Record speech → Whisper → LLM
        break
```

### Wake Word Architecture (Proposed)

```
Microphone (XVF3800)
    ↓
Hardware DSP (Beamforming)
    ↓
Wake Word Detector → Detects "hey aura" ONLY
    ↓
Visual Feedback (Aura eye activates)
    ↓
Silero VAD → Detects speech AFTER wake word
    ↓
Advanced Multi-Feature Filter
    ↓
Whisper Transcription (ONLY after wake word)
    ↓
LLM Processing (ONLY after wake word)
```

**New Flow:**
```python
# Two-stage detection
while True:
    audio_block, _ = stream.read(FRAME_SIZE)
    
    # Stage 1: Wake word detection (low CPU, always running)
    wake_word_detected = wake_word_model.process(audio_block)
    
    if wake_word_detected:
        # Wake word found! Activate listening mode
        print("[Wake Word] ✅ 'Hey Aura' detected")
        activate_listening_mode()  # Visual feedback
        
        # Stage 2: Now wait for actual speech
        while True:
            audio_block, _ = stream.read(FRAME_SIZE)
            vad_prob = model_vad(audio_block).item()
            
            if vad_prob > VAD_START_THRESHOLD:
                # Record speech → Whisper → LLM
                break
```

---

## Wake Word Detection Technologies

### 1. **Porcupine (Picovoice)** ⭐ Recommended

**What it is:**
- Production-ready wake word engine
- Optimized for embedded devices (Jetson compatible)
- Low latency (~100-200ms detection)
- Low CPU usage (~5-10% on Jetson)

**Pros:**
- ✅ Pre-trained models for common phrases ("Hey Aura" available)
- ✅ Can train custom wake words via Picovoice Console
- ✅ Runs entirely on-device (no cloud required)
- ✅ Supports multiple wake words simultaneously
- ✅ Commercial license available (free for personal projects)

**Cons:**
- ❌ Limited to short phrases (1-3 words)
- ❌ Custom wake words require training (takes time)
- ❌ Python bindings may need compilation on Jetson

**Installation:**
```bash
pip install pvporcupine

# For Jetson (ARM64), may need to build from source:
# git clone https://github.com/Picovoice/porcupine
# cd porcupine/binding/python
# python setup.py build_ext --inplace
```

**Usage Example:**
```python
import pvporcupine

# Initialize Porcupine with built-in "Hey Aura" model
porcupine = pvporcupine.create(
    keyword_paths=['path/to/hey-aura_en_linux_v3_0_0.ppn'],
    # Or use built-in: keyword_paths=[pvporcupine.KEYWORDS['hey aura']]
)

# Process audio frames (must be 16kHz mono, 16-bit PCM)
audio_frame = stream.read(512)  # 512 samples = 32ms at 16kHz
keyword_index = porcupine.process(audio_frame)

if keyword_index >= 0:
    print("[Porcupine] ✅ Wake word detected!")
```

**Performance:**
- CPU: ~5-10% on Jetson Orin
- Memory: ~50-100MB
- Latency: ~100-200ms
- Accuracy: ~95%+ with proper tuning

---

### 2. **Mycroft Precise**

**What it is:**
- Open-source wake word detection
- Uses neural networks (TensorFlow Lite)
- Train custom wake words with your own voice
- Community-driven

**Pros:**
- ✅ 100% free and open-source
- ✅ Train custom wake words locally
- ✅ Good for unique wake phrases
- ✅ Active community

**Cons:**
- ❌ Requires training (needs ~1000+ audio samples)
- ❌ Higher CPU usage than Porcupine (~15-20%)
- ❌ More complex setup
- ❌ Less optimized for embedded devices

**Installation:**
```bash
pip install precise-runner

# Train custom wake word (requires ~1000 audio samples):
# git clone https://github.com/MycroftAI/precise
# cd precise
# ./train.sh hey-aura /path/to/audio/samples/
```

**Usage Example:**
```python
from precise_runner import PreciseRunner, PreciseEngine

# Initialize Precise with trained model
engine = PreciseEngine('/path/to/hey-aura.pb')
runner = PreciseRunner(engine, on_activation=lambda: print("Wake word detected!"))

# Start listening
runner.start()
```

**Performance:**
- CPU: ~15-20% on Jetson Orin
- Memory: ~100-150MB
- Latency: ~200-300ms
- Accuracy: ~90-95% with good training data

---

### 3. **Custom Model (PyTorch/TensorFlow)**

**What it is:**
- Train your own wake word detection model
- Full control over architecture and training
- Can use existing models like Wav2Vec2, Whisper embeddings, etc.

**Pros:**
- ✅ Complete control
- ✅ Can fine-tune for your specific use case
- ✅ Can combine with existing Whisper pipeline

**Cons:**
- ❌ Requires ML expertise
- ❌ Needs large training dataset
- ❌ Higher CPU usage (depends on model size)
- ❌ Most complex to implement

**Architecture Options:**

**Option A: Keyword Spotting (KWS)**
- Small CNN model (e.g., TC-ResNet, Keyword Transformer)
- Input: Mel-spectrogram or MFCC features
- Output: Probability of wake word

**Option B: Whisper-based**
- Use Whisper's encoder to extract embeddings
- Classify embeddings with small MLP
- Reuses existing Whisper model (lower overhead)

**Usage Example (Simplified):**
```python
import torch
import torchaudio

class WakeWordDetector:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
    def process(self, audio_frame):
        # Convert to mel-spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=40
        )(audio_frame)
        
        # Run inference
        with torch.no_grad():
            prob = self.model(mel_spec.unsqueeze(0))
        
        return prob.item() > 0.8  # Threshold
```

**Performance:**
- CPU: 10-30% (depends on model)
- Memory: 100-500MB
- Latency: 200-500ms
- Accuracy: Varies (80-95% depending on training)

---

### 4. **Snowboy (Deprecated, Not Recommended)**

**What it is:**
- Previously popular wake word engine
- **Note: Project is deprecated** (last update 2019)

**Status:** ❌ **Not recommended** - Use Porcupine instead (same company, newer product)

---

## Integration Architecture

### Modified `listener.py` Structure

```python
# === listener.py with Wake Word Detection ===

import pvporcupine  # Or Mycroft Precise, custom model, etc.

# === Wake Word Configuration ===
ENABLE_WAKE_WORD = True  # Toggle wake word detection
WAKE_WORD_SENSITIVITY = 0.5  # 0.0-1.0 (lower = more strict)

# === Initialize Wake Word Detector ===
wake_word_detector = None
if ENABLE_WAKE_WORD:
    try:
        # Option 1: Porcupine
        wake_word_detector = pvporcupine.create(
            keyword_paths=['path/to/hey-aura.ppn'],
            sensitivities=[WAKE_WORD_SENSITIVITY]
        )
        print("[Wake Word] ✅ Porcupine initialized")
    except Exception as e:
        print(f"[Wake Word] ⚠️ Failed to initialize: {e}")
        print("[Wake Word] 💡 Falling back to VAD-only mode")
        ENABLE_WAKE_WORD = False

# === Wake Word Detection Function ===
def detect_wake_word(audio_frame):
    """
    Detect wake word in audio frame.
    Returns: (detected: bool, confidence: float)
    """
    if not ENABLE_WAKE_WORD or wake_word_detector is None:
        return False, 0.0
    
    try:
        # Porcupine expects int16 PCM, 512 samples
        if audio_frame.dtype != 'int16':
            audio_frame = (audio_frame * 32767).astype('int16')
        
        keyword_index = wake_word_detector.process(audio_frame)
        
        if keyword_index >= 0:
            return True, 1.0  # Wake word detected
        return False, 0.0
    except Exception as e:
        print(f"[Wake Word] ⚠️ Detection error: {e}")
        return False, 0.0

# === Modified Main Loop ===
def listen():
    global wake_word_detector
    
    # ... existing setup code ...
    
    wake_word_buffer = []  # Buffer for wake word detection
    listening_active = False  # True after wake word detected
    
    with stream:
        play_welcome_prompt(stream)
        
        while True:
            # Pause during TTS
            if is_playing():
                # ... existing TTS pause code ...
                listening_active = False  # Reset after TTS
                continue
            
            # === STAGE 1: Wake Word Detection (Always Running) ===
            if ENABLE_WAKE_WORD and not listening_active:
                try:
                    audio_block, _ = stream.read(FRAME_SIZE)
                    channel_audio = audio_block[:, MICROPHONE_CHANNEL]
                    
                    # Convert to int16 for Porcupine
                    audio_int16 = (channel_audio * 32767).astype('int16')
                    
                    # Detect wake word
                    wake_detected, confidence = detect_wake_word(audio_int16)
                    
                    if wake_detected:
                        print(f"\n[Wake Word] ✅ 'Hey Aura' detected (confidence: {confidence:.2f})")
                        listening_active = True
                        
                        # Visual feedback
                        from gui.aura_gui import set_wake_word_activated
                        set_wake_word_activated(True)
                        
                        # Optional: Play confirmation sound
                        # play_confirmation_sound()
                        
                        # Clear buffer
                        wake_word_buffer = []
                        
                        # Wait a moment before starting VAD (avoid wake word in transcription)
                        time.sleep(0.3)
                
                except Exception as e:
                    print(f"[Wake Word] ⚠️ Error: {e}")
                    continue
            
            # === STAGE 2: VAD + Speech Processing (Only After Wake Word) ===
            if listening_active:
                buffer = []
                silence_start = None
                
                # Wait for speech (existing VAD code)
                while True:
                    # ... existing VAD detection code (lines 496-533) ...
                    
                    audio_block, _ = stream.read(FRAME_SIZE)
                    channel_audio = audio_block[:, MICROPHONE_CHANNEL]
                    
                    vad_prob = model_vad(torch.from_numpy(channel_audio), SAMPLE_RATE).item()
                    
                    if vad_prob > VAD_START_THRESHOLD:
                        print(f"\n[VAD] 🔊 Speech detected after wake word")
                        break
                
                # Record speech (existing code)
                # ... lines 551-585 ...
                
                # Process audio
                full_audio = np.concatenate(buffer)
                mono = full_audio[:, 0]
                
                text = transcribe(mono)
                
                if text:
                    # Optional: Strip "hey aura" from transcription if present
                    text = strip_wake_word_from_text(text)
                    send_to_llm(text)
                
                # Reset for next wake word detection
                listening_active = False
                from gui.aura_gui import set_wake_word_activated
                set_wake_word_activated(False)
            
            # If wake word disabled, use existing VAD-only flow
            elif not ENABLE_WAKE_WORD:
                # ... existing VAD-only code ...
                pass
```

---

## Implementation Examples

### Example 1: Porcupine Integration (Recommended)

**File: `aura-control/core/wake_word_porcupine.py`**

```python
"""
Porcupine wake word detection integration
Requires: pip install pvporcupine
"""

import os
import numpy as np
import pvporcupine

class PorcupineWakeWord:
    def __init__(self, keyword_path=None, sensitivity=0.5):
        """
        Initialize Porcupine wake word detector.
        
        Args:
            keyword_path: Path to .ppn model file (or None for built-in)
            sensitivity: Detection sensitivity (0.0-1.0)
        """
        self.keyword_path = keyword_path
        self.sensitivity = sensitivity
        self.porcupine = None
        self.is_active = False
        
    def initialize(self):
        """Initialize Porcupine engine"""
        try:
            if self.keyword_path and os.path.exists(self.keyword_path):
                # Use custom model
                self.porcupine = pvporcupine.create(
                    keyword_paths=[self.keyword_path],
                    sensitivities=[self.sensitivity]
                )
                print(f"[Wake Word] ✅ Porcupine initialized with custom model: {self.keyword_path}")
            else:
                # Use built-in keywords (if available)
                # Check if 'hey aura' is available
                available_keywords = pvporcupine.KEYWORDS
                if 'hey aura' in available_keywords:
                    self.porcupine = pvporcupine.create(
                        keywords=['hey aura'],
                        sensitivities=[self.sensitivity]
                    )
                    print(f"[Wake Word] ✅ Porcupine initialized with built-in 'hey aura'")
                else:
                    raise ValueError("'hey aura' not available in Porcupine. Train custom model at https://console.picovoice.ai/")
            
            # Get required frame length
            self.frame_length = self.porcupine.frame_length
            self.sample_rate = self.porcupine.sample_rate
            
            print(f"[Wake Word]   Frame length: {self.frame_length} samples")
            print(f"[Wake Word]   Sample rate: {self.sample_rate} Hz")
            
            return True
        except Exception as e:
            print(f"[Wake Word] ❌ Failed to initialize Porcupine: {e}")
            return False
    
    def process(self, audio_frame):
        """
        Process audio frame for wake word detection.
        
        Args:
            audio_frame: numpy array of audio samples (int16 or float32)
            
        Returns:
            (detected: bool, confidence: float)
        """
        if self.porcupine is None:
            return False, 0.0
        
        try:
            # Convert to int16 if needed
            if audio_frame.dtype == 'float32' or audio_frame.dtype == 'float64':
                # Clamp to [-1, 1] range
                audio_frame = np.clip(audio_frame, -1.0, 1.0)
                # Convert to int16
                audio_frame = (audio_frame * 32767).astype('int16')
            
            # Ensure correct length
            if len(audio_frame) != self.frame_length:
                # Pad or truncate
                if len(audio_frame) < self.frame_length:
                    audio_frame = np.pad(audio_frame, (0, self.frame_length - len(audio_frame)))
                else:
                    audio_frame = audio_frame[:self.frame_length]
            
            # Process frame
            keyword_index = self.porcupine.process(audio_frame)
            
            if keyword_index >= 0:
                return True, 1.0
            return False, 0.0
            
        except Exception as e:
            print(f"[Wake Word] ⚠️ Processing error: {e}")
            return False, 0.0
    
    def release(self):
        """Release Porcupine resources"""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
```

**Integration in `listener.py`:**

```python
# Add at top of listener.py
from wake_word_porcupine import PorcupineWakeWord

# Initialize wake word detector
WAKE_WORD_ENABLED = os.getenv("ENABLE_WAKE_WORD", "false").lower() == "true"
WAKE_WORD_SENSITIVITY = float(os.getenv("WAKE_WORD_SENSITIVITY", "0.5"))

wake_word_detector = None
if WAKE_WORD_ENABLED:
    keyword_path = os.getenv("WAKE_WORD_MODEL_PATH", None)
    wake_word_detector = PorcupineWakeWord(
        keyword_path=keyword_path,
        sensitivity=WAKE_WORD_SENSITIVITY
    )
    if not wake_word_detector.initialize():
        wake_word_detector = None
        print("[Listener] ⚠️ Wake word detection disabled (fallback to VAD-only)")
```

---

### Example 2: Simple Keyword Matching (Fallback)

**If wake word detection fails, you can use simple text matching as a fallback:**

```python
def strip_wake_word_from_text(text):
    """
    Strip wake word phrases from transcribed text.
    This is a fallback if wake word detection isn't used.
    """
    wake_phrases = [
        "hey aura",
        "hey aura,",
        "hey aura.",
        "okay aura",
        "ok aura",
    ]
    
    text_lower = text.lower().strip()
    
    for phrase in wake_phrases:
        if text_lower.startswith(phrase):
            # Remove wake phrase and clean up
            text = text[len(phrase):].strip()
            # Remove leading punctuation
            text = text.lstrip(",.")
            return text
    
    return text
```

---

## Trade-offs and Considerations

### When to Use Wake Word Detection

✅ **Use wake word detection if:**
- Privacy is important (device in public spaces)
- Battery life is critical (mobile/embedded devices)
- False positives are problematic (TV, conversations triggering device)
- You want clear user control ("is it listening?")

❌ **Skip wake word detection if:**
- Latency must be minimal (<100ms)
- CPU resources are very limited (<5% available)
- Device is always in private, controlled environment
- Continuous listening is desired (always-on assistant)

### Performance Impact

| Metric | VAD Only | VAD + Wake Word |
|--------|----------|-----------------|
| CPU (idle) | ~5% | ~10-15% (Porcupine) |
| CPU (active) | ~30-40% | ~35-45% |
| Memory | ~200MB | ~250-300MB |
| Latency | ~500ms | ~600-800ms |
| Battery | Higher | Lower (less processing) |

### Privacy Considerations

**Wake Word Detection:**
- ✅ Audio processed only after wake word
- ✅ Can implement "hotword activation logging"
- ✅ User has clear indication of active listening

**VAD Only:**
- ⚠️ All speech is transcribed (even if not intended for device)
- ⚠️ Privacy concerns in shared spaces
- ⚠️ False positives mean unintended transcriptions

### Configuration Recommendations

**For Jetson Orin (Recommended):**
- Use **Porcupine** (optimized for ARM)
- Set sensitivity: `0.5-0.7` (tune based on environment)
- Enable wake word in `.env`: `ENABLE_WAKE_WORD=true`

**For Desktop/Server:**
- Use **Porcupine** or **Mycroft Precise**
- Can afford slightly higher CPU usage
- More flexibility in model choice

**For Low-Power Devices:**
- Consider **lightweight custom model**
- Use **simpler keyword matching** as fallback
- May need to disable some advanced features

---

## Next Steps

1. **Test Porcupine** on Jetson:
   ```bash
   pip install pvporcupine
   python -c "import pvporcupine; print(pvporcupine.KEYWORDS)"
   ```

2. **Train custom wake word** (if "hey aura" not available):
   - Visit: https://console.picovoice.ai/
   - Record ~100 samples of "hey aura"
   - Download `.ppn` model file

3. **Integrate into `listener.py`**:
   - Add wake word detection class
   - Modify main loop (two-stage detection)
   - Update GUI for wake word feedback

4. **Add configuration options**:
   - `.env`: `ENABLE_WAKE_WORD=true`
   - `aura_config.sh`: Toggle wake word, adjust sensitivity

5. **Test and tune**:
   - Adjust sensitivity based on false positive/false negative rate
   - Measure latency impact
   - Monitor CPU usage

---

## References

- **Porcupine Documentation**: https://github.com/Picovoice/porcupine
- **Mycroft Precise**: https://github.com/MycroftAI/precise
- **Wake Word Training Guide**: https://picovoice.ai/docs/quick-start/console-quick-start/
- **Jetson Optimization**: https://developer.nvidia.com/embedded/learn/getting-started-jetson

---

**Questions?** Check the codebase or open an issue for integration help!

