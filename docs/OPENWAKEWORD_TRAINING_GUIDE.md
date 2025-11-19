# OpenWakeWord Training Guide for "Hey Aura"

This guide explains how to train a custom OpenWakeWord model for "hey aura" that handles TTS echo.

## Overview

The training process creates a wake word model that:
- ✅ Detects "hey aura" when spoken by humans
- ❌ Ignores "hey aura" when played through TTS (echo rejection)
- ❌ Ignores similar-sounding phrases

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install openwakeword soundfile pyaudio scipy
   ```

2. **Configure ElevenLabs API** (for TTS echo generation):
   ```bash
   # Edit .env file
   ELEVENLABS_API_KEY=your_api_key_here
   ELEVENLABS_VOICE_ID=your_voice_id
   ```

## Training Process

### Step 1: Collect Training Data

Run the training script in collection mode:

```bash
cd /path/to/LedgerAI
python3 train_openwakeword_hey_aura.py --mode collect
```

**Device Selection:**
- The script will ask you to select a microphone device **once** at the start
- Your selection is saved to `data/wake_word_training/device_config.json`
- The same device will be reused for all recordings (no need to select each time)
- The script auto-detects preferred devices (reSpeaker, USB Audio, XVF3800, etc.)
- On subsequent runs, it will remember your preference

This will:
1. **Collect positive samples** (20+ recommended)
   - Record yourself saying "hey aura" naturally
   - Vary tone, speed, volume, and distance
   - Speak as you would in real use

2. **Collect negative samples** (30+ recommended)
   - Record other phrases (NOT "hey aura")
   - Include background noise, silence
   - Include similar-sounding phrases like "hey there", "hey siri", etc.

3. **Generate TTS echo samples** (20+ recommended)
   - **Default mode**: Plays TTS through speakers and records it back through microphone
   - Captures real echo/reverb from your environment (more realistic)
   - These are **negative samples** - model learns NOT to trigger on TTS
   - Uses your configured ElevenLabs voice
   - **Make sure speakers are on and microphone can hear them!**
   - Alternative: Use `--tts-direct` flag for direct generation (no echo, faster but less realistic)

### Step 2: Review Training Data

Check the collected samples:

```bash
ls -lh data/wake_word_training/positive/
ls -lh data/wake_word_training/negative/
ls -lh data/wake_word_training/negative_tts/
```

**Quality tips:**
- Ensure positive samples are clear and natural
- Ensure negative samples don't contain "hey aura"
- TTS samples should sound like your actual TTS output

### Step 3: Train the Model

OpenWakeWord training is typically done via Google Colab notebook:

1. **Format training data:**
   ```bash
   python3 train_openwakeword_hey_aura.py --mode train
   ```
   This prepares the data in the correct format.

2. **Use OpenWakeWord training notebook:**
   - Visit: https://github.com/dscripka/openWakeWord#training-custom-models
   - Upload the formatted training data from `data/wake_word_training/formatted/`
   - Follow the notebook instructions to train the model

3. **Download the trained model:**
   - Save as `hey_aura_v0.1.onnx` (or similar)
   - Place in `data/models/wake_words/`

### Alternative: CLI Training (if available)

If OpenWakeWord provides CLI training tools:

```bash
# Check if openwakeword CLI is available
openwakeword train --help

# Train using formatted data
openwakeword train \
  --positive-dir data/wake_word_training/formatted/positive \
  --negative-dir data/wake_word_training/formatted/negative \
  --output data/models/wake_words/hey_aura_v0.1.onnx \
  --wake-phrase "hey aura"
```

## Using the Trained Model

### Update OpenWakeWord Integration

Edit `aura-control/core/openwakeword_wake_word.py`:

```python
# Change the default model name
DEFAULT_MODEL = "hey_aura_v0.1"  # Your custom model

# Or specify model path
DEFAULT_MODEL = "/path/to/data/models/wake_words/hey_aura_v0.1.onnx"
```

### Test the Model

1. **Start the listener:**
   ```bash
   cd aura-control/core
   python3 listener.py
   ```

2. **Test wake word detection:**
   - Say "hey aura" - should trigger ✅
   - Play TTS saying "hey aura" - should NOT trigger ❌
   - Say similar phrases - should NOT trigger ❌

3. **Adjust threshold if needed:**
   - Edit `aura-control/core/state.py` or use GUI settings
   - Lower threshold = more sensitive (more false positives)
   - Higher threshold = less sensitive (may miss detections)

## Training Data Best Practices

### Positive Samples
- **Quantity:** 20-50 samples minimum
- **Diversity:**
  - Different speakers (if possible)
  - Different distances (near, far)
  - Different environments (quiet, noisy)
  - Different speaking styles (fast, slow, loud, quiet)
- **Quality:** Clear audio, minimal background noise

### Negative Samples
- **Quantity:** 30-100 samples minimum
- **Types:**
  - Other wake phrases ("hey siri", "hey google", etc.)
  - Similar phrases ("hey there", "hey you")
  - Just "aura" or just "hey"
  - Background noise, silence
  - Music, TV, other audio
- **TTS Echo:** Critical! Include 20+ TTS-generated "hey aura" samples

### TTS Echo Samples
- **Why critical:** Without these, model will trigger on TTS output
- **Generation modes:**
  - **Default (recommended):** Plays TTS through speakers and records echo/reverb
    - More realistic - captures actual room acoustics
    - Better training data for echo rejection
    - Requires speakers to be on and microphone to hear them
  - **Direct mode:** Generates TTS audio directly (no echo)
    - Faster but less realistic
    - Use with `--tts-direct` flag
- **Quantity:** 20-50 samples recommended
- **Setup:** Ensure speakers are on and positioned so microphone can pick up the audio

## Troubleshooting

### Model triggers on TTS
- **Solution:** Add more TTS negative samples
- **Check:** Ensure TTS samples use the same voice as production

### Model misses detections
- **Solution:** Lower threshold or add more positive samples
- **Check:** Ensure positive samples match your speaking style

### Model triggers on similar phrases
- **Solution:** Add more negative samples with similar phrases
- **Check:** Review negative samples for quality

### Training data insufficient
- **Minimum:** 10 positive, 20 negative samples
- **Recommended:** 30+ positive, 50+ negative samples
- **Best:** 50+ positive, 100+ negative samples

## File Structure

```
data/
├── wake_word_training/
│   ├── positive/              # Human "hey aura" recordings
│   ├── negative/               # Other phrases, noise
│   ├── negative_tts/          # TTS-generated "hey aura" (negative)
│   ├── formatted/             # Formatted for training
│   │   ├── positive/
│   │   └── negative/
│   └── dataset_manifest.json  # Training data manifest
└── models/
    └── wake_words/
        └── hey_aura_v0.1.onnx # Trained model (after training)
```

## Advanced: Fine-tuning

After initial training, you can fine-tune:

1. **Collect problematic samples:**
   - Record false positives (shouldn't trigger but does)
   - Record false negatives (should trigger but doesn't)

2. **Add to training data:**
   - False positives → add to negative samples
   - False negatives → add to positive samples

3. **Retrain model:**
   - Use updated dataset
   - May require fewer epochs if starting from previous model

## References

- OpenWakeWord GitHub: https://github.com/dscripka/openWakeWord
- Training Documentation: https://github.com/dscripka/openWakeWord#training-custom-models
- Model Format: ONNX (optimized for inference)

## Quick Start

```bash
# 1. Collect training data (interactive)
python3 train_openwakeword_hey_aura.py --mode collect

# 2. Generate TTS echo samples (automatic)
python3 train_openwakeword_hey_aura.py --mode tts-only

# 3. Prepare and train (requires Colab or CLI)
python3 train_openwakeword_hey_aura.py --mode train

# 4. Full pipeline (collect + train)
python3 train_openwakeword_hey_aura.py --mode full
```

