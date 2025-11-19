# Quick Start: Training "Hey Aura" Wake Word Model

## Overview

This is a quick reference for training a custom "Hey Aura" wake word model with TTS/echo rejection. For detailed information, see `TRAIN_HEY_AURA_WAKE_WORD.md`.

---

## Prerequisites

### Platform Support

**✅ Training CAN be done on Jetson NX/Orin NX:**
- Fully supported on ARM64 (aarch64) architecture
- Training is CPU-based, so it's slower (~30-60 min for 50 epochs) but works fine
- **Alternative**: Train on desktop/server (faster, ~5-15 min), then transfer model to Jetson

### Install Training Tools

```bash
# Activate virtual environment
source ~/aura-env/bin/activate

# Install training tools
pip install precise-runner precise

# On Jetson, you may need additional dependencies:
# System packages (via apt):
sudo apt-get install -y \
    python3-scipy \
    libhdf5-dev \
    python3-h5py \
    libopenblas-dev

# Python packages (via pip):
pip install cython
```

---

## Quick Training Workflow

### Step 1: Collect Training Data

```bash
cd ~/LedgerAI/setup/scripts

# Collect "Hey Aura" samples (200-500 recommended)
./collect_wake_word_data.sh wake-word

# Collect other speech samples (200-500 recommended)
./collect_wake_word_data.sh not-wake-word

# Collect TTS/echo samples (100-200 recommended) ⭐ CRITICAL
./collect_wake_word_data.sh tts-echo

# Collect background noise (50-100 recommended)
./collect_wake_word_data.sh noise
```

**Important**: The TTS/echo samples are critical for preventing false positives when TTS plays through speakers.

---

### Step 2: Train Model

```bash
# Train with default 50 epochs
./train_hey_aura.sh

# Or specify custom epochs
./train_hey_aura.sh 30
```

The script will:
- ✅ Combine negative training data (not-wake-word + tts-echo + noise)
- ✅ Train the model
- ✅ Convert to .pb format
- ✅ Test on validation set (if available)
- ✅ Install model to `~/precise-models/hey-aura.pb`

---

### Step 3: Deploy Model

**Option A: Via Settings GUI**
1. Open Settings → AI Model Settings
2. Set "Wake Word Model Path" to: `~/precise-models/hey-aura.pb`
3. Restart Aura

**Option B: Via Command Line**
```bash
# Update model path in state
python3 -c "
from aura_control.core.state import set_wake_word_model_path
set_wake_word_model_path('~/precise-models/hey-aura.pb')
"

# Restart Aura
sudo systemctl restart aura.service
```

---

## Data Collection Tips

### Wake Word Samples ("Hey Aura")
- ✅ Multiple speakers (different voices)
- ✅ Various distances and volumes
- ✅ Different intonations
- ✅ Different environments

### TTS/Echo Samples (Critical!)
- ✅ Record TTS responses through microphone
- ✅ **Include TTS saying "Hey Aura"** (should NOT trigger)
- ✅ Various TTS lengths and volumes
- ✅ Different TTS voices (if available)

### Not Wake Word Samples
- ✅ Other wake words ("Hey Google", "Alexa")
- ✅ Similar phrases ("Hey you", "Hey there")
- ✅ Common commands
- ✅ Random speech

### Noise Samples
- ✅ Room tone
- ✅ Fan/AC noise
- ✅ Background sounds

---

## Testing Checklist

After training, verify:

- [ ] ✅ Says "Hey Aura" → Triggers wake word
- [ ] ✅ Says "Hey Aura" quietly → Triggers wake word
- [ ] ✅ Says "Hey you" → Does NOT trigger
- [ ] ✅ TTS plays response → Does NOT trigger
- [ ] ✅ **TTS says "Hey Aura"** → Does NOT trigger (critical!)
- [ ] ✅ Background noise only → Does NOT trigger

---

## Troubleshooting

### High False Positive Rate on TTS
- **Solution**: Add more TTS/echo samples to training data
- **Critical**: Ensure TTS samples include TTS saying "Hey Aura"

### Low True Positive Rate
- **Solution**: Add more "Hey Aura" samples
- Check microphone levels and audio quality

### Model Too Sensitive
- **Solution**: Add more negative samples
- Increase threshold in `precise_wake_word.py`

### Model Not Sensitive Enough
- **Solution**: Add more positive samples
- Reduce threshold in `precise_wake_word.py`

---

## File Locations

- **Training Data**: `~/precise-training/`
- **Trained Model**: `~/precise-models/hey-aura.pb`
- **Training Scripts**: `~/LedgerAI/setup/scripts/`
- **Documentation**: `~/LedgerAI/setup/scripts/TRAIN_HEY_AURA_WAKE_WORD.md`

---

## Quick Commands Reference

```bash
# Collect data
./collect_wake_word_data.sh [wake-word|not-wake-word|tts-echo|noise]

# Train model
./train_hey_aura.sh [epochs]

# Check data counts
ls ~/precise-training/wake-word/*.wav | wc -l
ls ~/precise-training/tts-echo/*.wav | wc -l

# Test model manually
precise-test ~/precise-training/hey-aura.net \
    ~/precise-training/test/wake-word \
    ~/precise-training/test/combined-negative
```

---

## Success Criteria

A well-trained model should:
1. ✅ Detect "Hey Aura" reliably (>95% true positive rate)
2. ✅ Reject TTS/echo audio (>99% rejection rate)
3. ✅ Reject other speech (>98% rejection rate)
4. ✅ Handle various voice tones and environments
5. ✅ **Never trigger on TTS saying "Hey Aura"** (critical!)

---

**Key Success Factor**: Include sufficient TTS/echo samples in negative training data, especially samples where TTS says "Hey Aura" but should NOT trigger detection.

