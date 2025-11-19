# Training "Hey Aura" Wake Word Model with Echo/TTS Rejection

## Overview

This guide covers training a custom Mycroft Precise wake word model for "Hey Aura" that:
- ✅ Detects "Hey Aura" reliably
- ✅ Rejects TTS/echo audio (prevents false positives from speaker output)
- ✅ Works well on Jetson hardware
- ✅ Handles various voice tones and environments

---

## Prerequisites

### Platform Support

**✅ Training CAN be done on Jetson NX/Orin NX:**
- Mycroft Precise training works on ARM64 (aarch64) architecture
- Jetson NX/Orin NX are fully supported
- Training is CPU-based (not GPU-accelerated), so it will be slower than on desktop/server

**⚠️ Performance Considerations:**
- Training on Jetson NX: ~30-60 minutes for 50 epochs (depending on data size)
- Training on desktop/server: ~5-15 minutes for 50 epochs
- **Alternative**: Train on a more powerful machine, then transfer the `.pb` model file to Jetson

**💡 Recommendation:**
- For initial testing: Train on Jetson NX (convenient, all data already there)
- For production/final model: Consider training on desktop/server for faster iteration

### 1. Install Training Tools

```bash
# Activate virtual environment
source ~/aura-env/bin/activate

# Install training dependencies
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

**Note**: Mycroft Precise training uses TensorFlow, which should work on Jetson with ARM-compatible builds. If you encounter TensorFlow issues, you may need to install Jetson-optimized TensorFlow:

```bash
# For Jetson (if standard TensorFlow doesn't work)
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow
```

### 2. Verify Installation

```bash
# Check precise-train is available
precise-train --help

# Check precise-test is available
precise-test --help

# Verify TensorFlow works (on Jetson)
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

---

## Training Data Collection Strategy

### Critical: Echo/TTS Rejection Data

**The key to preventing false positives is including TTS/echo audio in negative training data.**

### Directory Structure

```
~/precise-training/
├── wake-word/              # Positive samples: "Hey Aura"
│   ├── user1_001.wav
│   ├── user1_002.wav
│   ├── user2_001.wav
│   └── ...
├── not-wake-word/          # Negative samples: Other speech
│   ├── other_phrases_001.wav
│   ├── other_phrases_002.wav
│   └── ...
├── tts-echo/               # ⭐ CRITICAL: TTS audio recordings
│   ├── tts_response_001.wav
│   ├── tts_response_002.wav
│   ├── tts_with_hey_aura_001.wav  # TTS saying "Hey Aura" (should NOT trigger)
│   └── ...
├── noise/                  # Background noise
│   ├── room_tone_001.wav
│   ├── fan_noise_001.wav
│   └── ...
└── test/                   # Validation set (same structure)
    ├── wake-word/
    ├── not-wake-word/
    ├── tts-echo/
    └── noise/
```

### Audio Requirements

- **Format**: WAV, 16kHz, 16-bit, mono
- **Duration**: 1-3 seconds per sample
- **Quality**: Clear recordings, minimal clipping

---

## Data Collection Steps

### Step 1: Record "Hey Aura" Samples (Positive Data)

**Goal**: 200-500 samples of "Hey Aura"

```bash
# Create directory
mkdir -p ~/precise-training/wake-word

# Record samples using arecord (on Jetson)
# Record 2 seconds, 16kHz, mono
arecord -D plughw:0,0 -f S16_LE -r 16000 -c 1 -d 2 ~/precise-training/wake-word/sample_001.wav
```

**Collection Tips**:
- ✅ Multiple speakers (different voices, ages, genders)
- ✅ Various distances from microphone (close, medium, far)
- ✅ Different speaking volumes (normal, quiet, loud)
- ✅ Different intonations (questioning, excited, tired)
- ✅ Various environments (quiet room, with background noise)
- ✅ Different times of day (voice changes throughout day)

**Target**: 200-500 samples minimum

---

### Step 2: Record "Not Wake Word" Samples (Negative Data)

**Goal**: 200-500 samples of other speech

```bash
mkdir -p ~/precise-training/not-wake-word
```

**What to Record**:
- ✅ Other wake words ("Hey Google", "Alexa", "Hey Siri")
- ✅ Similar phrases ("Hey you", "Hey there", "Hey buddy")
- ✅ Common commands ("What time is it", "Turn on lights")
- ✅ Random speech (conversations, reading text)
- ✅ Numbers, dates, names

**Target**: 200-500 samples

---

### Step 3: Record TTS/Echo Audio (Critical for Echo Rejection)

**Goal**: 100-200 samples of TTS audio captured by microphone

```bash
mkdir -p ~/precise-training/tts-echo
```

**How to Collect**:

1. **Record TTS Responses**:
   ```bash
   # Start recording
   arecord -D plughw:0,0 -f S16_LE -r 16000 -c 1 -d 5 ~/precise-training/tts-echo/tts_001.wav &
   
   # Trigger TTS response (ask Aura a question)
   # Let TTS play through speakers
   # Microphone will pick up the echo
   
   # Stop recording after TTS finishes
   ```

2. **Record TTS Saying "Hey Aura"**:
   - Use TTS to generate "Hey Aura" audio
   - Play through speakers
   - Record microphone input (will capture echo)
   - **This is critical**: Model must learn to reject TTS even when it says the wake word

3. **Record Various TTS Scenarios**:
   - ✅ Short responses (1-2 seconds)
   - ✅ Long responses (5-10 seconds)
   - ✅ Different TTS voices (if you have multiple)
   - ✅ Different volumes (quiet, normal, loud)
   - ✅ TTS with background noise

**Target**: 100-200 samples (minimum 50 samples of TTS saying "Hey Aura")

---

### Step 4: Record Background Noise

**Goal**: 50-100 samples of ambient noise

```bash
mkdir -p ~/precise-training/noise
```

**What to Record**:
- ✅ Room tone (silence with minimal background)
- ✅ Fan/AC noise
- ✅ Keyboard typing
- ✅ Paper rustling
- ✅ TV/radio in background
- ✅ Traffic noise (if applicable)

**Target**: 50-100 samples

---

## Data Preparation

### Convert Audio to Required Format

```bash
# Install sox for audio conversion
sudo apt-get install sox

# Convert all audio to 16kHz, 16-bit, mono WAV
for file in ~/precise-training/**/*.wav; do
    sox "$file" -r 16000 -b 16 -c 1 "${file%.wav}_converted.wav"
    mv "${file%.wav}_converted.wav" "$file"
done
```

### Split into Training/Test Sets

```bash
# Move 20% to test directory (maintain structure)
mkdir -p ~/precise-training/test/{wake-word,not-wake-word,tts-echo,noise}

# Example: Move 20% of wake-word samples
cd ~/precise-training/wake-word
total=$(ls *.wav | wc -l)
test_count=$((total / 5))
ls *.wav | shuf | head -n $test_count | xargs -I {} mv {} ../test/wake-word/

# Repeat for other directories
```

---

## Training Process

### Step 1: Initial Training (Wake Word vs Not Wake Word)

```bash
cd ~/precise-training

# Train initial model (10 epochs)
precise-train -e 10 hey-aura.net \
    wake-word \
    not-wake-word
```

**Expected Output**:
- Model file: `hey-aura.net`
- Training accuracy should be >90%

---

### Step 2: Add Echo/TTS Rejection Training

```bash
# Combine negative data (not-wake-word + tts-echo + noise)
mkdir -p combined-negative
cp not-wake-word/*.wav combined-negative/
cp tts-echo/*.wav combined-negative/
cp noise/*.wav combined-negative/

# Retrain with echo rejection
precise-train -e 20 hey-aura-echo-reject.net \
    wake-word \
    combined-negative
```

**Key**: Including TTS/echo in negative training teaches the model to reject it.

---

### Step 3: Fine-Tune with More Epochs

```bash
# Fine-tune for better accuracy (30-50 epochs)
# On Jetson NX: This may take 30-60 minutes
# On desktop/server: This may take 5-15 minutes
precise-train -e 50 hey-aura-final.net \
    wake-word \
    combined-negative
```

**Monitor Training**:
- Watch for overfitting (training accuracy >> test accuracy)
- If overfitting, reduce epochs or add more data
- **On Jetson**: Monitor system resources (`htop` or `jetson_stats`) - training is CPU-intensive
- **Performance Tip**: Start with fewer epochs (10-20) to test, then increase if needed

---

## Testing

### Step 1: Test on Validation Set

```bash
# Test model accuracy
precise-test hey-aura-final.net \
    test/wake-word \
    test/combined-negative
```

**Target Metrics**:
- ✅ Wake word detection: >95% true positive rate
- ✅ False positive rate: <1% (especially on TTS/echo)
- ✅ TTS rejection: >99% (should NOT trigger on TTS saying "Hey Aura")

---

### Step 2: Real-World Testing

```bash
# Convert .net to .pb format for deployment
precise-convert hey-aura-final.net hey-aura.pb
```

**Test Scenarios**:
1. ✅ Say "Hey Aura" → Should trigger
2. ✅ Say "Hey Aura" quietly → Should trigger
3. ✅ Say "Hey Aura" loudly → Should trigger
4. ✅ Say "Hey you" → Should NOT trigger
5. ✅ Play TTS response → Should NOT trigger
6. ✅ Play TTS saying "Hey Aura" → Should NOT trigger (critical!)
7. ✅ Background noise only → Should NOT trigger

---

## Deployment

### Step 1: Install Model

```bash
# Copy model to precise-models directory
cp hey-aura.pb ~/precise-models/hey-aura.pb

# Update permissions
chmod 644 ~/precise-models/hey-aura.pb
```

### Step 2: Configure Aura to Use Custom Model

**Option A: Via Settings GUI**
1. Open Settings → AI Model Settings
2. Set "Wake Word Model Path" to: `~/precise-models/hey-aura.pb`

**Option B: Via State Module**
```python
from core.state import set_wake_word_model_path
set_wake_word_model_path("~/precise-models/hey-aura.pb")
```

### Step 3: Restart Aura

```bash
# Restart Aura to load new model
sudo systemctl restart aura.service
# Or if running manually:
# Stop and restart main.py
```

---

## Advanced: Echo Rejection Techniques

### Technique 1: Data Augmentation

**Add echo simulation to training data**:

```python
# augment_echo.py
import numpy as np
import soundfile as sf

def add_echo(audio, delay=0.1, decay=0.3):
    """Add echo effect to simulate speaker feedback"""
    echo_samples = int(delay * 16000)  # 16kHz sample rate
    echoed = audio.copy()
    echoed[echo_samples:] += audio[:-echo_samples] * decay
    return np.clip(echoed, -1.0, 1.0)

# Apply to TTS samples
for wav_file in tts_echo_samples:
    audio, sr = sf.read(wav_file)
    echoed = add_echo(audio)
    sf.write(f"{wav_file}_echoed.wav", echoed, sr)
```

### Technique 2: Spectral Features

**Train model to recognize TTS spectral characteristics**:
- TTS audio has different spectral properties than human speech
- Include spectral analysis in training features
- Precise handles this automatically, but more TTS data helps

### Technique 3: Temporal Patterns

**TTS has different timing than natural speech**:
- TTS is more uniform in timing
- Natural speech has more variation
- Model learns these patterns from data

---

## Training on Jetson vs Desktop/Server

### Training on Jetson NX/Orin NX

**Advantages:**
- ✅ All training data already on device
- ✅ Can test immediately after training
- ✅ No file transfer needed
- ✅ Native ARM64 support

**Disadvantages:**
- ⚠️ Slower training (30-60 minutes for 50 epochs)
- ⚠️ CPU-intensive (may slow down other processes)
- ⚠️ Limited system resources during training

**Best For:**
- Initial testing and iteration
- Small datasets (<500 samples per category)
- When convenience > speed

### Training on Desktop/Server

**Advantages:**
- ✅ Much faster (5-15 minutes for 50 epochs)
- ✅ More system resources available
- ✅ Can train multiple models in parallel

**Disadvantages:**
- ⚠️ Need to transfer training data
- ⚠️ Need to transfer trained model back to Jetson
- ⚠️ May need to convert audio formats

**Best For:**
- Production/final model training
- Large datasets (>500 samples per category)
- When speed > convenience

### Transferring Model from Desktop to Jetson

```bash
# On desktop/server (after training):
scp hey-aura.pb aura@jetson-ip:~/precise-models/

# On Jetson:
# Model is now ready to use
```

---

## Troubleshooting

### Problem: TensorFlow Import Error on Jetson

**Solution**:
```bash
# Install Jetson-optimized TensorFlow
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow

# Or use CPU-only TensorFlow (slower but works)
pip install tensorflow-cpu
```

### Problem: Training Too Slow on Jetson

**Solutions**:
- Reduce epochs (start with 10-20, increase if needed)
- Reduce training data size (use subset for testing)
- Train on desktop/server, transfer model to Jetson
- Close other applications during training

### Problem: Out of Memory During Training

**Solutions**:
- Reduce training data size
- Use smaller batch size (if configurable)
- Close other applications
- Consider training on desktop/server with more RAM

### Problem: High False Positive Rate on TTS

**Solution**:
- Add more TTS/echo samples to negative training data
- Ensure TTS samples include TTS saying "Hey Aura"
- Increase training epochs
- Check audio quality (no clipping, proper levels)

### Problem: Low True Positive Rate

**Solution**:
- Add more "Hey Aura" samples (more speakers, more variations)
- Check audio quality (clear recordings)
- Reduce threshold in `precise_wake_word.py` (temporarily for testing)
- Ensure samples are properly formatted (16kHz, mono)

### Problem: Model Too Sensitive

**Solution**:
- Add more negative samples (especially noise)
- Increase threshold in code
- Reduce training epochs (may be overfitting)

### Problem: Model Not Sensitive Enough

**Solution**:
- Add more positive samples
- Check microphone levels (may be too quiet)
- Reduce threshold in code
- Increase training epochs

---

## Training Script Template

```bash
#!/bin/bash
# train_hey_aura.sh - Automated training script

set -e

TRAINING_DIR=~/precise-training
MODEL_NAME=hey-aura

echo "=========================================="
echo "  Training 'Hey Aura' Wake Word Model"
echo "=========================================="

# Check data exists
if [ ! -d "$TRAINING_DIR/wake-word" ]; then
    echo "❌ Error: Training data not found at $TRAINING_DIR"
    echo "   Please collect training data first (see TRAIN_HEY_AURA_WAKE_WORD.md)"
    exit 1
fi

# Prepare combined negative data
echo "📦 Preparing negative training data..."
mkdir -p "$TRAINING_DIR/combined-negative"
rm -f "$TRAINING_DIR/combined-negative"/*.wav
cp "$TRAINING_DIR/not-wake-word"/*.wav "$TRAINING_DIR/combined-negative/" 2>/dev/null || true
cp "$TRAINING_DIR/tts-echo"/*.wav "$TRAINING_DIR/combined-negative/" 2>/dev/null || true
cp "$TRAINING_DIR/noise"/*.wav "$TRAINING_DIR/combined-negative/" 2>/dev/null || true

echo "✅ Negative data prepared: $(ls "$TRAINING_DIR/combined-negative"/*.wav 2>/dev/null | wc -l) samples"

# Train model
echo "🎓 Training model (this may take a while)..."
cd "$TRAINING_DIR"
precise-train -e 50 "${MODEL_NAME}.net" \
    wake-word \
    combined-negative

# Convert to .pb format
echo "🔄 Converting to .pb format..."
precise-convert "${MODEL_NAME}.net" "${MODEL_NAME}.pb"

# Test model
echo "🧪 Testing model..."
if [ -d "$TRAINING_DIR/test" ]; then
    precise-test "${MODEL_NAME}.net" \
        test/wake-word \
        test/combined-negative
else
    echo "⚠️  Test directory not found, skipping validation"
fi

# Install model
echo "📦 Installing model..."
mkdir -p ~/precise-models
cp "${MODEL_NAME}.pb" ~/precise-models/
chmod 644 ~/precise-models/"${MODEL_NAME}.pb"

echo ""
echo "✅ Training complete!"
echo "   Model: ~/precise-models/${MODEL_NAME}.pb"
echo ""
echo "Next steps:"
echo "   1. Configure model path in Settings → AI Model Settings"
echo "   2. Restart Aura"
echo "   3. Test wake word detection"
```

---

## Resources

- **Mycroft Precise Documentation**: https://github.com/MycroftAI/mycroft-precise
- **Precise Training Guide**: https://mycroft-ai.gitbook.io/docs/mycroft-technologies/precise/precise-training
- **Precise Data Repository**: https://github.com/MycroftAI/precise-data

---

## Summary Checklist

- [ ] Install training tools (`precise`, `precise-runner`)
- [ ] Collect 200-500 "Hey Aura" samples
- [ ] Collect 200-500 "not wake word" samples
- [ ] **Collect 100-200 TTS/echo samples (CRITICAL)**
- [ ] Collect 50-100 noise samples
- [ ] Split into training/test sets (80/20)
- [ ] Train initial model (10 epochs)
- [ ] Train with echo rejection (20-50 epochs)
- [ ] Test on validation set
- [ ] Test in real-world scenarios
- [ ] Deploy model to `~/precise-models/hey-aura.pb`
- [ ] Configure Aura to use custom model
- [ ] Verify TTS rejection works (no false positives)

---

**Key Success Factor**: The model's ability to reject TTS/echo depends heavily on having sufficient TTS/echo samples in the negative training data, especially samples where TTS says "Hey Aura" but should NOT trigger detection.

