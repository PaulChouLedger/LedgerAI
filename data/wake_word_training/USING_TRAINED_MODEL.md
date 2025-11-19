# Using Your Trained Wake Word Model Files

After training, you'll have `.onnx` and possibly `.tflite` files. Here's what to do with them.

## Quick Start

**Use `.onnx` files** (ignore `.tflite` unless you only have that)

1. **Install the model:**
   ```bash
   cd ~/LedgerAI/data/wake_word_training
   ./install_trained_model.sh ~/Downloads/hey_aura_v0.1.onnx
   ```

2. **Test it:**
   ```bash
   cd ~/LedgerAI/aura-control/core
   python3 listener.py
   ```

3. **Say "hey aura"** - it should trigger! 🎉

---

## Detailed Instructions

## Which Format to Use

**✅ Use `.onnx` files (Recommended)**
- OpenWakeWord prefers ONNX format
- Better ARM64/Jetson support
- Faster inference
- This is what the system expects

**ℹ️ `.tflite` files**
- TensorFlow Lite format
- Can be used but requires conversion
- Not directly supported by OpenWakeWord
- You can ignore these if you have `.onnx`

## Step 1: Locate Your Trained Model Files

After training in Colab, you should have:
- `hey_aura_v0.1.onnx` (or similar name)
- Possibly `hey_aura_v0.1.tflite` (can ignore if you have .onnx)

**Download location:** Usually in `/content/` in Colab, or your Downloads folder after downloading.

## Step 2: Install the Model

### Option A: Use the Installation Script (Recommended)

```bash
cd ~/LedgerAI/data/wake_word_training

# If you have the .onnx file:
./install_trained_model.sh ~/Downloads/hey_aura_v0.1.onnx

# Or let it auto-detect from Downloads or current directory:
./install_trained_model.sh
```

The script will:
- ✅ Copy the model to the correct location
- ✅ Update the configuration automatically
- ✅ Verify the installation

### Option B: Manual Installation

```bash
# Create the wake words directory if it doesn't exist
mkdir -p ~/LedgerAI/data/models/wake_words

# Copy the .onnx file (adjust path as needed)
# If downloaded to your local machine first:
scp ~/Downloads/hey_aura_v0.1.onnx user@jetson:~/LedgerAI/data/models/wake_words/

# Or if you have direct access:
cp ~/Downloads/hey_aura_v0.1.onnx ~/LedgerAI/data/models/wake_words/
```

### Verify the file is in place:

```bash
ls -lh ~/LedgerAI/data/models/wake_words/
# Should show: hey_aura_v0.1.onnx
```

## Step 3: Configure the System to Use Your Model

### Option A: Update the Code (Recommended)

Edit `aura-control/core/openwakeword_wake_word.py`:

```python
# Change line 41 from:
DEFAULT_MODEL = "hey_mycroft_v0.1"

# To:
DEFAULT_MODEL = "hey_aura_v0.1"  # Your custom model (without .onnx extension)
```

**Important:** 
- Use the model name **without** the `.onnx` extension
- The file should be named `hey_aura_v0.1.onnx` in the directory
- The code will automatically look for `hey_aura_v0.1.onnx` when you specify `"hey_aura_v0.1"`

### Option B: Use Settings Dialog

1. Open Aura
2. Go to **Settings** → **🧠 AI Model Settings**
3. The wake word settings should detect your model automatically
4. Or configure it via `state.py`:

```python
# In aura-control/core/state.py or via settings
set_wake_word_model("hey_aura_v0.1")
```

## Step 4: Test Your Model

1. **Start the listener:**
   ```bash
   cd ~/LedgerAI/aura-control/core
   python3 listener.py
   ```

2. **Test wake word detection:**
   - Say "hey aura" - should trigger ✅
   - Play TTS saying "hey aura" - should NOT trigger ❌ (if trained with TTS samples)
   - Say similar phrases - should NOT trigger ❌

3. **Check the logs:**
   - Look for: `[OpenWakeWord] ✅ Custom model loaded: .../hey_aura_v0.1.onnx`
   - Watch for detection messages when you say "hey aura"

## Step 5: Adjust Threshold (If Needed)

If the model is:
- **Too sensitive** (triggers on false positives):
  - Increase threshold: Settings → AI Model Settings → Increase wake word threshold
  - Or edit `state.py`: `set_wake_word_sensitivity(0.7)` (lower sensitivity = higher threshold)

- **Not sensitive enough** (misses detections):
  - Decrease threshold: Settings → AI Model Settings → Decrease wake word threshold
  - Or edit `state.py`: `set_wake_word_sensitivity(0.9)` (higher sensitivity = lower threshold)

## File Structure

```
~/LedgerAI/
├── data/
│   └── models/
│       └── wake_words/
│           ├── hey_aura_v0.1.onnx  ← Your trained model (USE THIS)
│           └── hey_aura_v0.1.tflite ← Can ignore (not used)
└── aura-control/
    └── core/
        └── openwakeword_wake_word.py  ← Configure DEFAULT_MODEL here
```

## How Model Loading Works

The system checks for models in this order:

1. **Custom directory** (`~/LedgerAI/data/models/wake_words/`) - **CHECKED FIRST**
   - Looks for: `{model_name}.onnx` (e.g., `hey_aura_v0.1.onnx`)
   - This is where your trained models should go

2. **Built-in models** (openWakeWord's default location)
   - Only used if custom model not found
   - Default models like "hey_mycroft_v0.1" are stored in openWakeWord's cache

**So yes, the system knows to look in `/LedgerAI/data/models/wake_words/` for your custom models!**

The default model "hey_mycroft_v0.1" is a built-in model that comes with openWakeWord, so it's not in your custom directory. Your trained "hey_aura_v0.1" model should be placed in the custom directory.

## Troubleshooting

### Model not found

**Error:** `Failed to load model 'hey_aura_v0.1'`

**Solution:**
1. Check file exists: `ls -lh ~/LedgerAI/data/models/wake_words/hey_aura_v0.1.onnx`
2. Check file permissions: `chmod 644 ~/LedgerAI/data/models/wake_words/hey_aura_v0.1.onnx`
3. Verify the model name matches in `openwakeword_wake_word.py`

### Model loads but doesn't detect

**Possible causes:**
- Threshold too high → Lower it
- Model not trained well → Retrain with more/better samples
- Audio format mismatch → Ensure microphone is 16kHz

### Model triggers on everything

**Solution:**
- Increase threshold
- Retrain with more negative samples (especially TTS echo samples)

## What About .tflite Files?

**Short answer:** You can ignore `.tflite` files if you have `.onnx` files.

**Long answer:**
- `.tflite` is TensorFlow Lite format
- OpenWakeWord uses `.onnx` format
- If you only have `.tflite`, you'd need to convert it (complex, not recommended)
- **Recommendation:** Use the `.onnx` file from training

## Quick Reference

**Model file location:**
```
~/LedgerAI/data/models/wake_words/hey_aura_v0.1.onnx
```

**Configuration:**
```python
# In openwakeword_wake_word.py
DEFAULT_MODEL = "hey_aura_v0.1"
```

**Test command:**
```bash
cd ~/LedgerAI/aura-control/core && python3 listener.py
```

**Check if model is loaded:**
Look for this in logs:
```
[OpenWakeWord] 📁 Found custom model: .../hey_aura_v0.1.onnx
[OpenWakeWord] ✅ Custom model loaded: .../hey_aura_v0.1.onnx
```

## Next Steps

1. ✅ Copy `.onnx` file to `data/models/wake_words/`
2. ✅ Update `DEFAULT_MODEL` in `openwakeword_wake_word.py`
3. ✅ Test with `listener.py`
4. ✅ Adjust threshold if needed
5. ✅ Enjoy your custom "hey aura" wake word! 🎉

