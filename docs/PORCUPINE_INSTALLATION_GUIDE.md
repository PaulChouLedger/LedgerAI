# Porcupine Wake Word Installation Guide

## Overview

Porcupine wake word detection has been integrated into Aura. This guide covers installation and configuration.

---

## Installation

### Standard Installation (x86_64)

```bash
cd /path/to/LedgerAI/aura-control
pip install pvporcupine
```

### Jetson ARM64 Installation

For Jetson devices, you may need to build from source:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip build-essential

# Clone Porcupine repository
git clone https://github.com/Picovoice/porcupine.git
cd porcupine/binding/python

# Build and install
python3 setup.py build_ext --inplace
# Note: You may see a deprecation warning about license classifiers - this is harmless, ignore it
pip3 install .
```

**Alternative:** Try installing pre-built wheel if available:
```bash
pip install pvporcupine --no-cache-dir
```

---

## Configuration

### 1. Enable Wake Word Detection

**Wake word detection is controlled via the Settings dialog:**

1. Open Aura GUI
2. Click **Settings** button
3. Navigate to **AI Model Settings**
4. Toggle **Wake Word Detection** ON/OFF

The setting is saved automatically and persists across restarts.

### Configuration Options

| Setting | Default | Description | Location |
|---------|---------|-------------|----------|
| Wake Word Enabled | `false` | Enable/disable wake word detection | Settings → AI Model Settings |
| Sensitivity | `0.5` | Detection sensitivity (0.0-1.0) | Stored in `app_settings.json` |
| Model Path | `None` | Path to custom .ppn model file | Stored in `app_settings.json` |

**Note:** Settings are stored in `~/LedgerAI/data/app_settings.json` (not in `.env` file).

### Sensitivity Tuning

- **Lower (0.0-0.3)**: More strict, fewer false positives, may miss wake word
- **Medium (0.4-0.6)**: Balanced (recommended)
- **Higher (0.7-1.0)**: More sensitive, more false positives, catches wake word more easily

**To adjust sensitivity:** Edit `~/LedgerAI/data/app_settings.json` and set `wake_word_sensitivity` value (0.0-1.0).

---

## Getting a Wake Word Model

### Option 1: Use Built-in Keywords (if available)

Porcupine includes some built-in keywords. Check available keywords:

```python
import pvporcupine
print(pvporcupine.KEYWORDS)
```

If "hey aura" is not available, you'll need a custom model.

### Option 2: Train Custom Model (Recommended)

1. **Visit Picovoice Console**: https://console.picovoice.ai/
2. **Create Account** (free for personal projects)
3. **Train Custom Wake Word**:
   - Record ~100 samples of "hey aura" (or your preferred phrase)
   - Upload audio files
   - Download `.ppn` model file
4. **Save Model**:
   ```bash
   mkdir -p ~/LedgerAI/data/wake_word
   cp ~/Downloads/hey-aura_en_linux_v3_0_0.ppn ~/LedgerAI/data/wake_word/
   ```
5. **Configure**:
   ```bash
   # In .env
   WAKE_WORD_MODEL_PATH=/home/user/LedgerAI/data/wake_word/hey-aura_en_linux_v3_0_0.ppn
   ```

### Option 3: Use Closest Built-in Keyword

If you can't train a custom model, try using a similar built-in keyword:
- "hey siri" (if available)
- "hey google" (if available)
- "computer" (if available)

Then adjust your usage to match the keyword.

---

## Testing

### 1. Verify Installation

```bash
python3 -c "import pvporcupine; print('Porcupine installed:', pvporcupine.__version__)"
```

### 2. Test Wake Word Detection

```bash
cd /path/to/LedgerAI/aura-control/core
python3 -c "from wake_word import create_wake_word_detector; detector = create_wake_word_detector(); print('Wake word detector:', 'OK' if detector else 'Failed')"
```

### 3. Run Aura with Wake Word

```bash
cd /path/to/LedgerAI/aura-control/core
python3 main.py
```

**Expected output:**
```
[Wake Word] ✅ Porcupine initialized with built-in keyword: 'hey aura'
[Wake Word]   Frame length: 512 samples
[Wake Word]   Sample rate: 16000 Hz
[Wake Word]   Sensitivity: 0.5

[Audio] WAKE WORD PIPELINE
[Audio]   Hardware DSP → Wake Word → VAD → Whisper
```

### 4. Test Wake Word

1. Say "Hey Aura" (or your configured wake word)
2. You should see:
   ```
   [Wake Word] ✅ Wake word detected! (confidence: 1.00)
   [Wake Word] 🎤 Listening for speech...
   ```
3. Then speak your command
4. After processing, it returns to waiting for wake word:
   ```
   [Wake Word] 🔄 Waiting for wake word...
   ```

---

## Troubleshooting

### Issue: "Porcupine not available"

**Solution:**
```bash
pip install pvporcupine
```

### Issue: "No built-in 'hey aura' keyword found"

**Solution:**
1. Train custom model at https://console.picovoice.ai/
2. Download `.ppn` file
3. Save model path in `~/LedgerAI/data/app_settings.json`:
   ```json
   {
     "wake_word_model_path": "/path/to/your/hey-aura_en_linux_v3_0_0.ppn"
   }
   ```

### Issue: "ImportError on Jetson" or "Unsupported CPU: '0xd42'"

**Problem:** Porcupine doesn't recognize Jetson CPU architectures by default.

**Solution:** Patch Porcupine to support Jetson CPUs:

```bash
# Run the patch script
cd ~/LedgerAI/setup/scripts
python3 patch_porcupine_jetson.py
```

**Alternative manual patch:**

If the script doesn't work, manually edit the Porcupine util file:

1. Find Porcupine installation:
   ```bash
   python3 -c "import pvporcupine; import os; print(os.path.dirname(pvporcupine.__file__))"
   ```

2. Edit `_util.py` in that directory:
   ```bash
   nano /path/to/pvporcupine/_util.py
   ```

3. Find the `_pv_linux_machine` function and add Jetson support before the `raise NotImplementedError` line:
   ```python
   # Jetson CPU support
   jetson_cpus = ['0xd42', '0xd49', '0xd0b', '0xd07', '0xd08']  # Jetson Orin, Orin NX, TX1, TX2, Nano
   if cpu_part in jetson_cpus:
       return 'arm64'  # Jetson uses ARM64 architecture
   ```

**Note:** During build, you may see deprecation warnings about license classifiers - these are harmless and can be ignored.

### Issue: Build fails with compilation errors

**Possible solutions:**

1. **Install build dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-dev python3-pip build-essential cmake
   ```

2. **Try with pip install instead:**
   ```bash
   cd porcupine/binding/python
   pip3 install . --no-cache-dir
   ```

3. **Check Python version compatibility:**
   ```bash
   python3 --version  # Should be 3.8+
   ```

4. **If still failing, try pre-built wheel:**
   ```bash
   pip3 install pvporcupine --no-cache-dir
   ```

### Issue: Wake word not detecting

**Possible causes:**
1. **Sensitivity too low**: Edit `~/LedgerAI/data/app_settings.json` and increase `wake_word_sensitivity` (try 0.7)
2. **Wrong wake word**: Check if you're using the correct phrase
3. **Audio quality**: Ensure microphone is working and audio levels are good
4. **Model mismatch**: Verify model matches your language/accent
5. **Wake word disabled**: Check Settings → AI Model Settings → Wake Word Detection is ON

### Issue: Too many false positives

**Solution:**
- Edit `~/LedgerAI/data/app_settings.json` and decrease `wake_word_sensitivity` (try 0.3-0.4)
- Ensure quiet environment
- Consider training custom model with your voice

### Issue: High CPU usage

**Expected:** Porcupine uses ~5-10% CPU on Jetson Orin
**If higher:**
- Check if multiple instances are running
- Verify Porcupine version (should be >= 3.0.0)

---

## Architecture

### Pipeline Flow

```
Microphone (ReSpeaker)
    ↓
Hardware DSP (Beamforming)
    ↓
Porcupine Wake Word Detection ⭐ NEW
    ↓ (only if wake word detected)
Silero VAD
    ↓
Advanced Multi-Feature Filter
    ↓
Whisper STT
    ↓
LLM Processing
    ↓
ElevenLabs TTS
```

### Code Structure

- **`wake_word.py`**: Porcupine wrapper and initialization
- **`listener.py`**: Main loop with two-stage detection:
  1. Stage 1: Wake word detection (always running if enabled)
  2. Stage 2: VAD + speech processing (only after wake word)

---

## Performance

### Resource Usage (Jetson Orin)

| Metric | Value |
|--------|-------|
| CPU (idle) | ~5-10% |
| CPU (active) | ~35-45% (wake word + VAD + STT) |
| Memory | ~50-100MB additional |
| Latency | +100-200ms (wake word detection) |

### Comparison

| Mode | CPU (idle) | Privacy | False Positives |
|------|------------|--------|-----------------|
| VAD Only | ~5% | Lower | Higher |
| Wake Word + VAD | ~10% | Higher | Lower |

---

## Disabling Wake Word

To disable wake word detection and return to VAD-only mode:

```bash
# In .env
ENABLE_WAKE_WORD=false
```

Or simply don't set the variable (defaults to `false`).

---

## Next Steps

1. ✅ Install Porcupine
2. ✅ Configure `.env` with `ENABLE_WAKE_WORD=true`
3. ✅ Train/download wake word model
4. ✅ Test wake word detection
5. ✅ Tune sensitivity based on your environment
6. ✅ Enjoy privacy-enhanced voice assistant!

---

## References

- **Porcupine Documentation**: https://github.com/Picovoice/porcupine
- **Picovoice Console**: https://console.picovoice.ai/
- **Aura Wake Word Guide**: `docs/WAKE_WORD_DETECTION_GUIDE.md`
- **OVOS Analysis**: `docs/OVOS_INTEGRATION_ANALYSIS.md`

---

**Questions?** Check the codebase or open an issue!

