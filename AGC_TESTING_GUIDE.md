# AGC A/B Testing Guide

## Overview

The listener now has configurable flags to easily test **Hardware AGC** vs **Software AGC** to determine which gives better transcription results.

## Configuration Flags (in `listener.py`)

```python
# === AGC Testing Configuration ===
# Enable/disable hardware AGC (in ReSpeaker DSP chip)
USE_HARDWARE_AGC = True
HARDWARE_AGC_TARGET = 0.08  # Target RMS level (0.01-0.99)
HARDWARE_AGC_MAX_GAIN = 30.0  # Maximum gain in dB

# Enable/disable software AGC (in Python after audio capture)
USE_SOFTWARE_AGC = False
SOFTWARE_AGC_TARGET = 0.1  # Target RMS level for normalization
```

## Test Scenarios

### Test 1: Hardware AGC Only (Default) ✅ RECOMMENDED
```python
USE_HARDWARE_AGC = True
USE_SOFTWARE_AGC = False
```

**How it works:**
- Hardware high-pass filter removes low-frequency noise
- Hardware AGC amplifies clean signal in DSP chip
- Runs in real-time with zero CPU overhead
- AGC happens BEFORE audio reaches computer

**Expected output:**
```
[Hardware] ✅ HPF + AGC enabled (target=0.08, max_gain=30.0dB)
[Audio] ✅ Using HARDWARE AGC
```

### Test 2: Software AGC Only
```python
USE_HARDWARE_AGC = False
USE_SOFTWARE_AGC = True
```

**How it works:**
- Hardware high-pass filter removes low-frequency noise
- NO hardware AGC (raw audio levels)
- Software AGC normalizes audio in Python before transcription
- Simple gain adjustment + soft clipping

**Expected output:**
```
[Hardware] ✅ HPF enabled, AGC disabled
[Audio] ✅ Using SOFTWARE AGC
[Software AGC] 🎚️  RMS: 0.0234 → 0.1000 (gain=4.27x)
```

### Test 3: No AGC (Baseline)
```python
USE_HARDWARE_AGC = False
USE_SOFTWARE_AGC = False
```

**How it works:**
- Only hardware high-pass filter enabled
- Raw audio levels (no amplification)
- Use this to establish baseline transcription accuracy

**Expected output:**
```
[Hardware] ✅ HPF enabled, AGC disabled
[Audio] ⚠️  NO AGC ENABLED - audio may be too quiet
```

### Test 4: Both AGC (Not Recommended)
```python
USE_HARDWARE_AGC = True
USE_SOFTWARE_AGC = True
```

**How it works:**
- Hardware AGC amplifies in DSP
- Software AGC amplifies again in Python
- Double amplification - likely over-amplification

**Expected output:**
```
[Audio] ⚠️  BOTH AGC ENABLED - NOT RECOMMENDED
```

## Testing Procedure

### 1. Set Configuration
Edit `aura-control/listener.py` lines 24-30 with desired test scenario.

### 2. Restart Listener
```bash
# Stop current listener
pkill -f listener.py

# Start fresh
cd /Users/rcabello/Documents/GitHub/LedgerAI/aura-control
python3 listener.py
```

### 3. Test Phrases
Speak these test phrases at normal volume:

**Medical terms:**
- "What is pancreatitis?"
- "Explain myocardial infarction"
- "Tell me about diabetes"

**Quiet speech:**
- Speak from 6 feet away
- Speak very softly

**Normal speech:**
- "Hello, how are you today?"
- "What's the weather like?"

### 4. Evaluate Results

**Metrics to track:**

| Metric | What to Check |
|--------|---------------|
| **Transcription Accuracy** | Does Whisper correctly transcribe words? |
| **Medical Term Accuracy** | Are medical terms transcribed correctly? |
| **Quiet Speech** | Does it work at low volumes? |
| **Background Noise** | Does fan noise trigger false positives? |
| **Audio Levels** | Check RMS values in logs |
| **Speech Detection** | Does VAD reliably detect speech start/end? |

**Look for in logs:**
```
[Audio] RMS=0.037128, Peak=0.4594  # Audio levels
[Whisper] 📝 Transcribed: 'text'   # Accuracy
[Software AGC] RMS: 0.02 → 0.10    # Gain applied (if software AGC)
```

## Tuning Parameters

### Hardware AGC Tuning
If Hardware AGC is too quiet or too loud:

```python
# Too quiet → increase target
HARDWARE_AGC_TARGET = 0.15  # Higher = louder (default: 0.08)

# Quiet speech cut off → increase max gain
HARDWARE_AGC_MAX_GAIN = 40.0  # Higher = more boost (default: 30.0dB)
```

### Software AGC Tuning
If Software AGC is distorting or too quiet:

```python
# Adjust target level
SOFTWARE_AGC_TARGET = 0.15  # Higher = louder (default: 0.1)

# Max gain is hardcoded to 10x (20dB) - edit apply_software_agc() to change
```

## Comparison Table

| Feature | Hardware AGC | Software AGC |
|---------|--------------|--------------|
| **Processing Location** | ReSpeaker DSP chip | Python (post-capture) |
| **CPU Overhead** | Zero | Minimal |
| **Adaptive** | Yes (real-time) | No (simple gain) |
| **Latency** | None | Minimal |
| **Quality** | Professional DSP | Simple normalization |
| **When Applied** | Before audio capture | After audio capture |
| **Noise Handling** | Better (pre-amplification) | Worse (post-capture) |

## Expected Results

### Hardware AGC (Recommended)
- ✅ Consistent audio levels
- ✅ Good transcription accuracy
- ✅ Handles quiet speech well
- ✅ Minimal background noise amplification
- ✅ Zero CPU overhead

### Software AGC
- ✅ Consistent audio levels
- ⚠️  May amplify background noise
- ⚠️  Simple algorithm (not adaptive)
- ✅ Easy to tune parameters
- ⚠️  Post-capture (can't recover lost information)

### No AGC
- ❌ Inconsistent audio levels
- ❌ Quiet speech may fail transcription
- ✅ No amplification of background noise
- ✅ True baseline for testing

## Recommended Settings

After testing, the recommended configuration is:

```python
USE_HARDWARE_AGC = True
HARDWARE_AGC_TARGET = 0.08
HARDWARE_AGC_MAX_GAIN = 30.0
USE_SOFTWARE_AGC = False
```

**Why:**
- Hardware AGC processes audio optimally (filter → amplify)
- Professional DSP algorithm with real-time adaptation
- Zero CPU overhead
- No post-processing needed

## Troubleshooting

### Transcription Still Poor
1. **Check audio levels:**
   - Look for `[Audio] RMS=` in logs
   - Should be > 0.02 after AGC
   - Peak should be < 1.0 (no clipping)

2. **Test without AGC:**
   - Set both AGC flags to False
   - Check if raw audio RMS is too low (<0.01)
   - If yes, issue is audio input, not AGC

3. **Check Whisper model:**
   - Current: distil-small.en (fast, lower accuracy)
   - Consider: distil-large-v3 (better accuracy)

4. **Verify hardware configuration:**
   - Look for `[Hardware] ✅` in logs
   - If failed, ReSpeaker tuning library may not be working

### Software AGC Not Working
Check logs for:
```
[Software AGC] 🎚️  RMS: X.XXXX → Y.YYYY (gain=Z.ZZx)
```

If not appearing:
- Verify `USE_SOFTWARE_AGC = True`
- Check that speech is being detected (VAD working)

### Hardware AGC Not Working
Check logs for:
```
[Hardware] ✅ HPF + AGC enabled (target=0.08, max_gain=30.0dB)
```

If shows "AGC disabled":
- Verify `USE_HARDWARE_AGC = True`
- Check ReSpeaker is connected
- Verify tuning library is installed

## Data Collection

Create a test log to compare:

```
Test 1: Hardware AGC
- Phrase: "What is pancreatitis?"
- Transcribed: "What is pancreatitis?" ✅
- RMS: 0.037
- Accuracy: 100%

Test 2: Software AGC
- Phrase: "What is pancreatitis?"
- Transcribed: "What is Bankercitis?" ❌
- RMS: 0.095 (after AGC)
- Accuracy: 60%

Test 3: No AGC
- Phrase: "What is pancreatitis?"
- Transcribed: (empty - too quiet)
- RMS: 0.009
- Accuracy: 0%
```

## Next Steps

After determining which AGC works best:
1. Update default configuration in listener.py
2. Document the optimal settings
3. Consider adding automatic AGC selection based on audio levels
4. Test with various speaker distances and volumes

