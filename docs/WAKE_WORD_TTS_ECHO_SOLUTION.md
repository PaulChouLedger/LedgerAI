# Wake Word TTS Echo Solutions: Cooldown vs Training

## Problem

Wake word detector triggers on TTS echo after playback ends, causing false positives.

## Solution Options

### Option 1: Cooldown Period (Current Implementation)
- **Pros:** Simple, immediate fix, works with any model
- **Cons:** 
  - Introduces latency (2s after TTS ends before wake word works)
  - User can't interrupt TTS or speak immediately after
  - Not ideal UX - feels "unresponsive"
- **Best for:** Quick fix while improving training

### Option 2: Better Training (Recommended Long-term)
- **Pros:** 
  - No latency - model learns to ignore TTS
  - Better UX - works immediately after TTS
  - More robust - handles real-world echo/reverb
- **Cons:**
  - Requires retraining with proper TTS echo samples
  - May need more training data
  - Takes time to collect/retrain
- **Best for:** Production system

### Option 3: Hybrid Approach (Best)
- Use shorter cooldown (0.5-1s) + better training
- Cooldown catches immediate echo, training handles room reverb
- Provides safety net while model improves

### Option 4: Threshold Adjustment
- Increase threshold (less sensitive) to reduce false positives
- **Pros:** Simple, no latency
- **Cons:** May miss legitimate wake words from far away

## Current Situation Analysis

### Is TTS Echo Training Working?

Check your training data:
```bash
# Count TTS echo samples
ls data/wake_word_training/negative_tts/*.wav | wc -l

# Should have 20+ TTS echo samples
```

**If model still triggers on TTS, possible reasons:**

1. **Not enough TTS echo samples**
   - Need 30-50+ TTS echo samples as negative examples
   - Current: ~20 (may not be enough)

2. **TTS samples not representative**
   - Samples should match your actual TTS voice
   - Should include room echo/reverb (play through speakers)
   - Current: Using direct TTS generation (may not have echo)

3. **Training didn't complete properly**
   - Model may not have been trained with TTS samples
   - Check if TTS samples were included in training

4. **Threshold too low**
   - Current threshold: 0.5 (may be too sensitive)
   - Try increasing to 0.6-0.7

5. **Model quality**
   - Custom model may need more training epochs
   - May need better data augmentation

## Recommended Solution: Hybrid Approach

### Step 1: Reduce Cooldown (Immediate)

```python
# In listener.py
WAKE_WORD_POST_TTS_COOLDOWN = 0.5  # Reduce to 0.5s (from 2.0s)
```

This reduces latency while still catching immediate echo.

### Step 2: Improve Training (Long-term)

#### A. Collect More TTS Echo Samples

```bash
# Generate more TTS echo samples (play through speakers)
python3 train_openwakeword_hey_aura.py --mode tts-only --tts-samples 50

# This creates 50 samples with real echo/reverb
```

#### B. Verify Training Data Quality

1. **Check TTS samples have echo:**
   ```bash
   # Play a TTS sample to verify it has echo/reverb
   aplay data/wake_word_training/negative_tts/tts_echo_001.wav
   ```

2. **Ensure TTS samples are in training:**
   - TTS samples should be in `formatted/negative/` folder
   - Should have similar number as other negative samples

#### C. Retrain Model

1. Upload updated training data to Colab
2. Ensure TTS echo samples are in negative class
3. Train with more epochs if needed
4. Test model thoroughly with TTS playback

#### D. Adjust Threshold

After retraining, test different thresholds:

```python
# In openwakeword_wake_word.py or state.py
# Try different thresholds
thresholds_to_test = [0.5, 0.6, 0.7, 0.75]

# For each threshold:
# 1. Test with real "hey aura" (should trigger)
# 2. Test with TTS "hey aura" (should NOT trigger)
# 3. Choose threshold that balances both
```

### Step 3: Test and Verify

**Test Procedure:**
```bash
# 1. Start listener
cd aura-control/core
python3 listener.py

# 2. Test scenarios:
#    a) Say "hey aura" → should trigger ✅
#    b) Play TTS "hey aura" → should NOT trigger ❌
#    c) Say "hey aura" right after TTS → should work (after cooldown) ⚠️
#    d) Say "hey aura" from distance → should still work ✅
```

## Metrics to Track

### False Positive Rate
- How often TTS triggers wake word: **Target: 0%**
- Current: High (triggering on TTS echo)

### False Negative Rate  
- How often real "hey aura" is missed: **Target: <5%**
- Current: Unknown (need to test)

### Latency
- Time from wake word to detection: **Target: <500ms**
- Cooldown adds: 2s currently, 0.5s with reduction

### User Experience
- Can user interrupt TTS? **Ideal: Yes**
- Can user speak immediately after TTS? **Ideal: Yes**
- Current cooldown prevents this

## Implementation Recommendations

### Short Term (Now)
1. ✅ Reduce cooldown to 0.5s (keeps some protection)
2. ⚠️ Increase threshold to 0.6-0.7 (reduces false positives)
3. 📝 Monitor false positive rate

### Medium Term (This Week)
1. Collect 50+ TTS echo samples
2. Verify TTS samples have real echo/reverb
3. Retrain model with updated dataset
4. Test thoroughly

### Long Term (Production)
1. Model should NOT trigger on TTS at all
2. Remove cooldown entirely (or reduce to 0.1s safety margin)
3. Model handles all echo/reverb cases
4. Optimize threshold for best balance

## Code Changes

### Reduce Cooldown
```python
# listener.py
WAKE_WORD_POST_TTS_COOLDOWN = 0.5  # Reduced from 2.0
```

### Adjust Threshold (via Settings or code)
```python
# state.py or openwakeword_wake_word.py
# Higher threshold = less sensitive (fewer false positives)
DEFAULT_THRESHOLD = 0.6  # or 0.7
```

### Make Cooldown Configurable
```python
# Add to state.py settings
WAKE_WORD_COOLDOWN_ENABLED = True
WAKE_WORD_COOLDOWN_DURATION = 0.5  # seconds
```

## Conclusion

**Best approach:** Hybrid solution
- Short cooldown (0.5s) for immediate protection
- Better training with more TTS echo samples (30-50+)
- Proper threshold tuning
- Gradual improvement toward eliminating cooldown

The cooldown is a **band-aid** solution. The real fix is **better training** so the model learns to ignore TTS echo entirely.

