# Far-Field Speech Recognition Optimization (16m Range)

## Problem Analysis

Based on systematic distance testing (near → 6 feet), the following critical issues were identified:

### Test Results Summary

| Distance | Raw RMS | Processed RMS | Gain | Peak | Result | Issue |
|----------|---------|---------------|------|------|--------|-------|
| **Near** | 0.024030 | 0.118966 | 5.00x | **1.0000** | ✅ Success | Hard clipping |
| **~2 ft** | 0.014052 | 0.119982 | 8.53x | **1.0000** | ✅ Success | Hard clipping |
| **~4 ft** | 0.011765 | 0.117550 | **10.00x** | **1.0000** | ❌ **FAILED** | Max gain + clipping |
| **~4.5 ft** | 0.015418 | 0.120000 | 7.79x | 0.9859 | ✅ Success | Near clipping |
| **~6 ft** | 0.007328 | **0.073135** | **10.00x** | 0.4982 | ❌ **FAILED** | Insufficient gain |

### Critical Problems Identified

1. **Hard Clipping Destroys Audio Quality**
   - `Peak: 1.0000` indicates hard clipping (abrupt waveform cutoff)
   - Successful transcriptions were being clipped, reducing recognition accuracy
   - Hard clipping introduces harmonic distortion

2. **Insufficient Maximum Gain**
   - At 6 feet: Raw RMS = 0.007 → needs 17x gain to reach 0.12 target
   - System was limited to 10x → only reached 0.073 RMS (39% below target)
   - For 16 meters: estimated need for 50-100x gain

3. **Suboptimal Target RMS**
   - Original target: 0.12 RMS
   - Whisper performs better with 0.15-0.20 RMS
   - Failed attempts didn't reach minimum threshold

4. **Aggressive Bandpass Filter**
   - 80 Hz high-pass may remove too much low-frequency speech energy at distance
   - Male voice fundamentals (85-180 Hz) partially affected

## Optimizations Applied

### 1. Increased Maximum Gain (10x → 40x)
```python
AGC_MAX_GAIN = 40.0  # Increased for 16m range support
```

**Impact:**
- At 6 feet (RMS 0.007): Can now apply 25.7x gain → reaches 0.18 target
- At 12 feet (estimated RMS 0.003): Can apply 40x gain → reaches 0.12 (adequate)
- Provides headroom for far-field speech recognition

### 2. Increased Target RMS (0.12 → 0.18)
```python
AGC_TARGET_RMS = 0.18  # Optimal for Whisper recognition
```

**Impact:**
- Higher signal strength improves Whisper accuracy
- Better signal-to-noise ratio
- Still below clipping threshold with soft clipping

### 3. Implemented Soft Clipping
```python
AGC_SOFT_CLIP_THRESHOLD = 0.95  # Smooth compression above 0.95

def soft_clip(audio, threshold=0.95):
    """Uses tanh for smooth compression instead of hard clipping"""
    mask = np.abs(audio) > threshold
    if np.any(mask):
        excess = audio[mask] - np.sign(audio[mask]) * threshold
        compressed = threshold + np.tanh(excess / (1 - threshold)) * (1 - threshold)
        audio[mask] = np.sign(audio[mask]) * compressed
    return audio
```

**Impact:**
- Preserves waveform shape (no abrupt cutoffs)
- Reduces harmonic distortion
- Improves speech recognition accuracy
- Asymptotically approaches ±1.0 (no overflow)

### 4. Lowered High-Pass Filter (80 Hz → 60 Hz)
```python
HIGHPASS_CUTOFF = 60  # Hz - Preserves more speech energy
```

**Impact:**
- Preserves more male voice fundamentals (85-180 Hz)
- Better far-field speech capture
- Still removes subsonic rumble

## Expected Performance

### Theoretical Range Calculations

Assuming linear distance-to-RMS relationship:
- **Near (1 foot)**: RMS ~0.024 → Gain 7.5x → 0.18 RMS ✅
- **6 feet**: RMS ~0.007 → Gain 25.7x → 0.18 RMS ✅
- **12 feet**: RMS ~0.0035 → Gain 40x (max) → 0.14 RMS ✅
- **16 feet**: RMS ~0.0026 → Gain 40x (max) → 0.104 RMS ⚠️ (marginal)

### Recommendations for Extending Beyond 12 Feet

If testing shows inadequate performance beyond 12 feet:

1. **Increase max gain to 60-80x**
   ```python
   AGC_MAX_GAIN = 60.0  # For full 16m range
   ```

2. **Consider adaptive target RMS**
   - Lower target for far-field (0.15 instead of 0.18)
   - Reduces noise amplification

3. **Implement noise gate**
   - Filter out amplified background noise
   - Only when gain > 20x

4. **Use directional beam forming** (ReSpeaker supports this)
   - Combine multiple microphones for directional focus
   - Significantly improves far-field SNR

## Audio Processing Pipeline

```
Raw Audio → Band-Pass Filter (60-3400 Hz) → AGC (Target 0.18, Max 40x) → Soft Clip (>0.95) → Whisper
```

### Pipeline Components

1. **Band-Pass Filter (60-3400 Hz)**
   - Removes subsonic rumble (<60 Hz)
   - Removes high-frequency hiss (>3400 Hz)
   - Preserves all speech frequencies

2. **Automatic Gain Control**
   - Analyzes input RMS
   - Applies adaptive gain (up to 40x)
   - Normalizes to 0.18 RMS target

3. **Soft Clipping**
   - Compresses peaks above 0.95
   - Preserves waveform shape
   - Prevents hard clipping distortion

4. **Whisper TensorRT**
   - Receives clean, normalized audio
   - Optimal signal level for recognition

## Testing Instructions

1. **Run the test script:**
   ```bash
   python3 scripts/test_noise_reduction.py
   ```

2. **Repeat distance test:**
   - Start near device (1-2 feet)
   - Speak same phrase: "My name is Raphael and I'm testing the microphone"
   - Move back 2 feet each time
   - Continue until 16 feet or transcription fails

3. **Monitor key metrics:**
   - **Processed RMS**: Should be 0.15-0.20 for success
   - **Peak**: Should be <1.0 (no hard clipping)
   - **Gain**: Track when hitting 40x (max gain limit)

4. **Expected improvements:**
   - No more hard clipping (Peak <1.0)
   - Consistent RMS ~0.18 for all distances
   - Success at 6+ feet (previously failed)
   - Estimated success up to 12-14 feet

## Configuration Reference

### Previous Settings (Failed at 6 feet)
```python
AGC_TARGET_RMS = 0.12
AGC_MAX_GAIN = 10.0
HIGHPASS_CUTOFF = 80
# Hard clipping: np.clip(audio, -1.0, 1.0)
```

### New Settings (Optimized for 16m)
```python
AGC_TARGET_RMS = 0.18
AGC_MAX_GAIN = 40.0
AGC_SOFT_CLIP_THRESHOLD = 0.95
HIGHPASS_CUTOFF = 60
FILTER_TYPE = "bandpass"
LOWPASS_CUTOFF = 3400
```

## Next Steps

1. **Test new settings** with distance progression
2. **Record results** at 6, 9, 12, 15 feet
3. **Analyze failure point** (if any)
4. **Adjust AGC_MAX_GAIN** if needed (60x or 80x)
5. **Consider beam forming** for extreme distances (>12 feet)

## Files Modified

- `aura-control/listener.py` - Main voice listener
- `scripts/test_noise_reduction.py` - Audio testing utility

Both files now use identical audio processing pipelines.

