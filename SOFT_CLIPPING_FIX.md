# Soft Clipping Fix for Far-Field Speech Recognition

## Problem Identified

Based on your 8-foot distance test, transcription quality dropped significantly:

### Test Results (8 feet)
- **Raw RMS**: 0.007282
- **Processed RMS**: 0.178756
- **Gain**: 24.76x
- **Peak**: **1.0000** ⚠️ (HARD CLIPPING!)
- **Transcription**: ❌ "The Name of Cross Isle..." (completely wrong)
- **Expected**: "My name is Raphael and I'm testing the microphone."

### Root Cause
1. **Hard Clipping Distortion**: Even though we added soft clipping, peaks were still hitting exactly 1.0, destroying the waveform
2. **High Gain Amplification**: 24.76x gain amplifies both speech AND noise
3. **Waveform Destruction**: Hard clipping introduces harmonic distortion that confuses Whisper

## Solution Implemented

### 1. Two-Stage Soft Clipping

```python
def soft_clip(audio, threshold=0.85, max_peak=0.98):
    """
    Stage 1: Gradual compression above 0.85 (preserves waveform)
    Stage 2: Hard limit at 0.98 (prevents peak distortion)
    """
    # Stage 1: Soft compression using tanh
    mask = np.abs(audio) > 0.85
    if np.any(mask):
        excess = audio[mask] - np.sign(audio[mask]) * 0.85
        compressed = 0.85 + np.tanh(excess / 0.13) * 0.13
        audio[mask] = np.sign(audio[mask]) * compressed
    
    # Stage 2: Safety hard limit at 0.98
    audio = np.clip(audio, -0.98, 0.98)
    
    return audio
```

**How it works:**
- **0.0 to 0.85**: No modification (linear passthrough)
- **0.85 to 0.98**: Gradual compression using tanh (smooth curve)
- **Above 0.98**: Hard limit (safety ceiling, rarely triggered)

### 2. Improved AGC

```python
AGC_TARGET_RMS = 0.18        # Target signal level
AGC_MAX_GAIN = 40.0          # Allow high gain for far-field
HIGHPASS_CUTOFF = 60         # Preserve more speech energy
```

## Expected Improvements

### Before (Old Settings)
| Metric | Near | 8 ft |
|--------|------|------|
| Peak | 1.0000 🚨 | 1.0000 🚨 |
| Distortion | High | Severe |
| Result | ✅ Works | ❌ Failed |

### After (New Settings)
| Metric | Near | 8 ft |
|--------|------|------|
| Peak | ≤0.98 ✅ | ≤0.98 ✅ |
| Distortion | Minimal | Reduced |
| Result | ✅ Better | ✅ Should work |

## Key Changes Summary

1. **Soft Clipping Threshold**: 0.95 → **0.85**
   - Earlier compression starts = smoother waveform preservation

2. **Peak Ceiling**: 1.0 → **0.98**
   - Prevents hard clipping distortion entirely

3. **Two-Stage Approach**:
   - Gradual compression (0.85-0.98) preserves harmonics
   - Hard limit (0.98) prevents overflow

4. **High-Pass Filter**: 80 Hz → **60 Hz**
   - Preserves more low-frequency speech energy

## Testing Instructions

Repeat your 8-foot test with the same phrase:
```
"My name is Raphael and I'm testing the microphone."
```

### Expected Output (8 feet)
```
[Audio] 📊 RAW: RMS=0.007282, Peak=0.0638
[Audio] ✅ PROCESSED: RMS=0.178756, Peak=0.98 ← Should be ≤0.98 now!
[Audio] 📈 AMPLIFICATION: 0.007282 → 0.178756 (×24.55)
[Whisper] 📝 Transcribed: 'My name is Raphael and I'm testing the microphone.'
```

**Key indicator of success**: `Peak=0.98` or lower (not 1.0000!)

## Why This Works

### The Clipping Problem
Hard clipping at 1.0 creates a "flat top" waveform:
```
Original:     /\        /\
                \/        \/
                
Hard Clip:   /==\      /==\   ← Distorted!
                \/        \/
                
Soft Clip:   /‾\      /‾\     ← Preserved!
                \/        \/
```

### Harmonic Distortion
- **Hard clipping** at 1.0 introduces high-frequency harmonics that don't exist in speech
- **Soft clipping** at 0.85→0.98 gradually compresses peaks, preserving the natural harmonic structure
- Whisper's acoustic model recognizes natural speech patterns, not distorted ones

### Signal-to-Distortion Ratio
- At 24x gain, any clipping distortion is **already amplified 24x**
- By compressing gradually starting at 0.85, we minimize distortion before it gets amplified
- The 0.98 ceiling ensures peaks never reach the hard limit of 1.0

## Remaining Challenges

Even with soft clipping, far-field speech (8+ feet) faces these challenges:

1. **Signal-to-Noise Ratio (SNR)**
   - High gain (24x+) amplifies background noise
   - Room acoustics introduce reverb/echo
   
2. **Distance Attenuation**
   - Sound pressure drops with distance squared
   - Speech energy decreases significantly beyond 6 feet

3. **Potential Solutions** (if still having issues):
   - **Beam forming**: Use all 6 ReSpeaker mics for directional focus
   - **Noise gating**: Filter out amplified background noise
   - **Spectral subtraction**: Remove constant background noise spectrum
   - **Lower target RMS for far-field**: Accept quieter audio to reduce noise amplification

## Files Modified

- `aura-control/listener.py`
- `scripts/test_noise_reduction.py`

Both files now use identical two-stage soft clipping (0.85 threshold, 0.98 ceiling).

## Next Steps

1. **Test at 8 feet** - should now transcribe correctly with Peak ≤0.98
2. **Test at 12 feet** - push the limit
3. **If still having issues beyond 8 feet**, consider:
   - Implementing beam forming (directional microphone array)
   - Adding noise gate for high-gain scenarios
   - Using adaptive target RMS based on detected noise floor

## Technical Details

### Tanh Compression Function
```
y = threshold + tanh(x / range) * range

Where:
- threshold = 0.85 (where compression starts)
- range = max_peak - threshold = 0.13
- tanh asymptotically approaches ±1.0
```

### Compression Curve
```
Input  → Output
0.0    → 0.0  (no change)
0.5    → 0.5  (no change)
0.85   → 0.85 (threshold)
0.90   → 0.892 (slight compression)
0.95   → 0.930 (moderate compression)
1.0    → 0.956 (strong compression)
>1.0   → 0.98  (hard limited)
```

This ensures smooth, gradual compression that preserves speech characteristics while preventing distortion.

