# Chatterbox Voice Cloning Guide

## Ideal Voice Sample Requirements

For **zero-shot voice cloning** (what we use), Chatterbox TTS requires:

### Minimum Requirements
- **Duration:** At least **5 seconds** of clear speech
- **Format:** WAV, MP3, OGG, or M4A (WAV preferred)
- **Sample Rate:** 16kHz, 22.05kHz, or 44.1kHz
- **Quality:** Clean audio with minimal background noise

### Recommended Specifications
- **Duration:** **10-30 seconds** (more is better, but 5s minimum)
- **Format:** **WAV** (uncompressed, best quality)
- **Sample Rate:** **16kHz or 22.05kHz** (matches TTS output)
- **Channels:** **Mono** (single channel)
- **Bit Depth:** 16-bit or 24-bit

### Content Guidelines
- **Clear speech:** Natural, conversational tone
- **No background noise:** Record in quiet environment
- **No music or effects:** Pure voice only
- **Variety:** Include different phonemes and intonations
- **Natural pacing:** Not too fast, not too slow

## Recording the Ideal Sample

### Option 1: Record with Your Microphone

Use the XVF3800 microphone array to record a high-quality sample:

```bash
# Record a 10-15 second sample
cd ~/LedgerAI
python3 -c "
import sounddevice as sd
import soundfile as sf
import numpy as np

# Find XVF3800 device
devices = sd.query_devices()
device_idx = None
for i, d in enumerate(devices):
    if 'XVF3800' in d['name']:
        device_idx = i
        print(f'Found: {d[\"name\"]} (index {i})')
        break

if device_idx is None:
    print('XVF3800 not found')
    exit(1)

# Record 15 seconds at 16kHz, mono
print('Recording 15 seconds... Speak clearly!')
audio = sd.rec(int(15 * 16000), samplerate=16000, channels=1, device=device_idx)
sd.wait()

# Save as WAV
sf.write('assets/voice_samples/voice_clone_sample.wav', audio, 16000)
print('✅ Saved to assets/voice_samples/voice_clone_sample.wav')
"
```

### Option 2: Use Existing High-Quality Recording

If you have a professional recording:

```bash
# Convert to proper format if needed
ffmpeg -i input.wav \
  -ar 16000 \          # Resample to 16kHz
  -ac 1 \              # Convert to mono
  -sample_fmt s16 \    # 16-bit
  assets/voice_samples/voice_clone_sample.wav
```

### Option 3: Extract from Existing Audio

Extract a clean segment from existing audio:

```bash
# Extract 10 seconds starting at 5 seconds
ffmpeg -i assets/voice_samples/sample.wav \
  -ss 5 -t 10 \        # Start at 5s, duration 10s
  -ar 16000 \
  -ac 1 \
  assets/voice_samples/voice_clone_sample.wav
```

## Sample Scripts

### Script 1: Record Voice Clone Sample

Create `setup/scripts/record_voice_clone_sample.py`:

```python
#!/usr/bin/env python3
"""
Record a voice cloning sample for Chatterbox TTS
Records 10-15 seconds of clear speech
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys
import os

# Add workspace root to path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, workspace_root)

SAMPLE_RATE = 16000
DURATION = 15  # seconds
OUTPUT_FILE = os.path.join(workspace_root, "assets", "voice_samples", "voice_clone_sample.wav")

def find_xvf3800():
    """Find XVF3800 device"""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if 'XVF3800' in d['name']:
            return i, d['name']
    return None, None

def main():
    print("=" * 70)
    print("  Chatterbox Voice Cloning Sample Recorder")
    print("=" * 70)
    print()
    
    # Find device
    device_idx, device_name = find_xvf3800()
    if device_idx is None:
        print("❌ XVF3800 microphone not found")
        print("   Available devices:")
        for i, d in enumerate(sd.query_devices()):
            print(f"   {i}: {d['name']}")
        return
    
    print(f"✅ Found microphone: {device_name} (index {device_idx})")
    print()
    print("📝 Instructions:")
    print("   1. Speak clearly and naturally")
    print("   2. Include variety: different words, intonations")
    print("   3. Avoid background noise")
    print("   4. Duration: 10-15 seconds")
    print()
    print("💡 Suggested script:")
    print("   'Hello, my name is [Your Name]. I'm recording this sample")
    print("   for voice cloning. This audio will be used to create")
    print("   a personalized text-to-speech voice that sounds like me.'")
    print()
    
    input("Press ENTER when ready to record...")
    
    print()
    print("🎤 Recording 15 seconds...")
    print("   (Speak now!)")
    print()
    
    try:
        # Record
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device_idx,
            dtype='float32'
        )
        sd.wait()
        
        # Check if we got audio
        if np.max(np.abs(audio)) < 0.01:
            print("⚠️  Audio level very low - check microphone")
            return
        
        # Normalize (but don't clip)
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            print(f"⚠️  Audio may be clipping (max: {max_val:.3f})")
        else:
            # Normalize to 90% to avoid clipping
            audio = audio / max_val * 0.9
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
        
        # Get file info
        file_size = os.path.getsize(OUTPUT_FILE)
        duration = len(audio) / SAMPLE_RATE
        
        print()
        print("=" * 70)
        print("✅ Recording saved!")
        print("=" * 70)
        print(f"   File: {OUTPUT_FILE}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Size: {file_size / 1024:.1f} KB")
        print(f"   Sample Rate: {SAMPLE_RATE} Hz")
        print(f"   Channels: Mono")
        print()
        print("💡 Next steps:")
        print(f"   1. Test: python3 chatterbox-container/test_container.py")
        print(f"   2. Set in .env: CHATTERBOX_VOICE_SAMPLE={OUTPUT_FILE}")
        print(f"   3. Or use default location: assets/voice_samples/voice_clone_sample.wav")
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  Recording cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
```

### Script 2: Validate Voice Sample

Create `setup/scripts/validate_voice_sample.py`:

```python
#!/usr/bin/env python3
"""
Validate a voice cloning sample for Chatterbox TTS
Checks duration, format, quality, etc.
"""
import soundfile as sf
import numpy as np
import sys
import os

def validate_sample(file_path):
    """Validate voice cloning sample"""
    print("=" * 70)
    print("  Voice Sample Validation")
    print("=" * 70)
    print()
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Read audio file
        audio, sample_rate = sf.read(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        duration = len(audio) / sample_rate
        
        # Check channels
        if len(audio.shape) > 1:
            channels = audio.shape[1]
            # Convert to mono if stereo
            if channels > 1:
                audio = np.mean(audio, axis=1)
                print("⚠️  Stereo detected - will use mono conversion")
        else:
            channels = 1
        
        print(f"📄 File: {file_path}")
        print(f"   Size: {file_size / 1024:.1f} KB")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Channels: {channels}")
        print()
        
        # Validate requirements
        checks = []
        
        # Duration check
        if duration >= 5:
            print(f"✅ Duration: {duration:.2f}s (>= 5s minimum)")
            checks.append(True)
        else:
            print(f"❌ Duration: {duration:.2f}s (need >= 5s)")
            checks.append(False)
        
        # Sample rate check
        valid_rates = [16000, 22050, 44100, 48000]
        if sample_rate in valid_rates:
            print(f"✅ Sample Rate: {sample_rate} Hz (supported)")
            checks.append(True)
        else:
            print(f"⚠️  Sample Rate: {sample_rate} Hz (recommended: 16kHz, 22.05kHz, or 44.1kHz)")
            checks.append(True)  # Not critical, will be resampled
        
        # Audio level check
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.01:
            print(f"✅ Audio Level: {max_amplitude:.3f} (has signal)")
            checks.append(True)
        else:
            print(f"❌ Audio Level: {max_amplitude:.3f} (too quiet!)")
            checks.append(False)
        
        if max_amplitude > 0.95:
            print(f"⚠️  Warning: Audio may be clipping (max: {max_amplitude:.3f})")
        
        # Noise check (simple RMS check)
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0.05:
            print(f"✅ RMS Level: {rms:.3f} (good signal level)")
        elif rms > 0.01:
            print(f"⚠️  RMS Level: {rms:.3f} (acceptable but quiet)")
        else:
            print(f"❌ RMS Level: {rms:.3f} (too quiet, may be noise)")
        
        print()
        print("=" * 70)
        if all(checks):
            print("✅ Sample is VALID for voice cloning!")
            print("=" * 70)
            return True
        else:
            print("⚠️  Sample has issues - may not work well")
            print("=" * 70)
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default to sample.wav
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        file_path = os.path.join(workspace_root, "assets", "voice_samples", "sample.wav")
    else:
        file_path = sys.argv[1]
    
    validate_sample(file_path)
```

## Current Samples

Your current samples in `assets/voice_samples/`:
- `sample.wav` - Default sample (check duration/quality)
- `audio1.wav`, `startup.wav`, etc. - Various samples

## Recommended Approach

1. **Check existing samples:**
   ```bash
   python3 setup/scripts/validate_voice_sample.py assets/voice_samples/sample.wav
   ```

2. **Record a new ideal sample:**
   ```bash
   python3 setup/scripts/record_voice_clone_sample.py
   ```

3. **Use the best sample:**
   - Set in `.env`: `CHATTERBOX_VOICE_SAMPLE=assets/voice_samples/voice_clone_sample.wav`
   - Or replace `sample.wav` with your ideal sample

## Best Practices

1. **Record in quiet environment** - Minimize background noise
2. **Use good microphone** - XVF3800 is excellent for this
3. **Speak naturally** - Don't over-enunciate or speak too slowly
4. **Include variety** - Different words, sentences, intonations
5. **10-15 seconds ideal** - More than minimum, but not too long
6. **Test the sample** - Use `test_container.py` to verify it works

## Testing Your Sample

```bash
# Test with the container
cd chatterbox-container
python3 test_container.py

# Or test manually
curl -X POST http://localhost:11437/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of voice cloning",
    "voice_sample": "voice_clone_sample.wav",
    "exaggeration": 0.6
  }' \
  --output test_cloned.wav

aplay test_cloned.wav
```

## Summary

**Ideal Sample:**
- ✅ **10-15 seconds** of clear speech
- ✅ **WAV format**, 16kHz, mono
- ✅ **Clean audio** (no background noise)
- ✅ **Natural speech** (variety of words/intonations)
- ✅ **Good audio levels** (not too quiet, not clipping)

The code says "at least 5 seconds" but **10-15 seconds is ideal** for better voice cloning quality.
