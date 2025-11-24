# ChatterboxTTS Voice Cloning Guide

This guide explains how to clone a voice for ChatterboxTTS using samples from ElevenLabs or other sources.

## Overview

ChatterboxTTS supports **zero-shot voice cloning** using a reference audio sample. This allows you to:
- Clone voices from ElevenLabs samples
- Use your own voice recordings
- Create custom voices for AuraVision

### How Voice Cloning Works

**Key Points:**
- ✅ **No model training required** - ChatterboxTTS uses zero-shot cloning
- ✅ **Real-time processing** - Voice is cloned on-the-fly from the reference sample
- ⚠️ **Latency trade-off** - Voice cloning adds ~50-100ms latency vs default voice
- ✅ **Sub-200ms total latency** - Even with cloning, ChatterboxTTS maintains low latency

**How it works:**
1. You provide a reference audio sample (5+ seconds)
2. ChatterboxTTS processes the sample in real-time (no pre-training)
3. The voice characteristics are extracted and applied to new text
4. Speech is generated with the cloned voice characteristics

**Latency:**
- Default voice (no cloning): ~100-150ms
- With voice cloning: ~150-250ms (adds ~50-100ms overhead)
- Still faster than ElevenLabs API calls (which require internet)

## Requirements

1. **Audio Sample Requirements:**
   - Format: WAV file (recommended) or MP3
   - Duration: At least 5 seconds of clear speech
   - Quality: High quality, minimal background noise
   - Content: Natural speech (not singing or distorted audio)

2. **ChatterboxTTS Installation:**
   ```bash
   # Install setuptools first (fixes distutils compatibility)
   pip install setuptools
   
   # Then install chatterbox-tts
   pip install chatterbox-tts
   ```
   
   **If installation fails:** See [CHATTERBOX_INSTALLATION_FIX.md](CHATTERBOX_INSTALLATION_FIX.md) for troubleshooting.

## Method 1: Using ElevenLabs Samples

> **Quick start:** See [USING_ELEVENLABS_SAMPLES.md](USING_ELEVENLABS_SAMPLES.md) for the recommended approach using the generation script.

### Step 1: Generate Voice Sample from ElevenLabs

1. **Using ElevenLabs API:**
   ```python
   from elevenlabs.client import ElevenLabs
   from pydub import AudioSegment
   from io import BytesIO
   import os
   
   # Initialize ElevenLabs client
   client = ElevenLabs(api_key=YOUR_API_KEY)
   
   # Generate a sample (at least 5 seconds of text)
   text = "Hello, this is a voice sample for ChatterboxTTS voice cloning. " \
          "This sample should be at least five seconds long to work properly."
   
   # Generate audio
   audio_stream = client.text_to_speech.convert(
       voice_id=YOUR_VOICE_ID,
       text=text,
       output_format="mp3_44100_128"
   )
   
   # Save as WAV
   audio_bytes = b"".join(audio_stream)
   audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
   
   # Save to voice samples directory
   output_path = "assets/voice_samples/elevenlabs_clone.wav"
   os.makedirs(os.path.dirname(output_path), exist_ok=True)
   audio.export(output_path, format="wav")
   print(f"✅ Voice sample saved to: {output_path}")
   ```

2. **Using ElevenLabs Web Interface:**
   - Go to https://elevenlabs.io
   - Select your voice
   - Generate a sample with at least 5 seconds of text
   - Download as WAV or MP3
   - Convert to WAV if needed: `ffmpeg -i input.mp3 output.wav`

### Step 2: Place Sample in Correct Location

Place your voice sample in one of these locations:

1. **Default location (recommended):**
   ```
   assets/voice_samples/sample.wav
   ```

2. **Custom location:**
   - Set `CHATTERBOX_VOICE_SAMPLE` in your `.env` file:
     ```
     CHATTERBOX_VOICE_SAMPLE=/path/to/your/voice_sample.wav
     ```

### Step 3: Verify Voice Cloning

1. **Enable ChatterboxTTS:**
   - Open Settings → TTS Engine → Toggle to "Chatterbox"

2. **Test the voice:**
   - Ask AuraVision a question
   - The response should use the cloned voice from your sample

## Method 2: Using Your Own Voice Recording

### Recording Tips

1. **Environment:**
   - Quiet room with minimal background noise
   - Use a good quality microphone
   - Record at 44.1kHz or 48kHz sample rate

2. **Content:**
   - Speak naturally and clearly
   - Include varied intonation
   - At least 5-10 seconds of speech
   - Avoid background music or effects

3. **Processing:**
   ```bash
   # Convert to WAV if needed
   ffmpeg -i input.mp3 -ar 44100 -ac 1 output.wav
   
   # Normalize audio levels
   ffmpeg -i input.wav -af "loudnorm=I=-16:TP=-1.5:LRA=11" output.wav
   ```

### Place Recording

Save your recording to:
```
assets/voice_samples/my_voice.wav
```

Then set in `.env`:
```
CHATTERBOX_VOICE_SAMPLE=assets/voice_samples/my_voice.wav
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Optional: Path to voice cloning sample
CHATTERBOX_VOICE_SAMPLE=assets/voice_samples/sample.wav
```

### Voice Cloning Toggle

Voice cloning can be enabled/disabled in settings:
- **Enabled (default)**: Uses voice sample for cloning (adds ~50-100ms latency)
- **Disabled**: Uses ChatterboxTTS default voice (lower latency)

**To toggle:**
- Settings → TTS Engine → Voice Cloning toggle (when ChatterboxTTS is enabled)
- Or programmatically via `state.set_chatterbox_voice_cloning_enabled(False)`

### Default Behavior

- If `CHATTERBOX_VOICE_SAMPLE` is not set, ChatterboxTTS checks for:
  - `assets/voice_samples/sample.wav` (default location)
- If no sample is found, ChatterboxTTS uses its default voice
- Voice cloning is automatically enabled when a valid sample is found AND cloning is enabled
- You can disable voice cloning for lower latency even if a sample exists

## Troubleshooting

### Voice Cloning Not Working

1. **Check file path:**
   ```python
   import os
   print(f"Sample exists: {os.path.exists('assets/voice_samples/sample.wav')}")
   ```

2. **Verify audio format:**
   - Must be WAV format
   - Sample rate: 16kHz, 22.05kHz, 44.1kHz, or 48kHz
   - Channels: Mono or Stereo (will be converted automatically)

3. **Check audio duration:**
   - Minimum 5 seconds required
   - Longer samples (10-30 seconds) work better

4. **Check logs:**
   - Look for `[Speaker] 🎭 Using voice cloning from: ...` in logs
   - If you see `⚠️ Voice cloning not available`, the API may not support it

### Improving Voice Quality

1. **Better source audio:**
   - Use high-quality recordings
   - Ensure clear speech without distortion
   - Remove background noise

2. **Longer samples:**
   - 10-30 seconds work better than minimum 5 seconds
   - Include varied speech patterns

3. **Multiple samples:**
   - Currently supports single sample
   - Future versions may support multiple samples

## Advanced Usage

### Programmatic Voice Cloning

You can also clone voices programmatically:

```python
from chatterbox import ChatterboxTTS
import torchaudio as ta

# Initialize model
model = ChatterboxTTS.from_pretrained(
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Generate with voice cloning
text = "Hello, this is a test of voice cloning."
audio_prompt = "assets/voice_samples/sample.wav"

wav = model.generate(
    text,
    audio_prompt_path=audio_prompt,
    exaggeration=0.6  # Emotion intensity: 0.3 (monotone) to 0.7 (expressive)
)

# Save output
ta.save("output.wav", wav.squeeze(0).cpu(), model.sr)
```

### Emotion Control

The `exaggeration` parameter controls emotional intensity:
- `0.3`: Monotone, neutral
- `0.5`: Natural, balanced (default)
- `0.6`: More expressive
- `0.7`: Highly expressive

## Best Practices

1. **Sample Quality:**
   - Use the highest quality source audio available
   - Prefer WAV over MP3 for better quality
   - Normalize audio levels before using

2. **Sample Length:**
   - 10-20 seconds is optimal
   - Too short (<5s): May not clone well
   - Too long (>60s): Unnecessary, may slow processing

3. **Content Selection:**
   - Use natural conversational speech
   - Include varied intonation and emotion
   - Avoid technical jargon or unusual pronunciations

4. **Testing:**
   - Test with various text types
   - Compare with original ElevenLabs voice
   - Adjust if quality doesn't match expectations

## Example: Complete Workflow

```bash
# 1. Generate sample from ElevenLabs
python setup/scripts/generate_cached_prompts.py

# 2. Or download from ElevenLabs web interface
# Save to: assets/voice_samples/elevenlabs_clone.wav

# 3. Configure in .env (optional if using default location)
echo "CHATTERBOX_VOICE_SAMPLE=assets/voice_samples/elevenlabs_clone.wav" >> .env

# 4. Enable ChatterboxTTS in Settings
# Settings → TTS Engine → Toggle to "Chatterbox"

# 5. Test voice cloning
# Ask AuraVision a question and verify the voice matches your sample
```

## Latency Comparison

### TTS Engine Latency Breakdown

| Engine | Mode | Latency | Notes |
|--------|------|---------|-------|
| **ChatterboxTTS** | Default voice | ~100-150ms | Fastest, no internet needed |
| **ChatterboxTTS** | Cached voice embedding | ~100-150ms | **Same as default!** Pre-processed once |
| **ChatterboxTTS** | Real-time voice cloning | ~150-250ms | Adds ~50-100ms for real-time processing |
| **ElevenLabs** | Cloud API | ~200-500ms | Network latency + API processing |

**Speed Comparison:**
- ChatterboxTTS (Cached) is **2-5x faster** than ElevenLabs
- ChatterboxTTS (Real-time) is **1.5-3x faster** than ElevenLabs
- See [TTS_LATENCY_COMPARISON.md](TTS_LATENCY_COMPARISON.md) for detailed analysis

**Key Insight:** When voice embedding caching is available, cloned voices have **the same latency as the default voice** (~100-150ms) because the voice characteristics are pre-processed and cached. Only real-time cloning (when caching isn't available) adds latency.

### Voice Embedding Caching (Reduces Latency!)

**Good News:** The system automatically caches voice embeddings to eliminate real-time processing overhead!

**How it works:**
1. **First use:** Voice sample is processed once and voice embedding is extracted (~50-100ms overhead)
2. **Caching:** Voice embedding is saved to disk (`data/voice_cache/`)
3. **Subsequent uses:** Cached embedding is loaded instantly (no processing overhead)
4. **Result:** Cloned voice has **same latency as default voice** (~100-150ms)

**Cache invalidation:**
- Cache is automatically invalidated if the voice sample file changes
- Cache key is based on file path, modification time, and file size
- Old caches are automatically replaced when voice sample is updated

### Why Real-Time Cloning Adds Latency

When voice embedding caching is **not available** (e.g., API doesn't support it), real-time cloning requires:
1. Loading and processing the reference audio sample each time
2. Extracting voice characteristics from the sample
3. Applying those characteristics during synthesis

This adds approximately **50-100ms** compared to using the default voice or cached embedding.

### When to Use Voice Cloning

**Enable voice cloning when:**
- ✅ You want a specific voice (e.g., cloned from ElevenLabs)
- ✅ Voice quality/identity is more important than absolute lowest latency
- ✅ You're okay with ~150-250ms total latency

**Disable voice cloning when:**
- ✅ You want the lowest possible latency (~100-150ms)
- ✅ Default ChatterboxTTS voice is acceptable
- ✅ You're prioritizing speed over voice customization

### Performance Tips

1. **For lowest latency:** Disable voice cloning, use ChatterboxTTS default voice
2. **For best voice quality:** Enable voice cloning with a high-quality 10-20 second sample
3. **Balance:** Use voice cloning but keep sample file small (<5MB) for faster loading

## Notes

- Voice cloning requires ChatterboxTTS to be enabled (not ElevenLabs)
- Voice cloning can be toggled on/off in Settings (when ChatterboxTTS is enabled)
- The cloned voice is used for all TTS output when enabled
- Voice cloning adds ~50-100ms latency vs default voice
- Quality depends on source audio quality and sample length
- Zero-shot cloning means no model training needed - it's real-time
- Some ChatterboxTTS versions may have different APIs - check the library documentation

## Support

For issues or questions:
1. Check ChatterboxTTS documentation: https://github.com/resemble-ai/chatterbox
2. Verify your audio sample meets requirements
3. Check application logs for error messages
4. Ensure ChatterboxTTS is properly installed: `pip install chatterbox-tts`

