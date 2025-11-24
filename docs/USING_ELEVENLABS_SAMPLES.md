# Generating Voice Sample for ChatterboxTTS

This guide explains how to generate a high-quality voice sample from ElevenLabs for ChatterboxTTS voice cloning.

## Quick Answer

**Generate a new, longer sample (10-20 seconds) from ElevenLabs for optimal voice cloning!**

ChatterboxTTS uses zero-shot voice cloning, which means:
- ✅ One high-quality sample is sufficient
- ✅ **Longer samples (10-20 seconds) work MUCH better than short ones**
- ✅ The system automatically caches the voice embedding
- ✅ No model training required

## Recommended Approach: Generate New Sample

**Best approach:** Generate a new, longer sample (10-20 seconds) from ElevenLabs

```bash
python setup/scripts/generate_chatterbox_voice_sample.py
```

**Why this is best:**
- ✅ Longer samples (10-20s) clone much better than short ones
- ✅ Single continuous sample is more natural
- ✅ Guaranteed quality and format
- ✅ Takes ~30 seconds to generate
- ✅ Uses your existing ElevenLabs configuration

## Step-by-Step Guide

### Generate Voice Sample

**Run the generation script:**

```bash
python setup/scripts/generate_chatterbox_voice_sample.py
```

**What it does:**
1. Uses your ElevenLabs API key and voice ID from `.env`
2. Generates a 10-20 second high-quality sample
3. Formats it correctly for ChatterboxTTS (44.1kHz, mono)
4. Normalizes audio levels
5. Saves to `assets/voice_samples/sample.wav`
6. Ready to use immediately

**Requirements:**
- ElevenLabs API key configured in `.env` (ELEVENLABS_API_KEY)
- Voice ID configured in `.env` (ELEVENLABS_VOICE_ID) or uses default
- Internet connection (to generate from ElevenLabs)

### Alternative: Use Existing Sample

If you already have a high-quality WAV sample (10-20 seconds):

1. **Copy to correct location:**
   ```bash
   cp /path/to/your/sample.wav assets/voice_samples/sample.wav
   ```

2. **Or set custom path in `.env`:**
   ```bash
   CHATTERBOX_VOICE_SAMPLE=/path/to/your/sample.wav
   ```

**Sample requirements:**
- Duration: 10-20 seconds (ideal), minimum 5 seconds
- Format: WAV (uncompressed or PCM)
- Sample rate: 16kHz, 22.05kHz, 44.1kHz, or 48kHz
- Channels: Mono or Stereo (will be converted automatically)
- Quality: Clear, natural speech with minimal background noise

## What Happens Next

Once the sample is in place:

1. **First TTS request:**
   - Voice sample is processed once
   - Voice embedding is extracted (~50-100ms overhead)
   - Embedding is cached to `data/voice_cache/`

2. **Subsequent requests:**
   - Cached embedding is loaded instantly
   - No processing overhead
   - Same latency as default voice (~100-150ms)

3. **Automatic caching:**
   - Cache is created automatically
   - No manual steps required
   - Cache persists across restarts

## Sample Quality Guidelines

### What Makes a Good Sample?

✅ **Optimal:**
- 10-20 seconds of clear speech
- Natural conversational tone
- Varied intonation and emotion
- High audio quality (44.1kHz or 48kHz)
- Minimal background noise
- Good audio levels (normalized)

❌ **Avoid:**
- Samples shorter than 5 seconds
- Distorted or clipped audio
- Heavy background noise or music
- Unnatural speech patterns
- Very quiet or very loud audio

### Why Longer Samples Work Better

- **More voice data:** Longer samples capture more nuances
- **Natural flow:** Continuous speech is more natural than fragments
- **Better embedding:** More data = better voice characteristic extraction
- **Consistent quality:** Single sample ensures consistent voice

## File Format Requirements

Your WAV file should be:
- **Format:** WAV (uncompressed or PCM)
- **Sample rate:** 16kHz, 22.05kHz, 44.1kHz, or 48kHz
- **Channels:** Mono or Stereo (will be converted automatically)
- **Bit depth:** 16-bit or 24-bit
- **Duration:** Minimum 5 seconds, ideal 10-20 seconds

## Troubleshooting

### Script Can't Connect to ElevenLabs

```bash
# Check your API key is set
grep ELEVENLABS_API_KEY .env

# If not set, run:
./aura_config.sh
# Choose option 5 to configure TTS
```

### Sample Too Short

If the generated sample is less than 10 seconds:
- The script will still work (minimum 5 seconds)
- For better quality, edit the script to use longer text
- Or manually generate a longer sample from ElevenLabs

### Poor Voice Quality

If the cloned voice doesn't sound good:
- Ensure sample has clear, natural speech
- Check audio isn't distorted or noisy
- Try generating a new sample with different text
- Verify sample is 10-20 seconds long

### Cache Not Working

If voice embedding caching isn't working:
- Check `data/voice_cache/` directory exists
- Verify file permissions
- Check logs for error messages
- The system will fall back to real-time cloning if caching fails

## Summary

**Recommended workflow:**

1. ✅ **Generate a new longer sample** (10-20 seconds)
   ```bash
   python setup/scripts/generate_chatterbox_voice_sample.py
   ```

2. ✅ **Enable ChatterboxTTS** in Settings → TTS Engine

3. ✅ **Enable Voice Cloning** toggle (when ChatterboxTTS is enabled)

4. ✅ **Done!** Voice will be cloned and cached automatically

**Benefits:**
- ✅ Longer samples (10-20s) clone **much better** than short ones
- ✅ Single continuous sample is more natural
- ✅ Guaranteed quality and format
- ✅ Takes ~30 seconds to generate
- ✅ Automatic caching eliminates latency overhead

**You don't need to:**
- ❌ Manually process samples
- ❌ Create a model or train anything
- ❌ Manually cache the embedding
- ❌ Worry about file formats (script handles it)

The system handles everything automatically! 🎉
