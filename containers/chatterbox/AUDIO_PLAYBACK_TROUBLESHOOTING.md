# Audio Playback Troubleshooting

## Problem: `aplay` plays but no sound

When you run `aplay test_output_basic_synthesis.wav`, it completes without error but produces no sound. However, `main.py` with TTS works fine.

## Root Cause

**Device Selection Issue**: When you run `aplay` without specifying a device (`-D`), it uses the default ALSA device, which might not be your actual output device.

`speaker.py` works because it:
1. **Detects the correct device** (UACDemoV1.0 or USB Audio)
2. **Uses the specific device** with `-D plughw:CARD_INDEX,0`

## Solution

### Quick Test

Run the test script to find and use the correct device:

```bash
cd chatterbox-container
./test_audio_playback.sh
```

### Manual Testing

1. **Find your audio device:**
   ```bash
   aplay -l
   ```
   Look for "UACDemoV1.0" or "USB Audio" and note the card number.

2. **Test with the specific device:**
   ```bash
   # Replace 1 with your card number
   aplay -D plughw:1,0 test_output_basic_synthesis.wav
   ```

3. **Check what device speaker.py detected:**
   ```bash
   # When main.py starts, look for this line:
   # [Speaker] 🔍 Auto-detected output device: UACDemoV1.0 (card 1)
   ```

### Why `plughw:` instead of `hw:`?

The `plug` plugin automatically converts:
- **Sample rate** (24000 Hz → device's native rate)
- **Channels** (mono → stereo if needed)
- **Format** (if device requires different format)

This ensures compatibility. Without `plug`, you might need exact format matching.

## Common Issues

### 1. Volume is Muted

```bash
# Check volume
amixer -c 1 sget PCM

# Unmute and set volume
amixer -c 1 sset PCM unmute
amixer -c 1 sset PCM 80%
```

### 2. Wrong Device Selected

```bash
# List all devices
aplay -l

# Try each device manually
aplay -D plughw:0,0 test.wav
aplay -D plughw:1,0 test.wav
aplay -D plughw:2,0 test.wav
```

### 3. Device is Busy

```bash
# Check if device is in use
lsof | grep snd
fuser /dev/snd/*

# Kill processes using audio
# (Be careful - this might stop other audio)
```

### 4. Permissions Issue

```bash
# Check if you're in audio group
groups | grep audio

# If not, add yourself:
sudo usermod -a -G audio $USER
# Then logout and login again
```

### 5. PulseAudio Interference

The error you saw:
```
Expression 'PaAlsaStream_SetUpBuffers(...)' failed in 'pa_linux_alsa.c'
```

This is **harmless** - it's from a library (probably PyAudio or similar) trying to use PulseAudio, but `speaker.py` uses ALSA directly and doesn't need PulseAudio.

If PulseAudio is causing issues:
```bash
# Stop PulseAudio (if needed)
pulseaudio --kill

# Or configure it to not interfere with ALSA
# Edit /etc/pulse/default.pa and comment out:
# load-module module-alsa-sink
```

## Testing Different Formats

Your WAV files are:
- **Format**: Signed 16 bit Little Endian
- **Sample rate**: 24000 Hz (Chatterbox) or 8000 Hz (sample.wav)
- **Channels**: Mono

Test with different sample rates:

```bash
# Test with 24000 Hz (Chatterbox default)
aplay -D plughw:1,0 -r 24000 test_output_basic_synthesis.wav

# Test with 22050 Hz (speaker.py default)
aplay -D plughw:1,0 -r 22050 test_output_basic_synthesis.wav

# Test with 44100 Hz (common device rate)
aplay -D plughw:1,0 -r 44100 test_output_basic_synthesis.wav
```

The `plug` plugin should handle conversion automatically.

## Why main.py Works But Direct aplay Doesn't

| Method | Device Selection | Result |
|--------|-----------------|--------|
| `aplay file.wav` | Uses default ALSA device | ❌ No sound (wrong device) |
| `aplay -D plughw:1,0 file.wav` | Uses detected device | ✅ Should work |
| `main.py` (speaker.py) | Auto-detects and uses correct device | ✅ Works |

## Quick Fix Script

Create a helper script for easy testing:

```bash
#!/bin/bash
# play_wav.sh - Play WAV file using same device as speaker.py

# Auto-detect device (same as speaker.py)
CARD=$(aplay -l 2>/dev/null | grep -E "UACDemoV1.0|USB Audio" | head -1 | sed -n 's/.*card \([0-9]*\):.*/\1/p')

if [ -z "$CARD" ]; then
    echo "⚠️  No USB audio device found, using default"
    aplay "$1"
else
    echo "🎵 Playing on card $CARD (same as speaker.py)"
    aplay -D "plughw:$CARD,0" "$1"
fi
```

Usage:
```bash
chmod +x play_wav.sh
./play_wav.sh test_output_basic_synthesis.wav
```

## Summary

1. **Use the detected device**: `aplay -D plughw:CARD,0 file.wav`
2. **Check volume**: `amixer -c CARD sget PCM`
3. **Check device**: `aplay -l` to see all devices
4. **PulseAudio error is harmless** - speaker.py uses ALSA directly

The key is using the **same device selection** that `speaker.py` uses!
