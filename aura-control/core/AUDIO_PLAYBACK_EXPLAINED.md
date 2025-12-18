# Audio Playback in speaker.py - How It Works

## Overview

`speaker.py` uses **ALSA's `aplay`** for audio playback, **not PulseAudio's `paplay`**. This is why `paplay` results in no sound.

## How speaker.py Plays Audio

### 1. Device Detection
- Uses `aplay -l` to detect audio devices
- Looks for "UACDemoV1.0" or any USB audio device
- Stores device as `OUTPUT_CARD_INDEX`

### 2. Playback Command
The actual playback command is:
```bash
aplay -D plughw:CARD_INDEX,0 -f S16_LE -r 22050 -c 1
```

Where:
- `-D plughw:CARD_INDEX,0` - Uses ALSA's `plug` plugin for automatic format conversion
- `-f S16_LE` - Format: 16-bit signed little-endian PCM
- `-r 22050` - Sample rate: 22050 Hz
- `-c 1` - Channels: Mono (1 channel)

### 3. Why `plug` Plugin?
The `plug` plugin automatically converts:
- Sample rate (22050 Hz → device's native rate)
- Channels (mono → stereo if needed)
- Format (if device requires different format)

This ensures compatibility with any ALSA device.

## Why `paplay` Doesn't Work

`paplay` is a PulseAudio client, but:
1. **speaker.py uses ALSA directly** - bypasses PulseAudio
2. **Device might not be in PulseAudio** - ALSA devices aren't automatically exposed to PulseAudio
3. **Format mismatch** - PulseAudio might expect different format

## Testing Audio Playback

### Test with aplay (what speaker.py uses):

```bash
# 1. List available ALSA devices
aplay -l

# 2. Find your device (look for UACDemoV1.0 or USB Audio)
# Example output:
# card 1: UACDemoV1.0 [UACDemoV1.0], device 0: USB Audio [USB Audio]

# 3. Test playback with the same command speaker.py uses
# Replace CARD_INDEX with your card number (e.g., 1)
echo "test" | aplay -D plughw:1,0 -f S16_LE -r 22050 -c 1

# Or test with a WAV file
aplay -D plughw:1,0 test.wav
```

### Test with paplay (if you want PulseAudio):

```bash
# 1. Check if PulseAudio is running
pactl info

# 2. List PulseAudio sinks
pactl list sinks

# 3. Find your device sink name
# Look for "UACDemoV1.0" in the output

# 4. Test playback
paplay --device=YOUR_SINK_NAME test.wav

# Or use default sink
paplay test.wav
```

## Troubleshooting No Sound

### If `aplay` doesn't work:

1. **Check device detection:**
   ```bash
   aplay -l
   # Look for your device and note the card number
   ```

2. **Test device directly:**
   ```bash
   # Replace 1 with your card number
   aplay -D plughw:1,0 -f S16_LE -r 22050 -c 1 < /dev/zero
   # This should play silence (or noise if device works)
   ```

3. **Check permissions:**
   ```bash
   # Make sure user is in audio group
   groups
   # If not in audio group:
   sudo usermod -a -G audio $USER
   # Then logout/login
   ```

4. **Check device is not busy:**
   ```bash
   # Check if another process is using the device
   lsof | grep snd
   fuser /dev/snd/*
   ```

5. **Test with different format:**
   ```bash
   # Try without plug plugin (direct hardware access)
   aplay -D hw:1,0 -f S16_LE -r 22050 -c 1 test.wav
   
   # Try different sample rate
   aplay -D plughw:1,0 -f S16_LE -r 44100 -c 1 test.wav
   ```

### If `paplay` doesn't work:

1. **Check PulseAudio is running:**
   ```bash
   pulseaudio --check -v
   # If not running:
   pulseaudio --start
   ```

2. **Check device is in PulseAudio:**
   ```bash
   pactl list sinks | grep -A 10 "UACDemo"
   # If not found, device might not be exposed to PulseAudio
   ```

3. **Load ALSA sink (if device not in PulseAudio):**
   ```bash
   # Find your ALSA device
   aplay -l
   # Load it into PulseAudio (replace 1 with your card number)
   pactl load-module module-alsa-sink device=hw:1,0
   ```

4. **Test with correct format:**
   ```bash
   # paplay expects WAV format, not raw PCM
   paplay --format=s16le --rate=22050 --channels=1 test.wav
   ```

## Why speaker.py Uses ALSA Instead of PulseAudio

1. **Lower latency** - Direct ALSA access is faster
2. **More reliable** - ALSA is always available on Linux
3. **Device-specific** - Can target specific hardware directly
4. **Format control** - Full control over audio format

## Making paplay Work (If Needed)

If you want to use `paplay` instead of `aplay`, you would need to:

1. **Modify speaker.py** to use `paplay`:
   ```python
   # Instead of:
   proc = subprocess.Popen(
       ["aplay", "-D", alsa_device, "-f", "S16_LE", "-r", str(PCM_SAMPLE_RATE), "-c", "1"],
       ...
   )
   
   # Use:
   proc = subprocess.Popen(
       ["paplay", "--format=s16le", f"--rate={PCM_SAMPLE_RATE}", "--channels=1"],
       ...
   )
   ```

2. **Ensure PulseAudio is running and device is available**

3. **Handle format conversion** (PulseAudio might need WAV, not raw PCM)

## Recommended: Use aplay (Current Implementation)

The current implementation using `aplay` with the `plug` plugin is the best approach because:
- ✅ Works on all Linux systems
- ✅ Lower latency
- ✅ Automatic format conversion
- ✅ Direct hardware access
- ✅ No dependency on PulseAudio

## Quick Test Script

Create a test script to verify audio:

```bash
#!/bin/bash
# test_audio.sh

echo "=== Testing ALSA (aplay) ==="
aplay -l

echo ""
echo "=== Testing with plug plugin ==="
# Replace 1 with your card number
CARD=1
echo "Testing card $CARD..."
aplay -D plughw:$CARD,0 -f S16_LE -r 22050 -c 1 < /dev/zero &
sleep 1
kill %1 2>/dev/null

echo ""
echo "=== Testing PulseAudio (paplay) ==="
if command -v paplay &> /dev/null; then
    pactl list sinks | grep -A 5 "Name:"
    echo "Try: paplay test.wav"
else
    echo "paplay not found"
fi
```

## Summary

- **speaker.py uses `aplay` (ALSA)**, not `paplay` (PulseAudio)
- **Use `aplay -D plughw:CARD,0`** to test playback
- **`paplay` won't work** unless PulseAudio is configured and device is exposed
- **Current implementation is correct** - no changes needed for ALSA playback
