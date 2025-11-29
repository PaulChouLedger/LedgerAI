#!/bin/bash
# Test direct ALSA capture to see if device needs initialization

set +e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Direct ALSA Capture Test${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

# Find ALSA device
ALSA_DEVICE=$(arecord -l 2>/dev/null | grep -i "XVF3800\|reSpeaker" | head -1)
if [ -z "$ALSA_DEVICE" ]; then
    echo -e "${RED}❌ Device not found in ALSA${NC}"
    exit 1
fi

# Extract card number
CARD=$(echo "$ALSA_DEVICE" | sed -n 's/.*card \([0-9]*\):.*/\1/p')
echo -e "${YELLOW}[1]${NC} Found device at ALSA card $CARD"
echo "  $ALSA_DEVICE"
echo ""

# Test 1: Try direct arecord capture
echo -e "${YELLOW}[2]${NC} Testing direct ALSA capture (5 seconds)..."
echo "  Command: arecord -D hw:$CARD,0 -f S16_LE -r 16000 -c 2 -d 5 /tmp/test_capture.wav"
if arecord -D hw:$CARD,0 -f S16_LE -r 16000 -c 2 -d 5 /tmp/test_capture.wav 2>&1; then
    echo -e "  ✅ Capture completed"
    
    # Check file size
    SIZE=$(stat -f%z /tmp/test_capture.wav 2>/dev/null || stat -c%s /tmp/test_capture.wav 2>/dev/null)
    if [ -n "$SIZE" ] && [ "$SIZE" -gt 1000 ]; then
        echo -e "  ✅ File size: $SIZE bytes (looks good)"
        
        # Try to analyze RMS
        echo -e "  ${YELLOW}Analyzing audio levels...${NC}"
        python3 << EOF
import numpy as np
import wave
import sys

try:
    with wave.open('/tmp/test_capture.wav', 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        audio_float = audio.astype(np.float32) / 32768.0
        
        # Calculate RMS for channel 0
        if wf.getnchannels() >= 1:
            channel_0 = audio_float[::wf.getnchannels()]
            rms = np.sqrt(np.mean(channel_0**2))
            peak = np.max(np.abs(channel_0))
            
            print(f"  Channel 0 RMS: {rms:.4f}")
            print(f"  Channel 0 Peak: {peak:.4f}")
            
            if rms > 0.01:
                print("  ✅ Audio levels look good (RMS > 0.01)")
            else:
                print("  ⚠️  Audio levels very low (RMS < 0.01) - may be noise only")
        else:
            print("  ⚠️  Could not analyze audio")
except Exception as e:
    print(f"  ⚠️  Error analyzing audio: {e}")
EOF
    else
        echo -e "  ⚠️  File size very small: $SIZE bytes"
    fi
else
    echo -e "  ${RED}❌ Capture failed${NC}"
fi
echo ""

# Test 2: Check if device is "busy" or locked
echo -e "${YELLOW}[3]${NC} Checking device status..."
if [ -f "/proc/asound/card$CARD/stream0" ]; then
    echo "  Device stream info:"
    cat "/proc/asound/card$CARD/stream0" 2>/dev/null | head -20 | sed 's/^/    /' || echo "    (could not read)"
else
    echo "  ⚠️  Stream info not available"
fi
echo ""

# Test 3: Try Python sounddevice
echo -e "${YELLOW}[4]${NC} Testing Python sounddevice access..."
python3 << EOF
import sounddevice as sd
import numpy as np
import sys

try:
    # Find device
    devices = sd.query_devices()
    device_idx = None
    for i, dev in enumerate(devices):
        if 'XVF3800' in dev['name'] or 'reSpeaker' in dev['name']:
            device_idx = i
            print(f"  ✅ Found device: {dev['name']} (index {i})")
            print(f"    Channels: {dev['channels']}")
            print(f"    Sample rate: {dev['default_samplerate']}")
            break
    
    if device_idx is None:
        print("  ❌ Device not found in sounddevice")
        sys.exit(1)
    
    # Try to open stream and record a short sample
    print("  Attempting to record 1 second sample...")
    duration = 1.0
    sample_rate = 16000
    
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=2,
            device=device_idx,
            dtype='float32'
        )
        sd.wait()
        
        # Analyze
        channel_0 = recording[:, 0]
        rms = np.sqrt(np.mean(channel_0**2))
        peak = np.max(np.abs(channel_0))
        
        print(f"  ✅ Recording completed")
        print(f"    Channel 0 RMS: {rms:.4f}")
        print(f"    Channel 0 Peak: {peak:.4f}")
        
        if rms > 0.01:
            print("  ✅ Audio levels look good!")
        else:
            print("  ⚠️  Audio levels very low - may be noise only")
            
    except Exception as e:
        print(f"  ❌ Recording failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Test Complete${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""

