# Wyoming OpenWakeWord Container Integration

## Overview

The Wyoming OpenWakeWord container provides a more reliable wake word detection solution for Jetson devices. It runs in a separate container with optimized audio processing and uses the standardized Wyoming protocol.

## Benefits

- **Isolated Process**: Runs in its own container, preventing conflicts
- **Jetson Optimized**: Pre-built container optimized for Jetson hardware
- **Standardized Protocol**: Uses Wyoming protocol for reliable communication
- **Better Audio Handling**: Container handles audio stream management internally
- **More Reliable**: Less prone to PortAudio errors and stream issues

## Setup

### 1. Start the Wyoming Container

The container is already configured in `setup/docker-compose.yml`. Start it with:

```bash
cd ~/LedgerAI/setup
docker compose up -d wyoming-openwakeword
```

### 2. No Client Installation Needed!

The container includes everything needed. The client uses only Python standard library - no `pip install` required!

### 3. Ready to Use!

The `listener.py` is already configured to use the Wyoming container. Just start the container and restart Aura!

The Wyoming client is available in `aura-control/core/wyoming_wake_word.py`. To use it instead of the direct OpenWakeWord integration:

1. In `listener.py`, import the Wyoming client:
   ```python
   from wyoming_wake_word import create_wyoming_wake_word_detector
   ```

2. Replace the wake word detector initialization:
   ```python
   # Instead of:
   # wake_word_detector = create_wake_word_detector(...)
   
   # Use:
   wake_word_detector = create_wyoming_wake_word_detector()
   ```

3. The Wyoming client has the same interface (`process()` method), so the rest of the code should work without changes.

## Verification

### Check Container Status

```bash
docker compose ps wyoming-openwakeword
```

### Check Container Logs

```bash
docker compose logs -f wyoming-openwakeword
```

### Test Connection

The Wyoming client will automatically connect to `localhost:10400` when initialized. If connection fails, check:

1. Container is running: `docker compose ps`
2. Port is accessible: `netstat -an | grep 10400`
3. Container logs for errors: `docker compose logs wyoming-openwakeword`

## Configuration

### Container Image

The container uses the Jetson-optimized image from `dustynv/jetson-containers`:
- Image: `dustynv/wyoming-openwakeword:r36.2.0`
- Port: `10400` (TCP)
- Protocol: Wyoming over TCP/IP

### Wake Word Models

The container includes pre-trained OpenWakeWord models:
- `hey_jarvis` (default)
- `hey_mycroft`
- `hey_fire_fox`
- And more...

To use a different wake word, modify the container command in `docker-compose.yml` or configure it via Wyoming protocol messages.

## Troubleshooting

### Container Won't Start

1. Check Docker is running: `docker ps`
2. Check image exists: `docker images | grep wyoming`
3. Pull image if missing: `docker pull dustynv/wyoming-openwakeword:r36.2.0`

### Connection Refused

1. Ensure container is running: `docker compose ps`
2. Check port is not blocked: `sudo netstat -tulpn | grep 10400`
3. Verify network mode: Container uses `network_mode: host` so it should be accessible on `localhost:10400`

### Low Detection Accuracy

1. Check audio levels in container logs
2. Verify microphone is working: `arecord -d 5 test.wav && aplay test.wav`
3. Adjust microphone gain: `alsamixer` (press F4 for capture)

## Comparison: Direct vs Container

| Feature | Direct OpenWakeWord | Wyoming Container |
|---------|-------------------|-------------------|
| Setup Complexity | Simple (pip install) | Medium (container + client) |
| Reliability | Can have PortAudio issues | More stable (isolated process) |
| Jetson Optimization | Manual tuning needed | Pre-optimized |
| Audio Handling | Manual stream management | Handled by container |
| Debugging | Python logs | Container logs + Python logs |

## References

- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Jetson Containers - Wyoming OpenWakeWord](https://github.com/dusty-nv/jetson-containers/tree/master/packages/smart-home/wyoming/wyoming-openwakeword)
- [OpenWakeWord](https://github.com/dscripka/openWakeWord)

