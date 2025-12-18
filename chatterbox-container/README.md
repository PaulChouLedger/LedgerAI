# Chatterbox-TTS Container

Docker container for Chatterbox-TTS that installs from source to avoid dependency conflicts.

## Features

- ✅ **Installs from source** - Clones from `github.com/resemble-ai/chatterbox` and installs with `pip install -e .`
- ✅ **Avoids dependency conflicts** - Fresh installation in isolated container
- ✅ **GPU support** - Uses CUDA base image for GPU acceleration
- ✅ **Voice cloning** - Supports voice cloning from audio samples
- ✅ **REST API** - Simple Flask API for TTS synthesis

## Installation

### Build the container:

**Recommended: Use the build script** (uses pre-built PyTorch image, avoids source compilation):

```bash
cd chatterbox-container
./build.sh
```

**Or build manually:**

```bash
cd chatterbox-container
docker build --network=host --shm-size=8g -t chatterbox-tts:latest .
```

> **⚠️ Important for Jetson users:** If you're using jetson-containers and getting PyTorch build failures, see [BUILD_WITH_JETSON_CONTAINERS.md](BUILD_WITH_JETSON_CONTAINERS.md) for solutions. The Dockerfile already uses a pre-built PyTorch image, so you should build directly with Docker instead of using jetson-containers' PyTorch compilation.

### Run the container:

```bash
docker run -d \
  --name chatterbox-tts \
  --runtime=nvidia \
  --network=host \
  -v /path/to/voice/samples:/app/voice_samples \
  -v /path/to/voice/cache:/app/voice_cache \
  chatterbox-tts
```

Or add to `docker-compose.yml`:

```yaml
chatterbox-tts:
  build: ../chatterbox-container
  network_mode: host
  runtime: nvidia
  volumes:
    - ../assets/voice_samples:/app/voice_samples
    - ../data/voice_cache:/app/voice_cache
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Synthesize Text
```bash
POST /synthesize
Content-Type: application/json

{
  "text": "Hello, this is a test",
  "voice_sample": "sample.wav",  # Optional
  "exaggeration": 0.6  # Optional, default 0.6
}
```

Returns: WAV audio file

### Extract Voice Embedding
```bash
POST /voice/embedding
Content-Type: application/json

{
  "voice_sample_path": "/app/voice_samples/sample.wav"
}
```

## Testing the Container

### Quick Test Script

Run the included test script:

```bash
# Start the container first
cd setup
docker compose up -d chatterbox-tts

# Run the test script
cd ../chatterbox-container
python3 test_container.py
```

Or test manually:

```bash
# Health check
curl http://localhost:11437/health

# Synthesize text
curl -X POST http://localhost:11437/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test"}' \
  --output output.wav

# Play the audio (if available)
aplay output.wav
```

### Usage from Python

```python
import requests

# Synthesize text
response = requests.post(
    'http://localhost:11437/synthesize',
    json={
        'text': 'Hello, this is a test',
        'voice_sample': 'sample.wav',  # Optional
        'exaggeration': 0.6
    }
)

# Save audio
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### Testing Voice Cloning

If you have a voice sample:

```bash
# Extract voice embedding
curl -X POST http://localhost:11437/voice/embedding \
  -H "Content-Type: application/json" \
  -d '{"voice_sample_path": "/app/voice_samples/sample.wav"}'

# Synthesize with voice cloning
curl -X POST http://localhost:11437/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello with voice cloning", "voice_sample": "sample.wav"}' \
  --output output_cloned.wav
```

## Benefits of Source Installation

1. **No dependency conflicts** - Fresh install avoids version conflicts
2. **Latest features** - Get latest code from GitHub
3. **Customizable** - Can modify source if needed
4. **Isolated** - Doesn't affect host system dependencies

## Troubleshooting

### Build fails with jetson-containers (PyTorch compilation error)

If you see errors like:
```
ninja: build stopped: subcommand failed.
The command '/bin/sh -c /tmp/pytorch/install.sh || /tmp/pytorch/build.sh' returned a non-zero code: 1
```

**Solution:** Don't use jetson-containers to build PyTorch from source. Use the build script instead:
```bash
cd chatterbox-container
./build.sh
```

See [BUILD_WITH_JETSON_CONTAINERS.md](BUILD_WITH_JETSON_CONTAINERS.md) for detailed troubleshooting.

### Build fails (general)
- Ensure you have NVIDIA Docker runtime installed
- Check CUDA version compatibility (12.8)
- Ensure sufficient disk space (~10-15GB free)
- Try skipping model download: `SKIP_MODEL_DOWNLOAD=1 ./build.sh`

### Import errors
- Verify Chatterbox cloned correctly: `docker exec chatterbox-tts ls /tmp/chatterbox`
- Check installation: `docker exec chatterbox-tts python3 -c "from chatterbox import ChatterboxTTS"`

### GPU not detected
- Ensure `--runtime=nvidia` is set
- Check: `docker exec chatterbox-tts python3 -c "import torch; print(torch.cuda.is_available())"`

