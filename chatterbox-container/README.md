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

```bash
cd chatterbox-container
docker build -t chatterbox-tts .
```

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

## Usage from Python

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

## Benefits of Source Installation

1. **No dependency conflicts** - Fresh install avoids version conflicts
2. **Latest features** - Get latest code from GitHub
3. **Customizable** - Can modify source if needed
4. **Isolated** - Doesn't affect host system dependencies

## Troubleshooting

### Build fails
- Ensure you have NVIDIA Docker runtime installed
- Check CUDA version compatibility (12.1)

### Import errors
- Verify Chatterbox cloned correctly: `docker exec chatterbox-tts ls /tmp/chatterbox`
- Check installation: `docker exec chatterbox-tts python3 -c "from chatterbox import ChatterboxTTS"`

### GPU not detected
- Ensure `--runtime=nvidia` is set
- Check: `docker exec chatterbox-tts python3 -c "import torch; print(torch.cuda.is_available())"`

