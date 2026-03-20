# Chatterbox-TTS Container Testing Guide

This guide explains how to test the Chatterbox-TTS container independently before integrating it into the aura pipeline.

## Prerequisites

1. **Docker** with NVIDIA runtime support
2. **NVIDIA GPU** with CUDA support
3. **Python 3** with `requests` library installed

## Quick Start

### Option 1: Automated Test Script

Run the comprehensive test script:

```bash
cd chatterbox-container
python3 test_independent.py
```

This script will:
- Check Docker availability
- Build the container (if needed)
- Start the container
- Test all API endpoints
- Assess integration readiness
- Provide detailed test results

### Option 2: Manual Testing

#### Step 1: Build the Container

```bash
cd chatterbox-container
docker build -t chatterbox-tts .
```

Or use docker-compose:

```bash
cd setup
docker compose build chatterbox-tts
```

#### Step 2: Start the Container

Using docker-compose (recommended):

```bash
cd setup
docker compose up -d chatterbox-tts
```

Or manually:

```bash
docker run -d \
  --name chatterbox-tts \
  --runtime=nvidia \
  --network=host \
  -v /path/to/LedgerAI/shared:/shared \
  -v /path/to/LedgerAI/assets/voice_samples:/app/voice_samples \
  -v /path/to/LedgerAI/data/voice_cache:/app/voice_cache \
  chatterbox-tts
```

#### Step 3: Check Container Status

```bash
# Check if container is running
docker ps | grep chatterbox-tts

# Check container logs
docker logs chatterbox-tts

# Check container health
curl http://localhost:11437/health
```

#### Step 4: Run Tests

```bash
cd chatterbox-container
python3 test_container.py
```

Or test manually:

```bash
# Health check
curl http://localhost:11437/health

# Basic synthesis
curl -X POST http://localhost:11437/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test"}' \
  --output test_output.wav

# Test with voice cloning (if voice sample available)
curl -X POST http://localhost:11437/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello with voice cloning", "voice_sample": "sample.wav"}' \
  --output test_cloned.wav
```

## Test Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "service": "chatterbox-tts",
  "chatterbox_loaded": true,
  "can_import_chatterbox": true,
  "device": "cuda",
  "source_directory_exists": true
}
```

### 2. Text-to-Speech Synthesis

**Endpoint:** `POST /synthesize`

**Request:**
```json
{
  "text": "Hello, this is a test",
  "voice_sample": "sample.wav",  // Optional
  "exaggeration": 0.6  // Optional, default 0.6
}
```

**Response:** WAV audio file (binary)

### 3. Voice Embedding Extraction

**Endpoint:** `POST /voice/embedding`

**Request:**
```json
{
  "voice_sample_path": "/app/voice_samples/sample.wav"
}
```

**Response:**
```json
{
  "success": true,
  "voice_sample": "/app/voice_samples/sample.wav",
  "embedding_cached": true
}
```

## Integration Readiness Checklist

Before integrating into the aura pipeline, verify:

- [ ] Container builds successfully
- [ ] Container starts without errors
- [ ] Health check returns `status: "ok"`
- [ ] Basic synthesis works (text → audio)
- [ ] Voice cloning works (if needed)
- [ ] Latency is acceptable (< 5 seconds for short text)
- [ ] Audio quality is acceptable
- [ ] Container runs stably for extended periods

## Troubleshooting

### Container Won't Build

**Issue:** Build fails with dependency errors

**Solution:**
- Ensure you're using the correct base image (PyTorch with CUDA)
- Check that all dependencies in Dockerfile are available
- Review build logs: `docker build -t chatterbox-tts . 2>&1 | tee build.log`

### Container Won't Start

**Issue:** Container exits immediately

**Solution:**
- Check logs: `docker logs chatterbox-tts`
- Verify NVIDIA runtime: `docker info | grep -i nvidia`
- Ensure GPU is accessible: `nvidia-smi`

### CUDA Not Available

**Issue:** Health check shows `device: "cpu"` or CUDA errors

**Solution:**
- Verify NVIDIA Docker runtime: `docker run --rm --runtime=nvidia nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Check container has GPU access: `docker exec chatterbox-tts python3 -c "import torch; print(torch.cuda.is_available())"`

### Import Errors

**Issue:** `can_import_chatterbox: false` in health check

**Solution:**
- Verify Chatterbox was cloned: `docker exec chatterbox-tts ls /app/chatterbox`
- Check installation: `docker exec chatterbox-tts python3 -c "from chatterbox import ChatterboxTTS"`
- Review container logs for installation errors

### Synthesis Fails

**Issue:** `/synthesize` endpoint returns errors

**Solution:**
- Check container logs: `docker logs chatterbox-tts`
- Verify Chatterbox is initialized: Check health endpoint
- Test with simple text first: `{"text": "test"}`
- Check GPU memory: `nvidia-smi`

### High Latency

**Issue:** Synthesis takes > 10 seconds

**Solution:**
- First request may be slow (model loading)
- Subsequent requests should be faster
- Check GPU utilization: `nvidia-smi`
- Consider pre-warming the model

## Testing Voice Cloning

1. **Prepare voice sample:**
   - Use a WAV file with at least 5 seconds of clear speech
   - Place in `assets/voice_samples/` directory
   - Ensure file is accessible in container at `/app/voice_samples/`

2. **Extract embedding:**
   ```bash
   curl -X POST http://localhost:11437/voice/embedding \
     -H "Content-Type: application/json" \
     -d '{"voice_sample_path": "/app/voice_samples/sample.wav"}'
   ```

3. **Synthesize with cloning:**
   ```bash
   curl -X POST http://localhost:11437/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello", "voice_sample": "sample.wav"}' \
     --output cloned_output.wav
   ```

## Performance Benchmarks

Expected performance on NVIDIA GPU:

- **First request:** 5-15 seconds (model loading)
- **Subsequent requests:** 0.5-2 seconds (depending on text length)
- **Voice cloning:** +50-100ms overhead
- **Audio quality:** 22.05 kHz, 16-bit WAV

## Next Steps After Testing

Once the container passes all tests:

1. **Document integration points:**
   - Update `aura-control/core/speaker.py` to use HTTP API
   - Modify TTS engine selection logic
   - Add container URL configuration

2. **Test with aura pipeline:**
   - Start all containers: `docker compose up -d`
   - Test end-to-end voice interaction
   - Monitor performance and stability

3. **Optimize if needed:**
   - Adjust container resources
   - Tune synthesis parameters
   - Implement caching strategies

## Additional Resources

- Container README: `chatterbox-container/README.md`
- Docker Compose config: `setup/docker-compose.yml`
- Aura speaker module: `aura-control/core/speaker.py`
