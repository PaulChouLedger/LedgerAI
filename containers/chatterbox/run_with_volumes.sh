#!/bin/bash
# Run ChatterboxTTS container with all necessary volume mounts

docker run --rm --runtime=nvidia \
  -p 11437:11437 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/../assets/voice_samples:/app/voice_samples" \
  -v "$(pwd)/../assets/prompts:/app/prompts" \
  -v "$(pwd)/../shared:/shared" \
  -v "$(pwd)/../data/voice_cache:/app/voice_cache" \
  -v "$(pwd)/../.env:/app/.env" \
  chatterbox-tts
