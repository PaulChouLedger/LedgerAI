# Memory Container - Proactive AI Brain Component
# 
# Architecture:
# - Calls Whisper container via HTTP API (doesn't run Whisper locally)
# - Uses Sentence Transformers for embeddings (separate from Whisper)
# - Uses FAISS-CPU for semantic search on stored conversations
# Uses pre-built PyTorch image to avoid slow pip installs and get GPU support
FROM dustynv/pytorch:2.6-r36.4.0-cu128-24.04

# Install system dependencies
# Note: portaudio19-dev needed for background audio capture (sounddevice)
#       libsndfile1 needed for audio file I/O (soundfile)
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Verify PyTorch is installed and CUDA-enabled (from base image)
RUN python3 -c "import torch; assert hasattr(torch.version, 'cuda') and torch.version.cuda is not None, 'CUDA PyTorch required'; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Install Python dependencies (matching LLM container approach)
# Note: PyTorch is already included in base image (with CUDA support)
# sentence-transformers will automatically pull its dependencies (huggingface-hub, transformers, etc.)
# pip should detect existing torch and skip re-installing it
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple \
    flask \
    flask-cors \
    numpy>=1.26.0 \
    requests \
    sentence-transformers \
    faiss-cpu \
    watchdog \
    python-dotenv \
    sounddevice \
    soundfile \
    pydub

# Pre-download sentence transformer model (cached layer)
# This is for embeddings, NOT for transcription (Whisper handles transcription)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-distilroberta-v1')" && \
    echo "✅ all-distilroberta-v1 model pre-downloaded into image cache"

# Copy application files
COPY container_rest.py .
COPY memory_manager.py .
COPY proactive_analyzer.py .
COPY background_listener.py .
COPY __init__.py .

# Copy RAG module
COPY rag/ /app/rag/

# Create memory directory
RUN mkdir -p /app/data/memory

# Environment variables
ENV PORT=11438
ENV MEMORY_DIR=/app/data/memory
ENV WHISPER_SERVICE_URL=http://localhost:5000
ENV LLM_SERVICE_URL=http://localhost:11434
ENV AUDIO_DEVICE_NAME=reSpeaker

# Expose port
EXPOSE 11438

# Run the API
CMD ["python3", "/app/container_rest.py"]

