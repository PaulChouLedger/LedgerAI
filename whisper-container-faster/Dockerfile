#---
# name: faster-whisper
# group: audio
# depends: [ctranslate2, onnxruntime, huggingface_hub, numpy]
# requires: '>=34.1.0'
# test: test.py
# notes: Faster Whisper transcription with CTranslate2 (https://github.com/SYSTRAN/faster-whisper)
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    FASTER_WHISPER_VERSION \
    FASTER_WHISPER_BRANCH

WORKDIR /app

# Copy REST API and official dustynv scripts
COPY container_rest.py /app/container_rest.py
COPY *.sh /tmp/faster-whisper/

# Install faster-whisper using the official dustynv method
RUN /tmp/faster-whisper/install.sh || /tmp/faster-whisper/build.sh

# Install REST API dependencies
RUN pip3 install flask soundfile scipy

# Create cache directories
RUN mkdir -p /app/cache/whisper /root/.cache/huggingface

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
