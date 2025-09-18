# Dockerfile — aura-tts using Piper TTS as base
FROM dustynv/piper-tts:r36.4.0

WORKDIR /app

# Copy REST API script into container
COPY container_rest.py /app/container_rest.py
# Copy .env into container
COPY .env /app/.env
# Ensure model + config are in the correct path inside the image
COPY models /models

# Install Flask for REST API
RUN pip3 install --no-cache-dir --break-system-packages \
    flask soundfile python-dotenv \
    -i https://pypi.org/simple

EXPOSE 5002

CMD ["python3", "/app/container_rest.py"]
