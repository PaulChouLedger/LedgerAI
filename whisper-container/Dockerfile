FROM dustynv/whisper_trt:r36.3.0

WORKDIR /app

# Copy REST API
COPY container_rest.py /app/container_rest.py
#COPY cache/whisper /root/.cache/whisper
#COPY cache/whisper_trt /root/.cache/whisper_trt

# Patch both whisper_trt and OpenAI whisper cache dirs
RUN rm -f /root/.cache/whisper_trt && mkdir -p /root/.cache/whisper_trt && \
    rm -f /root/.cache/whisper && mkdir -p /root/.cache/whisper && \
    sed -i 's/os.makedirs(_CACHE_DIR)/os.makedirs(_CACHE_DIR, exist_ok=True)/' /opt/whisper_trt/whisper_trt/cache.py

# Install Flask and soundfile (avoid Jetson PyPI timeouts)
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir --index-url https://pypi.org/simple \
    flask soundfile numpy==1.26.4 scipy

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
