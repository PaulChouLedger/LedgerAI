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

# Install Flask and soundfile
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir --index-url https://pypi.org/simple \
    flask soundfile numpy==1.26.4 scipy

# Set memory optimization environment variables for TensorRT
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TRT_LOGGER_VERBOSITY=1

EXPOSE 5000
CMD ["python3", "/app/container_rest.py"]
