FROM dustynv/whisper_trt:r36.3.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip3 install --no-cache-dir uv --ignore-installed --index-url https://pypi.org/simple
ENV UV_SYSTEM_PYTHON=1  # This ensures uv uses the system Python interpreter

# Install minimal build dependencies (TensorRT/torch2trt should already be in base image)
# Install setuptools, whisper, and psutil first (required for whisper-trt build)
# Using uv pip install --system ensures they're available for --no-build-isolation
RUN uv pip install --system "openai-whisper @ git+https://github.com/openai/whisper.git"

# Copy package files
COPY pyproject.toml /app/
COPY src/ /app/src/
#COPY cache/whisper /root/.cache/whisper
#COPY cache/whisper_trt /root/.cache/whisper_trt

# Patch both whisper_trt and OpenAI whisper cache dirs
RUN rm -f /root/.cache/whisper_trt && mkdir -p /root/.cache/whisper_trt && \
    rm -f /root/.cache/whisper && mkdir -p /root/.cache/whisper && \
    sed -i 's/os.makedirs(_CACHE_DIR)/os.makedirs(_CACHE_DIR, exist_ok=True)/' /opt/whisper_trt/whisper_trt/cache.py

# Set library path for NVDLA and other Jetson libraries
# libnvdla_compiler.so is located at /usr/lib/aarch64-linux-gnu/nvidia/ (2024 bytes, valid)
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/opt/nvidia/lib:${LD_LIBRARY_PATH}

# Install the package using uv with no build isolation
# Enable preview feature for extra-build-dependencies (setuptools is defined there)
# This allows whisper-trt build to use TensorRT/torch2trt from base image
RUN uv sync --no-build-isolation

EXPOSE 5000
CMD ["uv", "run", "/app/src/aura_whisper/container_rest.py"]
