# Minimal TensorRT-LLM Container - Base image only for testing
# Based on Jetson AI Lab: https://www.jetson-ai-lab.com/tensorrt_llm.html
FROM dustynv/tensorrt_llm:0.12-r36.4.0

# Set working directory
WORKDIR /app

# Copy only the build script for testing
COPY scripts/build_tensorrt_engine.sh /app/scripts/build_tensorrt_engine.sh
RUN chmod +x /app/scripts/build_tensorrt_engine.sh

# Create models directory mount point
RUN mkdir -p /models
