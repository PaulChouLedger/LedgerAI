# TensorRT-LLM Container for LedgerAI
# Using pre-built dustynv/tensorrt_llm image for optimal performance
FROM dustynv/tensorrt_llm:0.12-r36.4.0

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install additional Python dependencies
RUN pip3 install --no-cache-dir \
    flask \
    requests \
    huggingface-hub \
    transformers

# Create app directory
WORKDIR /app

# Copy application files
COPY . /app/

# Create models directory
RUN mkdir -p /models

# Expose port
EXPOSE 11434

# Set default command
CMD ["python3", "container_rest.py"]
