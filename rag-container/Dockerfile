FROM faiss:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app

# Install dependencies in correct order for CUDA compatibility
RUN pip install --no-cache-dir --ignore-installed \
    torch==2.1.0 \
    transformers \
    sentence-transformers \
    flask \
    flask-cors

# Copy RAG code
COPY rag.py /app/
COPY container_rest.py /app/

# Create cache directories
RUN mkdir -p /app/cache/sentence_transformers
RUN mkdir -p /root/.cache/huggingface/hub

# Expose port
EXPOSE 11435

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers

# Run RAG service
CMD ["python3", "container_rest.py"]
