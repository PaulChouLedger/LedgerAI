FROM faiss:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app

# Install Python dependencies for RAG
RUN pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    numpy==1.24.4 \
    scipy==1.11.4 \
    torch==2.1.0 \
    transformers==4.35.2 \
    requests==2.31.0 \
    flask==2.3.3 \
    flask-cors==4.0.0

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
CMD ["python", "container_rest.py"]
