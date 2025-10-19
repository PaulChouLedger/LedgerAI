FROM faiss_lite:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app


# Install only missing dependencies
RUN pip install --no-cache-dir --ignore-installed \
    flask \
    flask-cors \
    requests \
    sentence-transformers \
    cuda-python \
    PyPDF2==3.0.1 \
    python-docx==0.8.11 \
    watchdog>=3.0.0 \
    metaphone

# Pre-download all-mpnet-base-v2 model (makes it WiFi-independent)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')" && \
    echo "✅ all-mpnet-base-v2 model pre-downloaded into image"

# Copy RAG service and modules
COPY container_rest.py /app/
COPY rag.py /app/
COPY ingest.py /app/
COPY benchmark_embedding_models.py /app/

# Expose port
EXPOSE 11435



# Run RAG service
CMD ["python3", "container_rest.py"]
