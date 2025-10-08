FROM faiss_lite:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app


# Install only missing dependencies
RUN pip install --no-cache-dir --ignore-installed \
    flask \
    flask-cors \
    requests \
    sentence-transformers \
    cuda-python

# Pre-download sentence transformer model (makes it WiFi-independent)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    echo "✅ Sentence transformer model pre-downloaded into image"

# Copy RAG service and module
COPY container_rest.py /app/
COPY rag.py /app/


# Expose port
EXPOSE 11435



# Run RAG service
CMD ["python3", "container_rest.py"]
