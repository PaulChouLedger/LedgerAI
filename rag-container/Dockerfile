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

# Pre-download all sentence transformer models (makes it WiFi-independent)
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
import time; \
models = ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'paraphrase-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1', 'all-mpnet-base-v2', 'all-distilroberta-v1']; \
print('📥 Pre-downloading all embedding models...'); \
[i for i, model_name in enumerate(models, 1) if (print(f'[{i}/{len(models)}] Downloading {model_name}...') or True) and (start_time := time.time()) and (SentenceTransformer(model_name) and print(f'✅ {model_name} downloaded in {time.time() - start_time:.1f}s'))]; \
print('🎉 All models pre-downloaded successfully!')" && \
echo "✅ All sentence transformer models pre-downloaded into image"

# Copy RAG service and modules
COPY container_rest.py /app/
COPY rag.py /app/
COPY ingest.py /app/
COPY benchmark_embedding_models.py /app/

# Expose port
EXPOSE 11435



# Run RAG service
CMD ["python3", "container_rest.py"]
