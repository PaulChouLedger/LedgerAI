FROM faiss_lite:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app


# Install dependencies with specific versions for compatibility
RUN pip install --no-cache-dir --ignore-installed \
    flask \
    flask-cors \
    requests \
    sentence-transformers==2.2.2 \
    torch==2.0.1 \
    transformers==4.30.2 \
    numpy==1.24.3

# Copy RAG service and module
COPY container_rest.py /app/
COPY rag.py /app/


# Expose port
EXPOSE 11435



# Run RAG service
CMD ["python3", "container_rest.py"]
