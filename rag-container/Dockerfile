FROM faiss:r36.4.tegra-aarch64-cu126-22.04

# Set working directory
WORKDIR /app

# Install only essential packages for container communication
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    requests

# Copy communication service
COPY container_rest.py /app/



# Expose port
EXPOSE 11435



# Run RAG service
CMD ["python3", "container_rest.py"]
