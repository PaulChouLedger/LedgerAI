# RAPIDS container with cuML and sentence-transformers
FROM cuml:r36.4.tegra-aarch64-cu126-22.04-cuda-python

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip --break-system-packages && \
    pip install --break-system-packages --ignore-installed \
        sentence-transformers \
        beautifulsoup4 \
        lxml \
        flask \
        cuml-cu12 \
        --extra-index-url=https://pypi.nvidia.com

CMD ["python3", "container_rest.py"]
