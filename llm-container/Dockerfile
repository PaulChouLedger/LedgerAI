FROM dustynv/llama_cpp:0.3.7-r36.4.0

WORKDIR /app

COPY container_rest.py /app/container_rest.py
COPY models/mistral.gguf /models/mistral.gguf
COPY models/qwen2.5-1.5b-instruct-q4_0.gguf /models/qwen2.5-1.5b-instruct-q4_0.gguf

ENV MODEL_NAME qwen2.5-1.5b-instruct-q4_0

RUN pip3 install --no-cache-dir flask --ignore-installed --index-url https://pypi.org/simple

EXPOSE 11434

CMD ["python3", "/app/container_rest.py"]
