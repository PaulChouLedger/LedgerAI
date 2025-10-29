# LLM Generic Container with TensorRT-LLM

A high-performance generic LLM container with TensorRT-LLM support for document-based question answering.

## Features

- **Dual-Model System**:
  - **Qwen2.5-7B-Instruct** (TensorRT-LLM) - For RAG and complex questions
  - **Llama-3.2-1B-Instruct** (TensorRT-LLM) - For simple tasks and greetings
- **RAG Integration**: Document-based question answering using FAISS
- **Streaming Support**: Real-time streaming responses
- **CPU FAISS Auto-Ingestion**: Automatic document processing and indexing
- **Fuzzy Matching**: Handles typos and transcription errors
- **Automatic Model Routing**: Selects appropriate model based on query complexity

## Model Requirements

Models must be pre-converted to TensorRT engine format:

- **Qwen2.5-7B-Instruct**: `/models/qwen2.5-7b-instruct-trt/engine`
- **Llama-3.2-1B-Instruct**: `/models/llama-3.2-1b-instruct-trt/engine`

**Fallback**: If TensorRT engines not found, falls back to GGUF format with llama.cpp

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Set model paths
MODEL_PATH_COMPLEX=/models/qwen2.5-7b-instruct-trt/engine
MODEL_PATH_SIMPLE=/models/llama-3.2-1b-instruct-trt/engine
```

### 2. Build Container

```bash
docker build -t llm-generic-container .
```

### 3. Run Container

```bash
docker run -p 11434:11434 \
  -v /path/to/models:/models \
  -v /path/to/documents:/app/data/input \
  llm-generic-container
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Chat (Non-streaming)
```bash
POST /chat
Content-Type: application/json

{
  "prompt": "What is artificial intelligence?",
  "session_id": "user123",
  "use_rag": true,
  "force_complex": false,  # Force Qwen2.5-7B
  "max_tokens": 512
}
```

### Chat (Streaming)
```bash
POST /chat
Content-Type: application/json

{
  "prompt": "Explain quantum computing",
  "stream": true,
  "use_rag": true
}
```

## Model Routing Logic

The system automatically selects the appropriate model:

**Qwen2.5-7B-Instruct** (Complex Model) is used for:
- RAG queries (when `use_rag: true` and documents found)
- Long prompts (>50 words)
- Complex questions ("what is", "why", "how", "explain")
- Technical/academic queries

**Llama-3.2-1B-Instruct** (Simple Model) is used for:
- Greetings and casual conversation
- Short prompts (<50 words)
- Simple questions
- Non-RAG queries

## Configuration

### Environment Variables

- `MODEL_PATH_COMPLEX`: Path to Qwen2.5-7B TensorRT engine
- `MODEL_PATH_SIMPLE`: Path to Llama-3.2-1B TensorRT engine
- `N_CTX_COMPLEX`: Context window for complex model (default: 4096)
- `N_CTX_SIMPLE`: Context window for simple model (default: 2048)
- `CHAT_FORMAT_COMPLEX`: Chat format for Qwen (default: mistral-instruct)
- `CHAT_FORMAT_SIMPLE`: Chat format for Llama (default: llama-3)
- `LLM_TEMPERATURE`: Sampling temperature (default: 0.7)
- `LLM_TOP_P`: Top-p sampling (default: 0.85)
- `LLM_TOP_K`: Top-k sampling (default: 30)
- `LLM_REPEAT_PENALTY`: Repeat penalty (default: 1.15)

## Expected Performance

### Qwen2.5-7B-Instruct (TensorRT-LLM)
- **Latency**: ~2-3 seconds for 50 tokens (RAG queries)
- **Throughput**: ~15-20 tokens/sec
- **Memory**: ~4-5 GB GPU
- **Use Case**: Complex reasoning, RAG, multi-document synthesis

### Llama-3.2-1B-Instruct (TensorRT-LLM)
- **Latency**: ~0.5-1 second for 50 tokens
- **Throughput**: ~30-50 tokens/sec
- **Memory**: ~1-2 GB GPU
- **Use Case**: Greetings, simple tasks, quick responses

## Model Conversion

Models need to be converted to TensorRT format. See TensorRT-LLM documentation for conversion scripts.

## Directory Structure

```
llm-generic-container/
├── container_rest.py          # Main REST API
├── fuzzy_matcher.py           # Fuzzy matching for typos
├── tensorrt_llm_wrapper.py   # TensorRT-LLM wrapper
├── tensorrt_models_config.py # Model configurations
├── rag/                       # RAG client and FAISS integration
├── requirements.txt           # Dependencies
├── Dockerfile                # Container image
└── README.md                # This file
```

## Requirements

- Python 3.8+
- TensorRT-LLM (via dustynv/tensorrt_llm:0.12-r36.4.0 base image)
- FAISS (CPU or GPU)
- sentence-transformers
- Fuzzy matching libraries

See `requirements.txt` for full list.

## Troubleshooting

### Model Not Found
- Check model paths in environment variables
- Verify TensorRT engines exist
- System will fallback to GGUF if TensorRT engines not found

### TensorRT-LLM Not Available
- Container falls back to llama.cpp automatically
- Check base image includes TensorRT-LLM
- Verify CUDA/GPU access

### Performance Issues
- Ensure models are in TensorRT format for best performance
- Check GPU memory availability
- Monitor model routing (complex vs simple)
