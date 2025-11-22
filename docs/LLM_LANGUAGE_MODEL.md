# LLM (Large Language Model) Component

## Overview

The LLM component provides conversational AI capabilities using quantized models via `llama.cpp`. Two variants exist: **Generic** (general conversation) and **Medical** (specialized medical assessments).

## Architecture

### Containers
- **Generic Container**: `llm-container/container_rest.py`
- **Medical Container**: `llm-medical-container/container_rest.py`
- **Port**: `11434` (both containers use same port, only one runs at a time)
- **Framework**: llama.cpp (CPU/GPU inference)

## Core Models

### Generic LLM
- **Default Model**: `Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`
- **Context Window**: 8192 tokens (supports RAG context)
- **Quantization**: Q4_K_M (4-bit quantization)
- **Chat Format**: Qwen format

### Medical LLM
- **Default Model**: `Qwen2.5-1.5B-Instruct.Q4_K_M-medical.gguf`
- **Same base model** as generic, specialized for medical use
- **Uses Advanced Medical Navigator** for complex reasoning

## Model Loading Logic

### Path Resolution Priority

1. **App Settings** (`/app/data/app_settings.json`):
   - Checks `llm_model` field
   - Constructs path: `/models/{filename}`
   - Validates file exists

2. **Environment Variable** (`SIMPLE_MODEL_PATH`):
   - Set by Dockerfile
   - Fallback if settings not found

3. **Default Fallback**:
   - Generic: `/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`
   - Medical: `/models/Qwen2.5-1.5B-Instruct.Q4_K_M-medical.gguf`

### GPU Acceleration

**Configuration**:
```python
n_gpu_layers = -1  # -1 = offload all layers to GPU
n_threads = 8      # CPU threads for remaining work
n_batch = 256      # Batch size (reduced for lower latency)
```

**Benefits**:
- Faster token generation
- Lower latency responses
- Better throughput

## Conversation Handling

### Generic LLM Flow

**Function**: `handle_conversation(prompt, session_id, memory_context, stream)`

**Process**:
1. **RAG Decision**:
   - Checks if query matches RAG content (`quick_content_match`)
   - Skips RAG for personal/conversational queries
   - Uses RAG only when relevant content exists

2. **RAG Context Retrieval** (if enabled):
   - Queries RAG with k=3 top results
   - Builds context string from chunks
   - Formats context for LLM prompt

3. **Conversation Memory** (optional):
   - Conversation memory index for long-term context
   - Embedding-based similarity search
   - Activation keywords trigger memory retrieval

4. **Message Construction**:
   - **With RAG/Memory**: System prompt includes context
   - **Without Context**: Direct conversational system prompt
   - User message appended

5. **LLM Generation**:
   - **RAG Mode**: Max 150 tokens (concise, focused)
   - **Direct Mode**: Max 500 tokens (full conversation)

### Medical LLM Flow

**Function**: `/chat-tts` endpoint → `generate_medical_response()`

**Process**:
1. **Session Management**:
   - Gets or creates session for `session_id`
   - Maintains conversation state
   - Handles session reset

2. **Advanced Medical Navigator**:
   - Uses hybrid LLM/RAG/FAISS approach
   - Dynamic condition ranking
   - Smart question selection
   - OLDCARTS-based questioning
   - Guideline matching

3. **Streaming Response**:
   - Tokens yielded as generated
   - Sentence splitting for TTS
   - `<sentence_start>` / `<sentence_end>` tags
   - Garbage detection and filtering

## Streaming Architecture

### Token Streaming

**Generic Container**:
```python
def stream_llm_response(messages, max_tokens):
    stream = llm_chat_simple(messages, max_tokens=max_tokens, stream=True)
    for chunk in stream:
        # Normalize chunk (dict/string)
        # Yield content tokens
        yield content
```

**Medical Container**:
```python
def generate_medical_response():
    result = navigator.process_message(session_id, prompt, stream=True)
    if isinstance(result, tuple):
        response_dict, token_stream = result
        for token in token_stream:
            yield token
```

### Sentence Tagging

**Tags**:
- `<sentence_start>`: Beginning of new sentence
- `<sentence_end>`: End of sentence
- `<pause>`: Pause marker for TTS

**Logic**:
1. Buffers tokens until sentence boundary
2. Detects sentence endings: `. ! ? :`
3. Special handling for list items (`-` triggers new sentence)
4. Yields complete sentences to TTS

### Word Boundary Detection

**Function**: `_word_stream_from_chunks(chunk_iter)`

**Process**:
1. Buffers raw LLM chunks
2. Detects word boundaries (spaces, punctuation)
3. Yields complete words (no sub-word splits)
4. Flushes remaining buffer at end

## Response Filtering

### Garbage Detection

**Function**: `filter_think_blocks(generator)`

**Logic**:
1. Accumulates output tokens
2. Checks every 100 characters
3. Calculates character repetition ratio
4. If >60% same character → garbage detected
5. Provides fallback response

**Fallback**:
```
"I'm sorry, I had trouble processing that. Could you tell me more about what's going on?"
```

### Think Block Removal

- Filters `<think>` tags
- Removes internal reasoning tokens
- Only user-facing content passes through

## LLM Configuration

### Generation Parameters

```python
LLM_TEMPERATURE_SIMPLE = 0.7  # Generic: 0.7 (creative)
LLM_TEMPERATURE_SIMPLE = 0.4  # Medical: 0.4 (deterministic)
LLM_TOP_P = 0.95
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1
LLM_NUM_PREDICT_DEFAULT = 500
```

### Context Window

- **Generic**: `SIMPLE_N_CTX = 8192`
- **Medical**: `SIMPLE_N_CTX = 8192`
- Supports RAG context (typically 1-3 chunks)

### Stop Sequences

**Reasoning Prevention**:
```python
reasoning_stop_sequences = [
    "\n\nHere's a",
    "\n\nHere is a",
    "\n\nAlternatively:",
    # ... more patterns
]
```

Prevents model from generating internal explanations instead of direct answers.

## Session Management

### Generic LLM Sessions

**Conversation Memory**:
- Stores conversation history
- Semantic similarity search
- Activation keywords: `["hey aura"]`
- Activation window: 15 seconds
- Top-K retrieval: 3 most relevant memories

### Medical LLM Sessions

**State Management**:
- Session files: `/app/data/sessions/{session_id}.json`
- Maintains:
  - Mode (triage, clinician, etc.)
  - Condition tracking
  - Step index
  - Answers collected
  - Flags (emergency, etc.)
  - User name
  - Conversation history

**Session Lifecycle**:
1. **Creation**: `get_or_create_session(session_id)`
2. **Update**: State saved after each interaction
3. **Reset**: `reset_session(session_id)` clears state
4. **Cleanup**: Inactive sessions (>2 hours) auto-removed

## RAG Integration

### RAG Mode Resolution

**Configuration**: `app_settings.json` → `rag_mode`

**Options**:
- `"CPU"`: CPU FAISS within LLM container
- `"GPU"`: External RAG container (port 11435)
- `"OFF"`: No RAG

### RAG Usage Logic

**Generic LLM**:
1. **Content Match Check**: `rag_client.quick_content_match(prompt)`
2. **Skip Personal Queries**: Day, schedule, "how are you"
3. **Search & Augment**: Retrieves k=3 chunks, builds context
4. **Prompt Construction**: Context + user question

**Medical LLM**:
1. **Navigator Integration**: Advanced Medical Navigator handles RAG
2. **Semantic Search**: FAISS-based retrieval
3. **Guideline Matching**: Medical guideline chunks prioritized
4. **Hybrid Approach**: LLM reasoning + RAG evidence

## API Endpoints

### Generic Container

**`POST /chat-tts`**:
- Streaming endpoint for TTS
- Returns Server-Sent Events (SSE)
- Sentence-tagged output

**`POST /chat-tg`**:
- Telegram bot endpoint
- Non-streaming JSON response
- Backward compatible

**`POST /voice/transcript`**:
- Passive transcript ingestion
- Conversation memory indexing
- Activation keyword detection

**`GET /health`**:
- Health check
- Model loading status

### Medical Container

**`POST /chat-tts`**:
- Main chat endpoint
- Streaming response
- Advanced Medical Navigator integration

**`POST /chat-tg`**:
- Telegram endpoint
- Non-streaming
- Session-aware

**`GET /health`**:
- Health check
- Model status

## Thread Safety

### LLM Lock

**Implementation**:
```python
llm_lock = threading.Lock()

with llm_lock:
    response = llm_simple.create_chat_completion(**params)
```

**Purpose**:
- Prevents concurrent LLM calls
- Ensures thread-safe model access
- Prevents GPU memory conflicts

## Code Locations

- **Generic Container**: `llm-container/container_rest.py`
- **Medical Container**: `llm-medical-container/container_rest.py`
- **Conversation Manager**: `llm-container/conversation_manager.py`
- **Medical Navigator**: `llm-medical-container/advanced_medical_navigator.py`

## Dependencies

- `llama-cpp-python`: Python bindings for llama.cpp
- `numpy`: Array operations
- `flask`: HTTP server
- `requests`: HTTP client (for RAG)

