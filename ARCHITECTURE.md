# Aura Voice Assistant - System Architecture

## Overview
Aura is a real-time voice assistant system built with a microservices architecture. The system processes audio input through speech-to-text, generates contextual responses using an LLM, and outputs synthesized speech, all orchestrated by a central control application.

## Microservices

### 1. Whisper Container (Speech-to-Text)
- **Port**: 5000
- **Purpose**: Converts audio input to text using OpenAI's Whisper model
- **Technology**: Faster-whisper with CUDA acceleration
- **API Endpoint**: `POST /transcribe`
- **Input**: Audio data (WAV/raw audio)
- **Output**: Transcribed text

### 2. LLM Container (Language Model)
- **Port**: 11434
- **Purpose**: Generates contextual responses to user queries
- **Technology**: llama-cpp with Qwen2.5-1.5B model
- **API Endpoint**: `POST /chat`
- **Input**: Text prompts
- **Output**: Streaming text responses
- **Features**: Streaming responses, GPU acceleration (32 layers)

### 3. TTS Container (Text-to-Speech) - **UNUSED**
- **Port**: 5002
- **Purpose**: Converts text responses to audio output (built but not integrated)
- **Technology**: Hybrid ElevenLabs + Piper TTS with ONNX models
- **API Endpoint**: `POST /speak`
- **Input**: Text to synthesize
- **Output**: Audio files (MP3/WAV)
- **Models**: en_US-amy-low voice model (Piper fallback)
- **Status**: **Dead code** - not launched or used in main application

### 4. RAG Container (Vector Search/RAG)
- **Port**: 5003
- **Purpose**: Provides document context via vector similarity search and RAG
- **Technology**: Modern RAG stack with cuDF, cuVS, LlamaIndex, and GPU embeddings
- **API Endpoints**: 
  - `POST /rag` - Query with RAG context
  - `GET /health` - Health check
  - `POST /rebuild` - Rebuild vector index
  - `GET /stats` - System statistics
  - `GET /config` - Configuration parameters
- **Input**: Natural language queries
- **Output**: Contextual responses with retrieved document chunks
- **Status**: **Fully integrated** - Primary response generation with LLM fallback

## Core Application: aura-control

The `aura-control` directory contains the main orchestrating application that:

### Components:
- **`main.py`**: Entry point and container health monitoring (Whisper + RAG + LLM)
- **`aura_gui.py`**: PyQt5 GUI with "aura eye" visual indicator
- **`listener.py`**: Real-time audio input with 6-channel VAD and Silero VAD
- **`speaker.py`**: ElevenLabs TTS with RAG integration and LLM fallback
- **`chat.py`**: Simple chat utilities (filler phrases, sentence splitting)
- **`fingerprint.py`**: Wake word detection system (not integrated)
- **`state.py`**: Global state management (playback, shutdown, restart flags)

### Key Features:
- **Real-time Audio Processing**: 6-channel microphone array with Silero VAD
- **Streaming TTS**: ElevenLabs cloud TTS with SSML and emotion detection
- **Playback Coordination**: Smart mic pausing during TTS to prevent feedback
- **Chunked Responses**: Sentence-based LLM streaming with natural speech flow
- **Visual Feedback**: PyQt5 GUI with pulsing "aura eye" indicator
- **Document-aware Conversation**: **RAG fully integrated** with LLM fallback
- **Wake Word Detection**: Fingerprint-based system (built but not integrated)

## Data Flow / Architecture Flow

```
┌─────────────┐    ┌──────────────┐     ┌─────────────┐
│ Microphone  │───▶│ aura-control │───▶│ Audio Out   │
│ (Hardware)  │    │ (Orchestrator)│    │ (Speakers)  │
└─────────────┘    └──────────────┘     └─────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Processing Pipeline                      │
│                                                         │
│ 1. Audio Input (listener.py)                            │
│    ├─ 6-Channel Microphone Array (ReSpeaker 4)        │
│    ├─ Real-time VAD (Silero VAD on channel 0)         │
│    ├─ 32ms Frame Processing (512 samples)              │
│    ├─ Silence Detection (500ms timeout)                │
│    └─ Mono Audio Extraction (channel 0 only)           │
│                                                         │
│ 2. Speech-to-Text (whisper-container)                   │
│    ├─ HTTP POST /transcribe                             │
│    └─ Whisper Model (distil-small.en)                   │
│                                                         │
│ 3. RAG Processing (rag-container) [Primary]             │
│    ├─ HTTP POST /rag                                    │
│    ├─ Document Chunking (cuDF)                          │
│    ├─ GPU Embeddings (HuggingFace Transformers)         │
│    ├─ Vector Search (cuVS)                              │
│    ├─ Context Retrieval (LlamaIndex)                    │
│    ├─ LLM Integration (Qwen2.5-1.5B)                    │
│    └─ Contextual Response Generation                     │
│                                                         │
│ 4. LLM Fallback (llm-container) [Backup]               │
│    ├─ HTTP POST /chat                                   │
│    ├─ Qwen2.5-1.5B Model (32 GPU layers)               │
│    ├─ Streaming Response (token by token)               │
│    └─ Direct response (no context)                      │
│                                                         │
│ 5. Text-to-Speech (speaker.py)                          │
│    ├─ ElevenLabs API (cloud TTS)                        │
│    ├─ SSML Processing (emotion, breaks, emphasis)       │
│    ├─ Streaming Audio Generation                         │
│    └─ Real-time PCM Output                              │
│                                                         │
│ 6. Audio Playback (speaker.py)                          │
│    ├─ Queue Management                                  │
│    ├─ Chunked Playback                                  │
│    ├─ Volume Control                                    │
│    └─ Playback State Management                         │
└─────────────────────────────────────────────────────────┘
```

## Multi-Threaded Architecture

The Aura system uses a sophisticated multi-threaded design orchestrated by `main.py` to handle concurrent operations while maintaining thread safety and proper coordination.

### Thread Management Strategy:

#### **Main Thread (GUI Thread):**
- **Purpose**: PyQt5 GUI event loop (required by Qt framework)
- **Responsibilities**: 
  - GUI rendering and user interaction
  - Visual feedback (pulsing "aura eye")
  - Application lifecycle management
- **Blocking**: Runs `run_gui_loop()` which blocks until application exit

#### **Background Threads (Daemon Threads):**

1. **Container Management Thread** (`start_services()`)
   - **Purpose**: Monitors Docker container health and warm-up
   - **Responsibilities**:
     - Health checks for Whisper, RAG, and LLM containers
     - Warm-up procedures for RAG and LLM services
     - Start other background services
   - **Lifecycle**: Runs once during startup, then exits

2. **Fingerprint Monitor Thread** (`start_fingerprint_monitor()`)
   - **Purpose**: Wake word detection (currently stub implementation)
   - **Responsibilities**:
     - Continuous audio monitoring for wake words
     - Fingerprint matching against TTS playback
   - **Status**: Built but not integrated into main flow

3. **Audio Listener Thread** (`listen()`)
   - **Purpose**: Real-time audio input processing
   - **Responsibilities**:
     - 6-channel microphone input
     - VAD (Voice Activity Detection) with Silero model
     - Audio buffering and transcription triggering
     - Whisper container communication
   - **Lifecycle**: Continuous loop until application shutdown

4. **TTS Playback Thread** (`playback_loop()`)
   - **Purpose**: Audio output and TTS processing
   - **Responsibilities**:
     - Queue-based sentence processing
     - ElevenLabs API communication
     - SSML processing and audio generation
     - Real-time audio streaming to speakers
   - **Lifecycle**: Continuous loop processing `SENTENCE_QUEUE`

### Thread Coordination Mechanisms:

#### **Global State Management** (`state.py`):
```python
# Playback state coordination
_playing = False  # Prevents mic feedback during TTS

# Shutdown coordination
shutdown_requested = False  # Graceful exit signaling

# Listener restart coordination
restart_listener_flag = False  # Dynamic restart capability
```

#### **Thread Safety Patterns**:

1. **Playback State Synchronization**:
   - `is_playing()`: Audio listener checks before processing
   - `set_playing(True/False)`: TTS thread updates state
   - **Purpose**: Prevents microphone feedback during TTS playback

2. **Queue-based Communication**:
   - `SENTENCE_QUEUE`: Thread-safe queue for TTS chunks
   - **Producer**: LLM streaming thread (via `speak_llm_response()`)
   - **Consumer**: TTS playback thread (`playback_loop()`)

3. **Stream Management**:
   - Audio listener properly stops/starts microphone stream
   - Prevents resource conflicts between input/output

#### **Thread Lifecycle Coordination**:

```
Main Thread (GUI)
├─ Launches Container Management Thread
│  └─ Starts Fingerprint Monitor Thread
│  └─ Starts Audio Listener Thread
├─ TTS Playback Thread (auto-started on import)
└─ GUI Event Loop (blocks until exit)
```

### Critical Thread Interactions:

#### **Audio Feedback Prevention**:
```python
# In listener.py
if is_playing():
    stream.stop()  # Pause microphone
    while is_playing():
        time.sleep(0.1)  # Wait for TTS to finish
    stream.start()  # Resume microphone
```

#### **RAG Response Flow**:
```python
# RAG query → TTS queue → Audio output
speak_llm_response(text)  # Main thread
├─ Attempts RAG query (rag-container)
├─ Falls back to direct LLM if RAG fails
├─ Enqueues complete sentences
└─ TTS thread processes queue asynchronously
```

### Thread Safety Considerations:

1. **Daemon Threads**: All background threads are daemon threads, ensuring clean shutdown
2. **Global State**: Thread-safe boolean flags for coordination
3. **Queue Communication**: Thread-safe queue for producer-consumer pattern
4. **Resource Management**: Proper stream lifecycle management
5. **Error Isolation**: Thread failures don't crash main application

### Performance Characteristics:

- **Low Latency**: Streaming responses reduce perceived delay
- **Concurrent Processing**: Audio input/output can overlap
- **Resource Efficiency**: Daemon threads auto-cleanup on exit
- **Fault Tolerance**: Individual thread failures are isolated

## Container Orchestration

### Startup Sequence:
1. **GUI Launch**: PyQt5 interface with pulsing "aura eye" visual
2. **Container Health Monitoring**: 
   - Check Whisper container (port 5000)
   - Check RAG container (port 5003) 
   - Check LLM container (port 11434)
   - Health checks via HTTP endpoints (10s timeout)
3. **Service Initialization**:
   - TTS warm-up (ElevenLabs API test)
   - RAG warm-up (health check to rag-container)
   - LLM warm-up (test prompt to Qwen model)
   - Fingerprint monitor (wake word detection - stub)
   - Audio listener (6-channel VAD + conversation loop)

### Container Configuration:
- **Orchestration**: Docker Compose manages all containers
- **Network**: Shared localhost network for inter-service communication
- **Volumes**: Shared directory (`/shared`) for document storage and vector indices
- **Environment**: Configurable via environment variables in docker-compose.yml

### Health Monitoring:
- HTTP endpoint polling with 10-second timeout
- Graceful degradation if containers fail to start
- Status codes 200/404 considered "healthy"
- RAG system warm-up on startup

### Shutdown:
- Signal handler for graceful exit (Ctrl+C)
- Docker Compose handles container lifecycle
- Clean shutdown of all services

## Communication Patterns

### Inter-Service Communication:
- **HTTP REST APIs** between all services
- **Shared volume** (`/shared`) for file-based communication
- **Streaming responses** for real-time audio processing
- **Queue-based** audio playback management

### Data Formats:
- **Audio**: WAV/raw audio, 16kHz sample rate
- **Text**: UTF-8 strings with streaming support
- **Embeddings**: NumPy arrays for vector operations
- **Config**: Environment variables and JSON payloads

## Current Limitations

1. **TTS Container**: Built but completely unused - dead code
2. **Fingerprint System**: Wake word detection built but not integrated
3. **ElevenLabs Dependency**: No local TTS fallback, requires internet
4. **Hardcoded Hardware**: Only works with specific ReSpeaker 4 Mic Array
5. **Error Handling**: Basic retry logic, limited fault tolerance
6. **Scaling**: Single-instance containers, no load balancing
7. **Monitoring**: Basic logging, no metrics collection
8. **RAG Fallback**: Direct LLM fallback loses document context

## Development Notes

- **Docker Compose**: **Now actively used** for container orchestration
- **Environment**: Python 3.10 with CUDA support
- **Dependencies**: Modern RAG stack (cuDF, cuVS, LlamaIndex, HuggingFace Transformers)
- **Hardware**: Optimized for NVIDIA GPUs and ReSpeaker 4 Mic Array
- **TTS Architecture**: Dual system (ElevenLabs cloud + Piper local) but only cloud used
- **Audio Processing**: 6-channel input with beamformed channel 0 for VAD/transcription
- **State Management**: Global state coordination between listener and speaker modules
- **RAG Architecture**: Clean separation with dedicated rag-container microservice

## RAG System Architecture

The new RAG implementation in `rag-container` provides a modern, GPU-accelerated document retrieval and response generation system.

### RAG Components:

#### **Document Processing** (`document_processor.py`):
- **Technology**: cuDF for GPU-accelerated text processing
- **Features**: 
  - Intelligent chunking with sentence boundary detection
  - Configurable chunk size (default: 512 tokens)
  - Overlap handling for context preservation
  - Support for TXT, PDF, and HTML files
- **Input**: Raw documents from `/shared/input` directory
- **Output**: Structured chunks with metadata (doc_id, chunk_id, position)

#### **Embedding Engine** (`embedding_engine.py`):
- **Technology**: HuggingFace Transformers with GPU acceleration
- **Model**: Configurable (default: `sentence-transformers/all-MiniLM-L6-v2`)
- **Features**:
  - Batch processing for efficiency
  - GPU memory optimization
  - Normalized embeddings for cosine similarity
- **Input**: Text chunks
- **Output**: 384-dimensional embedding vectors

#### **Vector Store** (`vector_store.py`):
- **Technology**: cuVS for GPU-accelerated vector search
- **Features**:
  - Approximate Nearest Neighbor (ANN) search
  - Metadata storage alongside vectors
  - Configurable similarity thresholds
  - Persistent index storage
- **Input**: Embedding vectors + metadata
- **Output**: Similarity search results with scores

#### **LlamaIndex Integration** (`llamaindex_adapter.py`):
- **Purpose**: Bridges cuVS with LlamaIndex framework
- **Features**:
  - Custom VectorStore adapter
  - Query result formatting
  - Metadata preservation
- **Input**: LlamaIndex query objects
- **Output**: Formatted search results

#### **RAG Engine** (`rag_engine.py`):
- **Purpose**: Orchestrates the complete RAG pipeline
- **Features**:
  - Document ingestion and indexing
  - Query processing with context retrieval
  - LLM integration for response generation
  - Error handling and fallback mechanisms
- **Input**: Natural language queries
- **Output**: Contextual responses with source attribution

### Configuration System (`config.py`):

The RAG system uses Pydantic for configuration management with 9 critical tunable parameters:

```python
class AuraRAGConfig:
    chunk_size: int = 512                    # Text chunk size
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32           # GPU batch processing
    top_k: int = 3                          # Number of chunks to retrieve
    score_threshold: float = 0.7            # Similarity threshold
    llm_url: str = "http://localhost:11434" # LLM container endpoint
    llm_timeout: int = 30                   # Request timeout
    input_dir: str = "/shared/input"        # Document directory
    index_path: str = "/shared/vector_index" # Index storage
    debug: bool = False                     # Debug logging
```

### API Endpoints:

- **`POST /rag`**: Primary query endpoint with RAG processing
- **`GET /health`**: Health check for container monitoring
- **`POST /rebuild`**: Rebuild vector index from documents
- **`GET /stats`**: System statistics and performance metrics
- **`GET /config`**: Current configuration parameters

### Integration Flow:

1. **Document Ingestion**: Documents placed in `/shared/input` are automatically processed
2. **Index Building**: cuDF chunks documents, embeddings generated, cuVS index built
3. **Query Processing**: User queries → embeddings → vector search → context retrieval
4. **Response Generation**: Retrieved context + query → LLM → contextual response
5. **Fallback**: If RAG fails, direct LLM query without context

### Performance Characteristics:

- **GPU Acceleration**: All vector operations on GPU via cuDF/cuVS
- **Batch Processing**: Efficient embedding generation
- **Persistent Storage**: Index survives container restarts
- **Configurable Quality**: Tunable similarity thresholds and chunk sizes
- **Fault Tolerance**: Graceful fallback to direct LLM queries
