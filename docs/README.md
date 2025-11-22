# LedgerAI Component Documentation

## Overview

This documentation provides detailed explanations of each component in the LedgerAI system, divided by function. The system is a comprehensive conversational AI platform with speech-to-text, language model, retrieval-augmented generation, text-to-speech, chatbot, and GUI components.

## Component Documentation

### 1. [SST (Speech-to-Text)](SST_SPEECH_TO_TEXT.md)
The SST component handles converting spoken audio into text using faster-whisper in a Docker container.

**Key Features**:
- GPU-accelerated transcription
- Medical vocabulary support
- Audio normalization for optimal accuracy
- Wake word integration
- Advanced speech filtering

### 2. [LLM (Large Language Model)](LLM_LANGUAGE_MODEL.md)
The LLM component provides conversational AI using quantized models via llama.cpp. Available in Generic and Medical variants.

**Key Features**:
- GPU-accelerated inference
- Streaming response generation
- Session management
- RAG integration
- Conversation memory

### 3. [RAG (Retrieval Augmented Generation)](RAG_RETRIEVAL_AUGMENTED_GENERATION.md)
The RAG component provides semantic search over document embeddings using FAISS, with both GPU and CPU implementations.

**Key Features**:
- GPU-accelerated search (faiss_lite)
- CPU-based embedded search
- Fuzzy name matching
- Medical guideline retrieval
- Auto-ingestion from files

### 4. [TTS (Text-to-Speech)](TTS_TEXT_TO_SPEECH.md)
The TTS component converts text responses into natural speech using the ElevenLabs API, with sentence batching and ALSA playback.

**Key Features**:
- Streaming response handling
- Sentence batching for efficiency
- SSML markup support
- Initials merging
- Audio playback via ALSA

### 5. [Chat Bot / Conversation Management](CHATBOT_CONVERSATION_MANAGEMENT.md)
The chatbot component orchestrates the entire conversation flow, integrating SST, LLM, RAG, and TTS into a cohesive system.

**Key Features**:
- Wake word detection
- Voice activity detection
- Audio processing pipeline
- State synchronization
- Error handling

### 6. [GUI (User Interface)](GUI_USER_INTERFACE.md)
The GUI component provides a circular, touch-friendly interface with visual feedback for all system states.

**Key Features**:
- Circular border animations
- State-based visual feedback
- Dialog system
- Custom keyboard
- Window management

## System Architecture

### Component Flow

```
User Speech
    ↓
Hardware DSP (XVF3800) → Audio Processing
    ↓
Wake Word Detection (OpenWakeWord) → Optional
    ↓
VAD (Silero VAD) → Speech Detection
    ↓
SST (Whisper) → Text Transcription
    ↓
Chat Bot Logic → Conversation Management
    ↓
RAG (FAISS) → Context Retrieval (if needed)
    ↓
LLM (Qwen2.5) → Response Generation
    ↓
TTS (ElevenLabs) → Speech Synthesis
    ↓
Audio Playback → User Hears Response
```

### Container Architecture

```
┌─────────────────┐
│  Whisper        │  Port: 5000
│  Container      │  GPU-accelerated
└─────────────────┘

┌─────────────────┐
│  LLM Container  │  Port: 11434
│  (Medical/      │  GPU-accelerated
│   Generic)      │
└─────────────────┘

┌─────────────────┐
│  RAG Container  │  Port: 11435 (GPU mode)
│  (Optional)     │  or embedded in LLM (CPU)
└─────────────────┘

┌─────────────────┐
│  Main Process   │  Orchestrates all
│  (main.py)      │  components
└─────────────────┘
```

## Key Concepts

### 1. Streaming Architecture
- **LLM Streaming**: Tokens generated and sent immediately
- **Sentence Tagging**: `<sentence_start>` / `<sentence_end>` markers
- **TTS Batching**: Sentences batched for efficient API calls
- **Low Latency**: First audio starts as soon as possible

### 2. State Management
- **GUI States**: Visual feedback for all system states
- **Session States**: Conversation state persistence
- **Audio States**: Playing/listening/muted states
- **Synchronization**: State updates across all components

### 3. Error Handling
- **Graceful Degradation**: Continues operation on errors
- **Health Checks**: Container health monitoring
- **Retry Logic**: Automatic retries with backoff
- **User Feedback**: Error messages via GUI

### 4. Performance Optimization
- **GPU Acceleration**: Wherever possible (Whisper, LLM, RAG)
- **Caching**: Model caching, prompt caching
- **Batching**: TTS batching, embedding batching
- **Lazy Loading**: Components loaded on demand

## Configuration

### Environment Variables
- `.env` file in workspace root
- API keys, model paths, mode settings
- Volume control, timeout settings

### Settings File
- `data/app_settings.json`
- LLM mode (Medical/Generic)
- RAG mode (CPU/GPU/OFF)
- Model selection
- Wake word settings

## Dependencies

### Core Dependencies
- PyQt5 (GUI)
- PyTorch (ML models)
- llama.cpp (LLM inference)
- FAISS (Vector search)
- faster-whisper (SST)
- ElevenLabs SDK (TTS)
- Docker (Containers)

### System Dependencies
- ALSA (Audio)
- PortAudio (Audio capture)
- X11 (Display)
- CUDA (GPU acceleration)

## Development

### Code Structure
```
LedgerAI/
├── aura-control/          # Main application
│   ├── core/             # Core modules (listener, speaker, main)
│   ├── gui/              # GUI components
│   ├── server/           # Web servers
│   └── wallet/           # Wallet integration
├── whisper-container/     # SST container
├── llm-container/         # Generic LLM container
├── llm-medical-container/ # Medical LLM container
├── rag-container/         # RAG container (GPU mode)
├── setup/                 # Setup scripts
└── docs/                  # Documentation
```

### Running the System
1. Configure `.env` file
2. Run `python3 aura-control/core/main.py`
3. Complete welcome setup (WiFi)
4. System starts containers automatically
5. GUI shows visual feedback

## Troubleshooting

### Common Issues

**Container Not Starting**:
- Check Docker daemon running
- Verify container images built
- Check port availability

**Audio Not Working**:
- Verify XVF3800 connected
- Check ALSA device detection
- Verify audio permissions

**GUI Not Showing**:
- Check DISPLAY environment variable
- Verify X11 connection
- Check window focus

**Transcription Failing**:
- Verify Whisper container running
- Check GPU availability
- Verify audio normalization

## Further Reading

- [SST Component Details](SST_SPEECH_TO_TEXT.md)
- [LLM Component Details](LLM_LANGUAGE_MODEL.md)
- [RAG Component Details](RAG_RETRIEVAL_AUGMENTED_GENERATION.md)
- [TTS Component Details](TTS_TEXT_TO_SPEECH.md)
- [Chat Bot Details](CHATBOT_CONVERSATION_MANAGEMENT.md)
- [GUI Details](GUI_USER_INTERFACE.md)

