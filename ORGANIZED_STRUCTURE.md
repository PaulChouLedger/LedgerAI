# LedgerAI - Organized Project Structure

## 📁 **Complete Directory Organization**

The LedgerAI project has been reorganized for better maintainability and clarity.

---

## 🏗️ **Root Directory Structure**

```
📁 LedgerAI/
├── 📁 assets/                    # Static assets and media files
│   ├── 📁 prompts/              # Audio prompt files
│   └── 📁 voice_samples/        # Voice sample recordings
├── 📁 aura-control/             # GUI and control interface
│   ├── 📁 core/                 # Core functionality
│   │   ├── listener.py          # Audio input handling
│   │   ├── speaker.py           # Audio output handling
│   │   ├── state.py             # Application state management
│   │   ├── main.py              # Main application entry point
│   │   ├── example_button_script.py # Example scripts
│   │   └── generate_doc_chunks.py # Document processing
│   ├── 📁 gui/                  # Graphical user interface components
│   │   ├── aura_gui.py          # Main GUI application
│   │   ├── circular_border.py   # UI border components
│   │   ├── custom_keyboard.py   # Custom input components
│   │   ├── file_upload_dialog.py # File upload interface
│   │   ├── metamask_payment_dialog.py # Payment dialogs
│   │   ├── payment_dialog.py    # Payment processing UI
│   │   └── wallet_dialog.py     # Wallet management UI
│   ├── 📁 server/               # Network servers and bots
│   │   ├── telegram_bot.py      # Telegram bot integration
│   │   └── web_upload_server.py # Web upload server
│   ├── 📁 wallet/               # Wallet and crypto integration
│   │   ├── metamask_integration.py # MetaMask wallet integration
│   │   ├── native_wallet.py     # Native wallet implementation
│   │   └── wallet_integration.py # General wallet integration
│   ├── 📁 requirements/         # Different requirement files
│   │   ├── requirements.txt     # Main requirements
│   │   ├── requirements_gdrive.txt # Google Drive requirements
│   │   └── requirements_upload.txt # Upload service requirements
│   └── 📁 utils/                # Utility functions
│       └── __init__.py          # Utils package initialization
├── 📁 data/                     # Application data and embeddings
│   ├── 📁 embeddings/           # FAISS embeddings and vectors
│   ├── 📁 fillers/              # Audio filler files
│   ├── 📁 input/                # Input documents for processing
│   ├── 📁 parsed/               # Parsed document text
│   ├── ingest_state.json        # Ingestion state tracking
│   └── wallet_address.txt       # Wallet configuration
├── 📁 docs/                     # Documentation and guides
│   ├── ADVANCED_FILTER_SUMMARY.md
│   ├── AUTO_INGEST_GUIDE.md
│   ├── CIRCULAR_BORDER_SYSTEM.md
│   ├── CLINICIAN_MODE_ROADMAP.md
│   ├── DYNAMIC_RAG_IMPROVEMENTS.md
│   ├── DYNAMIC_TRIAGE_SYSTEM.md
│   ├── INSTALLATION_GUIDE.md
│   ├── INTENT_CLASSIFICATION_SYSTEM.md
│   ├── LLM_GARBAGE_OUTPUT_FIX.md
│   ├── MODULAR_ARCHITECTURE.md
│   ├── NFC_WALLET_AUTH_GUIDE.md
│   ├── PIPELINE_SYNC_SUMMARY.md
│   ├── RAG_PHONETIC_MATCHING.md
│   ├── README.md                # Main project documentation
│   ├── TRIAGE_CONTEXT_BUG_FIX.md
│   ├── TRIAGE_FIXES_SUMMARY.md
│   ├── TRIAGE_SIMPLIFICATION_PLAN.md
│   ├── WALLET_INTEGRATION_GUIDE.md
│   └── WHISPER_NAME_GUIDANCE.md
├── 📁 hardware/                 # Hardware-specific configurations
├── 📁 llm-container/            # Language model container
│   ├── 📁 synonyms/             # Medical synonym dictionaries
│   ├── 📁 triage_defs/         # Medical triage definitions
│   ├── casual.py                # Casual conversation mode
│   ├── clinician.py             # Basic clinician mode
│   ├── container_rest.py        # Main container REST API
│   ├── Dockerfile               # Container configuration
│   ├── dynamic_triage.py        # Dynamic triage system
│   ├── enhanced_clinician.py    # Advanced clinician mode
│   ├── intent_classifier.py     # Intent classification
│   ├── nlg.py                   # Natural language generation
│   ├── outcome_cache.py         # Outcome caching
│   ├── requirements.txt         # Container requirements
│   ├── router.py                # Mode routing logic
│   ├── thinker.py               # Knowledge query mode
│   ├── triage.py                # Medical triage system
│   ├── unified_medical_mode.py  # Unified medical assistant
│   └── validation.py            # Input validation
├── 📁 medical/                  # Medical data and processing
│   ├── clinician_rag.py         # Medical RAG system
│   ├── ENHANCED_CLINICIAN_INTEGRATION.md
│   ├── medical_data_ingestion.py # Medical data collection
│   ├── medical_update_scheduler.py # Automated medical updates
│   ├── MEDICAL_DATA_SYSTEM_GUIDE.md
│   ├── MEDICAL_VOCABULARY_SYSTEM.md
│   └── requirements_medical.txt # Medical-specific requirements
├── 📁 rag-container/            # RAG (Retrieval-Augmented Generation)
│   ├── container_rest.py        # RAG container API
│   ├── Dockerfile               # RAG container config
│   ├── ingest.py                # Document ingestion
│   ├── rag.py                   # RAG implementation
│   └── rebuild_embeddings.py    # Embedding rebuild utility
├── 📁 setup/                    # Setup and installation scripts
│   ├── docker-compose.yml       # Docker composition
│   ├── HARDWARE_CONFIGURATION_GUIDE.md
│   ├── install_dependencies.sh  # Dependency installation
│   ├── install_jetson.sh         # Jetson-specific setup
│   ├── install_wallet_integration.sh # Wallet setup
│   ├── scripts/                 # Additional setup scripts
│   ├── setup_alsa_jetson.sh      # Audio setup for Jetson
│   └── setup_infura.sh          # Infura blockchain setup
├── 📁 shared/                   # Shared resources between containers
│   ├── input_audio              # Shared audio input
│   └── output_audio             # Shared audio output
├── 📁 tests/                    # Test files and validation
│   ├── enhanced_clinician_demo.py # Enhanced clinician demo
│   ├── test_medical_fixes.py    # Medical data ingestion fixes test
│   ├── test_medical_routing.py   # Medical routing test
│   └── test_unified_medical_mode.py # Unified medical mode test
└── 📁 utils/                    # General utility functions
    └── audio_utils.py           # Audio processing utilities
```

---

## 🎯 **Organization Benefits**

### **📋 By Function**
- **Clear separation** of concerns
- **Easy navigation** for developers
- **Logical grouping** of related functionality

### **🔧 Maintainability**
- **Faster development** - find files quickly
- **Easier debugging** - related files are together
- **Better collaboration** - clear file ownership

### **📦 Modularity**
- **Independent components** can be developed separately
- **Clean imports** with proper path management
- **Scalable architecture** for future growth

---

## 🚀 **Import Path Updates**

All import statements have been updated to use the new organized structure:

```python
# Before: sys.path.append('llm-container')
sys.path.append(str(Path(__file__).parent.parent / "llm-container"))

# After: Direct imports work correctly
from enhanced_clinician import EnhancedClinicianSession
```

---

## ✅ **Ready for Production**

The project is now **fully organized** and **production-ready** with:

- ✅ **Clear file organization** by function
- ✅ **Proper import paths** throughout codebase
- ✅ **Comprehensive documentation** structure
- ✅ **Scalable architecture** for future development

**Perfect foundation for continued development!** 🎉
