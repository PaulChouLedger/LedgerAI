# Unified Configuration Guide

**One `.env` file to control EVERYTHING in Aura**

---

## 🎯 The Problem (Before)

```
llm-medical-container/.env    ← LLM settings here
rag-container/.env            ← RAG settings here?
whisper-container/.env        ← Whisper settings here??
aura-control/.env             ← GUI settings here???

❌ Too many files
❌ Hard to manage
❌ Easy to forget which file controls what
```

## ✅ The Solution (Now)

```
.env    ← ONE file at root level
        Controls EVERYTHING:
        • EHR integration toggle
        • LLM model selection
        • RAG search parameters
        • TTS settings
        • Debug flags
        • ALL settings in one place!
```

---

## 🚀 Quick Start

### Method 1: Interactive Config Manager (Easiest!)

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Run the config manager
./aura_config.sh
```

You'll see a nice menu:

```
========================================================================
   CURRENT AURA CONFIGURATION
========================================================================

🏥 EHR INTEGRATION
  ○ Disabled

🧠 LLM MODELS
  Complex Model: Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
  Simple Model:  Llama-3.2-1B-Instruct-Q4_K_M.gguf
  Context Size:  8192
  Temperature:   0.6

📚 RAG SEARCH
  Threshold:     0.3
  Top K:         3
  Phonetic:      true

🔊 TEXT-TO-SPEECH
  ❌ API Key not set

🐛 DEBUGGING
  Debug Mode:    false
  Log Level:     INFO

========================================================================

QUICK ACTIONS:
  1) Toggle EHR (on/off)
  2) Configure EHR settings
  3) Configure LLM models
  4) Configure RAG search
  5) Edit .env file directly
  6) Restart Docker containers
  7) Exit

Enter choice [1-7]:
```

### Method 2: Command Line (Fast!)

```bash
# Toggle EHR integration
./aura_config.sh ehr on
./aura_config.sh ehr off

# Show all settings
./aura_config.sh show

# Edit .env directly
./aura_config.sh edit
```

### Method 3: Edit .env Manually

```bash
# Open in your editor
nano .env
# or
code .env
# or
vim .env
```

---

## 📋 Configuration Reference

### 🏥 EHR Integration

Control FHIR/SystmOne integration:

```bash
# Toggle EHR on/off (most important!)
EHR_INTEGRATION_ENABLED=false    # OFF for development
EHR_INTEGRATION_ENABLED=true     # ON for testing/production

# FHIR server
SYSTMONE_FHIR_URL=https://hapi.fhir.org/baseR4          # Test
SYSTMONE_FHIR_URL=https://api.systmone.nhs.uk/fhir     # Production
```

**Quick commands:**
```bash
./aura_config.sh ehr on    # Enable EHR
./aura_config.sh ehr off   # Disable EHR
```

---

### 🧠 LLM Models

Choose which models to use:

```bash
# Complex model (for reasoning, diagnosis)
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
CHAT_FORMAT=mistral-instruct
N_CTX=8192

# Simple model (for templates, simple tasks)
SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SIMPLE_CHAT_FORMAT=llama-3
SIMPLE_N_CTX=2048

# Generation parameters
LLM_TEMPERATURE=0.6          # Creativity (0.0 = focused, 1.0 = creative)
LLM_TOP_P=0.85               # Nucleus sampling
LLM_TOP_K=30                 # Top-K sampling
LLM_REPEAT_PENALTY=1.15      # Penalize repetition
```

**Common model options:**
```bash
# Mistral 7B (recommended for medical)
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf

# Llama 3.1 8B
MODEL_PATH=/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Smaller/faster models
MODEL_PATH=/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

### 📚 RAG Search

Control how RAG searches documents:

```bash
# How strict should matching be?
RAG_THRESHOLD=0.3    # Lower = more results, Higher = stricter

# Number of results to return
RAG_TOP_K=3          # Usually 3-5 is good

# Advanced features
RAG_USE_RERANKING=false              # Slower but better results
RAG_USE_PHONETIC_MATCHING=true       # Helps with medical terms
```

**Adjusting threshold:**
```bash
# Very loose (lots of results)
RAG_THRESHOLD=0.1

# Balanced (recommended)
RAG_THRESHOLD=0.3

# Strict (only very relevant)
RAG_THRESHOLD=0.5
```

---

### 🔊 Text-to-Speech

```bash
# ElevenLabs API key (required for TTS)
ELEVENLABS_API_KEY=your_api_key_here

# Get your key from: https://elevenlabs.io/
```

---

### 🐛 Debugging

```bash
# Enable debug output
DEBUG_MODE=true

# Log level
LOG_LEVEL=DEBUG     # Very verbose
LOG_LEVEL=INFO      # Normal (recommended)
LOG_LEVEL=WARNING   # Quiet
LOG_LEVEL=ERROR     # Very quiet

# RAG-specific debugging
RAG_DEBUG=true      # Shows RAG search details
```

---

## 🔄 Common Workflows

### Workflow 1: Normal Development (EHR OFF)

```bash
# 1. Make sure EHR is off
./aura_config.sh ehr off

# 2. Verify settings
./aura_config.sh show

# 3. Start Aura
cd aura-control
python main.py

# Develop normally - no EHR calls, fast iteration
```

### Workflow 2: Test EHR Integration

```bash
# 1. Turn EHR on
./aura_config.sh ehr on

# 2. Restart containers
docker-compose restart

# 3. Test
cd aura-control
python main.py

# Watch for EHR logs:
# [EHR] 🏥 Integration enabled
# [EHR] ✅ Found patient
```

### Workflow 3: Change LLM Model

```bash
# 1. Open config manager
./aura_config.sh

# 2. Choose option 3 (Configure LLM models)

# 3. Update model path

# 4. Restart LLM container
docker-compose restart llm
```

### Workflow 4: Adjust RAG Sensitivity

```bash
# If RAG returns too many irrelevant results:
./aura_config.sh

# Choose option 4 (Configure RAG)
# Increase threshold to 0.4 or 0.5

# Restart RAG container
docker-compose restart rag
```

---

## 📁 File Structure

```
LedgerAI/
├── .env                    ← MASTER CONFIG (controls everything)
├── .env.example            ← Template (copy to .env)
├── aura_config.sh          ← Interactive config manager
├── docker-compose.yml      ← Loads .env automatically
│
├── llm-medical-container/
│   └── (no .env needed!)   ← Uses root .env
│
├── rag-container/
│   └── (no .env needed!)   ← Uses root .env
│
└── whisper-container/
    └── (no .env needed!)   ← Uses root .env
```

**Everything uses the root `.env` file!**

---

## 🔍 How It Works

### Docker Compose Magic

Your `docker-compose.yml` now looks like this:

```yaml
services:
  llm:
    env_file:
      - .env  # ← Loads root .env file
    environment:
      - EHR_INTEGRATION_ENABLED=${EHR_INTEGRATION_ENABLED}
      - MODEL_PATH=${MODEL_PATH}
      # ... all variables passed through

  rag:
    env_file:
      - .env  # ← Same file!
    environment:
      - RAG_THRESHOLD=${RAG_THRESHOLD}
      - RAG_TOP_K=${RAG_TOP_K}
      # ... all variables passed through
```

**All containers read from the same `.env` file!**

### What Happens When You Change Settings

```
1. Edit .env (manually or with aura_config.sh)
    ↓
2. Values stored in .env file
    ↓
3. docker-compose reads .env
    ↓
4. Passes values to containers as environment variables
    ↓
5. Containers use the values

Note: Restart required for changes to take effect!
```

---

## 🎛️ Quick Reference

### Check Current Settings

```bash
# Show all settings
./aura_config.sh show

# Or just look at the file
cat .env | grep -v "^#" | grep -v "^$"
```

### Toggle EHR

```bash
# Turn on
./aura_config.sh ehr on

# Turn off
./aura_config.sh ehr off

# Or edit directly
nano .env
# Change: EHR_INTEGRATION_ENABLED=true
```

### Restart After Changes

```bash
# Restart all containers
docker-compose restart

# Or restart specific ones
docker-compose restart llm    # For LLM/EHR changes
docker-compose restart rag    # For RAG changes
```

---

## 🔧 Troubleshooting

### Problem: Changes not taking effect

**Solution:**
```bash
# 1. Verify .env file exists
ls -la .env

# 2. Check values are set correctly
./aura_config.sh show

# 3. Restart containers (REQUIRED!)
docker-compose restart

# 4. Check container logs
docker logs aura-llm | grep -i ehr
docker logs aura-rag | grep -i threshold
```

### Problem: .env file doesn't exist

**Solution:**
```bash
# Copy from example
cp .env.example .env

# Or use config manager to create it
./aura_config.sh
```

### Problem: Don't know which setting to change

**Solution:**
```bash
# Use interactive config manager
./aura_config.sh

# Shows current values and guides you through changes
```

---

## 💡 Best Practices

### 1. Keep .env.example Updated

When you add new settings:
```bash
# 1. Add to .env.example (the template)
# 2. Add to docker-compose.yml (pass to containers)
# 3. Update this documentation
```

### 2. Don't Commit .env to Git

Your `.gitignore` should have:
```
.env
```

Only commit `.env.example`!

### 3. Document Your Changes

In `.env`, add comments:
```bash
# Changed for testing EHR integration - 2025-10-21
EHR_INTEGRATION_ENABLED=true
```

### 4. Use Config Manager for Complex Changes

```bash
# Instead of manually editing 5 variables:
./aura_config.sh

# Use the menu - it's safer and easier
```

---

## 🌟 Summary

**You now have ONE config file that controls:**
- ✅ EHR integration (on/off)
- ✅ LLM model selection
- ✅ RAG search parameters
- ✅ TTS settings
- ✅ Debug flags
- ✅ Everything else!

**Easy to manage with:**
```bash
./aura_config.sh    # Interactive menu
./aura_config.sh show    # View settings
./aura_config.sh ehr on  # Quick toggle
nano .env           # Direct edit
```

**No more hunting through multiple files!** 🎉

---

## 📖 See Also

- `EHR_TOGGLE_GUIDE.md` - Detailed EHR integration guide
- `EHR_STEP_BY_STEP_WALKTHROUGH.md` - Learn EHR integration
- `.env.example` - Template with all options
- `docker-compose.yml` - See how .env is loaded

---

**Questions? Run:** `./aura_config.sh`

