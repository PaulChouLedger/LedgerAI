# Quick Reference: Unified Configuration

**ONE file controls EVERYTHING: `.env`**

---

## 🎯 The Solution

```
OLD WAY (confusing):                NEW WAY (simple):
┌─────────────────────┐            ┌─────────────────────┐
│ llm-container/.env  │            │                     │
│ rag-container/.env  │────────>   │    .env (root)      │
│ whisper/.env        │            │                     │
│ aura-control/.env   │            │  Controls ALL!      │
└─────────────────────┘            └─────────────────────┘
     4 files!                            1 file!
```

---

## 🚀 Quick Commands

```bash
# Interactive manager (easiest!)
./aura_config.sh

# Toggle EHR on/off
./aura_config.sh ehr on
./aura_config.sh ehr off

# View all settings
./aura_config.sh show

# Edit directly
nano .env
```

---

## 📋 Most Important Settings

### Toggle EHR Integration

```bash
# In .env file:
EHR_INTEGRATION_ENABLED=false    # OFF (normal development)
EHR_INTEGRATION_ENABLED=true     # ON (testing/production)

# Quick toggle:
./aura_config.sh ehr on     # Turn ON
./aura_config.sh ehr off    # Turn OFF
```

### Choose LLM Model

```bash
# In .env file:
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
MODEL_PATH=/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf
MODEL_PATH=/models/your-custom-model.gguf

# Or use interactive:
./aura_config.sh
# Choose option 3 (Configure LLM models)
```

### Adjust RAG Search

```bash
# In .env file:
RAG_THRESHOLD=0.3    # Balanced (recommended)
RAG_THRESHOLD=0.1    # Loose (more results)
RAG_THRESHOLD=0.5    # Strict (fewer results)

RAG_TOP_K=3          # Number of results

# Or use interactive:
./aura_config.sh
# Choose option 4 (Configure RAG search)
```

---

## 🔄 Apply Changes

**IMPORTANT:** After editing `.env`, restart containers!

```bash
# Restart all
docker-compose restart

# Or just what changed:
docker-compose restart llm    # For LLM/EHR settings
docker-compose restart rag    # For RAG settings
```

---

## 📊 What's in .env?

```bash
.env contains:

🏥 EHR Integration
   • Toggle on/off
   • FHIR server URL
   • NHS credentials

🧠 LLM Models
   • Complex model path
   • Simple model path
   • Temperature, context size

📚 RAG Search
   • Threshold
   • Top K results
   • Phonetic matching

🔊 TTS
   • ElevenLabs API key

🐛 Debug
   • Log level
   • Debug mode
```

---

## ✅ Quick Checklist

Before you start developing:

- [ ] `.env` file exists (copy from `.env.example`)
- [ ] `EHR_INTEGRATION_ENABLED=false` (for normal development)
- [ ] LLM model paths correct
- [ ] Restart containers after changes

---

## 🆘 Troubleshooting

```bash
# Settings not working?
docker-compose restart

# Don't know what to set?
./aura_config.sh

# Want to see current values?
./aura_config.sh show

# File missing?
cp .env.example .env
```

---

## 🎓 Full Documentation

For complete details, see:
- `UNIFIED_CONFIG_GUIDE.md` - Complete configuration guide
- `EHR_TOGGLE_GUIDE.md` - EHR integration toggle
- `.env.example` - All available options

---

**Remember:** ONE file (`.env`) controls EVERYTHING! 🎉

