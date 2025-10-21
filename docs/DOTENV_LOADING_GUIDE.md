# .env Loading Guide - Unified Configuration

**Where each component loads the `.env` file from**

---

## 📁 One .env File, Multiple Loaders

```
/Users/rcabello/Documents/GitHub/LedgerAI/
├── .env                           ← ONE file at root
│
├── aura-control/
│   ├── core/
│   │   ├── speaker.py             ← Loads from root .env ✅
│   │   └── main.py                ← Loads from root .env ✅
│   └── server/
│       └── telegram_bot.py        ← Loads from root .env ✅
│
├── llm-medical-container/
│   └── (uses docker-compose)      ← Gets .env via docker ✅
│
├── rag-container/
│   └── (uses docker-compose)      ← Gets .env via docker ✅
│
└── whisper-container/
    └── (uses docker-compose)      ← Gets .env via docker ✅
```

---

## 🔧 How Each Component Loads .env

### 1. **speaker.py** (TTS)

```python
# Location: aura-control/core/speaker.py

# Loads from workspace root (2 levels up)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(workspace_root, '.env')
load_dotenv(dotenv_path)

# Uses:
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
```

**Needs:**
- `ELEVENLABS_API_KEY=sk_your_key`
- `ELEVENLABS_VOICE_ID=default`

---

### 2. **telegram_bot.py** (Telegram)

```python
# Location: aura-control/server/telegram_bot.py

# Loads from workspace root (2 levels up)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(workspace_root, '.env')
load_dotenv(dotenv_path)

# Uses:
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
```

**Needs:**
- `TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHI...`

---

### 3. **Docker Containers** (LLM, RAG, Whisper)

```yaml
# Location: setup/docker-compose.yml

services:
  llm:
    env_file:
      - ../.env  # Loads root .env
    environment:
      - EHR_INTEGRATION_ENABLED=${EHR_INTEGRATION_ENABLED}
      - MODEL_PATH=${MODEL_PATH}
      - N_CTX=${N_CTX}
      # ... etc
  
  rag:
    env_file:
      - ../.env  # Same root .env
    environment:
      - RAG_THRESHOLD=${RAG_THRESHOLD}
      # ... etc
```

**Needs:**
- All LLM settings (`MODEL_PATH`, `N_CTX`, etc.)
- All RAG settings (`RAG_THRESHOLD`, `RAG_TOP_K`, etc.)
- EHR settings (`EHR_INTEGRATION_ENABLED`, etc.)

---

## 📋 Complete Variable Reference

### Variables Used by Python Files (aura-control)

| Variable | Used By | Required? |
|----------|---------|-----------|
| `ELEVENLABS_API_KEY` | speaker.py | ✅ Yes |
| `ELEVENLABS_VOICE_ID` | speaker.py | Optional (defaults to "default") |
| `TELEGRAM_BOT_TOKEN` | telegram_bot.py | Only if using Telegram |

### Variables Used by Docker Containers

| Variable | Used By | Required? |
|----------|---------|-----------|
| `EHR_INTEGRATION_ENABLED` | llm container | Optional (defaults to false) |
| `SYSTMONE_FHIR_URL` | llm container | Only if EHR enabled |
| `MODEL_PATH` | llm container | ✅ Yes |
| `N_CTX` | llm container | ✅ Yes |
| `SIMPLE_MODEL_PATH` | llm container | ✅ Yes |
| `SIMPLE_N_CTX` | llm container | ✅ Yes |
| `RAG_ENABLED` | llm container | Optional (defaults to true) |
| `RAG_THRESHOLD` | rag container | Optional (defaults to 0.3) |
| `RAG_TOP_K` | rag container | Optional (defaults to 3) |

---

## ✅ Fixed Files

### File 1: `speaker.py`
**Status:** ✅ Fixed  
**Loads from:** `/Users/rcabello/Documents/GitHub/LedgerAI/.env`  
**Variables:** `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`

### File 2: `telegram_bot.py`
**Status:** ✅ Fixed  
**Loads from:** `/Users/rcabello/Documents/GitHub/LedgerAI/.env`  
**Variables:** `TELEGRAM_BOT_TOKEN`

### File 3: `main.py`
**Status:** 🔍 Let me check if it needs fixing too

---

## 🧪 Quick Test

### Test 1: Check .env Location

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Should exist at root
ls -la .env

# Should show:
-rw-r--r-- 1 user user 1234 Oct 21 14:30 .env
```

### Test 2: Check API Keys Are Set

```bash
# View current settings
./aura_config.sh show

# Should show:
🔊 TEXT-TO-SPEECH
  ✅ API Key configured    ← Should be green checkmark
  
💬 TELEGRAM BOT
  ○ Not configured         ← Yellow if not using Telegram (OK)
```

### Test 3: Run Aura

```bash
cd aura-control
python main.py

# Should NOT get:
# ❌ AssertionError: Missing ElevenLabs credentials
```

---

## 🔧 If You Still Get Errors

### Error: "Missing ElevenLabs credentials"

**Fix:**
```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Set API key
./aura_config.sh
# Option 5 → Option 1 → Paste your key
```

### Error: "TELEGRAM_BOT_TOKEN not found"

**If you're using Telegram:**
```bash
./aura_config.sh
# Option 6 → Option 1 → Paste your token
```

**If you're NOT using Telegram:**
```bash
# Don't run the Telegram bot!
# Just run main.py without starting telegram_bot.py
```

---

## 📊 Summary

**Both files now fixed to use unified `.env`:**
- ✅ `aura-control/core/speaker.py` → loads from root `.env`
- ✅ `aura-control/server/telegram_bot.py` → loads from root `.env`
- ✅ Docker containers → get `.env` via docker-compose
- ✅ Helpful error messages if keys are missing

**Your .env location:**
```
/Users/rcabello/Documents/GitHub/LedgerAI/.env
```

**Configure with:**
```bash
./aura_config.sh
```

**All your API keys and settings in ONE place!** 🎉

---

Want me to check if `main.py` or any other files also need updating?
