# Aura Setup Checklist

**Everything you need to configure before running Aura**

---

## ✅ Pre-Flight Checklist

### 1. **Create .env File** (1 minute)

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Check if .env exists
ls -la .env

# If not, create it
cp .env.example .env
```

---

### 2. **Configure ElevenLabs API Key** (2 minutes) - REQUIRED

```bash
# Method 1: Interactive
./aura_config.sh
# Choose option 5 (Configure TTS)
# Choose option 1 (Set API key)
# Paste your key from https://elevenlabs.io/

# Method 2: Direct edit
nano .env
# Change: ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
# To:     ELEVENLABS_API_KEY=sk_your_real_key_here
```

**Without this, Aura won't start!**

---

### 3. **Configure Telegram Bot** (Optional, 2 minutes)

**Only if using Telegram interface:**

```bash
./aura_config.sh
# Choose option 6 (Configure Telegram bot)
# Get token from @BotFather on Telegram
```

**Skip if only using voice interface!**

---

### 4. **Configure LLM Models** (Optional)

Check if model paths are correct for your system:

```bash
./aura_config.sh
# Choose option 3 (Configure LLM models)
# Verify paths match your actual model files
```

**Default values usually work!**

---

### 5. **Configure RAG** (Optional)

Adjust search sensitivity if needed:

```bash
./aura_config.sh
# Choose option 4 (Configure RAG search)
# Default RAG_THRESHOLD=0.3 is usually good
```

---

### 6. **EHR Integration** (Optional, for future)

**Keep OFF for now:**

```bash
./aura_config.sh show

# Verify:
🏥 EHR INTEGRATION
  ○ Disabled    ← Should be OFF
```

**Turn ON only when ready to test EHR!**

---

## 🚀 Start Aura

After configuration:

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI/aura-control
python main.py
```

**Should start without errors!**

---

## 📋 Minimum Required Settings

Your `.env` MUST have:

```bash
# Required for TTS (voice)
ELEVENLABS_API_KEY=sk_your_real_key_here

# Required for LLM
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
N_CTX=8192
SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SIMPLE_N_CTX=2048

# Optional but recommended
ELEVENLABS_VOICE_ID=default
RAG_ENABLED=true
RAG_THRESHOLD=0.3
EHR_INTEGRATION_ENABLED=false
```

---

## 🆘 Common Errors & Fixes

### Error 1: "Missing ElevenLabs credentials"

```bash
AssertionError: Missing ElevenLabs credentials
```

**Fix:**
```bash
./aura_config.sh
# Option 5 → Option 1 → Enter your API key
```

---

### Error 2: "TELEGRAM_BOT_TOKEN not found"

```bash
RuntimeError: Missing Telegram bot token!
```

**Fix (if using Telegram):**
```bash
./aura_config.sh
# Option 6 → Option 1 → Enter your token
```

**Fix (if NOT using Telegram):**
```bash
# Don't run telegram_bot.py
# Just run main.py - it's optional!
```

---

### Error 3: Model file not found

```bash
[LLM] ❌ Model file not found: /models/...
```

**Fix:**
```bash
./aura_config.sh
# Option 3 → Check/update model paths
```

---

## ✨ Quick Setup (3 minutes)

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# 1. Create .env (if needed)
cp .env.example .env

# 2. Configure TTS (REQUIRED)
./aura_config.sh
# Choose: 5 → 1 → Paste API key → Back

# 3. View settings
./aura_config.sh show

# Should see:
# ✅ API Key configured

# 4. Start Aura
cd aura-control
python main.py
```

---

## 📊 Files That Load .env

| File | Location | Loads From | Variables Used |
|------|----------|------------|----------------|
| `speaker.py` | aura-control/core/ | Root .env ✅ | ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID |
| `telegram_bot.py` | aura-control/server/ | Root .env ✅ | TELEGRAM_BOT_TOKEN |
| `main.py` | aura-control/core/ | Root .env ✅ | MODEL_PATH, N_CTX, CHAT_FORMAT |
| All Docker containers | Via docker-compose | Root .env ✅ | All settings |

**Everyone uses the same .env at the root level!**

---

## 🎯 Summary

**ALL components now load from:**
```
/Users/rcabello/Documents/GitHub/LedgerAI/.env
```

**Configure with:**
```bash
./aura_config.sh
```

**Required settings:**
- ✅ `ELEVENLABS_API_KEY` (must be real key, not placeholder)
- ✅ Model paths correct for your system

**Optional settings:**
- Telegram bot token (only if using Telegram)
- EHR integration (turn on when ready)
- RAG parameters (defaults are good)

**Everything configured in ONE place!** 🎉

