# TTS Setup Guide - ElevenLabs Configuration

**Quick fix for "Missing ElevenLabs credentials" error**

---

## ❌ The Error

```
AssertionError: Missing ElevenLabs credentials
```

or

```
RuntimeError: Missing ElevenLabs API key!
```

**This means:** You need to configure your ElevenLabs API key for text-to-speech!

---

## ✅ Quick Fix (2 minutes)

### Method 1: Using Config Manager (Easiest)

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Run config manager
./aura_config.sh

# Choose option 5 (Configure TTS)
Enter choice [0-9]: 5

# Choose option 1 (Set API key)
Choice [1-4]: 1

# Paste your API key
Enter ElevenLabs API key: sk_your_api_key_here

✅ API key saved
```

### Method 2: Edit .env Directly

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Edit .env file
nano .env

# Find this line:
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Change to:
ELEVENLABS_API_KEY=sk_your_actual_api_key_here
ELEVENLABS_VOICE_ID=default

# Save and exit (Ctrl+X, Y, Enter)
```

---

## 🔑 Get Your API Key

### Step 1: Sign up at ElevenLabs

Go to: **https://elevenlabs.io/**

### Step 2: Get Your API Key

1. Sign in to your account
2. Click your profile (top right)
3. Go to "Profile" or "API Keys"
4. Copy your API key (starts with `sk_`)

### Step 3: Add to Aura

Use either Method 1 or Method 2 above to add your key!

---

## 📋 Configuration Details

### Variable Names

Aura uses these variable names (backwards compatible with old names):

```bash
# New unified names (preferred):
ELEVENLABS_API_KEY=sk_your_key_here
ELEVENLABS_VOICE_ID=default

# Old names (still work for backwards compatibility):
ELEVEN_API_KEY=sk_your_key_here
ELEVEN_VOICE_ID=default
```

**Use the NEW names (ELEVENLABS_*) in your .env file!**

### Where is .env Located?

```
/Users/rcabello/Documents/GitHub/LedgerAI/.env  ← Root level!
```

**NOT** in:
- ❌ `aura-control/.env`
- ❌ `llm-medical-container/.env`
- ❌ `rag-container/.env`

**The unified .env at the ROOT level is used by everything!**

---

## 🧪 Test It Works

### Step 1: Check .env File

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Check if API key is set
grep ELEVENLABS_API_KEY .env

# Should show:
ELEVENLABS_API_KEY=sk_your_key_here (not "your_elevenlabs_api_key_here")
```

### Step 2: Test Aura

```bash
cd aura-control
python main.py
```

**Should start without error!**

If you still see the error, the API key is not set correctly.

---

## 🔍 Troubleshooting

### Problem 1: Still getting "Missing credentials" error

**Check:**

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# 1. Does .env exist?
ls -la .env

# 2. Is API key set?
grep ELEVENLABS_API_KEY .env

# 3. Is it the placeholder or real key?
# Should be: ELEVENLABS_API_KEY=sk_XXXXXXXXX
# NOT:       ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

**Solution:**
```bash
# Use config manager to set it
./aura_config.sh
# Choose option 5, then option 1
```

---

### Problem 2: .env file doesn't exist

**Solution:**
```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Create from template
cp .env.example .env

# Then configure
./aura_config.sh
```

---

### Problem 3: API key not recognized

**Make sure you're using the NEW variable name:**

```bash
# Correct (unified config):
ELEVENLABS_API_KEY=sk_your_key

# Old name (still works but use new one):
ELEVEN_API_KEY=sk_your_key
```

**Update your .env to use `ELEVENLABS_API_KEY`!**

---

## 📖 How It Works

### Where speaker.py Loads From

```python
# speaker.py now loads from workspace root .env
workspace_root = /Users/rcabello/Documents/GitHub/LedgerAI/
dotenv_path = workspace_root/.env
load_dotenv(dotenv_path)
```

**Location hierarchy:**
```
/Users/rcabello/Documents/GitHub/LedgerAI/
├── .env                    ← Loads from HERE!
└── aura-control/
    └── core/
        └── speaker.py      ← This file
```

---

## 🎯 Complete .env Example

Your `.env` should have:

```bash
# ============================================================================
# AURA UNIFIED CONFIGURATION
# ============================================================================

# 🏥 EHR INTEGRATION
EHR_INTEGRATION_ENABLED=false
SYSTMONE_FHIR_URL=https://hapi.fhir.org/baseR4

# 🧠 LLM MODELS
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
N_CTX=8192

# 📚 RAG
RAG_ENABLED=true
RAG_THRESHOLD=0.3

# 🔊 TEXT-TO-SPEECH (IMPORTANT!)
ELEVENLABS_API_KEY=sk_your_actual_key_here    ← MUST be set!
ELEVENLABS_VOICE_ID=default

# 💬 TELEGRAM BOT (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# ... rest of config ...
```

---

## ✅ Success Checklist

After configuration, you should have:

- [x] `.env` file exists at `/Users/rcabello/Documents/GitHub/LedgerAI/.env`
- [x] `ELEVENLABS_API_KEY=sk_XXXXXXXXX` (real key, not placeholder)
- [x] `ELEVENLABS_VOICE_ID=default` (or your chosen voice)
- [x] `python main.py` starts without error
- [x] TTS works when Aura speaks

---

## 🚀 Quick Commands

```bash
# Set API key (easiest)
cd /Users/rcabello/Documents/GitHub/LedgerAI
./aura_config.sh
# Choose option 5

# Check if set
grep ELEVENLABS_API_KEY .env

# Test Aura
cd aura-control
python main.py
```

---

## 📞 Common Questions

**Q: Where do I get an API key?**  
A: https://elevenlabs.io/ (sign up, go to Profile → API Keys)

**Q: Is it free?**  
A: ElevenLabs has a free tier with limited characters per month

**Q: Which voice should I use?**  
A: Start with `default`, or choose from the config manager (option 5 → option 2)

**Q: Do I need to restart Docker?**  
A: No! TTS is in the main app (aura-control), not in Docker containers

**Q: Can I use a different TTS service?**  
A: Currently only ElevenLabs is supported. You'd need to modify `speaker.py` for alternatives.

---

**Quick fix:** `./aura_config.sh` → Option 5 → Set API key → Done! 🎉

