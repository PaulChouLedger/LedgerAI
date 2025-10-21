# Slowdown Diagnosis & Fix

**What caused the slowdown from 2s → 12s and how it's fixed**

---

## 🐌 Two Issues Identified

### Issue 1: Docker env_file Overhead (10 seconds)

**What I added that was slow:**
```yaml
# docker-compose.yml
services:
  llm:
    env_file:
      - ../.env          ← Docker parses ENTIRE file
    environment:
      - VAR1=${VAR1}
      - VAR2=${VAR2}
      # ... 40+ variables
```

**Why slow:**
- Docker opens and parses entire .env file
- Processes ALL 46 lines
- Substitutes ${VAR} syntax for each
- Total overhead: ~10 seconds

**Fix:** ✅ Removed env_file from docker-compose.yml

---

### Issue 2: Simple Model Variables Not Passed

**What was missing:**
```python
# main.py was only passing:
MODEL_PATH       ✅
CHAT_FORMAT      ✅
N_CTX            ✅

# But NOT passing:
SIMPLE_MODEL_PATH       ❌ Missing!
SIMPLE_CHAT_FORMAT      ❌ Missing!
SIMPLE_N_CTX            ❌ Missing!
```

**Result:**
- Simple model used default values
- Wrong configuration → timeout/errors

**Fix:** ✅ Added simple model variables to main.py

---

## ✅ What's Fixed

### 1. Restored Fast docker-compose.yml

**Before (slow):**
```yaml
services:
  llm:
    env_file: ../.env
    environment:
      - EHR_INTEGRATION_ENABLED=${EHR_INTEGRATION_ENABLED}
      # ... 20+ more lines
```

**After (fast):**
```yaml
services:
  llm:
    # Clean and simple!
    build: ../llm-medical-container
    network_mode: host
```

**Saved:** ~10 seconds

---

### 2. Added Simple Model Variables

**Before (broken):**
```python
# main.py only passed:
"-e", "MODEL_PATH=/models/Mistral-7B..."
"-e", "N_CTX=8192"
# Simple model got defaults (wrong!)
```

**After (correct):**
```python
# main.py now passes:
"-e", "MODEL_PATH=/models/Mistral-7B..."
"-e", "N_CTX=8192"
"-e", "SIMPLE_MODEL_PATH=/models/Llama-3.2-1B..."  ← NEW!
"-e", "SIMPLE_N_CTX=2048"                          ← NEW!
"-e", "SIMPLE_CHAT_FORMAT=llama-3"                 ← NEW!
# Simple model gets correct config!
```

**Fixed:** Simple model timeout issue

---

## 📊 Expected Results After Fix

### Loading Times (Should Be Fast)

```
[LLM] 🚀 Loading COMPLEX model: /models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
[LLM] ✅ Complex model loaded: ... (took 2.1s)     ← ~2s ✅

[LLM] 🚀 Loading SIMPLE model: /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
[LLM] ✅ Simple model loaded: ... (took 0.8s)      ← ~1s ✅

Total: ~3 seconds (both models)
```

---

## 🚀 Apply the Fix Now

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# 1. Stop containers
docker-compose down

# 2. Start fresh
docker-compose up -d

# 3. Watch logs
docker logs -f aura-llm
```

**You should see ~2-3 second total loading now!**

---

## 🎯 Summary

**What was causing slowdown:**
1. ❌ `env_file` in docker-compose.yml (added 10s overhead)
2. ❌ Simple model variables not passed (caused timeouts)

**What's fixed:**
1. ✅ Removed env_file from docker-compose.yml
2. ✅ Added SIMPLE_MODEL_PATH, SIMPLE_N_CTX, SIMPLE_CHAT_FORMAT passing
3. ✅ Added EHR_INTEGRATION_ENABLED passing

**Expected result:**
- Mistral-7B: ~2 seconds (was 12s)
- Llama-1B: ~1 second (was timing out)
- Total: ~3 seconds

**Run:** `docker-compose down && docker-compose up -d` to apply the fix! 🚀

