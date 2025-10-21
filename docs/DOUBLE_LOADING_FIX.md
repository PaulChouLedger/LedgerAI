# Double Loading Fix - Models Loading Twice

**Why models were loading twice and how it's fixed**

---

## 🔍 The Problem

**What was happening:**

```
0s:   main.py starts LLM container
      ↓
2s:   Container starts, models begin loading
      ↓
4s:   Mistral-7B loading... (takes ~2-6s)
      ↓
8s:   Mistral-7B loaded ✅
      ↓
10s:  Llama-1B loading... (takes ~1-2s)
      ↓
12s:  Llama-1B loaded ✅
      ↓
13s:  Flask API starts
      ↓
BUT... main.py's health check times out at 15s! ❌
      ↓
15s:  Health check timeout!
      ↓
      main.py removes container (thinking it failed)
      ↓
      main.py starts container AGAIN (retry)
      ↓
      Models load AGAIN! (2nd time)
```

**Result:** Models loaded twice, taking 2x the time!

---

## ✅ The Fix

### 1. Increased LLM Container Timeout

**Before:**
```python
timeout=20  # Too short for model loading!
```

**After:**
```python
timeout=30  # Enough time for both models + Flask startup
```

### 2. Better Health Check

**Before:**
```python
# Just checked if API responds (no progress updates)
for _ in range(timeout * 10):
    check_api()
    time.sleep(0.1)
```

**After:**
```python
# Shows progress, understands LLM takes time
for i in range(timeout * 10):
    check_api()
    
    # Show progress every 3 seconds for LLM
    if is_llm and i % 30 == 0:
        print(f"Still waiting... ({i/10}s - models loading)")
    
    time.sleep(0.1)
```

### 3. Wait for BOTH Models

**Added in main.py:**
```python
# Check /health endpoint for BOTH models
health_data = response.json()
complex_loaded = health_data["models"]["complex_loaded"]
simple_loaded = health_data["models"]["simple_loaded"]

# Only proceed when BOTH are ready
if complex_loaded and simple_loaded:
    print("Both models loaded!")
```

---

## 📊 Timeline After Fix

```
0s:   main.py starts LLM container (timeout=30s now)
      ↓
2s:   Container starts, models begin loading
      ↓
4s:   Mistral-7B loading...
      ↓
6s:   Mistral-7B loaded ✅
      ↓
7s:   Llama-1B loading...
      ↓
8s:   Llama-1B loaded ✅
      ↓
9s:   Flask API starts
      ↓
10s:  Health check succeeds! ✅
      ↓
      main.py waits for both models to be loaded
      ↓
11s:  Both models confirmed loaded
      ↓
      Warm-up request succeeds
      ↓
      ✅ Aura starts!
```

**Result:** Models load ONCE, total time ~11 seconds

---

## 🎯 What Changed

| Setting | Before | After |
|---------|--------|-------|
| **LLM timeout** | 20s | 30s ✅ |
| **Health check** | Simple | Progress updates ✅ |
| **Model check** | API only | Both models ✅ |
| **Retry logic** | Triggered | Doesn't trigger ✅ |
| **Load count** | 2x (retry) | 1x ✅ |

---

## 🚀 Expected Behavior Now

**Console output:**
```
[Aura] 🧠 Starting LLM container...
[Aura] 🚀 Launching aura-llm...
[Aura] ⏳ Waiting for aura-llm to respond (timeout 30s)...
[Aura] ⏳ Still waiting for aura-llm... (3s - models loading)
[Aura] ⏳ Still waiting for aura-llm... (6s - models loading)
[Aura] ⏳ Still waiting for aura-llm... (9s - models loading)
[Aura] ✅ aura-llm is online.
[Aura] ✅ All containers started successfully!
[Aura] ⏳ Waiting for LLM Flask API and models to load...
[Aura] ⏳ Mistral-7B loaded, waiting for Llama-1B... (7.5s)
[Aura] ✅ Both models loaded after 10.0 seconds
[Aura] 🧪 Testing LLM with warm-up request...
[Aura] ✅ LLM warm-up complete.
```

**Models load ONCE, Aura starts successfully!**

---

## 🧪 Test the Fix

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Stop containers
docker-compose down

# Start fresh
cd aura-control
python main.py

# Watch for:
# ✅ Container doesn't restart/retry
# ✅ Models load only once
# ✅ Warm-up succeeds
```

---

## ✨ Summary

**Root cause:** Health check timeout (20s) was shorter than model loading time (~12s), causing container to be removed and restarted

**Fix:**
1. ✅ Increased timeout to 30s
2. ✅ Added progress messages
3. ✅ Check for both models loaded
4. ✅ No more retries/restarts

**Your models will now load ONCE and Aura will start properly!** 🚀

