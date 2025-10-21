# Startup Timing Fix - LLM Container Loading

**Fix for "Connection refused" errors during Aura startup**

---

## ❌ The Problem

When starting Aura, you might see:

```
[LLM] ✅ Complex model loaded: /models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf (took 12.1s)
[LLM] ✅ Simple model loaded: /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf (took 5.2s)

[Aura] ⚠️ LLM warm-up attempt 1 failed: Connection refused
[Aura] ⚠️ LLM warm-up attempt 2 failed: Connection refused
[Aura] ⚠️ LLM warm-up attempt 3 failed: Connection refused
[Aura] ❌ LLM warm-up failed. Aborting.
```

**Why this happens:**
- Models are loading (slow - takes 10-20 seconds)
- Flask API hasn't started listening yet
- Aura tries to connect too early
- Connection refused!

---

## ✅ The Fix

**Updated `main.py` with better timing:**

### 1. **Wait for Health Endpoint**

```python
# New logic in main.py:

# Wait up to 60 seconds for LLM API to respond
print("[Aura] ⏳ Waiting for LLM Flask API to be ready...")
print("[Aura] 💡 Models are loading in background (Mistral-7B + Llama-1B)...")

for attempt in range(12):  # 12 * 5 = 60 seconds max
    try:
        response = requests.get("http://localhost:11434/health", timeout=2)
        if response.status_code == 200:
            print(f"[Aura] ✅ LLM API ready after {(attempt + 1) * 5} seconds")
            break
    except:
        pass
    
    time.sleep(5)
```

### 2. **Simplified Warm-up**

```python
# warm_up_llm() now just does a test request
# No duplicate waiting - health check already done!
```

---

## 🕐 Timeline Explained

### What Happens During Startup

```
0s:   Docker run aura-llm
      ↓
2s:   Container starts
      Python starts loading
      ↓
5s:   Flask app starting
      Starting to load Mistral-7B model...
      ↓
10s:  Mistral-7B still loading...
      ↓
15s:  Mistral-7B loaded! ✅
      Starting to load Llama-1B model...
      ↓
20s:  Llama-1B loaded! ✅
      Flask app NOW listening on port 11434
      /health endpoint ready
      ↓
25s:  Aura's health check succeeds
      "LLM API ready after 25 seconds"
      ↓
30s:  Warm-up test request succeeds
      "LLM warm-up complete"
      ↓
      ✅ Ready to use!
```

**Total time:** ~25-30 seconds for LLM to be fully ready

---

## 🔧 What Changed in main.py

### Before (Too Impatient)

```python
# Started containers
time.sleep(5)  # Only 5 seconds!
warm_up_llm()  # Try to connect - FAILS!
```

### After (Patient)

```python
# Started containers
# Wait for health endpoint (up to 60 seconds)
for attempt in range(12):
    check http://localhost:11434/health
    if success:
        break
    sleep(5)

# Now warm up (API is ready)
warm_up_llm()  # SUCCESS!
```

---

## 📊 Health Check Details

### What `/health` Endpoint Returns

```json
{
  "status": "ok",
  "service": "aura-llm",
  "models": {
    "complex_loaded": true,     ← Mistral-7B ready
    "simple_loaded": true,      ← Llama-1B ready
    "complex_path": "/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
    "simple_path": "/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  }
}
```

**Aura waits for `status: "ok"` before proceeding!**

---

## 🧪 Testing the Fix

### Normal Startup (Should Work Now)

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI/aura-control
python main.py
```

**Expected output:**
```
[Aura] 🚀 Starting all containers...
[Aura] ✅ All containers started successfully!
[Aura] ⏳ Waiting for LLM Flask API to be ready...
[Aura] 💡 Models are loading in background (Mistral-7B + Llama-1B)...
[Aura] ⏳ Still waiting for LLM API... (5s elapsed)
[Aura] ⏳ Still waiting for LLM API... (10s elapsed)
[Aura] ⏳ Still waiting for LLM API... (15s elapsed)
[Aura] ⏳ Still waiting for LLM API... (20s elapsed)
[Aura] ✅ LLM API ready after 25 seconds
[Aura] 🧪 Testing LLM with warm-up request...
[Aura] ✅ LLM warm-up complete.
[Aura] ✅ Core services started successfully!
```

---

## ⏱️ How Long Does Startup Take?

**Total startup time:**

| Component | Time | What's Happening |
|-----------|------|------------------|
| Docker containers start | 2-5s | Launching containers |
| Models load | 15-25s | Loading Mistral-7B + Llama-1B |
| Flask API ready | 1-2s | REST API starts listening |
| Warm-up test | 2-3s | Test request |
| **Total** | **~25-35s** | **Normal for your setup** |

**This is expected!** Large models take time to load.

---

## 🆘 If It Still Fails

### Check 1: Container Logs

```bash
# Check what's happening in LLM container
docker logs aura-llm

# Look for:
✅ [LLM] ✅ Complex model loaded
✅ [LLM] ✅ Simple model loaded
✅ [Aura-LLM] 🚀 Starting Aura LLM Container

# Or errors:
❌ [LLM] ❌ Model file not found
❌ Out of memory
```

### Check 2: Port Availability

```bash
# Check if port 11434 is available
netstat -tuln | grep 11434

# Or
lsof -i :11434
```

### Check 3: Manual Health Check

```bash
# After models load, test health endpoint manually
curl http://localhost:11434/health

# Should return:
{"status":"ok","models":{"complex_loaded":true,"simple_loaded":true}}
```

---

## 🎯 Summary

**What was fixed:**
- ✅ Better wait logic (up to 60 seconds for models to load)
- ✅ Health check before warm-up
- ✅ Clear progress messages
- ✅ Simplified warm-up (no duplicate waiting)

**Expected behavior:**
- Models load (takes 15-25 seconds - normal!)
- Health check succeeds
- Warm-up succeeds
- Aura starts!

**If startup takes 25-35 seconds, that's NORMAL for large models!**

---

Try running Aura again and watch for the progress messages:

```bash
python main.py
```

It should wait patiently for the models to load now! 🚀

