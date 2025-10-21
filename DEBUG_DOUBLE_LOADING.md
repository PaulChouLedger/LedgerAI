# Debug: Why Models Appear to Load Twice

**Diagnostic steps to find why you're seeing models load twice**

---

## 🔍 Possible Causes

### 1. Multiple Containers Running

**Check if you have duplicate containers:**

```bash
docker ps -a | grep aura-llm
```

**Should see:**
```
aura-llm   Up X minutes   (only ONE)
```

**If you see TWO containers:**
```
aura-llm      Up X minutes
aura-llm-old  Up Y minutes
```

**Then both are outputting logs!**

**Fix:**
```bash
docker stop $(docker ps -a -q --filter name=aura-llm)
docker rm $(docker ps -a -q --filter name=aura-llm)
```

---

### 2. Logs from Multiple Runs

**Are you looking at cumulative logs?**

The logs might show:
- First run (yesterday): Models load
- Second run (today): Models load again

**But they're from DIFFERENT container startups!**

**To verify:**
```bash
# Clear old logs
docker-compose down

# Start fresh with timestamp
docker-compose up -d
date  # Note the time

# Run Aura
python main.py

# Check logs from ONLY this startup
docker logs aura-llm --since="1m"  # Only last 1 minute
```

---

### 3. Container Auto-Restart Policy

**Check restart policy:**

```bash
docker inspect aura-llm | grep -i restart
```

**Should see:**
```
"RestartPolicy": {"Name": "no"}
```

**If you see "always" or "unless-stopped":**
```
"RestartPolicy": {"Name": "always"}  ← Container auto-restarts!
```

**Then Docker is restarting the container automatically!**

**Fix:** Change in docker-compose.yml or remove --restart flag

---

### 4. Main.py Called Twice

**Are you running main.py twice accidentally?**

```bash
# Check for multiple main.py processes
ps aux | grep "python.*main.py"
```

**Should see:**
```
Only ONE process
```

**If multiple:**
```bash
# Kill old ones
pkill -f "python.*main.py"

# Start fresh
python main.py
```

---

### 5. Dockerfile Loads Models at Build Time

**Check if models load during docker build:**

```bash
grep "Llama(" llm-medical-container/Dockerfile
```

**Should be:**
```
(nothing found - models should NOT load in Dockerfile)
```

**If models load in Dockerfile:**
- They load once during build
- Then again at runtime

---

## 🧪 Clean Test Procedure

**To get definitive answer:**

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# 1. Stop everything
docker-compose down
docker rm -f $(docker ps -aq) 2>/dev/null

# 2. Clear Docker logs
docker system prune -f

# 3. Start ONE container manually
docker run -it --rm \
  --network=host \
  --name aura-llm-test \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/shared:/shared \
  aura-llm:latest

# Watch for "Loading COMPLEX model" message
# Should appear ONCE
# If it appears TWICE, there's a bug in container_rest.py
```

---

## 🎯 What to Look For

In your **container logs**, models should load in this order:

```
[Clinician] ✅ Medical RAG available
[Clinician] ✅ Adaptive diagnostic engine imported
[Clinician] ✅ Loaded 486 medical terms
[LLM] 🚀 Loading COMPLEX model: ...        ← Line A (1st time - EXPECTED)
[LLM] ✅ Complex model loaded: (5.3s)
[LLM] 🚀 Loading SIMPLE model: ...
[LLM] ✅ Simple model loaded: (1.4s)
[Aura-LLM] 🚀 Starting Aura LLM Container
* Running on http://127.0.0.1:11434

(Container is now running and waiting for requests)

... later, when warmup request arrives ...

[Aura-LLM] 💬 Session: warmup
[Clinician] 🩺 Starting unified medical session
[Engine] 📚 LOADING MEDICAL GUIDELINES  ← Guideline loading (EXPECTED)
[Clinician] ✅ Adaptive engine initialized

(NO second model loading should happen here!)
```

---

## ❓ Key Question

**After you see:**
```
[Aura-LLM] 🚀 Starting Aura LLM Container
* Running on http://127.0.0.1:11434
```

**Do you see ANOTHER:**
```
[LLM] 🚀 Loading COMPLEX model: ...  ← Second time?
```

**If YES:**
- Something in the code is reloading models
- OR container is restarting
- OR there are two containers

**If NO:**
- The logs you showed are from two separate runs
- Models only load once per container start ✅

---

Let me know what you see after the singleton fix and I'll investigate further!

