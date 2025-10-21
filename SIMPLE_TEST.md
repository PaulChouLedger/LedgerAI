# Simple Test - Check Model Loading

**Quick test to see if models load once or twice**

---

## 🧪 Run This Test

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI

# Stop containers
docker stop aura-llm aura-rag aura-whisper 2>/dev/null
docker rm aura-llm aura-rag aura-whisper 2>/dev/null

# Start ONLY LLM container and watch logs
docker run -it --rm \
  --network=host \
  --name aura-llm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/shared:/shared \
  aura-llm:latest
```

---

## 👀 What to Watch For

**Models should load ONCE:**

```
[LLM] 🚀 Loading COMPLEX model: /models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
[LLM] ✅ Complex model loaded: ... (took 5.3s)
[LLM] 🚀 Loading SIMPLE model: /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
[LLM] ✅ Simple model loaded: ... (took 1.4s)
[Aura-LLM] 🚀 Starting Aura LLM Container
* Running on http://127.0.0.1:11434

(Container waits for requests)
```

**Then in ANOTHER terminal, send warmup request:**

```bash
# Send test request
curl -X POST http://localhost:11434/chat-tts \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello","session_id":"test"}'
```

**Back in first terminal, watch if models load AGAIN:**

```
If you see:
[LLM] 🚀 Loading COMPLEX model...  ← SECOND time

Then there's a bug in container_rest.py

If you DON'T see it:
Models only load once ✅ (the double loading was from viewing old logs)
```

---

## 🎯 Expected Result

**Container starts:** Models load once (6-7 seconds total)  
**Warmup request:** No model loading (just processes request)  
**Result:** ✅ Models loaded once, container works

---

Try this test and let me know what you see!

