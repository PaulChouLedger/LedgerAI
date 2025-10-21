# Docker Compose Configuration Note

## Why docker-compose.yml Doesn't Use env_file

**Original approach was FASTER!**

---

## The Issue

**When we added:**
```yaml
services:
  llm:
    env_file:
      - ../.env
    environment:
      - EHR_INTEGRATION_ENABLED=${EHR_INTEGRATION_ENABLED}
      # ... 20+ more variables ...
```

**Result:** Model loading slowed from 2s → 12s! ❌

**Why:** Docker has to parse .env file and inject all variables at container start

---

## The Solution

**Keep docker-compose.yml simple:**
```yaml
services:
  llm:
    # No env_file!
    # No environment section!
    # Let main.py handle it
```

**How configuration works instead:**

### 1. LLM Container (via main.py)

```python
# main.py passes only needed variables:
cmd = [
    "docker", "run", "-d",
    "-e", f"MODEL_PATH={model_path}",      # Only if set in .env
    "-e", f"CHAT_FORMAT={chat_format}",    # Only if set in .env
    "-e", f"N_CTX={n_ctx}",                # Only if set in .env
    "aura-llm:latest"
]
```

**Selective passing = FAST!**

### 2. Python Files (speaker.py, telegram_bot.py)

```python
# Load .env directly:
load_dotenv('/path/to/root/.env')
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
```

**Direct loading = FAST!**

---

## Configuration Flow

```
.env file at root
    ├── speaker.py reads directly (Python)
    ├── telegram_bot.py reads directly (Python)
    ├── main.py reads, then passes to containers (selective)
    └── Containers receive only what they need (fast!)
```

---

## Summary

**✅ Original approach (fast):**
- main.py selectively passes variables to Docker
- Only 3-5 variables passed per container
- Fast startup (2s)

**❌ env_file approach (slow):**
- Docker loads entire .env file
- Parses 40+ variables
- Injects all into container
- Slow startup (12s)

**Solution:** Keep docker-compose.yml simple, let main.py handle variable passing!

---

**Your 2-second load time is back!** 🚀

