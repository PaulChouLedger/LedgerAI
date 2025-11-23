# Memory Container Diagnostic Guide

## Quick Check: Is Memory Container Working?

### 1. Check if Memory Container is Running

```bash
docker ps | grep memory
```

Expected output:
```
CONTAINER ID   IMAGE              STATUS
abc123def456   memory-container   Up 2 minutes
```

If not running:
```bash
cd setup
docker-compose up -d memory
```

### 2. Check Memory Container Health

```bash
curl http://localhost:11438/health
```

Expected output:
```json
{
  "status": "healthy",
  "service": "memory-container",
  "listener_enabled": false,
  "memory_stats": {
    "total_conversations": 0,
    "total_embeddings": 0
  }
}
```

### 3. Check if Memory Container is Receiving Data

```bash
# Watch memory container logs in real-time
docker logs -f memory-container
```

Then trigger a wake word and look for:
- `📥 Received conversation to store`
- `💾 Storing conversation`
- `✅ Stored conversation`

### 4. Check Main Listener Logs

Look for memory forwarding messages:
```bash
# In your main Aura logs, look for:
[Memory] 📤 Forwarding transcription to memory container
[Memory] ✅ Forwarded to memory container
```

If you see:
- `[Memory] Memory forwarding disabled` → MEMORY_ENABLED=false
- `[Memory] Memory container unavailable` → Container not running or wrong URL
- Nothing at all → Import failed or MEMORY_AVAILABLE=False

## Common Issues

### Issue 1: Memory Container Not Running

**Symptoms:**
- No memory container in `docker ps`
- No logs from memory container

**Fix:**
```bash
cd setup
docker-compose up -d memory
docker logs memory-container
```

### Issue 2: MEMORY_ENABLED Disabled

**Symptoms:**
- Logs show: `[Memory] Memory forwarding disabled`

**Fix:**
```bash
# Check environment variable
echo $MEMORY_ENABLED

# Enable it
export MEMORY_ENABLED=true

# Or add to .env file
echo "MEMORY_ENABLED=true" >> .env
```

### Issue 3: Import Failed Silently

**Symptoms:**
- No memory logs at all
- `MEMORY_AVAILABLE` might be False

**Check:**
```python
# In Python console or add to listener.py temporarily
try:
    from memory_integration import forward_to_memory
    print("✅ Memory integration imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

**Fix:**
- Ensure `memory_integration.py` is in `aura-control/core/`
- Check Python path is correct

### Issue 4: Wrong URL

**Symptoms:**
- `[Memory] Memory container unavailable: Connection refused`

**Check:**
```bash
# Test if memory container is accessible
curl http://localhost:11438/health
```

**Fix:**
```bash
# Set correct URL
export MEMORY_CONTAINER_URL=http://localhost:11438

# Or in .env
echo "MEMORY_CONTAINER_URL=http://localhost:11438" >> .env
```

### Issue 5: Log Level Too High

**Symptoms:**
- Memory logs exist but are at DEBUG level
- Only seeing INFO/ERROR, not DEBUG messages

**Fix:**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Or check current log level in memory_integration.py
# Ensure logger is configured correctly
```

## Test Memory Container Manually

### Test 1: Store Conversation

```bash
curl -X POST http://localhost:11438/store \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I treat pneumonia?", "source": "test"}'
```

Expected response:
```json
{
  "status": "stored",
  "conversation_id": "abc123..."
}
```

Check logs:
```bash
docker logs memory-container | tail -20
```

Should see:
```
[memory-container] 📥 Received conversation to store
[MemoryManager] 💾 Storing conversation...
[MemoryManager] ✅ Stored conversation
```

### Test 2: Check Stats

```bash
curl http://localhost:11438/stats
```

Should show:
```json
{
  "total_conversations": 1,
  "total_embeddings": 1,
  "index_size": 1
}
```

### Test 3: Search

```bash
curl -X POST http://localhost:11438/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pneumonia", "k": 5, "threshold": 0.5}'
```

## Enable Debug Logging

### In Main Listener

Add to your startup script or environment:
```bash
export LOG_LEVEL=DEBUG
```

Or check if memory_integration logger is configured:
```python
# In memory_integration.py, ensure logger is set up
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Add this temporarily
```

### In Memory Container

```bash
# Set in docker-compose.yml
environment:
  - LOG_LEVEL=DEBUG

# Or when running
LOG_LEVEL=DEBUG docker-compose up memory
```

## Expected Log Flow

When wake word is triggered, you should see:

**Main Listener:**
```
[Memory] 📤 Forwarding transcription to memory container (source: wake_word)
[Memory] ✅ Forwarded to memory container (ID: abc123, 0.023s)
```

**Memory Container:**
```
[memory-container] 📥 Received conversation to store (source: wake_word)
[MemoryManager] 💾 Storing conversation...
[MemoryManager] ✅ Stored conversation (ID: abc123)
[memory-container] 🔍 Analyzing conversation for suggestions...
```

## Quick Diagnostic Script

Create `test_memory.sh`:
```bash
#!/bin/bash

echo "=== Memory Container Diagnostic ==="
echo ""

echo "1. Checking if memory container is running..."
docker ps | grep memory || echo "❌ Memory container not running"

echo ""
echo "2. Checking health endpoint..."
curl -s http://localhost:11438/health | jq '.' || echo "❌ Cannot reach memory container"

echo ""
echo "3. Checking stats..."
curl -s http://localhost:11438/stats | jq '.' || echo "❌ Cannot get stats"

echo ""
echo "4. Testing store endpoint..."
curl -s -X POST http://localhost:11438/store \
  -H "Content-Type: application/json" \
  -d '{"text": "test conversation", "source": "diagnostic"}' | jq '.' || echo "❌ Store failed"

echo ""
echo "5. Checking environment..."
echo "MEMORY_ENABLED: ${MEMORY_ENABLED:-not set}"
echo "MEMORY_CONTAINER_URL: ${MEMORY_CONTAINER_URL:-not set}"

echo ""
echo "=== Diagnostic Complete ==="
```

Run it:
```bash
chmod +x test_memory.sh
./test_memory.sh
```

