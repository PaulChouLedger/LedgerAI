# Memory Container Quick Start

## ✅ Container is Running!

Your memory container is now running and ready. Here's what to do next:

## 1. Verify It's Working

```bash
# Check health
curl http://localhost:11438/health

# Check stats
curl http://localhost:11438/stats
```

## 2. Start Background Listener (Optional)

The background listener continuously transcribes audio even without wake word:

```bash
curl -X POST http://localhost:11438/start
```

**Note:** This requires audio device access. If you get device conflicts, you can:
- Disable background listener (it's optional)
- Use transcription forwarding from main listener instead (already enabled)

## 3. Test Storage

Trigger a wake word conversation, then check:

```bash
# View recent conversations
curl http://localhost:11438/recent?hours=1&limit=10

# Check stats
curl http://localhost:11438/stats
```

## 4. Monitor Logs

```bash
# Watch memory container logs
docker logs -f memory-container

# Or filter for key events
docker logs -f memory-container 2>&1 | grep -E "📥|💾|🔍|💡|✅"
```

## Current Status

Based on your logs:
- ✅ Container running on port 11438
- ✅ MemoryManager initialized
- ✅ Embedding model loaded
- ✅ ProactiveAnalyzer ready
- ✅ BackgroundListener ready (not started yet)
- ⚠️ No conversations stored yet (expected - will populate as you use Aura)

## What Happens Next

1. **Wake word conversations** → Automatically forwarded to memory container
2. **Storage** → Conversations vectorized and stored
3. **Analysis** → After 5+ conversations, proactive suggestions will be generated
4. **Suggestions** → Spoken via TTS when insights are found

## Enable/Disable

You can toggle memory container in:
**Settings → AI Model Settings → Memory Container** (ON/OFF)

The setting is saved to `~/LedgerAI/data/app_settings.json`

## Troubleshooting

### Container Not Receiving Data

Check if memory forwarding is enabled:
```bash
# In Aura logs, look for:
[Memory] 📤 Forwarding transcription to memory container
```

If you don't see this:
1. Check Settings → AI Model Settings → Memory Container is ON
2. Restart Aura after changing setting

### Background Listener Conflicts

If background listener conflicts with main listener:
```bash
# Stop background listener
curl -X POST http://localhost:11438/stop
```

Transcription forwarding from main listener will still work.

### No Suggestions Appearing

Suggestions require:
- At least 5 conversations stored
- Similar conversations found (similarity >= 0.65)
- 60+ seconds since last suggestion (cooldown)

Check with:
```bash
curl http://localhost:11438/stats
```

## Next Steps

1. ✅ Container is running - you're all set!
2. Use Aura normally - conversations will be stored automatically
3. After 5+ conversations, proactive suggestions will start appearing
4. Monitor logs to see the system in action

Enjoy your proactive AI assistant! 🧠✨

