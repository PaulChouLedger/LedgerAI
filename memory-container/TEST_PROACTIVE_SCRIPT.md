# Proactive Memory Component Test Script

This script helps you test the proactive memory component by having conversations that build context and trigger proactive suggestions.

## Prerequisites

1. **Memory container running**: `docker ps | grep memory`
2. **At least 5 conversations needed** for proactive suggestions
3. **Similarity threshold**: 0.65 (conversations need to be similar)
4. **Cooldown**: 60 seconds between suggestions

## Test Script

### Phase 1: Build Context (5+ Conversations)

Start with these conversations to build up memory:

#### Conversation 1 (Medical Context)
**You say:** "Hey Aura, I'm treating a patient with pneumonia. What antibiotics should I consider?"

**Expected:** Aura responds with antibiotic recommendations.

---

#### Conversation 2 (Similar Medical Context)
**You say:** "Hey Aura, I have another pneumonia case. The patient is allergic to penicillin. What are my options?"

**Expected:** Aura responds with alternative antibiotics.

---

#### Conversation 3 (Related Medical Context)
**You say:** "Hey Aura, for community-acquired pneumonia, what's the first-line treatment?"

**Expected:** Aura responds with first-line treatment options.

---

#### Conversation 4 (Similar Topic)
**You say:** "Hey Aura, what's the recommended duration for treating pneumonia with antibiotics?"

**Expected:** Aura responds with treatment duration guidelines.

---

#### Conversation 5 (Related Topic)
**You say:** "Hey Aura, my patient with pneumonia isn't improving after 3 days. What should I do?"

**Expected:** Aura responds with next steps.

---

### Phase 2: Trigger Proactive Suggestion

After 5+ conversations about pneumonia, try a related but slightly different topic:

#### Conversation 6 (Should Trigger Proactive Suggestion)
**You say:** "Hey Aura, I'm seeing a patient with a respiratory infection."

**Expected Behavior:**
1. Aura responds to your current question
2. **Proactive suggestion should appear** (after 60-second cooldown):
   - "Excuse me, have you thought about..."
   - "Based on our previous conversations about pneumonia..."
   - Suggestion relates to similar cases you've discussed

---

## Alternative Test: Project Management Context

If medical context doesn't work, try a different domain:

### Phase 1: Build Context

1. **You say:** "Hey Aura, I'm having trouble managing my project timeline."

2. **You say:** "Hey Aura, my project is behind schedule. How can I catch up?"

3. **You say:** "Hey Aura, I need help prioritizing tasks in my project."

4. **You say:** "Hey Aura, my team is struggling with project deadlines."

5. **You say:** "Hey Aura, what's the best way to track project progress?"

### Phase 2: Trigger Suggestion

6. **You say:** "Hey Aura, I'm starting a new project and need advice."

**Expected:** Proactive suggestion about project management based on previous conversations.

---

## Verification Steps

### 1. Check Memory Container Stats

```bash
curl http://localhost:11438/stats
```

**Expected output:**
```json
{
  "total_conversations": 6,
  "total_embeddings": 6,
  "index_size": 6
}
```

### 2. Check Recent Conversations

```bash
curl http://localhost:11438/recent?hours=1&limit=10
```

**Expected:** See your test conversations listed.

### 3. Check for Similar Conversations

```bash
curl -X POST http://localhost:11438/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pneumonia treatment", "k": 3}'
```

**Expected:** Returns similar conversations with similarity scores.

### 4. Monitor Memory Container Logs

```bash
docker logs -f memory-container
```

**Look for:**
- `📥 Received conversation to store`
- `✅ Stored conversation`
- `🔍 Analyzing conversation for suggestions...`
- `💡 Generated proactive suggestion`
- `💡 Suggestion sent to TTS`

---

## Troubleshooting

### No Suggestions Appearing?

1. **Check conversation count:**
   ```bash
   curl http://localhost:11438/stats | jq '.total_conversations'
   ```
   - Need at least 5 conversations

2. **Check similarity:**
   - Conversations need to be similar (threshold: 0.65)
   - Try using related keywords/topics

3. **Check cooldown:**
   - Wait 60+ seconds between suggestions
   - Previous suggestion may have triggered recently

4. **Check logs:**
   ```bash
   docker logs memory-container 2>&1 | grep -E "💡|🔍|suggestion"
   ```

### Suggestions Not Being Spoken?

1. **Check if suggestion was generated:**
   ```bash
   docker logs memory-container 2>&1 | grep "💡 Generated"
   ```

2. **Check TTS integration:**
   - Suggestions are sent to TTS endpoint
   - Check if TTS is working: `curl http://localhost:11437/health` (if available)

3. **Check shared file:**
   ```bash
   cat /shared/memory_suggestion.txt
   ```
   - Suggestions may be written to shared file if TTS endpoint unavailable

---

## Quick Test (Minimal)

If you want a quick test with just 3 conversations:

1. **You say:** "Hey Aura, I need help with pneumonia treatment."

2. **You say:** "Hey Aura, what antibiotics work for pneumonia?"

3. **You say:** "Hey Aura, I'm treating a patient with pneumonia."

**Note:** This may not trigger proactive suggestions (needs 5+ conversations), but will test basic storage and similarity search.

---

## Expected Timeline

- **Conversations 1-5**: Stored, vectorized, but no suggestions (below threshold)
- **Conversation 6+**: Should trigger analysis and proactive suggestion
- **After suggestion**: 60-second cooldown before next suggestion

---

## Success Criteria

✅ **Conversations are stored** (check `/stats`)
✅ **Similar conversations are found** (check `/search`)
✅ **Proactive suggestions are generated** (check logs)
✅ **Suggestions are spoken** (hear TTS output)

---

## Notes

- **Similarity threshold**: 0.65 (adjustable in `proactive_analyzer.py`)
- **Minimum conversations**: 5 (adjustable in `proactive_analyzer.py`)
- **Cooldown**: 60 seconds (adjustable in `proactive_analyzer.py`)
- **Enable DEBUG logging** for more visibility:
  ```bash
  LOG_LEVEL=DEBUG docker compose restart memory
  ```

Good luck testing! 🧠✨

