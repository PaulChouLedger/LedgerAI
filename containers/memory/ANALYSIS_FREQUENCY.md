# Analysis Trigger Frequency

## Current Implementation

Analysis is triggered **immediately** when a new conversation is stored, but with rate limiting to prevent spam.

## Trigger Points

### 1. Event-Driven Analysis (Primary)

Analysis is triggered **immediately** when:

1. **Wake word transcription** is forwarded to memory container
2. **Background listener** transcribes audio
3. **Manual storage** via `/store` API endpoint

**Code location:**
- `container_rest.py` - `store_conversation()` endpoint
- `container_rest.py` - `_on_transcription_callback()` for background listener

**Flow:**
```
New Conversation Stored
    ↓
Immediately triggers analyze_and_suggest()
    ↓
Checks cooldown (60 seconds)
    ↓
If allowed, performs analysis
    ↓
Generates suggestion if insights found
```

### 2. Manual Analysis

Can be manually triggered via:
```bash
curl -X POST http://localhost:11438/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "conversation to analyze"}'
```

## Rate Limiting

### Suggestion Cooldown

**Default: 60 seconds**

Even though analysis runs immediately, suggestions are rate-limited:

```python
self.suggestion_cooldown = 60.0  # Don't suggest more than once per minute
```

**Behavior:**
- Analysis runs immediately for every conversation
- But suggestion is only generated if:
  - At least 60 seconds have passed since last suggestion
  - Similar conversations are found (similarity >= 0.65)
  - At least 5 conversations exist in memory (increased for more context)

**Example:**
```
10:30:00 - Conversation 1 stored → Analysis runs → Suggestion generated ✅
10:30:15 - Conversation 2 stored → Analysis runs → Cooldown active, no suggestion ⏳
10:30:45 - Conversation 3 stored → Analysis runs → Cooldown active, no suggestion ⏳
10:31:05 - Conversation 4 stored → Analysis runs → Suggestion generated ✅ (60s passed)
```

### Periodic Analysis (Not Currently Active)

There's a `analyze_recent_activity()` method with a 30-second interval, but it's **not currently being called**:

```python
self.analysis_interval = 30.0  # Analyze every 30 seconds
```

This could be used for periodic pattern detection, but is currently unused.

## Configuration

### Current Settings

```python
# In proactive_analyzer.py
self.suggestion_cooldown = 60.0  # Seconds between suggestions
self.similarity_threshold = 0.65  # Minimum similarity for matches
self.min_conversations_for_analysis = 5  # Minimum conversations needed (increased for more context)
```

### How to Change

**Option 1: Modify code directly**
```python
# In proactive_analyzer.py __init__
self.suggestion_cooldown = 120.0  # 2 minutes
```

**Option 2: Make configurable via environment**

Add to `proactive_analyzer.py`:
```python
self.suggestion_cooldown = float(os.environ.get("SUGGESTION_COOLDOWN", "60.0"))
```

Then in `docker-compose.yml`:
```yaml
environment:
  - SUGGESTION_COOLDOWN=120.0
```

## Analysis Frequency Summary

| Event | Analysis Trigger | Suggestion Generated? |
|-------|-----------------|----------------------|
| New conversation stored | ✅ Immediately | ✅ If cooldown expired & insights found |
| Background transcription | ✅ Immediately | ✅ If cooldown expired & insights found |
| Wake word transcription | ✅ Immediately | ✅ If cooldown expired & insights found |
| Manual `/analyze` call | ✅ Immediately | ✅ If cooldown expired & insights found |

## Example Timeline

```
10:30:00 - User: "I'm having trouble with my project"
           → Stored → Analysis → Suggestion: "Have you tried X?" ✅

10:30:15 - User: "What about Y?"
           → Stored → Analysis → Cooldown active (45s remaining) ⏳

10:30:30 - Background: "Maybe I should check Z"
           → Stored → Analysis → Cooldown active (30s remaining) ⏳

10:31:05 - User: "That didn't work"
           → Stored → Analysis → Suggestion: "Based on your previous issues..." ✅
           (60+ seconds passed since last suggestion)
```

## Recommendations

### For More Frequent Suggestions

Reduce cooldown:
```python
self.suggestion_cooldown = 30.0  # 30 seconds
```

### For Less Frequent Suggestions

Increase cooldown:
```python
self.suggestion_cooldown = 120.0  # 2 minutes
```

### For Immediate Analysis Only (No Cooldown)

Remove cooldown check:
```python
# Comment out cooldown check in analyze_and_suggest()
# if time_since_last < self.suggestion_cooldown:
#     return None
```

**Warning:** This may generate too many suggestions and be annoying to users.

## Periodic Analysis (Future Enhancement)

To enable periodic analysis every 30 seconds:

1. Add a background thread in `container_rest.py`:
```python
def _periodic_analysis_thread():
    while True:
        time.sleep(30)
        if analyzer:
            analyzer.analyze_recent_activity()
```

2. Start thread in `initialize_service()`:
```python
threading.Thread(target=_periodic_analysis_thread, daemon=True).start()
```

This would analyze recent activity patterns even without new conversations.

## Summary

- **Analysis runs**: Immediately when conversation is stored
- **Suggestions generated**: Only if:
  - 60+ seconds since last suggestion (cooldown)
  - Similar conversations found (similarity >= 0.65)
  - At least 5 conversations in memory (increased for more context)
- **Frequency**: Event-driven (not periodic)
- **Rate limiting**: 60-second cooldown prevents spam

