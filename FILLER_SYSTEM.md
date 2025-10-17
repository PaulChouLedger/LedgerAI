# Thinking Filler System

## Overview

The **Thinking Filler System** provides natural, synchronized filler phrases while the LLM generates responses. This reduces perceived latency and makes interactions feel more natural for both **voice (TTS)** and **chatbot (Telegram)** modes.

## Key Principle: Perfect Synchronization

**CRITICAL**: Voice and chatbot MUST output the **SAME** message.

- **Voice Mode**: Plays pre-recorded audio file
- **Chatbot Mode**: Displays text message
- **Both**: Use the same filler from the same ID

## Architecture

```
User says something
     ↓
Engine needs time to think (LLM call)
     ↓
Immediately return filler:
     ├─ Text: "Let me think..."
     └─ Audio: /data/fillers/thinking/think_1.wav
     ↓
[LLM generates response]
     ↓
Return actual response
```

## File Structure

```
llm-container/
  thinking_fillers.py          # Master library of fillers
  
data/fillers/thinking/
  think_1.wav                  # "Let me think..."
  think_2.wav                  # "One moment..."
  opening_1.wav                # "I understand..."
  question_1.wav               # "Let me think about what to ask next..."
  location_1.wav               # "Let me make sure I understand the location..."
  diagnosis_1.wav              # "Let me review everything..."
  manifest.json                # Mapping of IDs to text and audio files

scripts/
  generate_filler_audio.py     # Script to generate all audio samples
```

## Filler Categories

### 1. **Opening** (`'opening'`)
Used when starting a medical assessment
- "I understand..."
- "Okay..."
- "Alright..."
- "I see..."

### 2. **Question Generation** (`'question_generation'`)
Used when generating clinical questions
- "Let me think about what to ask next..."
- "One moment while I consider the best question..."
- "Give me a second to think..."
- "Hmm, let me see..."
- "Let me figure out what to ask..."

### 3. **Location Clarification** (`'location_clarification'`)
Used when clarifying anatomical locations
- "Let me make sure I understand the location..."
- "I want to be more specific about the location..."
- "Let me clarify where exactly..."
- "Hold on, let me get more detail..."

### 4. **Diagnosis** (`'diagnosis'`)
Used when formulating final diagnosis
- "Let me review everything..."
- "Give me a moment to put this together..."
- "Let me analyze your symptoms..."
- "Hmm, let me think through this..."

## Filler Data Structure

Each filler has three components:

```python
{
    'id': 'think_1',                           # Unique identifier
    'text': "Let me think...",                 # For chatbot/logging
    'audio_path': '/data/.../think_1.wav'     # For voice (if exists)
}
```

## Usage in Code

### In Diagnostic Engine

```python
from thinking_fillers import get_filler

# Before any LLM call:
filler = get_filler('question_generation', use_audio=True)

# Return immediately to user:
return {
    'success': True,
    'question': actual_question,  # Generated after LLM finishes
    'status': 'questioning',
    'filler': filler              # Played/sent BEFORE question
}
```

### In Container REST API

```python
result = engine.process_answer(user_answer)

if 'filler' in result:
    # Voice: Play audio file if available
    if 'audio_path' in result['filler']:
        play_audio(result['filler']['audio_path'])
    
    # Chatbot: Send text immediately
    send_telegram_message(result['filler']['text'])

# Then handle the actual response
handle_response(result['question'])
```

## Generating Audio Samples

### Prerequisites

1. LLM container must be running on port 5001
2. TTS endpoint must be available: `http://localhost:5001/tts`

### Generate All Fillers

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI
python3 scripts/generate_filler_audio.py
```

### Output

```
/data/fillers/thinking/
  think_1.wav
  think_2.wav
  ...
  opening_1.wav
  ...
  question_1.wav
  ...
  manifest.json   # Maps IDs to text and files
```

### Manifest Format

```json
{
  "description": "Pre-recorded thinking filler audio samples",
  "total_fillers": 23,
  "fillers": [
    {
      "id": "think_1",
      "text": "Let me think...",
      "audio_file": "think_1.wav",
      "file_size": 24576
    },
    ...
  ]
}
```

## Integration Points

### 1. **adaptive_diagnostic_engine.py**

Fillers are generated at these points:
- `start_assessment()`: Opening filler before generating empathetic statement
- `_ask_next_clinical_question()`: Question generation filler before OLDCARTS questions
- `_ask_next_clinical_question()`: Question generation filler before associated symptom questions
- `_process_clinical_answer()`: Location clarification filler before LLM generates clarification

### 2. **container_rest.py**

The REST API should:
1. Check for `filler` key in engine response
2. If voice mode: Stream audio file first (if exists), then actual response
3. If chatbot mode: Send text immediately, then actual response

### 3. **telegram_bot.py**

The Telegram bot should:
1. Check for `filler` key in engine response
2. Send `filler['text']` as first message
3. Show typing indicator while waiting for actual response
4. Send actual response as second message

## Benefits

✅ **Reduces perceived latency** - User gets immediate feedback
✅ **More natural conversation** - Mimics human thinking patterns
✅ **Synchronized behavior** - Voice and chatbot use same messages
✅ **Scalable** - Easy to add more fillers with different contexts
✅ **Flexible** - Falls back to text if audio doesn't exist

## Future Enhancements

- [ ] Add fillers for more contexts (triage, red flag screening)
- [ ] Generate multiple voice variations per filler (different intonations)
- [ ] Use emotion detection to select appropriate filler tone
- [ ] Add background ambient sounds for longer thinking pauses
- [ ] Implement progressive fillers for very long LLM calls ("Still thinking...", "Almost there...")

## Maintenance

### Adding New Fillers

1. Add to `thinking_fillers.py`:
   ```python
   MEDICAL_FILLERS = {
       'my_new_context': [
           {'id': 'mynew_1', 'text': "Custom filler text..."},
           {'id': 'mynew_2', 'text': "Another filler..."},
       ]
   }
   ```

2. Regenerate audio:
   ```bash
   python3 scripts/generate_filler_audio.py
   ```

3. Use in code:
   ```python
   filler = get_filler('my_new_context')
   ```

### Testing

```bash
# Test filler library
cd /Users/rcabello/Documents/GitHub/LedgerAI/llm-container
python3 thinking_fillers.py

# Output:
# General fillers:
#   - [think_1] Let me think...
#   - [think_2] One moment...
#   ...
```

## Notes

- Fillers are selected **randomly** from their category each time
- Audio files are **optional** - system works with text-only
- All fillers are designed to be **< 2 seconds** to avoid over-delaying
- Fillers should be **natural and conversational**, not robotic

