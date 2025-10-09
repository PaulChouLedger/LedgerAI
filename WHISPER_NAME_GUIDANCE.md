# Whisper Name Guidance System

## 🎯 **Problem Solved**

Whisper was transcribing "Rafael" as "Raphael" because:
1. They sound identical phonetically
2. Whisper is trained on English text where "Raphael" is more common
3. The model defaults to the more common spelling

### **The Circular Dependency Problem**

Initial approach had a fatal flaw:
1. User says: "My name is Rafael"
2. Whisper transcribes: "Raphael" (wrong)
3. System learns: "Raphael" (wrong)
4. System reinforces: "Raphael" forever ❌

**We can't learn from Whisper if Whisper is wrong!**

## ✅ **Solution: RAG-Based Name Correction + Dynamic Initial Prompts**

**Two-part solution:**

1. **Use RAG as phonetic dictionary** - Extract all person names from documents
2. **Correct Whisper's mistakes** - Match phonetically to find correct spelling
3. **Guide future transcriptions** - Use corrected name in `initial_prompt`

---

## 📋 **How It Works**

### **Step 1: User Introduces Themselves**

```
User: "My name is Rafael"
Whisper: Transcribes as "Raphael" (wrong spelling due to phonetics)
Listener: Detects pattern "My name is Raphael"
System: Checks RAG database for phonetically similar names
RAG: Finds "Rafael Cabello" - phonetic match! ('RFL' == 'RFL')
System: Corrects spelling: "Raphael" → "Rafael" ✅
System: Saves user_name = "Rafael" (correct spelling from database)
```

**Key Innovation:** Uses RAG database as a phonetic dictionary to correct Whisper's spelling!

### **Step 2: Future Transcriptions Use Name Guidance**

```
User: "Who is Rafael?"
Listener: Sends to Whisper with initial_prompt = "Rafael is speaking. This is a medical conversation with proper names."
Whisper: Now strongly biased toward "Rafael" spelling ✅
```

### **Step 3: Consistent Correct Spelling**

All future transcriptions will prefer "Rafael" over "Raphael" automatically!

---

## 🔧 **Implementation Details**

### **Whisper Container** (`whisper-container-faster/container_rest.py`)

**Added:**
- Accepts optional `initial_prompt` form parameter
- Uses custom prompt or falls back to default

```python
custom_prompt = request.form.get("initial_prompt", INITIAL_PROMPT)
segments, _ = model.transcribe(audio, initial_prompt=custom_prompt)
```

### **RAG Container** (`rag-container/container_rest.py`)

**Added:**
- `/rag/names` endpoint - returns all person names from database
- Extracts patterns like "Rafael Cabello", "Bob Carella", etc.
- Used as phonetic dictionary for name correction

### **Listener** (`aura-control/listener.py`)

**Added:**
1. Global `user_name` variable
2. `extract_user_name()` function - detects name patterns
3. `correct_name_from_rag()` function - **corrects spelling using RAG database**
4. Custom initial_prompt in `transcribe()` - guides Whisper

```python
# Patterns detected:
- "My name is Rafael"
- "I'm Rafael"  
- "I am Rafael"
- "This is Rafael"
- "Call me Rafael"
```

**Name correction flow:**
```python
# 1. Whisper transcribes (possibly wrong)
text = "My name is Raphael"  # Wrong spelling

# 2. Extract detected name
detected = "Raphael"

# 3. Check RAG database for phonetic match
rag_names = get_rag_names()  # ["Rafael Cabello", "Bob Carella", ...]
corrected = phonetic_match("Raphael", rag_names)
# Returns: "Rafael" ✅

# 4. Use corrected name
user_name = "Rafael"

# 5. Guide future transcriptions
if user_name:
    data["initial_prompt"] = f"{user_name} is speaking. This is a medical conversation with proper names."
```

---

## 🚀 **Usage**

### **1. Rebuild Whisper Container**

```bash
cd ~/LedgerAI
docker compose stop whisper-container
docker compose build whisper-container  
docker compose up -d whisper-container
```

### **2. Test the System**

```bash
# Start listener
python3 aura-control/main.py

# Say: "My name is Rafael"
# Output: [Listener] 👤 User name detected: 'Rafael' (will guide Whisper spelling)

# Say: "Who is Rafael?"  
# Output: [Whisper] 🎯 Using name guidance: 'Rafael'
# Result: Transcribes as "Rafael" not "Raphael" ✅
```

---

## 📊 **Detection Patterns**

The system detects names from these patterns:

| Pattern | Example | Detected Name |
|---------|---------|--------------|
| `my name is X` | "My name is Rafael" | Rafael |
| `i'm X` | "I'm Rafael" | Rafael |
| `i am X` | "I am Rafael" | Rafael |
| `this is X` | "This is Rafael" | Rafael |
| `call me X` | "Call me Rafael" | Rafael |

*Case-insensitive, capitalizes first letter automatically*

---

## 🔍 **How Initial Prompts Work**

Whisper uses the initial prompt to **bias the model** toward certain spellings:

```python
# Without initial_prompt:
"Who is Rafael?" → "Who is Raphael?" (wrong)

# With initial_prompt = "Rafael is speaking":
"Who is Rafael?" → "Who is Rafael?" (correct) ✅
```

The initial prompt acts as **context** that influences:
- Spelling choices
- Proper name recognition
- Technical term preferences

---

## 🧪 **Testing Different Names**

### **Rafael vs Raphael**
```bash
Say: "My name is Rafael"
Then: "Who is Rafael?"
Result: ✅ "Rafael" (not "Raphael")
```

### **Katherine vs Catherine**
```bash
Say: "My name is Katherine"
Then: "Tell me about Katherine"
Result: ✅ "Katherine" (not "Catherine")
```

### **Jon vs John**
```bash
Say: "I'm Jon"
Then: "Jon is here"
Result: ✅ "Jon" (not "John")
```

---

## 🛠️ **Advanced Configuration**

### **Adjust Initial Prompt Template**

In `listener.py`:
```python
# Current:
data["initial_prompt"] = f"{user_name} is speaking. This is a medical conversation with proper names."

# Custom (more specific):
data["initial_prompt"] = f"Speaker: {user_name}. Medical consultation with technical terminology."
```

### **Add More Detection Patterns**

```python
patterns = [
    r"my name is ([A-Z][a-z]+)",
    r"people call me ([A-Z][a-z]+)",  # Add this
    r"you can call me ([A-Z][a-z]+)", # Add this
]
```

---

## 📝 **Limitations**

1. **First introduction might be wrong** - System learns from transcription, so if Whisper gets it wrong the first time, it will save the wrong name
2. **Single name only** - Currently tracks one name at a time
3. **Pattern-based** - Only detects explicit introductions, not implicit references

---

## 🔧 **Troubleshooting**

### **Name not being detected?**

Check logs for:
```
[Listener] 👤 User name detected: 'Rafael' (will guide Whisper spelling)
```

If not appearing, the pattern might not match. Add debug:
```python
print(f"[Debug] Checking text: {text}")
extract_user_name(text)
```

### **Still transcribing wrong?**

Check if initial_prompt is being sent:
```
[Whisper] 🎯 Using name guidance: 'Rafael'
```

If not, check `user_name` variable:
```python
print(f"[Debug] user_name = {user_name}")
```

### **Reset learned name**

Restart the listener - `user_name` resets to `None`

---

## 📚 **Related Documentation**

- Whisper initial_prompt: https://github.com/openai/whisper/discussions/117
- faster-whisper docs: https://github.com/guillaumekln/faster-whisper
- Name spelling biases in ASR: Common issue with phonetically similar names

---

**Last Updated:** October 9, 2025

**Files Modified:**
- `whisper-container-faster/container_rest.py` - Accept custom prompts
- `aura-control/listener.py` - Extract names and send guidance

