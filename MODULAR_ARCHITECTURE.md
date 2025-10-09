# Aura Modular Architecture - Clean Separation of Modes

## Overview

The conversation system is refactored into separate, focused modules for each conversation mode. This makes the system:
- ✅ **Easier to maintain** - Each mode in its own file
- ✅ **Easier to test** - Test modes independently
- ✅ **Easier to extend** - Add new modes without touching existing code
- ✅ **Cleaner code** - No giant monolithic files

---

## File Structure

```
llm-container/
├── container_rest.py      # Main Flask app (routing only)
├── router.py              # 🎯 Intelligent mode selection
├── casual.py              # 💬 CASUAL mode - greetings
├── thinker.py             # 🧠 THINKER mode - knowledge queries + RAG
├── triage.py              # 🏥 TRIAGE mode - hardcoded diagnostic (existing)
├── clinician.py           # 🩺 CLINICIAN mode - RAG-powered diagnosis (new)
└── medical_guidelines/    # 📚 Multi-organ medical knowledge (future)
    ├── cardiovascular/
    ├── respiratory/
    ├── neurology/
    ├── gastroenterology/
    └── ... (more organ systems)
```

---

## Module Responsibilities

### `router.py` - The Orchestrator 🎯

**Purpose:** Decide which mode to use

**Logic:**
```python
def route_prompt(prompt, state, session_id):
    # 1. Check for active session
    if has_active_session(state):
        return continue_current_mode(state)
    
    # 2. Classify new prompt
    if is_casual_trigger(prompt):
        return CASUAL
    elif is_thinker_trigger(prompt):
        return THINKER
    elif USE_CLINICIAN and is_clinician_trigger(prompt):
        return CLINICIAN
    elif detect_triage_condition(prompt):
        return TRIAGE
    else:
        return THINKER  # Default
```

**Key Functions:**
- `route_prompt()` - Main routing logic
- `get_active_mode()` - Check for active sessions
- `format_mode_info()` - Mode metadata

---

### `casual.py` - Simple Greetings 💬

**Triggers:**
- Standalone greetings: "Hello", "Hi", "Good morning"
- Small talk: "How are you?", "What's up?"

**Behavior:**
- Brief, friendly responses
- No RAG, no memory
- Quick turnaround

**Key Functions:**
- `is_casual_trigger()` - Pattern matching for greetings
- `handle_casual()` - Generate friendly response
- `stream_casual_response()` - Streaming version

**Example:**
```
User: "Hello Aura"
→ "Hi there! What can I do for you?"
```

---

### `thinker.py` - Knowledge Engine 🧠

**Triggers:**
- Information queries: "What is X?", "Who is Y?"
- Explanations: "Explain diabetes", "How does the brain work?"
- Details: "Tell me everything about Rafael Cabello"

**Behavior:**
- **ALWAYS searches RAG** for relevant documents
- Comprehensive, detailed responses
- Educational and insightful tone
- Synthesizes information from multiple sources

**Key Functions:**
- `is_thinker_trigger()` - Detect knowledge queries
- `handle_thinker()` - Search RAG + generate response
- `search_rag()` - RAG API wrapper

**Example:**
```
User: "Who is Bob Carella?"
→ [Searches RAG, finds document about Bob]
→ "Bob Carella is the Co-Founder and Chief Financial Officer of LedgerAI. 
   He brings extensive experience in finance, blockchain, and enterprise strategy,
   having previously served as Global Head of Payroll at Binance.US..."
```

---

### `triage.py` - Hardcoded Diagnostics 🏥

**Triggers:**
- Medical symptoms: "I have chest pain", "My head hurts"
- When: `USE_CLINICIAN_MODE = False` (current baseline)

**Behavior:**
- Structured questions from JSON definitions
- Fixed question order per condition
- Severity scoring
- SOAP-style recap at end

**Key Functions:**
- `detect_condition()` - Match symptoms to conditions
- `get_steps()` - Load triage steps
- `process_triage_step()` - Handle answers
- `generate_triage_completion()` - Final recap

**Status:** ✅ Keep intact as working baseline

**Example:**
```
User: "I have chest pain"
→ "When did the chest pain start?"
User: "An hour ago"
→ "Is the pain sharp, dull, or crushing?"
User: "Crushing"
→ "Does it radiate to your arm or jaw?"
... (structured questions continue)
```

---

### `clinician.py` - Intelligent Diagnosis 🩺

**Triggers:**
- Medical symptoms: "I have chest pain", "I'm experiencing dizziness"
- When: `USE_CLINICIAN_MODE = True` (future)

**Behavior:**
- **Searches RAG** for medical guidelines
- **Generates intelligent questions** based on clinical reasoning
- **Adapts questioning** based on responses
- **Builds differential diagnosis**
- Thinks like a real doctor

**Key Classes:**
- `ClinicianSession` - Manages diagnostic conversation

**Key Functions:**
- `is_clinician_trigger()` - Detect medical symptoms
- `start_session()` - Begin intelligent diagnostic
- `process_response()` - Continue diagnostic conversation
- `_search_medical_guidelines()` - RAG search for clinical info
- `_generate_next_question()` - LLM + RAG → intelligent questions

**Status:** 🚧 Framework complete, needs medical guideline database

**Example (Future):**
```
User: "I have chest pain"
→ [Searches RAG: "chest pain differential diagnosis guidelines"]
→ "I understand you're experiencing chest pain - that's important to evaluate. 
   Can you describe the pain? Is it sharp, dull, crushing, or pressure-like?"

User: "It's crushing and goes down my left arm"
→ [Searches RAG: "acute coronary syndrome presentation"]
→ "Crushing pain with left arm radiation is very concerning. Did this come on 
   suddenly or gradually? And are you experiencing any sweating or shortness of breath?"

User: "Suddenly about 30 minutes ago, and yes I'm sweating"
→ [Searches RAG: "ACS emergency management"]
→ "This is an emergency situation suggestive of a possible heart attack. 
   You need to call 911 immediately or have someone take you to the ER right now..."
```

---

## Conversation Flow

### Entry Point: `/chat` Endpoint

```python
@app.route("/chat", methods=["POST"])
def chat():
    prompt = get_prompt_from_request()
    session_id = get_session_id()
    state = load_state(session_id)
    
    # Route to appropriate mode
    mode, updated_state = route_prompt(prompt, state, session_id)
    save_state(updated_state, session_id)
    
    # Dispatch to mode handler
    if mode == ConversationMode.CASUAL:
        return handle_casual_mode(prompt, session_id)
    
    elif mode == ConversationMode.THINKER:
        return handle_thinker_mode(prompt, session_id)
    
    elif mode == ConversationMode.TRIAGE:
        return handle_triage_mode(prompt, state, session_id)
    
    elif mode == ConversationMode.CLINICIAN:
        return handle_clinician_mode(prompt, state, session_id)
    
    else:
        return handle_fallback()
```

---

## Session State Examples

### CASUAL (Stateless)
```python
{
    "mode": "casual"
    # No persistence needed
}
```

### THINKER (Stateless)
```python
{
    "mode": "thinker"
    # No persistence needed, RAG handles context
}
```

### TRIAGE (Stateful)
```python
{
    "mode": "triage",
    "condition": "chest_pain",
    "step_index": 3,
    "answers": ["severe", "crushing", "yes"],
    "flags": {"emergency": True},
    "user_name": "Rafael"
}
```

### CLINICIAN (Stateful)
```python
{
    "mode": "clinician",
    "chief_complaint": "I have chest pain",
    "conversation_history": [
        {"role": "patient", "content": "crushing pain"},
        {"role": "clinician", "content": "does it radiate?"},
        {"role": "patient", "content": "yes to left arm"}
    ],
    "findings": {
        "severity": "severe",
        "onset": "acute",
        "radiation": "left arm",
        "associated_sx": ["sweating", "dyspnea"]
    },
    "differential_diagnoses": ["ACS", "PE", "Dissection"],
    "current_focus": "cardiovascular"
}
```

---

## Migration Timeline

### Phase 1: Refactor (Current)
- [x] Create modular files (casual.py, thinker.py, router.py, clinician.py)
- [ ] Refactor container_rest.py to use router
- [ ] Extract triage logic to triage.py
- [ ] Test all modes work independently

### Phase 2: Build Medical Database
- [ ] Create medical_guidelines/ directory structure
- [ ] Populate with 20-30 common chief complaints
- [ ] Ingest into RAG system
- [ ] Verify RAG retrieval quality

### Phase 3: Beta Test Clinician
- [ ] Set `USE_CLINICIAN_MODE = True`
- [ ] Test clinician mode with real queries
- [ ] Compare quality vs Triage
- [ ] Iterate on questioning logic

### Phase 4: Parallel Operation
- [ ] Run Clinician as primary
- [ ] Keep Triage as fallback
- [ ] Monitor and compare both
- [ ] Gather feedback

### Phase 5: Full Migration
- [ ] Clinician handles all medical queries
- [ ] Triage becomes legacy/fallback only
- [ ] Eventually deprecate triage_defs/

---

## Benefits of Modular Architecture

### Before (Monolithic)
```
container_rest.py (1400+ lines)
└── Everything in one file
    ├── Casual handling
    ├── Knowledge queries
    ├── Triage logic
    ├── State management
    ├── RAG calls
    └── Response generation
```
❌ Hard to maintain
❌ Hard to test
❌ Hard to extend
❌ Coupling between modes

### After (Modular)
```
router.py (150 lines)
└── Mode selection only

casual.py (100 lines)
└── Just greetings

thinker.py (150 lines)  
└── Just knowledge + RAG

triage.py (600 lines)
└── Just hardcoded triage

clinician.py (400 lines)
└── Just intelligent diagnosis
```
✅ Easy to maintain
✅ Easy to test
✅ Easy to extend
✅ Clean separation

---

## API Consistency

All modes return the same streaming format:

```
<sentence_start>
Sentence content here.
<sentence_end>
<sentence_start>
Next sentence here.
<sentence_end>
```

This ensures the speaker/TTS system works identically across all modes.

---

## Future Enhancements

### Overnight Analysis (Your Vision)
- Store all conversations in vector DB
- Nightly processing:
  - Analyze patterns
  - Identify recurring issues
  - Search for solutions in RAG
  - Generate insights
- Morning briefing with findings

### Mode Integration
- THINKER can answer follow-up questions after TRIAGE/CLINICIAN
- Seamless transitions between modes
- Context carries over when appropriate

### Smart Mode Switching
- Mid-conversation mode changes
- "By the way, who is Rafael Cabello?" during medical discussion
- Temporary THINKER query → return to CLINICIAN/TRIAGE

