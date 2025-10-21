# Current vs FHIR Data Flow - Detailed Explanation

## 🔍 Current Data Flow (WITHOUT EHR Integration)

### What Happens When You Run `main.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  1. START: python main.py                                       │
│                                                                  │
│  • Starts Docker containers:                                    │
│    - Whisper (STT) on port 5000                                │
│    - LLM Medical on port 11434                                 │
│    - RAG on port 11435                                         │
│                                                                  │
│  • Starts GUI                                                   │
│  • Starts listener (microphone)                                │
│  • Starts speaker (TTS playback)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. VOICE INPUT: Patient speaks                                 │
│                                                                  │
│  "I have chest pain"                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. LISTENER.PY (aura-control/core/listener.py)                 │
│                                                                  │
│  • Captures audio from microphone                               │
│  • Sends audio to Whisper container:                            │
│    POST http://localhost:5000/transcribe                        │
│                                                                  │
│  • Receives transcription: "I have chest pain"                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. SPEAKER.PY (aura-control/core/speaker.py)                   │
│                                                                  │
│  Function: speak_llm_response(prompt, context)                  │
│                                                                  │
│  • Sends text to LLM container:                                 │
│    POST http://localhost:11434/chat-tts                         │
│    {                                                            │
│      "prompt": "I have chest pain",                            │
│      "chat_id": "voice_session"                                │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. CONTAINER_REST.PY (llm-medical-container)                   │
│                                                                  │
│  Endpoint: /chat-tts                                            │
│                                                                  │
│  • Receives request                                             │
│  • Routes to clinician mode:                                    │
│    handle_clinician_response(prompt, session_id, llm_chat...)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. CLINICIAN_MODE.PY (llm-medical-container)                   │
│                                                                  │
│  Class: ClinicianSession                                        │
│                                                                  │
│  • Processes medical query                                      │
│  • Uses adaptive diagnostic engine                              │
│  • Generates follow-up questions                                │
│  • Collects OLDCARTS data                                       │
│                                                                  │
│  • Saves state to:                                              │
│    /app/data/sessions/{session_id}.json                         │
│                                                                  │
│    Example:                                                     │
│    {                                                            │
│      "chief_complaint": "chest pain",                          │
│      "questions_asked": [                                       │
│        "When did the pain start?",                             │
│        "Where is the pain located?"                            │
│      ],                                                         │
│      "responses_received": [                                    │
│        "2 hours ago",                                          │
│        "center of chest"                                       │
│      ],                                                         │
│      "symptoms_collected": ["chest pain", "dyspnea"],          │
│      "urgency_score": 8.5,                                     │
│      "completed": false                                         │
│    }                                                            │
│                                                                  │
│  ⚠️ NOTE: Data is ONLY saved to local JSON file                │
│  ⚠️ NO CONNECTION to SystmOne/EHR happens here!                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. RESPONSE STREAMING                                          │
│                                                                  │
│  • Clinician mode returns question:                             │
│    "Can you describe the pain? Is it sharp or crushing?"       │
│                                                                  │
│  • Streams back through container_rest.py                       │
│  • Returns to speaker.py                                        │
│  • speaker.py converts to speech (TTS)                          │
│  • Plays audio to user                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  8. LOOP CONTINUES                                              │
│                                                                  │
│  • Patient answers → listener captures → back to step 3        │
│  • Process repeats until assessment complete                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ❌ What's MISSING in Current System

**The data stays in your local system:**

```
Patient Assessment Data
    ↓
Saved to: /app/data/sessions/voice_session.json
    ↓
❌ STOPS HERE
    ↓
NOT saved to SystmOne EHR
NOT visible to doctors/clinicians
NOT part of patient's medical record
```

---

## ✅ Enhanced Data Flow (WITH FHIR Integration)

### Modified Flow with EHR

```
┌─────────────────────────────────────────────────────────────────┐
│  1-5: SAME AS BEFORE                                            │
│  (main.py → listener → speaker → container_rest)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. ENHANCED CLINICIAN_MODE.PY                                  │
│                                                                  │
│  NEW: On session start with NHS Number                          │
│                                                                  │
│  def start_assessment(chief_complaint, nhs_number=None):        │
│                                                                  │
│      # Existing code (unchanged)                                │
│      self.chief_complaint = chief_complaint                     │
│                                                                  │
│      # NEW: EHR Integration                                     │
│      if EHR_ENABLED and nhs_number:                            │
│                                                                  │
│          # 1. Find patient in SystmOne                          │
│          patient = ehr_client.search_patient(nhs_number)        │
│                                                                  │
│          # 2. Create encounter (consultation)                   │
│          encounter = ehr_client.create_encounter(patient.id)    │
│                                                                  │
│          # Store for later use                                  │
│          self.ehr_patient_id = patient.id                       │
│          self.ehr_encounter_id = encounter.id                   │
│                                                                  │
│      # Continue with normal assessment...                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. DURING ASSESSMENT (as symptoms are collected)               │
│                                                                  │
│  NEW: After each symptom is identified                          │
│                                                                  │
│  def record_symptom(symptom_code, symptom_name, details):      │
│                                                                  │
│      # Existing: Save to local state                            │
│      self.symptoms_collected.append(symptom_name)              │
│                                                                  │
│      # NEW: Save to SystmOne in real-time                       │
│      if EHR_ENABLED:                                           │
│          ehr_client.create_observation(                        │
│              patient_id=self.ehr_patient_id,                   │
│              encounter_id=self.ehr_encounter_id,               │
│              code=symptom_code,  # SNOMED: 29857009           │
│              display=symptom_name,  # "Chest pain"            │
│              value=details  # "Severity 8/10, crushing"       │
│          )                                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  8. ASSESSMENT COMPLETE                                         │
│                                                                  │
│  NEW: When session finishes                                     │
│                                                                  │
│  def finalize_assessment():                                     │
│                                                                  │
│      # Existing: Generate summary                               │
│      summary = self._generate_summary()                         │
│                                                                  │
│      # NEW: Save to SystmOne                                    │
│      if EHR_ENABLED:                                           │
│                                                                  │
│          # Save consultation summary                            │
│          ehr_client.create_document(                           │
│              patient_id=self.ehr_patient_id,                   │
│              encounter_id=self.ehr_encounter_id,               │
│              title="Aura AI Consultation Summary",            │
│              content=summary                                    │
│          )                                                      │
│                                                                  │
│          # Close encounter                                      │
│          ehr_client.close_encounter(self.ehr_encounter_id)     │
│                                                                  │
│      return summary                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  9. DATA NOW IN SYSTMONE                                        │
│                                                                  │
│  ✅ Patient record updated                                      │
│  ✅ Encounter logged                                            │
│  ✅ Symptoms documented with SNOMED codes                       │
│  ✅ AI summary attached                                         │
│  ✅ Available for clinician review                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 What You Need to Modify

### Answer to Your Question:

**Q: "Will main.py automatically do the FHIR API call?"**

**A: NO - You need to modify `clinician_mode.py` to add FHIR integration**

`main.py` stays the same - it just starts containers and the GUI.

**The FHIR calls happen inside the LLM container, specifically in `clinician_mode.py`**

---

## 📝 Step-by-Step Integration Guide

### Step 1: Copy the Example Code to LLM Container

```bash
# The example code is already in llm-medical-container/
cd /Users/rcabello/Documents/GitHub/LedgerAI/llm-medical-container

# You have:
# - ehr_integration_example.py (the FHIR client)
# - requirements_ehr.txt (dependencies)
```

### Step 2: Install Dependencies in Container

**Option A: Install locally for testing**
```bash
cd llm-medical-container
pip install -r requirements_ehr.txt
```

**Option B: Add to Dockerfile (for production)**

Edit `llm-medical-container/Dockerfile`:

```dockerfile
# Add to requirements
COPY requirements_ehr.txt /app/
RUN pip install -r requirements_ehr.txt
```

### Step 3: Modify `clinician_mode.py`

Add EHR integration to the existing file:

```python
# llm-medical-container/clinician_mode.py

# At the top of the file, add imports:
from ehr_integration_example import SimpleFHIRClient, validate_nhs_number
import os

# Add after existing imports:
EHR_INTEGRATION_ENABLED = os.getenv("EHR_INTEGRATION_ENABLED", "false").lower() == "true"

class ClinicianSession:
    """Enhanced with EHR integration"""
    
    def __init__(self, session_id: str, llm_chat_fn: Callable, llm_chat_simple_fn: Callable = None):
        # Existing initialization code...
        self.session_id = session_id
        self.llm_chat_fn = llm_chat_fn
        self.llm_chat_simple_fn = llm_chat_simple_fn or llm_chat_fn
        
        # ... existing attributes ...
        
        # NEW: EHR integration attributes
        self.ehr_enabled = EHR_INTEGRATION_ENABLED
        self.ehr_client = None
        self.ehr_patient_id = None
        self.ehr_encounter_id = None
        
        if self.ehr_enabled:
            fhir_url = os.getenv("SYSTMONE_FHIR_URL", "https://hapi.fhir.org/baseR4")
            self.ehr_client = SimpleFHIRClient(fhir_url)
            print(f"[EHR] 🏥 Integration enabled: {fhir_url}")
```

### Step 4: Add EHR Methods to ClinicianSession

Add these new methods to the `ClinicianSession` class:

```python
class ClinicianSession:
    # ... existing code ...
    
    def start_ehr_session(self, nhs_number: str):
        """
        Initialize EHR session with patient NHS Number
        
        Call this at the START of an assessment if you have NHS Number
        """
        if not self.ehr_enabled or not self.ehr_client:
            return False
        
        try:
            # Validate NHS Number
            if not validate_nhs_number(nhs_number):
                print(f"[EHR] ❌ Invalid NHS Number: {nhs_number}")
                return False
            
            # Find patient in SystmOne
            patient = self.ehr_client.search_patient(nhs_number)
            
            if not patient:
                print(f"[EHR] ⚠️ Patient not found: {nhs_number}")
                return False
            
            self.ehr_patient_id = patient.id
            print(f"[EHR] ✅ Found patient: {patient.name[0].family if patient.name else 'Unknown'}")
            
            # Create encounter (consultation)
            encounter = self.ehr_client.create_encounter(
                patient_id=patient.id,
                encounter_type="virtual"
            )
            
            if encounter:
                self.ehr_encounter_id = encounter.id
                print(f"[EHR] ✅ Started encounter: {encounter.id}")
                return True
            
        except Exception as e:
            print(f"[EHR] ❌ Error starting EHR session: {e}")
        
        return False
    
    def save_symptom_to_ehr(self, symptom_code: str, symptom_name: str, symptom_details: str):
        """
        Save a symptom observation to SystmOne
        
        Call this DURING assessment as you collect symptoms
        """
        if not self.ehr_enabled or not self.ehr_encounter_id:
            return False
        
        try:
            observation = self.ehr_client.create_observation(
                patient_id=self.ehr_patient_id,
                encounter_id=self.ehr_encounter_id,
                code=symptom_code,
                display=symptom_name,
                value=symptom_details
            )
            
            if observation:
                print(f"[EHR] ✅ Saved symptom: {symptom_name}")
                return True
                
        except Exception as e:
            print(f"[EHR] ⚠️ Failed to save symptom: {e}")
        
        return False
    
    def finalize_ehr_session(self, summary: str):
        """
        Save consultation summary and close EHR encounter
        
        Call this at the END of assessment
        """
        if not self.ehr_enabled or not self.ehr_encounter_id:
            return False
        
        try:
            # Save consultation summary
            document = self.ehr_client.create_document(
                patient_id=self.ehr_patient_id,
                encounter_id=self.ehr_encounter_id,
                title="Aura AI Consultation Summary",
                content=summary
            )
            
            if document:
                print(f"[EHR] ✅ Saved consultation summary")
            
            # Close encounter
            success = self.ehr_client.close_encounter(self.ehr_encounter_id)
            
            if success:
                print(f"[EHR] ✅ Consultation closed")
                print(f"[EHR] 🏥 All data saved to SystmOne")
                return True
                
        except Exception as e:
            print(f"[EHR] ❌ Error finalizing EHR session: {e}")
        
        return False
```

### Step 5: Modify Your Workflow Methods

Update the methods that handle the assessment:

```python
class ClinicianSession:
    # ... existing code ...
    
    def process_medical_query(self, prompt: str, nhs_number: str = None):
        """
        Enhanced to support EHR integration
        
        Args:
            prompt: Patient's message
            nhs_number: Optional NHS Number for EHR integration
        """
        
        # If this is first interaction and NHS Number provided
        if nhs_number and not self.ehr_encounter_id:
            self.start_ehr_session(nhs_number)
        
        # Your existing assessment logic...
        # (Keep all your current code here)
        
        # When you identify a symptom, also save to EHR
        # Example:
        if self.dynamic_assessment:
            # You detected chest pain
            symptom_code = "29857009"  # SNOMED code
            symptom_name = "Chest pain"
            symptom_details = f"Severity: {severity}, Character: {character}"
            
            # Save to local state (existing)
            self.dynamic_assessment.symptoms_collected.append(symptom_name)
            
            # NEW: Also save to EHR
            self.save_symptom_to_ehr(symptom_code, symptom_name, symptom_details)
        
        # ... rest of your existing code ...
        
        return response
```

### Step 6: Configure Environment Variables

Create/edit `llm-medical-container/.env`:

```bash
# Enable EHR integration
EHR_INTEGRATION_ENABLED=true

# FHIR server URL
# For testing (no auth needed):
SYSTMONE_FHIR_URL=https://hapi.fhir.org/baseR4

# For production (when ready):
# SYSTMONE_FHIR_URL=https://api.systmone.nhs.uk/fhir
# NHS_CLIENT_ID=your_client_id
# NHS_CLIENT_SECRET=your_client_secret
```

### Step 7: Test It

**Test with existing workflow:**

```bash
# 1. Install dependencies
cd llm-medical-container
pip install -r requirements_ehr.txt

# 2. Enable EHR in .env
echo "EHR_INTEGRATION_ENABLED=true" >> .env
echo "SYSTMONE_FHIR_URL=https://hapi.fhir.org/baseR4" >> .env

# 3. Rebuild and restart container
docker-compose down
docker-compose build llm
docker-compose up -d

# 4. Run main.py as normal
cd ../aura-control
python main.py
```

**What happens now:**

```
Patient: "I have chest pain"
    ↓
Aura: Processes with clinician_mode
    ↓
[EHR] ✅ Found patient: Smith
[EHR] ✅ Started encounter: 12345
[EHR] ✅ Saved symptom: Chest pain
    ↓
Aura: Asks follow-up questions...
    ↓
[EHR] ✅ Saved symptom: Dyspnea
[EHR] ✅ Saved consultation summary
[EHR] ✅ Consultation closed
[EHR] 🏥 All data saved to SystmOne
```

---

## 🎛️ Control Flow Summary

### Without NHS Number (Normal Aura)

```python
# In speaker.py (unchanged):
speak_llm_response("I have chest pain", context="")
    ↓
# LLM container processes
# Saves to: /app/data/sessions/voice_session.json
# NO EHR calls
```

### With NHS Number (EHR Enabled)

```python
# Modified speaker.py (optional enhancement):
speak_llm_response(
    prompt="I have chest pain",
    context="",
    nhs_number="9434765870"  # NEW parameter
)
    ↓
# LLM container processes
# Saves to: /app/data/sessions/voice_session.json (local)
# ALSO saves to: SystmOne via FHIR API (remote)
```

---

## 📊 Where is Data Stored?

### Current System (Local Only)

```
/app/data/sessions/
├── voice_session.json         # Current conversation
├── telegram_12345.json        # Telegram user
└── session_abc123.json        # Another session

Contents:
{
  "chief_complaint": "chest pain",
  "symptoms_collected": ["chest pain", "dyspnea"],
  "urgency_score": 8.5,
  "questions_asked": [...],
  "responses_received": [...]
}
```

### Enhanced System (Local + EHR)

```
LOCAL:
/app/data/sessions/voice_session.json
    ↓
    Same as before (unchanged)

PLUS

REMOTE (SystmOne):
Patient/9434765870
├── Encounter/67890
│   ├── Observation/123 (Chest pain - SNOMED: 29857009)
│   ├── Observation/124 (Dyspnea - SNOMED: 267036007)
│   └── DocumentReference/456 (AI consultation summary)
```

---

## ⚡ Quick Answer to Your Question

**"Will main.py automatically do the FHIR API call?"**

❌ **NO** - `main.py` does NOT automatically do FHIR calls

✅ **YES** - You need to:
1. Add FHIR client code to `clinician_mode.py`
2. Enable with environment variable
3. Optionally pass NHS Number to the assessment

**The FHIR calls happen inside the LLM container during clinician mode assessment**

---

## 🚀 Next Steps

1. ✅ **Understand the flow** (you're here!)
2. 📝 **Modify `clinician_mode.py`** (add methods from Step 4)
3. 🔧 **Set environment variables** (enable EHR)
4. 🧪 **Test with HAPI FHIR** (no NHS credentials needed)
5. 🏥 **Deploy to NHS** (when ready for production)

---

**Want help implementing Step 4?** I can modify your `clinician_mode.py` file directly!

