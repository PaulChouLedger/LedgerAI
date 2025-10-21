# EHR Integration Quick Start Guide

**Goal:** Integrate Aura Medical Chatbot with SystmOne EHR in the UK NHS

---

## 🚀 Quick Overview

```
Aura Chatbot → FHIR API → NHS Authentication → SystmOne EHR
```

**What you'll do:**
1. Use **FHIR R4** API (healthcare data standard)
2. Authenticate via **NHS CIS2** (OAuth 2.0)
3. Save consultation data to **SystmOne**
4. Comply with **NHS regulations** (GDPR, clinical safety)

---

## 📋 Prerequisites Checklist

### Technical
- [ ] Python 3.8+
- [ ] Understand REST APIs
- [ ] Basic OAuth 2.0 knowledge
- [ ] Familiarity with healthcare data (optional but helpful)

### Regulatory (before production)
- [ ] Register with NHS Digital API Platform
- [ ] Appoint Clinical Safety Officer
- [ ] Complete NHS DSPT (Data Security toolkit)
- [ ] SNOMED CT license (free for NHS use)
- [ ] Data sharing agreement with NHS trust

---

## 🛠️ Installation

### 1. Install Python Dependencies

```bash
cd llm-medical-container
pip install -r requirements_ehr.txt
```

Key libraries:
- `fhir.resources` - FHIR client
- `authlib` - OAuth 2.0 for NHS authentication
- `requests` - HTTP client

### 2. Test the Example

```bash
python ehr_integration_example.py
```

This will:
- Connect to HAPI FHIR test server (public sandbox)
- Create a test patient
- Create an encounter (consultation)
- Save symptom observations
- Save consultation summary
- Close the encounter

Expected output:
```
✅ Created test patient: 12345
✅ Created encounter: 67890
✅ Saved: Chest pain
✅ Saved: Dyspnea
✅ Saved consultation summary
✅ Encounter closed
```

---

## 🔑 Key Concepts

### 1. FHIR Resources

**FHIR** = Fast Healthcare Interoperability Resources

**Key resources for Aura:**

| Resource | Purpose | When to Use |
|----------|---------|-------------|
| **Patient** | Patient demographics, NHS Number | Start of consultation |
| **Encounter** | Consultation session | Create at start, close at end |
| **Observation** | Symptoms, vital signs | Each symptom reported |
| **DocumentReference** | Consultation summary | End of consultation |
| **QuestionnaireResponse** | Structured symptom data | OLDCARTS assessment |

### 2. NHS Number

**10-digit unique patient identifier** in the UK NHS

Format: `943 476 5870` (spaces optional)

**Validation:** Uses Modulus 11 algorithm

```python
from ehr_integration_example import validate_nhs_number

is_valid = validate_nhs_number("9434765870")  # True
```

### 3. SNOMED CT Codes

**Clinical terminology** for symptoms, diagnoses, procedures

Examples:
- `29857009` - Chest pain
- `267036007` - Dyspnea (shortness of breath)
- `21522001` - Abdominal pain

**Where to find codes:**
- https://termbrowser.nhs.uk/ (official NHS browser)
- https://browser.ihtsdotools.org/ (international SNOMED browser)

---

## 📖 Basic Usage Examples

### Example 1: Find Patient by NHS Number

```python
from ehr_integration_example import SimpleFHIRClient

# Connect to SystmOne FHIR endpoint
client = SimpleFHIRClient("https://api.systmone.nhs.uk/fhir")

# Search for patient
patient = client.search_patient("9434765870")

if patient:
    print(f"Found: {patient.name[0].given[0]} {patient.name[0].family}")
else:
    print("Patient not found")
```

### Example 2: Create Consultation

```python
# Create encounter (consultation session)
encounter = client.create_encounter(
    patient_id=patient.id,
    encounter_type="virtual"
)

print(f"Started consultation: {encounter.id}")
```

### Example 3: Save Symptom

```python
# Save symptom observation
observation = client.create_observation(
    patient_id=patient.id,
    encounter_id=encounter.id,
    code="29857009",  # SNOMED code for chest pain
    display="Chest pain",
    value="Severity: 8/10, crushing, radiates to left arm"
)

print(f"Saved symptom: {observation.id}")
```

### Example 4: Save Consultation Summary

```python
# Save AI-generated summary
summary_text = """
Patient reports chest pain, severity 8/10, crushing character.
Duration 2 hours. Associated shortness of breath.

Assessment: Possible acute coronary syndrome
Urgency: Emergency
Recommendation: Immediate A&E evaluation
"""

document = client.create_document(
    patient_id=patient.id,
    encounter_id=encounter.id,
    title="Aura AI Consultation Summary",
    content=summary_text
)

print(f"Saved summary: {document.id}")
```

### Example 5: Close Consultation

```python
# Mark encounter as finished
client.close_encounter(encounter.id)
print("Consultation closed")
```

---

## 🔐 Authentication (Production)

### NHS CIS2 OAuth 2.0 Flow

```python
from authlib.integrations.requests_client import OAuth2Session

# Your credentials (from NHS Digital API Portal)
client_id = "your_client_id"
client_secret = "your_client_secret"
redirect_uri = "https://your-app.com/callback"

# Initialize OAuth session
oauth = OAuth2Session(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="openid profile patient/*.read encounter/*.write"
)

# Step 1: Get authorization URL
auth_url, state = oauth.create_authorization_url(
    "https://auth.national.nhs.uk/authorize"
)

print(f"Login URL: {auth_url}")
# Redirect user to this URL for login

# Step 2: Exchange code for token (after user logs in)
token = oauth.fetch_token(
    "https://auth.national.nhs.uk/token",
    authorization_response=callback_url  # Full callback URL with code
)

access_token = token['access_token']

# Step 3: Use token in API requests
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/fhir+json"
}
```

---

## 🔗 Integration with Aura Clinician Mode

### Modify `clinician_mode.py`

```python
# llm-medical-container/clinician_mode.py

from ehr_integration_example import SimpleFHIRClient
import os

# Feature flag
EHR_ENABLED = os.getenv("EHR_INTEGRATION_ENABLED", "false") == "true"

class ClinicianSession:
    def __init__(self, session_id, llm_chat, llm_chat_simple):
        # ... existing code ...
        
        # EHR client
        if EHR_ENABLED:
            self.ehr_client = SimpleFHIRClient(
                os.getenv("SYSTMONE_FHIR_URL")
            )
            self.ehr_encounter_id = None
    
    def start_assessment(self, chief_complaint, nhs_number=None):
        """Start assessment with EHR integration"""
        
        # Normal Aura assessment
        self.chief_complaint = chief_complaint
        
        # Create EHR encounter if enabled
        if EHR_ENABLED and nhs_number:
            patient = self.ehr_client.search_patient(nhs_number)
            
            if patient:
                encounter = self.ehr_client.create_encounter(patient.id)
                self.ehr_encounter_id = encounter.id
                print(f"[EHR] Created encounter: {encounter.id}")
    
    def save_symptom(self, symptom_code, symptom_name, symptom_value):
        """Save symptom to EHR"""
        
        if EHR_ENABLED and self.ehr_encounter_id:
            self.ehr_client.create_observation(
                patient_id=self.patient_id,
                encounter_id=self.ehr_encounter_id,
                code=symptom_code,
                display=symptom_name,
                value=symptom_value
            )
    
    def finalize_assessment(self):
        """Complete assessment and save to EHR"""
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save to EHR
        if EHR_ENABLED and self.ehr_encounter_id:
            self.ehr_client.create_document(
                patient_id=self.patient_id,
                encounter_id=self.ehr_encounter_id,
                title="Aura AI Consultation Summary",
                content=summary
            )
            
            self.ehr_client.close_encounter(self.ehr_encounter_id)
            print("[EHR] Consultation saved to SystmOne")
        
        return summary
```

---

## 🧪 Testing Strategy

### 1. Development Testing (HAPI FHIR)

Use public HAPI FHIR test server:

```python
client = SimpleFHIRClient("https://hapi.fhir.org/baseR4")
# No authentication required
# Data is temporary (reset periodically)
```

### 2. NHS Test Environment

Request access to NHS test environment:
- Contact NHS Digital API Support
- Use test NHS Numbers (not real patients)
- Test OAuth flow with test credentials

### 3. Unit Tests

```bash
pytest tests/test_ehr_integration.py -v
```

Example test:
```python
def test_create_encounter():
    client = SimpleFHIRClient("https://hapi.fhir.org/baseR4")
    
    # Create test patient
    patient = create_test_patient(client)
    
    # Create encounter
    encounter = client.create_encounter(patient.id)
    
    assert encounter is not None
    assert encounter.status == "in-progress"
    assert encounter.subject.reference == f"Patient/{patient.id}"
```

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Aura Medical Chatbot                       │
│                                                              │
│  Patient: "I have chest pain"                               │
│      ↓                                                       │
│  Clinician Mode: OLDCARTS assessment                        │
│      ↓                                                       │
│  Collect: onset, location, severity, character, etc.        │
│      ↓                                                       │
│  Generate: differential diagnosis, urgency level            │
│      ↓                                                       │
│  Create: consultation summary                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ FHIR API Call
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  EHR Integration Layer                       │
│                                                              │
│  1. Find Patient (NHS Number) → Patient/123                 │
│  2. Create Encounter → Encounter/456                        │
│  3. Save Observations (symptoms) → Observation/789          │
│  4. Save Document (summary) → DocumentReference/101         │
│  5. Close Encounter → status: "finished"                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ OAuth 2.0 Bearer Token
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   SystmOne FHIR API                          │
│              (https://api.systmone.nhs.uk/fhir)              │
│                                                              │
│  • Validates authentication                                 │
│  • Checks permissions (RBAC)                                │
│  • Saves data to patient record                             │
│  • Logs access (audit trail)                                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   SystmOne EHR Database                      │
│                                                              │
│  Patient Record Updated:                                    │
│  • New encounter logged                                     │
│  • Symptoms documented                                      │
│  • AI consultation summary attached                         │
│  • Available for clinician review                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Common Pitfalls & Solutions

### 1. Invalid NHS Number

**Problem:** NHS Number validation fails

**Solution:**
```python
from ehr_integration_example import validate_nhs_number

# Always validate before API call
if not validate_nhs_number(nhs_number):
    raise ValueError("Invalid NHS Number")
```

### 2. Missing SNOMED Codes

**Problem:** Don't know which SNOMED code to use

**Solution:**
- Use NHS SNOMED browser: https://termbrowser.nhs.uk/
- Search symptom name (e.g., "chest pain")
- Use the code in your API call

### 3. Authentication Expired

**Problem:** Access token expired (401 Unauthorized)

**Solution:**
```python
# Refresh token before expiry
if token_expires_soon():
    new_token = oauth.refresh_token(
        "https://auth.national.nhs.uk/token",
        refresh_token=refresh_token
    )
    access_token = new_token['access_token']
```

### 4. FHIR Validation Errors

**Problem:** API returns 400 (Bad Request) with validation error

**Solution:**
- Check resource structure matches FHIR UK Core profiles
- Use `fhir.resources` library to validate locally:
```python
from fhir.resources.patient import Patient

try:
    patient = Patient(**patient_data)
    # Valid FHIR resource
except Exception as e:
    print(f"Invalid FHIR: {e}")
```

---

## 🎯 Next Steps

### For Development
1. Run example: `python ehr_integration_example.py`
2. Read full guide: `docs/SYSTMONE_EHR_INTEGRATION_GUIDE.md`
3. Integrate with clinician mode
4. Write unit tests

### For Production
1. Register with NHS Digital API Platform
2. Contact TPP (SystmOne vendor) for API access
3. Complete clinical safety assessment
4. Obtain SNOMED CT license
5. Complete NHS DSPT
6. Deploy to test environment
7. Pilot with NHS trust
8. Production deployment

---

## 📚 Resources

### Official Documentation
- **NHS Digital Developer Portal:** https://digital.nhs.uk/developer
- **FHIR UK Core:** https://simplifier.net/HL7FHIRUKCoreR4
- **SNOMED Browser:** https://termbrowser.nhs.uk/
- **NHS DSPT:** https://www.dsptoolkit.nhs.uk/

### Support
- **NHS API Support:** api.management@nhs.net
- **TPP (SystmOne):** https://tpp-uk.com/contact/
- **Clinical Safety:** safetyguidance@nhsx.nhs.uk

### Test Environments
- **HAPI FHIR Test Server:** https://hapi.fhir.org/baseR4
- **FHIR Validator:** https://validator.fhir.org/

---

## 🔍 FAQ

**Q: Do I need NHS credentials to test?**  
A: No, use HAPI FHIR test server for initial development. NHS credentials needed for production.

**Q: How long does NHS onboarding take?**  
A: 6-12 months for full production deployment.

**Q: Can I use this with other UK EHRs (EMIS, Vision)?**  
A: Yes! FHIR is a standard. Replace endpoint URL and follow vendor-specific auth.

**Q: Is SNOMED CT license free?**  
A: Yes, free for NHS use. Apply via NHS TRUD portal.

**Q: What about GDPR compliance?**  
A: Complete NHS DSPT, sign data sharing agreements, implement consent management.

**Q: Can I deploy Aura as a medical device?**  
A: Requires MHRA registration as Class IIa medical device. Consult regulatory expert.

---

**Ready to start?** Run the example and explore the full integration guide!

```bash
python llm-medical-container/ehr_integration_example.py
```

**Questions?** See full guide: `docs/SYSTMONE_EHR_INTEGRATION_GUIDE.md`

