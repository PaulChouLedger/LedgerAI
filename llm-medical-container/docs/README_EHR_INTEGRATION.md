# EHR Integration - SystmOne FHIR API

This directory contains the EHR (Electronic Health Record) integration components for Aura Medical Chatbot, specifically designed for **SystmOne** (UK NHS).

---

## 📁 Files Overview

### Documentation
- **`docs/SYSTMONE_EHR_INTEGRATION_GUIDE.md`** - Comprehensive integration guide (60+ pages)
  - Architecture design
  - NHS regulatory requirements (GDPR, clinical safety)
  - FHIR implementation details
  - Security & authentication (NHS CIS2)
  - Code examples
  - Testing & deployment strategies
  - Compliance checklist

- **`docs/EHR_INTEGRATION_QUICKSTART.md`** - Quick start guide for developers
  - 15-minute introduction
  - Key concepts (FHIR, NHS Number, SNOMED CT)
  - Basic usage examples
  - Common pitfalls & solutions

### Code
- **`ehr_integration_example.py`** - Working implementation example
  - Complete consultation workflow
  - FHIR client implementation
  - NHS Number validation
  - SNOMED CT mapping examples
  - Runnable demo with HAPI FHIR test server

### Dependencies
- **`requirements_ehr.txt`** - Python dependencies for EHR integration
  - fhir.resources (FHIR client)
  - authlib (OAuth 2.0 for NHS authentication)
  - requests (HTTP client)
  - And more...

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ehr.txt
```

### 2. Run the Example

```bash
python ehr_integration_example.py
```

This will:
1. Connect to HAPI FHIR test server (public sandbox)
2. Create a test patient with NHS Number
3. Create an encounter (consultation)
4. Save symptom observations (chest pain, dyspnea)
5. Save AI-generated consultation summary
6. Close the encounter

**Expected Output:**
```
======================================================================
EXAMPLE: Aura Medical Chatbot → SystmOne Integration
======================================================================

Step 1: Creating test patient...
✅ Created test patient: 12345

Step 2: Creating encounter (consultation)...
✅ Created encounter: 67890

Step 3: Saving symptom observations...
✅ Saved: Chest pain
✅ Saved: Dyspnea

Step 4: Saving consultation summary...
✅ Saved consultation summary: 98765

Step 5: Closing encounter...
✅ Encounter closed: 67890

======================================================================
WORKFLOW COMPLETE
======================================================================

Patient ID: 12345
Encounter ID: 67890
Observations Created: 2
Document ID: 98765

All data has been saved to the FHIR server.
In production, this would appear in SystmOne for clinician review.
```

### 3. Read the Guides

- **New to healthcare IT?** Start with `docs/EHR_INTEGRATION_QUICKSTART.md`
- **Ready to implement?** Read `docs/SYSTMONE_EHR_INTEGRATION_GUIDE.md`

---

## 🎯 What Does This Integration Do?

### Data Flow

```
┌────────────────────────┐
│   Aura Chatbot         │
│   "I have chest pain"  │
└───────────┬────────────┘
            │
            │ Collects: symptoms, severity, duration, etc.
            ↓
┌────────────────────────┐
│  EHR Integration       │
│  (ehr_integration.py)  │
└───────────┬────────────┘
            │
            │ FHIR API calls
            ↓
┌────────────────────────┐
│   SystmOne EHR         │
│   Patient Record       │
└────────────────────────┘
```

### Specific Actions

1. **Find Patient** - Retrieve patient by NHS Number
2. **Create Encounter** - Start new consultation session
3. **Save Symptoms** - Record symptoms as FHIR Observations (with SNOMED codes)
4. **Save Summary** - Store AI-generated consultation summary as DocumentReference
5. **Close Encounter** - Mark consultation as finished

### Result

All data appears in SystmOne for clinician review:
- ✅ Encounter logged
- ✅ Symptoms documented with SNOMED codes
- ✅ AI consultation summary attached
- ✅ Available for GP/clinician decision-making

---

## 🔑 Key Technologies

### 1. FHIR (Fast Healthcare Interoperability Resources)

**What:** International standard for healthcare data exchange  
**Why:** Used by NHS Digital for all modern integrations  
**Version:** FHIR R4 with UK Core extensions

**Resources Used:**
- `Patient` - Patient demographics
- `Encounter` - Consultation session
- `Observation` - Symptoms, vital signs
- `DocumentReference` - Consultation summaries

### 2. NHS CIS2 (Care Identity Service 2)

**What:** NHS authentication service  
**How:** OAuth 2.0 / OpenID Connect  
**Why:** Secure access control for healthcare data

### 3. SNOMED CT (Systematized Nomenclature of Medicine)

**What:** Clinical terminology (codes for symptoms, diagnoses, procedures)  
**Example:** `29857009` = Chest pain  
**Why:** Mandatory for NHS clinical systems

### 4. NHS Number

**What:** 10-digit unique patient identifier in UK NHS  
**Format:** `943 476 5870`  
**Validation:** Modulus 11 algorithm

---

## 🏗️ Architecture

### Current Aura Architecture

```
Aura Control (GUI)
    ↓
Whisper (STT) → LLM Medical Container → RAG System
                       ↓
               Clinician Mode
               (Adaptive Diagnostic)
```

### With EHR Integration

```
Aura Control (GUI)
    ↓
Whisper (STT) → LLM Medical Container → RAG System
                       ↓
               Clinician Mode
               (Adaptive Diagnostic)
                       ↓
                [NEW] EHR Integration Layer
                       ↓
                  FHIR API
                       ↓
                NHS Authentication
                       ↓
                  SystmOne EHR
```

---

## 📖 Usage Examples

### Example 1: Basic Patient Lookup

```python
from ehr_integration_example import SimpleFHIRClient, validate_nhs_number

# Validate NHS Number
nhs_number = "9434765870"
if not validate_nhs_number(nhs_number):
    raise ValueError("Invalid NHS Number")

# Connect to SystmOne
client = SimpleFHIRClient("https://api.systmone.nhs.uk/fhir")

# Find patient
patient = client.search_patient(nhs_number)

if patient:
    print(f"Found: {patient.name[0].family}, {patient.name[0].given[0]}")
    print(f"DOB: {patient.birthDate}")
    print(f"Gender: {patient.gender}")
```

### Example 2: Complete Consultation Workflow

```python
from ehr_integration_example import SimpleFHIRClient

client = SimpleFHIRClient("https://api.systmone.nhs.uk/fhir")

# 1. Find patient
patient = client.search_patient("9434765870")

# 2. Create encounter
encounter = client.create_encounter(patient.id)

# 3. Save symptoms
client.create_observation(
    patient_id=patient.id,
    encounter_id=encounter.id,
    code="29857009",  # SNOMED code
    display="Chest pain",
    value="Severity 8/10, crushing, radiates to left arm"
)

# 4. Save consultation summary
summary = "Patient reports acute chest pain... [AI assessment]"
client.create_document(
    patient_id=patient.id,
    encounter_id=encounter.id,
    title="Aura AI Consultation",
    content=summary
)

# 5. Close encounter
client.close_encounter(encounter.id)

print("✅ Consultation saved to SystmOne")
```

### Example 3: Integration with Clinician Mode

```python
# In clinician_mode.py

from ehr_integration_example import SimpleFHIRClient
import os

class ClinicianSession:
    def __init__(self, session_id, llm_chat, llm_chat_simple):
        # ... existing code ...
        
        # Add EHR integration
        self.ehr_enabled = os.getenv("EHR_INTEGRATION_ENABLED") == "true"
        if self.ehr_enabled:
            self.ehr_client = SimpleFHIRClient(
                os.getenv("SYSTMONE_FHIR_URL")
            )
    
    def start_assessment(self, chief_complaint, nhs_number=None):
        # Start normal assessment
        self.chief_complaint = chief_complaint
        
        # Create EHR encounter
        if self.ehr_enabled and nhs_number:
            patient = self.ehr_client.search_patient(nhs_number)
            encounter = self.ehr_client.create_encounter(patient.id)
            self.ehr_encounter_id = encounter.id
    
    def finalize_assessment(self):
        # Generate summary
        summary = self._generate_summary()
        
        # Save to EHR
        if self.ehr_enabled:
            self.ehr_client.create_document(
                patient_id=self.patient_id,
                encounter_id=self.ehr_encounter_id,
                title="Aura AI Consultation",
                content=summary
            )
            self.ehr_client.close_encounter(self.ehr_encounter_id)
        
        return summary
```

---

## 🧪 Testing

### Development Testing (No NHS Credentials Required)

Use **HAPI FHIR Test Server** (public sandbox):

```python
# Test with public FHIR server
client = SimpleFHIRClient("https://hapi.fhir.org/baseR4")

# No authentication required
# Data resets periodically
# Perfect for development
```

### Unit Tests

```bash
# Run tests
pytest tests/test_ehr_integration.py -v

# With coverage
pytest tests/test_ehr_integration.py --cov=ehr_integration --cov-report=html
```

### Integration Tests

Request access to NHS test environment:
1. Register at https://digital.nhs.uk/developer
2. Apply for test credentials
3. Use test NHS Numbers (not real patients)

---

## 🔐 Security & Compliance

### Required Before Production

- [ ] **Clinical Safety Officer** appointed
- [ ] **DCB0129** clinical risk assessment completed
- [ ] **DCB0160** deployment risk assessment completed
- [ ] **NHS DSPT** (Data Security toolkit) completed
- [ ] **GDPR compliance** documented
- [ ] **Data sharing agreement** with NHS trust
- [ ] **SNOMED CT license** obtained (free for NHS use)
- [ ] **Penetration testing** completed
- [ ] **Audit logging** implemented

### Environment Variables (Production)

```bash
# config.env

# SystmOne FHIR endpoint
SYSTMONE_FHIR_URL=https://api.systmone.nhs.uk/fhir

# NHS CIS2 authentication
NHS_CLIENT_ID=your_client_id
NHS_CLIENT_SECRET=your_client_secret
NHS_REDIRECT_URI=https://your-app.nhs.uk/callback

# Feature flags
EHR_INTEGRATION_ENABLED=true

# Security
TOKEN_ENCRYPTION_KEY=your_secure_key_from_kms
```

---

## 📚 Additional Resources

### Official NHS Resources
- **NHS Digital Developer Portal:** https://digital.nhs.uk/developer
- **FHIR UK Core:** https://simplifier.net/HL7FHIRUKCoreR4
- **GP Connect:** https://digital.nhs.uk/services/gp-connect
- **SNOMED Browser:** https://termbrowser.nhs.uk/
- **NHS DSPT:** https://www.dsptoolkit.nhs.uk/

### SystmOne / TPP
- **TPP Website:** https://tpp-uk.com/
- **Contact:** integrations@tpp-uk.com
- **Support:** support@tpp-uk.com

### FHIR Resources
- **FHIR Specification:** https://www.hl7.org/fhir/
- **HAPI FHIR Test Server:** https://hapi.fhir.org/baseR4
- **FHIR Validator:** https://validator.fhir.org/

### Python Libraries
- **fhir.resources:** https://pypi.org/project/fhir.resources/
- **authlib:** https://docs.authlib.org/
- **requests:** https://requests.readthedocs.io/

---

## 🗺️ Implementation Roadmap

### Phase 1: Development (Weeks 1-4)
- [x] Study EHR integration guide
- [x] Install dependencies
- [x] Run example code
- [ ] Integrate with clinician mode
- [ ] Write unit tests

### Phase 2: NHS Onboarding (Weeks 5-12)
- [ ] Register with NHS Digital API Platform
- [ ] Contact TPP for SystmOne API access
- [ ] Appoint Clinical Safety Officer
- [ ] Begin clinical risk assessment

### Phase 3: Compliance (Weeks 13-20)
- [ ] Complete NHS DSPT
- [ ] Obtain SNOMED CT license
- [ ] Sign data sharing agreements
- [ ] Finalize Clinical Safety Case

### Phase 4: Testing (Weeks 21-28)
- [ ] Deploy to NHS test environment
- [ ] User acceptance testing with NHS trust
- [ ] Security & penetration testing
- [ ] Clinical validation

### Phase 5: Pilot (Weeks 29-36)
- [ ] Pilot with single NHS practice
- [ ] Gather feedback
- [ ] Iterate on design
- [ ] Monitor performance

### Phase 6: Production (Week 37+)
- [ ] Production deployment
- [ ] Ongoing monitoring
- [ ] Continuous improvement
- [ ] Scale to more practices

**Estimated Timeline:** 9-12 months from start to full production

---

## ❓ FAQ

**Q: Do I need NHS credentials to test the example?**  
A: No, the example uses HAPI FHIR public test server (no auth required).

**Q: How long does NHS onboarding take?**  
A: Typically 6-12 months for full production deployment.

**Q: Is SNOMED CT license expensive?**  
A: No, it's free for NHS use. Apply via NHS TRUD portal.

**Q: Can I use this with other UK EHRs (EMIS, Vision)?**  
A: Yes! FHIR is a standard. Just change the endpoint URL and follow vendor-specific auth.

**Q: What about patient consent?**  
A: Implement consent management as part of GDPR compliance. Document in DSPT.

**Q: Is Aura considered a medical device?**  
A: If making clinical decisions, yes (Class IIa). Requires MHRA registration. Consult regulatory expert.

**Q: What if SystmOne API is down?**  
A: Implement retry logic (already in code) and fallback to local storage. Don't block clinical workflow.

**Q: How much does SystmOne API access cost?**  
A: Contact TPP directly. Pricing varies by usage and agreement type.

---

## 🆘 Support & Contact

### NHS Digital API Support
- **Email:** api.management@nhs.net
- **Portal:** https://digital.nhs.uk/developer/support

### TPP (SystmOne Vendor)
- **Website:** https://tpp-uk.com/contact/
- **Integration Team:** integrations@tpp-uk.com
- **Technical Support:** support@tpp-uk.com

### Clinical Safety
- **NHS Digital:** safetyguidance@nhsx.nhs.uk
- **Consult a qualified Clinical Safety Officer**

---

## 📝 License & Disclaimer

**Disclaimer:** This integration code is provided as-is for educational and development purposes. It is NOT certified for clinical use without proper regulatory approval, clinical safety assessment, and NHS compliance.

**Before Production Use:**
- Complete all regulatory requirements
- Obtain necessary licenses and approvals
- Conduct thorough clinical validation
- Implement comprehensive audit logging
- Establish clinical governance

**For Clinical Deployment:** Consult with NHS Digital, clinical safety experts, and regulatory advisors.

---

**Version:** 1.0  
**Last Updated:** October 21, 2025  
**Author:** Aura Medical Team  
**Status:** Development / Reference Implementation

---

**Ready to get started?** Run the example and explore the integration guides! 🚀

```bash
python ehr_integration_example.py
```

