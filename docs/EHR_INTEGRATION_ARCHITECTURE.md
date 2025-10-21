# EHR Integration Architecture - Visual Reference

## System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         AURA MEDICAL CHATBOT                            │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │   Whisper    │  │  LLM Medical │  │  RAG System  │                │
│  │   (STT)      │  │  Container   │  │ (Guidelines) │                │
│  │  Port 5051   │  │  Port 11434  │  │ Port 11435   │                │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │
│         │                 │                  │                         │
│         └─────────────────┼──────────────────┘                         │
│                           │                                            │
│  ┌────────────────────────▼──────────────────────────┐                │
│  │        Clinician Mode (Adaptive Diagnostic)       │                │
│  │  • OLDCARTS symptom assessment                    │                │
│  │  • Differential diagnosis generation              │                │
│  │  • Clinical urgency assessment                    │                │
│  │  • Consultation summary generation                │                │
│  └────────────────────────┬──────────────────────────┘                │
│                           │                                            │
└───────────────────────────┼────────────────────────────────────────────┘
                            │
                            │ REST API calls
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                   EHR INTEGRATION LAYER (NEW)                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │              SimpleFHIRClient / SystmOneClient                │     │
│  │                                                               │     │
│  │  Methods:                                                     │     │
│  │  • search_patient(nhs_number)                                │     │
│  │  • create_encounter(patient_id)                              │     │
│  │  • create_observation(symptom_data)                          │     │
│  │  • create_document(consultation_summary)                     │     │
│  │  • close_encounter(encounter_id)                             │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                  NHS Authentication Module                    │     │
│  │                                                               │     │
│  │  • OAuth 2.0 / OpenID Connect                                │     │
│  │  • NHS CIS2 (Care Identity Service)                          │     │
│  │  • Token management & refresh                                │     │
│  │  • Secure credential storage                                 │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                    SNOMED CT Mapping                          │     │
│  │                                                               │     │
│  │  • Symptom → SNOMED code translation                         │     │
│  │  • Clinical terminology validation                           │     │
│  │  • UK terminology standards                                  │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                   Audit & Compliance                          │     │
│  │                                                               │     │
│  │  • Access logging (who, what, when)                          │     │
│  │  • Clinical safety monitoring                                │     │
│  │  • GDPR compliance tracking                                  │     │
│  │  • NHS IG Toolkit requirements                               │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            │ HTTPS + OAuth 2.0 Bearer Token
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                    SYSTMONE FHIR API (NHS)                             │
│                 https://api.systmone.nhs.uk/fhir                       │
│                                                                         │
│  Endpoints:                                                             │
│  • GET  /Patient?identifier={nhs_number}                               │
│  • POST /Encounter                                                     │
│  • POST /Observation                                                   │
│  • POST /DocumentReference                                             │
│  • PUT  /Encounter/{id}                                                │
│                                                                         │
│  FHIR Resources (UK Core):                                             │
│  • Patient (demographics + NHS Number)                                 │
│  • Encounter (consultation session)                                    │
│  • Observation (symptoms, vital signs)                                 │
│  • DocumentReference (clinical documents)                              │
│  • QuestionnaireResponse (structured assessments)                      │
│                                                                         │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            │ Database writes
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                      SYSTMONE EHR DATABASE                             │
│                    (Patient Medical Records)                           │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │  Patient Record  │  │    Encounter     │  │   Observations   │    │
│  │                  │  │                  │  │                  │    │
│  │  NHS Number:     │  │  Date/Time:      │  │  Chest pain      │    │
│  │  9434765870      │  │  2025-10-21      │  │  (SNOMED:        │    │
│  │                  │  │                  │  │   29857009)      │    │
│  │  Name:           │  │  Type:           │  │                  │    │
│  │  John Smith      │  │  Telemedicine    │  │  Severity: 8/10  │    │
│  │                  │  │                  │  │                  │    │
│  │  DOB:            │  │  Status:         │  │  Character:      │    │
│  │  1980-01-01      │  │  Finished        │  │  Crushing        │    │
│  │                  │  │                  │  │                  │    │
│  │  GP Practice:    │  │  Provider:       │  │  Duration:       │    │
│  │  NHS Trust XYZ   │  │  Aura AI         │  │  2 hours         │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              Consultation Summary Document                      │  │
│  │                                                                 │  │
│  │  Generated by: Aura Medical AI Assistant                       │  │
│  │  Date: 2025-10-21 14:30:00                                     │  │
│  │                                                                 │  │
│  │  CHIEF COMPLAINT: Chest pain                                   │  │
│  │                                                                 │  │
│  │  ASSESSMENT:                                                    │  │
│  │  Patient reports crushing chest pain, severity 8/10...         │  │
│  │                                                                 │  │
│  │  DIFFERENTIAL DIAGNOSIS:                                       │  │
│  │  1. Acute Coronary Syndrome (ACS) - HIGH PROBABILITY           │  │
│  │  2. Pulmonary Embolism (PE)                                    │  │
│  │  3. Aortic Dissection                                          │  │
│  │                                                                 │  │
│  │  URGENCY: EMERGENCY                                            │  │
│  │                                                                 │  │
│  │  RECOMMENDATION:                                               │  │
│  │  Immediate emergency department evaluation recommended.        │  │
│  │  Call 999 or proceed to nearest A&E immediately.               │  │
│  │                                                                 │  │
│  │  [Available for clinician review and action]                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Sequence

### Complete Consultation Workflow

```
┌─────────┐                ┌──────────┐              ┌─────────┐              ┌──────────┐
│ Patient │                │   Aura   │              │   EHR   │              │ SystmOne │
│         │                │ Chatbot  │              │ Layer   │              │   API    │
└────┬────┘                └────┬─────┘              └────┬────┘              └────┬─────┘
     │                          │                         │                        │
     │ "I have chest pain"      │                         │                        │
     ├─────────────────────────>│                         │                        │
     │                          │                         │                        │
     │                          │ OLDCARTS Assessment     │                        │
     │                          │ • Onset?               │                        │
     │                          │ • Location?            │                        │
     │<─────────────────────────┤ • Duration?            │                        │
     │                          │ • Character?           │                        │
     │ "2 hours ago, crushing,  │ • Severity?            │                        │
     │  center of chest"        │                         │                        │
     ├─────────────────────────>│                         │                        │
     │                          │                         │                        │
     │                          │ Generate differential   │                        │
     │                          │ diagnosis & summary     │                        │
     │                          │                         │                        │
     │                          │ search_patient(NHS#)    │                        │
     │                          ├────────────────────────>│                        │
     │                          │                         │ GET /Patient?id=...    │
     │                          │                         ├───────────────────────>│
     │                          │                         │                        │
     │                          │                         │ Patient resource       │
     │                          │                         │<───────────────────────┤
     │                          │ Patient found           │                        │
     │                          │<────────────────────────┤                        │
     │                          │                         │                        │
     │                          │ create_encounter()      │                        │
     │                          ├────────────────────────>│                        │
     │                          │                         │ POST /Encounter        │
     │                          │                         ├───────────────────────>│
     │                          │                         │                        │
     │                          │                         │ Encounter/123 created  │
     │                          │                         │<───────────────────────┤
     │                          │ Encounter ID: 123       │                        │
     │                          │<────────────────────────┤                        │
     │                          │                         │                        │
     │                          │ create_observation()    │                        │
     │                          │ (chest pain)            │                        │
     │                          ├────────────────────────>│                        │
     │                          │                         │ POST /Observation      │
     │                          │                         ├───────────────────────>│
     │                          │                         │ Observation/456 saved  │
     │                          │                         │<───────────────────────┤
     │                          │                         │                        │
     │                          │ create_document()       │                        │
     │                          │ (consultation summary)  │                        │
     │                          ├────────────────────────>│                        │
     │                          │                         │ POST /DocumentRef      │
     │                          │                         ├───────────────────────>│
     │                          │                         │ Document/789 saved     │
     │                          │                         │<───────────────────────┤
     │                          │                         │                        │
     │                          │ close_encounter()       │                        │
     │                          ├────────────────────────>│                        │
     │                          │                         │ PUT /Encounter/123     │
     │                          │                         │ (status=finished)      │
     │                          │                         ├───────────────────────>│
     │                          │                         │ Encounter closed       │
     │                          │                         │<───────────────────────┤
     │                          │ Success                 │                        │
     │                          │<────────────────────────┤                        │
     │                          │                         │                        │
     │ "Emergency - go to A&E"  │                         │                        │
     │<─────────────────────────┤                         │                        │
     │                          │                         │                        │

[Data now available in SystmOne for GP/clinician review]
```

---

## Integration Points

### 1. Patient Identification

```
Aura Input: NHS Number (e.g., "9434765870")
    ↓
Validation: Modulus 11 check
    ↓
FHIR Query: GET /Patient?identifier=https://fhir.nhs.uk/Id/nhs-number|9434765870
    ↓
Response: Patient FHIR resource with demographics
```

### 2. Encounter Creation

```
Aura: Start of consultation
    ↓
FHIR: Create Encounter resource
    ↓
Type: Telemedicine consultation (SNOMED: 448337001)
Status: in-progress
    ↓
Returns: Encounter ID for subsequent data linkage
```

### 3. Symptom Recording

```
Aura Output: 
{
    "symptom": "Chest pain",
    "severity": "8/10",
    "character": "crushing",
    "duration": "2 hours"
}
    ↓
SNOMED Mapping: "Chest pain" → 29857009
    ↓
FHIR: Create Observation resource
    ↓
Linked to: Patient + Encounter
```

### 4. Summary Storage

```
Aura Generated Summary:
- Chief complaint
- OLDCARTS details
- Differential diagnoses
- Urgency assessment
- Recommendations
    ↓
FHIR: Create DocumentReference
    ↓
Content: Base64-encoded text
Type: Clinical consultation report (SNOMED: 371531000)
```

### 5. Consultation Closure

```
Aura: Assessment complete
    ↓
FHIR: Update Encounter
    ↓
Status: in-progress → finished
Period.end: Current timestamp
    ↓
Result: Complete consultation record in SystmOne
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Clinician)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ 1. Login request
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    NHS CIS2 (Authentication)                     │
│                                                                  │
│  Login Page:                                                     │
│  • Username/password                                             │
│  • Multi-factor authentication (MFA)                             │
│  • NHS Smartcard (optional)                                      │
│                                                                  │
│  OAuth 2.0 Authorization Code Flow:                              │
│  1. Authorization request                                        │
│  2. User authenticates                                           │
│  3. Authorization code issued                                    │
│  4. Exchange code for access token                               │
│  5. Refresh token for extended sessions                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ 2. Access token (JWT)
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Aura EHR Integration                        │
│                                                                  │
│  Token Management:                                               │
│  • Secure storage (encrypted)                                    │
│  • Automatic refresh before expiry                               │
│  • Revocation on logout                                          │
│                                                                  │
│  Request Headers:                                                │
│  Authorization: Bearer {access_token}                            │
│  Accept: application/fhir+json                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ 3. FHIR API request
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SystmOne FHIR API                             │
│                                                                  │
│  Security Checks:                                                │
│  1. Token validation                                             │
│  2. Scope verification (patient/*.read, encounter/*.write)       │
│  3. Role-based access control (RBAC)                             │
│  4. Data access audit logging                                    │
│  5. Rate limiting                                                │
│                                                                  │
│  If authorized:                                                  │
│    → Process request                                             │
│    → Return FHIR resource                                        │
│  If unauthorized:                                                │
│    → Return 401 Unauthorized                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Compliance Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                   NHS COMPLIANCE REQUIREMENTS                    │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                     ↓
┌───────────────┐   ┌───────────────┐   ┌────────────────┐
│  Clinical     │   │  Information  │   │ Data Protection│
│  Safety       │   │  Governance   │   │ & GDPR         │
│               │   │               │   │                │
│ • DCB0129     │   │ • NHS DSPT    │   │ • Data Privacy │
│ • DCB0160     │   │ • IG Toolkit  │   │ • Consent Mgmt │
│ • Hazard Log  │   │ • Staff Train │   │ • Breach Plan  │
│ • CSO         │   │ • RBAC        │   │ • DPO          │
│ • Incidents   │   │ • Audit Trail │   │ • DPIA         │
└───────────────┘   └───────────────┘   └────────────────┘
        │                    │                     │
        └────────────────────┼─────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Technical Standards                            │
│                                                                  │
│  • FHIR UK Core (FHIR R4 + UK extensions)                       │
│  • SNOMED CT UK Edition                                          │
│  • NHS Number validation                                         │
│  • dm+d (Dictionary of Medicines & Devices)                     │
│  • NHS Data Dictionary                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEVELOPMENT                               │
│                                                                  │
│  Local Machine                                                   │
│  • Docker containers (Whisper, LLM, RAG)                        │
│  • HAPI FHIR test server (public)                               │
│  • No NHS authentication                                         │
│  • Mock data                                                     │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ git push
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     NHS TEST ENVIRONMENT                         │
│                                                                  │
│  NHS Digital Test Platform                                       │
│  • Aura containers deployed                                      │
│  • SystmOne test instance                                        │
│  • NHS CIS2 test authentication                                  │
│  • Test NHS Numbers (synthetic patients)                         │
│  • UAT with NHS trust                                            │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ Approval + Go-live
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    NHS PRODUCTION ENVIRONMENT                    │
│                                                                  │
│  NHS Digital Production Platform (or NHS Trust infrastructure)   │
│  • Aura production deployment                                    │
│  • SystmOne production instance                                  │
│  • NHS CIS2 production authentication                            │
│  • Real NHS Numbers (live patients)                              │
│  • 24/7 monitoring & support                                     │
│  • Disaster recovery & backups                                   │
│  • Clinical governance oversight                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Technologies Summary

| Technology | Purpose | Version/Standard |
|-----------|---------|------------------|
| **FHIR** | Healthcare data exchange | R4 (FHIR UK Core) |
| **OAuth 2.0** | Authentication/Authorization | OpenID Connect |
| **NHS CIS2** | NHS identity service | OAuth 2.0 |
| **SNOMED CT** | Clinical terminology | UK Edition |
| **NHS Number** | Patient identifier | 10-digit, Modulus 11 |
| **Python** | Implementation language | 3.8+ |
| **fhir.resources** | FHIR Python library | 6.5.0+ |
| **authlib** | OAuth client library | 1.2.1+ |
| **requests** | HTTP client | 2.31.0+ |
| **SystmOne** | EHR system (TPP) | FHIR API |

---

## Next Steps

### For Developers
1. Review this architecture
2. Read quick start guide
3. Run example code
4. Integrate with clinician mode

### For Project Managers
1. Contact NHS Digital for API access
2. Contact TPP for SystmOne integration
3. Appoint Clinical Safety Officer
4. Begin compliance activities

### For Clinical Safety
1. Complete DCB0129 risk assessment
2. Create hazard log
3. Document clinical safety case
4. Establish incident reporting

### For Information Governance
1. Complete NHS DSPT
2. Document data flows
3. Implement audit logging
4. Create data sharing agreements

---

**Document Version:** 1.0  
**Date:** October 21, 2025  
**Status:** Reference Architecture

