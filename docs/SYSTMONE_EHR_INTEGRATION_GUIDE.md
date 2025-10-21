# SystmOne EHR Integration Guide

## Overview

This guide provides a comprehensive approach to integrating the Aura Medical Chatbot with **SystmOne**, the electronic health record (EHR) system used by the UK's National Health Service (NHS).

**SystmOne** is a unified clinical and administrative system that creates a single, shared record for each patient across primary care, community care, and social care settings in the UK.

---

## Table of Contents

1. [Integration Architecture](#integration-architecture)
2. [NHS & UK Regulatory Requirements](#nhs--uk-regulatory-requirements)
3. [Integration Approaches](#integration-approaches)
4. [FHIR Implementation](#fhir-implementation)
5. [Security & Authentication](#security--authentication)
6. [Code Implementation](#code-implementation)
7. [Testing & Deployment](#testing--deployment)
8. [Compliance Checklist](#compliance-checklist)

---

## Integration Architecture

### Current Aura Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Aura Medical Chatbot                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Whisper    │  │ LLM Medical  │  │   RAG System    │   │
│  │   (STT)      │  │  Container   │  │  (Guidelines)   │   │
│  │  Port 5051   │  │  Port 11434  │  │  Port 11435     │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│         ↓                  ↓                    ↓           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Clinician Mode (Adaptive Diagnostic)        │   │
│  │  • Medical symptom assessment                       │   │
│  │  • OLDCARTS questioning                             │   │
│  │  • Differential diagnosis                           │   │
│  │  • Clinical urgency assessment                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    [INTEGRATION LAYER]
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      SystmOne EHR                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Patient    │  │  Encounter   │  │   Clinical      │   │
│  │   Records    │  │   Notes      │  │   Coding        │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Aura Medical Chatbot                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           New: EHR Integration Service              │   │
│  │                                                      │   │
│  │  • FHIR Client (fhirclient / fhir.resources)        │   │
│  │  • Patient context management                       │   │
│  │  • Clinical data synchronization                    │   │
│  │  • Encounter documentation                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                   ┌──────────────────┐
                   │  FHIR API Layer  │
                   │  (NHS Standard)  │
                   └──────────────────┘
                              ↓
           ┌──────────────────────────────────────┐
           │         NHS Authentication           │
           │  • NHS Identity (Smartcard)          │
           │  • OAuth 2.0 / OpenID Connect        │
           │  • Role-Based Access Control (RBAC)  │
           └──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              SystmOne FHIR API Endpoint                      │
│  • RESTful FHIR R4 API                                       │
│  • Standard FHIR resources (Patient, Encounter, etc.)        │
└─────────────────────────────────────────────────────────────┘
```

---

## NHS & UK Regulatory Requirements

### 1. **Clinical Safety (DCB0129 & DCB0160)**

**DCB0129** - Clinical Risk Management: Manufacture of Health IT Systems
- You MUST have a **Clinical Safety Officer (CSO)**
- Perform **Clinical Risk Management** activities
- Maintain a **Clinical Safety Case** with hazard logs
- Document all potential patient safety risks

**DCB0160** - Clinical Risk Management: Deployment and Use
- Required for organizations deploying health IT
- Assess risks in the deployment environment
- Ongoing hazard monitoring and incident reporting

**Action Items:**
- [ ] Appoint a qualified Clinical Safety Officer
- [ ] Complete clinical risk assessment
- [ ] Create hazard log for all integration points
- [ ] Document mitigation strategies
- [ ] Establish incident reporting process

### 2. **Data Protection & GDPR**

**UK GDPR & Data Protection Act 2018:**
- Patient data is **Special Category Data** (Article 9)
- Requires explicit consent or legal basis
- Data minimization and purpose limitation
- Right to access, rectification, and erasure
- Data breach notification (72 hours)

**NHS Data Security and Protection Toolkit (DSPT):**
- Annual self-assessment required
- 10 mandatory standards
- Evidence-based assurance
- Used for contract awards

**Action Items:**
- [ ] Register with ICO (Information Commissioner's Office)
- [ ] Complete DSPT assessment
- [ ] Implement data processing agreement (DPA) with NHS
- [ ] Document legal basis for processing
- [ ] Implement patient consent management
- [ ] Create data breach response plan

### 3. **Information Governance (IG)**

**NHS Digital Standards:**
- IG Toolkit compliance
- Staff training requirements
- Access controls and audit trails
- Confidentiality and data sharing agreements

**NHS Care Record Guarantee:**
- Patients control who sees their records
- Right to request information withheld
- Access must be role-appropriate

**Action Items:**
- [ ] Complete IG training for all staff
- [ ] Implement role-based access controls (RBAC)
- [ ] Create comprehensive audit logging
- [ ] Establish data sharing agreements (DSA)

### 4. **Interoperability Standards**

**NHS Interoperability Toolkit:**
- **FHIR UK Core** - UK-specific FHIR profiles
- **SNOMED CT** - Clinical terminology (mandatory)
- **dm+d** - Dictionary of Medicines and Devices
- **NHS Number** - Unique patient identifier

**CareConnect Profiles (now superseded by FHIR UK Core):**
- UK extensions to FHIR resources
- NHS-specific coding systems

### 5. **Clinical Coding Requirements**

You MUST use NHS-approved clinical terminologies:

- **SNOMED CT UK Edition** - Primary clinical terminology
- **Read Codes** - Legacy (being phased out)
- **ICD-10** - Disease classification
- **OPCS-4** - Procedure codes
- **dm+d** - Medications

**License Required:**
- SNOMED CT UK Edition license (free for NHS use)
- Apply via NHS Digital's Technology Reference Data Update Distribution (TRUD)

---

## Integration Approaches

### Option 1: **FHIR RESTful API Integration** (Recommended)

**Pros:**
- ✅ Modern, standardized approach
- ✅ Well-documented with NHS FHIR API specifications
- ✅ Supports granular data access
- ✅ Widely supported by EHR vendors
- ✅ Future-proof (NHS Digital mandate)

**Cons:**
- ⚠️ Requires FHIR expertise
- ⚠️ Complex authentication (NHS Identity)
- ⚠️ Need to handle FHIR resources correctly

**When to Use:**
- New integrations (NHS Digital recommendation)
- Cloud-based deployments
- Multi-EHR integrations

### Option 2: **HL7 v2 Messaging**

**Pros:**
- ✅ Mature, battle-tested standard
- ✅ Widely supported by legacy systems
- ✅ Real-time event notifications

**Cons:**
- ⚠️ Legacy standard (being replaced by FHIR)
- ⚠️ Complex message parsing
- ⚠️ Less flexible than FHIR

**When to Use:**
- Legacy system integration
- Real-time ADT (Admit/Discharge/Transfer) events
- Lab result notifications

### Option 3: **SystmOne Vendor-Specific API**

**Contact TPP (The Phoenix Partnership):**
- SystmOne is developed by **TPP (The Phoenix Partnership)**
- TPP provides vendor-specific APIs and integration pathways
- May offer **API Portal** access for developers

**Steps:**
1. Contact TPP Integration Team: https://tpp-uk.com/contact/
2. Request API documentation and access
3. Apply for developer credentials
4. Negotiate integration agreement

**Pros:**
- ✅ Optimized for SystmOne
- ✅ Potentially simpler than FHIR
- ✅ Vendor support available

**Cons:**
- ⚠️ Vendor lock-in
- ⚠️ May require commercial agreement
- ⚠️ Documentation may be restricted

### Option 4: **NHS Digital API Integration (GP Connect)**

**GP Connect:**
- NHS Digital's interoperability program
- Access GP records across systems
- Uses FHIR-based APIs
- Requires NHS Spine integration

**Components:**
- **Access Record HTML** - View patient record
- **Access Record Structured** - FHIR resources
- **Appointment Management** - Book/manage appointments
- **Send Document** - Upload clinical documents

**Steps:**
1. Register with NHS Digital API Platform
2. Complete assurance process (technical & clinical)
3. Obtain NHS Smartcard for staff
4. Integrate with NHS Spine (authentication/authorization)
5. Implement FHIR API clients

**Pros:**
- ✅ National standard
- ✅ Works across multiple GP systems (including SystmOne)
- ✅ NHS-backed and supported

**Cons:**
- ⚠️ Complex onboarding (6-12 months)
- ⚠️ Requires NHS Spine integration
- ⚠️ Strict clinical safety assurance

---

## FHIR Implementation

### Understanding FHIR UK Core

**FHIR UK Core** is the UK-specific implementation of FHIR, mandated by NHS Digital.

**Key Resources for Medical Chatbot:**

1. **Patient** - Patient demographics and NHS Number
2. **Encounter** - Clinical consultation or interaction
3. **Condition** - Diagnosis or clinical problem
4. **Observation** - Clinical findings (vital signs, symptoms)
5. **QuestionnaireResponse** - Structured symptom data
6. **DocumentReference** - Consultation notes/summaries

### FHIR R4 Python Libraries

**Recommended: `fhir.resources`**
```bash
pip install fhir.resources
```

**Alternative: `fhirclient`**
```bash
pip install fhirclient
```

### FHIR Resource Examples

#### 1. Patient Resource (Read)

```python
from fhir.resources.patient import Patient
import requests

def get_patient(nhs_number, fhir_server_url, access_token):
    """
    Retrieve patient by NHS Number from SystmOne FHIR API
    
    Args:
        nhs_number: NHS Number (10-digit identifier)
        fhir_server_url: Base URL of FHIR server
        access_token: OAuth 2.0 bearer token
    
    Returns:
        Patient FHIR resource
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/fhir+json"
    }
    
    # Search for patient by NHS Number (identifier)
    url = f"{fhir_server_url}/Patient"
    params = {
        "identifier": f"https://fhir.nhs.uk/Id/nhs-number|{nhs_number}"
    }
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    
    bundle = response.json()
    
    if bundle.get('total', 0) == 0:
        raise ValueError(f"Patient not found: {nhs_number}")
    
    # Get first matching patient
    patient_resource = bundle['entry'][0]['resource']
    patient = Patient(**patient_resource)
    
    return patient
```

#### 2. Encounter Resource (Create)

```python
from fhir.resources.encounter import Encounter, EncounterParticipant
from fhir.resources.reference import Reference
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.period import Period
from datetime import datetime

def create_encounter(patient_id, practitioner_id, location_id, fhir_server_url, access_token):
    """
    Create an encounter (consultation) in SystmOne
    
    This represents the chatbot interaction as a clinical encounter
    """
    encounter = Encounter(
        status="in-progress",
        class_fhir=Coding(
            system="http://terminology.hl7.org/CodeSystem/v3-ActCode",
            code="VR",  # Virtual encounter
            display="Virtual"
        ),
        type=[
            CodeableConcept(
                coding=[
                    Coding(
                        system="http://snomed.info/sct",
                        code="448337001",  # Telemedicine consultation
                        display="Telemedicine consultation"
                    )
                ]
            )
        ],
        subject=Reference(reference=f"Patient/{patient_id}"),
        participant=[
            EncounterParticipant(
                individual=Reference(reference=f"Practitioner/{practitioner_id}")
            )
        ],
        period=Period(
            start=datetime.utcnow().isoformat()
        ),
        location=[
            {
                "location": Reference(reference=f"Location/{location_id}")
            }
        ],
        serviceProvider=Reference(reference="Organization/your-org-id")
    )
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json"
    }
    
    response = requests.post(
        f"{fhir_server_url}/Encounter",
        json=encounter.dict(),
        headers=headers
    )
    response.raise_for_status()
    
    created_encounter = Encounter(**response.json())
    return created_encounter
```

#### 3. QuestionnaireResponse (Submit Symptom Data)

```python
from fhir.resources.questionnaireresponse import QuestionnaireResponse, QuestionnaireResponseItem, QuestionnaireResponseItemAnswer
from fhir.resources.reference import Reference

def create_questionnaire_response(patient_id, encounter_id, symptom_data, fhir_server_url, access_token):
    """
    Submit structured symptom assessment data
    
    Maps Aura's OLDCARTS data to FHIR QuestionnaireResponse
    
    Args:
        symptom_data: Dict from clinician_mode session
            {
                "chief_complaint": "chest pain",
                "onset": "2 hours ago",
                "location": "center of chest",
                "duration": "continuous",
                "character": "crushing",
                "aggravating_factors": "movement",
                "relieving_factors": "rest",
                "timing": "constant",
                "severity": "8/10"
            }
    """
    items = []
    
    for key, value in symptom_data.items():
        if value:
            items.append(
                QuestionnaireResponseItem(
                    linkId=key,
                    text=key.replace("_", " ").title(),
                    answer=[
                        QuestionnaireResponseItemAnswer(
                            valueString=str(value)
                        )
                    ]
                )
            )
    
    questionnaire_response = QuestionnaireResponse(
        status="completed",
        subject=Reference(reference=f"Patient/{patient_id}"),
        encounter=Reference(reference=f"Encounter/{encounter_id}"),
        authored=datetime.utcnow().isoformat(),
        source=Reference(reference=f"Patient/{patient_id}"),
        item=items
    )
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json"
    }
    
    response = requests.post(
        f"{fhir_server_url}/QuestionnaireResponse",
        json=questionnaire_response.dict(),
        headers=headers
    )
    response.raise_for_status()
    
    return QuestionnaireResponse(**response.json())
```

#### 4. Observation (Record Vital Signs / Symptoms)

```python
from fhir.resources.observation import Observation
from fhir.resources.quantity import Quantity

def create_symptom_observation(patient_id, encounter_id, symptom_code, symptom_value, fhir_server_url, access_token):
    """
    Create observation for a specific symptom
    
    Example: Record "chest pain" as clinical observation
    """
    observation = Observation(
        status="final",
        category=[
            CodeableConcept(
                coding=[
                    Coding(
                        system="http://terminology.hl7.org/CodeSystem/observation-category",
                        code="survey",
                        display="Survey"
                    )
                ]
            )
        ],
        code=CodeableConcept(
            coding=[
                Coding(
                    system="http://snomed.info/sct",
                    code=symptom_code,  # e.g., "29857009" for chest pain
                    display=symptom_value
                )
            ]
        ),
        subject=Reference(reference=f"Patient/{patient_id}"),
        encounter=Reference(reference=f"Encounter/{encounter_id}"),
        effectiveDateTime=datetime.utcnow().isoformat(),
        valueString=symptom_value
    )
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json"
    }
    
    response = requests.post(
        f"{fhir_server_url}/Observation",
        json=observation.dict(),
        headers=headers
    )
    response.raise_for_status()
    
    return Observation(**response.json())
```

#### 5. DocumentReference (Save Consultation Summary)

```python
from fhir.resources.documentreference import DocumentReference, DocumentReferenceContent
from fhir.resources.attachment import Attachment
import base64

def create_consultation_summary(patient_id, encounter_id, summary_text, fhir_server_url, access_token):
    """
    Save the AI-generated consultation summary to SystmOne
    
    This creates a clinical document that clinicians can review
    """
    # Encode summary as base64
    summary_bytes = summary_text.encode('utf-8')
    summary_base64 = base64.b64encode(summary_bytes).decode('utf-8')
    
    document_reference = DocumentReference(
        status="current",
        type=CodeableConcept(
            coding=[
                Coding(
                    system="http://snomed.info/sct",
                    code="371531000",  # Clinical consultation report
                    display="Clinical consultation report"
                )
            ]
        ),
        subject=Reference(reference=f"Patient/{patient_id}"),
        context={
            "encounter": [Reference(reference=f"Encounter/{encounter_id}")]
        },
        date=datetime.utcnow().isoformat(),
        author=[
            Reference(display="Aura Medical AI Assistant")
        ],
        description="AI-assisted symptom assessment summary",
        content=[
            DocumentReferenceContent(
                attachment=Attachment(
                    contentType="text/plain",
                    data=summary_base64,
                    title="Aura Consultation Summary"
                )
            )
        ]
    )
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json"
    }
    
    response = requests.post(
        f"{fhir_server_url}/DocumentReference",
        json=document_reference.dict(),
        headers=headers
    )
    response.raise_for_status()
    
    return DocumentReference(**response.json())
```

---

## Security & Authentication

### NHS Identity & Access Management

**NHS CIS2 (Care Identity Service 2):**
- Modern authentication service for NHS applications
- Replaces legacy NHS Smartcard authentication
- Uses **OpenID Connect** and **OAuth 2.0**

**Authentication Flow (OAuth 2.0 Authorization Code):**

```
1. User (Clinician) initiates login
   ↓
2. Redirect to NHS CIS2 login page
   ↓
3. User authenticates (username/password + MFA)
   ↓
4. NHS CIS2 redirects back with authorization code
   ↓
5. Exchange code for access token
   ↓
6. Use access token to call SystmOne FHIR API
```

### Implementation: OAuth 2.0 Client

```python
from authlib.integrations.requests_client import OAuth2Session
import os

class NHSAuthClient:
    """
    NHS CIS2 OAuth 2.0 client for accessing SystmOne FHIR API
    """
    
    def __init__(self):
        # NHS CIS2 endpoints (production)
        self.authorization_endpoint = "https://auth.national.nhs.uk/authorize"
        self.token_endpoint = "https://auth.national.nhs.uk/token"
        
        # Your application credentials (from NHS Digital API Portal)
        self.client_id = os.getenv("NHS_CLIENT_ID")
        self.client_secret = os.getenv("NHS_CLIENT_SECRET")
        self.redirect_uri = os.getenv("NHS_REDIRECT_URI", "https://your-app.com/callback")
        
        # Required scopes for FHIR API access
        self.scope = "openid profile patient/*.read encounter/*.write observation/*.write"
        
        self.session = OAuth2Session(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scope
        )
    
    def get_authorization_url(self):
        """
        Generate NHS login URL
        
        Returns:
            Tuple of (authorization_url, state)
        """
        authorization_url, state = self.session.create_authorization_url(
            self.authorization_endpoint
        )
        return authorization_url, state
    
    def fetch_token(self, authorization_response):
        """
        Exchange authorization code for access token
        
        Args:
            authorization_response: Full callback URL with code
        
        Returns:
            Token dict with access_token, refresh_token, etc.
        """
        token = self.session.fetch_token(
            self.token_endpoint,
            authorization_response=authorization_response
        )
        return token
    
    def get_access_token(self):
        """Get current access token"""
        return self.session.token.get('access_token')
    
    def refresh_access_token(self, refresh_token):
        """
        Refresh expired access token
        
        Args:
            refresh_token: Refresh token from initial auth
        
        Returns:
            New token dict
        """
        token = self.session.refresh_token(
            self.token_endpoint,
            refresh_token=refresh_token
        )
        return token
```

### Secure Token Storage

```python
import json
from cryptography.fernet import Fernet
import os

class SecureTokenStore:
    """
    Encrypted storage for NHS access tokens
    Complies with NHS Data Security standards
    """
    
    def __init__(self, encryption_key=None):
        if encryption_key is None:
            # In production, load from secure key management service (AWS KMS, Azure Key Vault)
            encryption_key = os.getenv("TOKEN_ENCRYPTION_KEY")
            if not encryption_key:
                raise ValueError("TOKEN_ENCRYPTION_KEY must be set")
        
        self.cipher = Fernet(encryption_key.encode())
        self.token_file = "/secure/nhs_tokens.enc"  # Encrypted file location
    
    def save_token(self, user_id, token_data):
        """
        Save encrypted access token
        
        Args:
            user_id: NHS user identifier (e.g., smartcard UUID)
            token_data: Dict with access_token, refresh_token, expires_at
        """
        # Load existing tokens
        all_tokens = self._load_all_tokens()
        
        # Add/update this user's token
        all_tokens[user_id] = token_data
        
        # Encrypt and save
        plaintext = json.dumps(all_tokens).encode()
        encrypted = self.cipher.encrypt(plaintext)
        
        with open(self.token_file, 'wb') as f:
            f.write(encrypted)
    
    def get_token(self, user_id):
        """
        Retrieve decrypted access token
        
        Args:
            user_id: NHS user identifier
        
        Returns:
            Token data dict or None if not found
        """
        all_tokens = self._load_all_tokens()
        return all_tokens.get(user_id)
    
    def _load_all_tokens(self):
        """Load and decrypt all tokens"""
        if not os.path.exists(self.token_file):
            return {}
        
        with open(self.token_file, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
```

### NHS Number Validation

```python
def validate_nhs_number(nhs_number):
    """
    Validate NHS Number using Modulus 11 algorithm
    
    NHS Numbers are 10 digits with check digit validation
    
    Args:
        nhs_number: String or int of NHS Number
    
    Returns:
        True if valid, False otherwise
    """
    # Remove spaces and convert to string
    nhs_str = str(nhs_number).replace(" ", "").replace("-", "")
    
    # Must be exactly 10 digits
    if len(nhs_str) != 10 or not nhs_str.isdigit():
        return False
    
    # Modulus 11 check
    total = 0
    for i in range(9):
        digit = int(nhs_str[i])
        factor = 10 - i
        total += digit * factor
    
    remainder = total % 11
    check_digit = 11 - remainder
    
    # If check digit is 11, it becomes 0
    if check_digit == 11:
        check_digit = 0
    
    # Check digit 10 is invalid
    if check_digit == 10:
        return False
    
    # Compare calculated check digit with actual
    return check_digit == int(nhs_str[9])


# Example usage
valid = validate_nhs_number("9434765870")  # True
invalid = validate_nhs_number("1234567890")  # False
```

---

## Code Implementation

### New Module: `ehr_integration.py`

Create a new container or service for EHR integration:

```python
# llm-medical-container/ehr_integration.py
"""
EHR Integration Service for Aura Medical Chatbot
Handles bidirectional communication with SystmOne via FHIR API
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
import os
from fhir.resources.patient import Patient
from fhir.resources.encounter import Encounter
from fhir.resources.observation import Observation
from fhir.resources.documentreference import DocumentReference
import logging

logger = logging.getLogger(__name__)


class SystmOneClient:
    """
    FHIR client for SystmOne EHR integration
    """
    
    def __init__(
        self,
        fhir_server_url: str = None,
        auth_client = None
    ):
        """
        Initialize SystmOne FHIR client
        
        Args:
            fhir_server_url: Base URL of SystmOne FHIR endpoint
            auth_client: NHS authentication client (NHSAuthClient instance)
        """
        self.fhir_server_url = fhir_server_url or os.getenv(
            "SYSTMONE_FHIR_URL",
            "https://api.systmone.nhs.uk/fhir"  # Example URL - verify with TPP
        )
        self.auth_client = auth_client
        
        # Session for connection pooling
        self.session = requests.Session()
        
        logger.info(f"[EHR] Initialized SystmOne client: {self.fhir_server_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with authentication"""
        access_token = self.auth_client.get_access_token()
        
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        }
    
    def get_patient(self, nhs_number: str) -> Optional[Patient]:
        """
        Retrieve patient by NHS Number
        
        Args:
            nhs_number: 10-digit NHS Number
        
        Returns:
            Patient FHIR resource or None
        """
        try:
            params = {
                "identifier": f"https://fhir.nhs.uk/Id/nhs-number|{nhs_number}"
            }
            
            response = self.session.get(
                f"{self.fhir_server_url}/Patient",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            bundle = response.json()
            
            if bundle.get('total', 0) == 0:
                logger.warning(f"[EHR] Patient not found: {nhs_number}")
                return None
            
            patient_resource = bundle['entry'][0]['resource']
            patient = Patient(**patient_resource)
            
            logger.info(f"[EHR] Retrieved patient: {nhs_number}")
            return patient
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[EHR] Error retrieving patient: {e}")
            return None
    
    def create_encounter(
        self,
        patient_id: str,
        practitioner_id: str,
        encounter_type: str = "virtual"
    ) -> Optional[Encounter]:
        """
        Create a new encounter (consultation session)
        
        Args:
            patient_id: FHIR Patient ID
            practitioner_id: FHIR Practitioner ID (supervising clinician)
            encounter_type: Type of encounter ("virtual", "in-person", etc.)
        
        Returns:
            Created Encounter resource or None
        """
        try:
            from fhir.resources.encounter import Encounter, EncounterParticipant
            from fhir.resources.reference import Reference
            from fhir.resources.codeableconcept import CodeableConcept
            from fhir.resources.coding import Coding
            from fhir.resources.period import Period
            
            encounter = Encounter(
                status="in-progress",
                class_fhir=Coding(
                    system="http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    code="VR",
                    display="Virtual"
                ),
                type=[
                    CodeableConcept(
                        coding=[
                            Coding(
                                system="http://snomed.info/sct",
                                code="448337001",  # Telemedicine consultation
                                display="Telemedicine consultation with Aura AI"
                            )
                        ]
                    )
                ],
                subject=Reference(reference=f"Patient/{patient_id}"),
                participant=[
                    EncounterParticipant(
                        individual=Reference(reference=f"Practitioner/{practitioner_id}")
                    )
                ],
                period=Period(start=datetime.utcnow().isoformat())
            )
            
            response = self.session.post(
                f"{self.fhir_server_url}/Encounter",
                json=encounter.dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            created_encounter = Encounter(**response.json())
            logger.info(f"[EHR] Created encounter: {created_encounter.id}")
            
            return created_encounter
            
        except Exception as e:
            logger.error(f"[EHR] Error creating encounter: {e}")
            return None
    
    def save_symptom_assessment(
        self,
        patient_id: str,
        encounter_id: str,
        symptom_data: Dict
    ) -> bool:
        """
        Save Aura's symptom assessment to SystmOne
        
        Args:
            patient_id: FHIR Patient ID
            encounter_id: FHIR Encounter ID
            symptom_data: Dict from clinician_mode session
                {
                    "chief_complaint": "chest pain",
                    "onset": "2 hours ago",
                    "location": "center of chest",
                    "duration": "continuous",
                    "character": "crushing",
                    "severity": "8/10",
                    "differential_diagnoses": ["ACS", "PE"],
                    "urgency": "emergency"
                }
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map symptoms to SNOMED codes
            symptom_observations = self._map_symptoms_to_observations(
                symptom_data,
                patient_id,
                encounter_id
            )
            
            # Create each observation
            for obs_data in symptom_observations:
                observation = Observation(**obs_data)
                
                response = self.session.post(
                    f"{self.fhir_server_url}/Observation",
                    json=observation.dict(),
                    headers=self._get_headers()
                )
                response.raise_for_status()
            
            logger.info(f"[EHR] Saved {len(symptom_observations)} symptom observations")
            return True
            
        except Exception as e:
            logger.error(f"[EHR] Error saving symptom assessment: {e}")
            return False
    
    def _map_symptoms_to_observations(
        self,
        symptom_data: Dict,
        patient_id: str,
        encounter_id: str
    ) -> List[Dict]:
        """
        Map Aura symptom data to FHIR Observations with SNOMED codes
        
        This is a simplified example - in production, use comprehensive
        SNOMED CT mapping and the NHS terminology service
        """
        from fhir.resources.reference import Reference
        from fhir.resources.codeableconcept import CodeableConcept
        from fhir.resources.coding import Coding
        
        observations = []
        
        # Example SNOMED mappings (simplified)
        snomed_mappings = {
            "chest pain": "29857009",
            "abdominal pain": "21522001",
            "headache": "25064002",
            "dyspnea": "267036007",  # Shortness of breath
            "nausea": "422587007"
        }
        
        chief_complaint = symptom_data.get("chief_complaint", "")
        
        # Find matching SNOMED code
        snomed_code = None
        for symptom, code in snomed_mappings.items():
            if symptom.lower() in chief_complaint.lower():
                snomed_code = code
                break
        
        if snomed_code:
            observation = {
                "resourceType": "Observation",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "survey",
                                "display": "Survey"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": snomed_code,
                            "display": chief_complaint
                        }
                    ]
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "encounter": {"reference": f"Encounter/{encounter_id}"},
                "effectiveDateTime": datetime.utcnow().isoformat(),
                "valueString": f"Severity: {symptom_data.get('severity', 'unknown')}, "
                              f"Character: {symptom_data.get('character', 'unknown')}"
            }
            observations.append(observation)
        
        return observations
    
    def save_consultation_summary(
        self,
        patient_id: str,
        encounter_id: str,
        summary_text: str
    ) -> bool:
        """
        Save AI-generated consultation summary as DocumentReference
        
        Args:
            patient_id: FHIR Patient ID
            encounter_id: FHIR Encounter ID
            summary_text: Full consultation summary from Aura
        
        Returns:
            True if successful
        """
        try:
            import base64
            from fhir.resources.documentreference import DocumentReference, DocumentReferenceContent
            from fhir.resources.attachment import Attachment
            from fhir.resources.reference import Reference
            from fhir.resources.codeableconcept import CodeableConcept
            from fhir.resources.coding import Coding
            
            # Encode summary as base64
            summary_base64 = base64.b64encode(summary_text.encode('utf-8')).decode('utf-8')
            
            document_reference = DocumentReference(
                status="current",
                type=CodeableConcept(
                    coding=[
                        Coding(
                            system="http://snomed.info/sct",
                            code="371531000",  # Clinical consultation report
                            display="Clinical consultation report"
                        )
                    ]
                ),
                subject=Reference(reference=f"Patient/{patient_id}"),
                context={
                    "encounter": [Reference(reference=f"Encounter/{encounter_id}")]
                },
                date=datetime.utcnow().isoformat(),
                author=[Reference(display="Aura Medical AI Assistant")],
                description="AI-assisted symptom assessment summary",
                content=[
                    DocumentReferenceContent(
                        attachment=Attachment(
                            contentType="text/plain",
                            data=summary_base64,
                            title="Aura Consultation Summary"
                        )
                    )
                ]
            )
            
            response = self.session.post(
                f"{self.fhir_server_url}/DocumentReference",
                json=document_reference.dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            logger.info(f"[EHR] Saved consultation summary for encounter {encounter_id}")
            return True
            
        except Exception as e:
            logger.error(f"[EHR] Error saving consultation summary: {e}")
            return False
    
    def close_encounter(self, encounter_id: str) -> bool:
        """
        Mark encounter as finished
        
        Args:
            encounter_id: FHIR Encounter ID
        
        Returns:
            True if successful
        """
        try:
            # First, retrieve the encounter
            response = self.session.get(
                f"{self.fhir_server_url}/Encounter/{encounter_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            encounter = Encounter(**response.json())
            
            # Update status and end time
            encounter.status = "finished"
            if encounter.period:
                encounter.period.end = datetime.utcnow().isoformat()
            
            # Update the encounter
            response = self.session.put(
                f"{self.fhir_server_url}/Encounter/{encounter_id}",
                json=encounter.dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            logger.info(f"[EHR] Closed encounter: {encounter_id}")
            return True
            
        except Exception as e:
            logger.error(f"[EHR] Error closing encounter: {e}")
            return False


# Singleton instance
_systmone_client = None

def get_systmone_client(auth_client=None) -> SystmOneClient:
    """
    Get or create SystmOne FHIR client (singleton)
    
    Args:
        auth_client: NHS authentication client
    
    Returns:
        SystmOneClient instance
    """
    global _systmone_client
    
    if _systmone_client is None:
        _systmone_client = SystmOneClient(auth_client=auth_client)
    
    return _systmone_client
```

### Integration with `clinician_mode.py`

Modify the clinician mode to optionally save data to SystmOne:

```python
# Add to llm-medical-container/clinician_mode.py

from ehr_integration import get_systmone_client
import os

# Feature flag for EHR integration
EHR_INTEGRATION_ENABLED = os.getenv("EHR_INTEGRATION_ENABLED", "false").lower() == "true"

class ClinicianSession:
    """Existing ClinicianSession class with EHR integration"""
    
    def __init__(self, session_id, llm_chat, llm_chat_simple):
        # ... existing init code ...
        
        # EHR integration state
        self.ehr_patient_id = None
        self.ehr_encounter_id = None
        self.ehr_enabled = EHR_INTEGRATION_ENABLED
        
        if self.ehr_enabled:
            self.ehr_client = get_systmone_client()
    
    def start_assessment(self, chief_complaint, patient_nhs_number=None, practitioner_id=None):
        """
        Start symptom assessment with optional EHR integration
        
        Args:
            chief_complaint: Patient's chief complaint
            patient_nhs_number: NHS Number (for EHR integration)
            practitioner_id: Supervising practitioner ID
        """
        # Existing assessment logic...
        self.chief_complaint = chief_complaint
        
        # EHR Integration: Create encounter
        if self.ehr_enabled and patient_nhs_number and practitioner_id:
            try:
                # Get patient from SystmOne
                patient = self.ehr_client.get_patient(patient_nhs_number)
                
                if patient:
                    self.ehr_patient_id = patient.id
                    
                    # Create encounter
                    encounter = self.ehr_client.create_encounter(
                        patient_id=patient.id,
                        practitioner_id=practitioner_id
                    )
                    
                    if encounter:
                        self.ehr_encounter_id = encounter.id
                        print(f"[EHR] Created encounter {encounter.id} for patient {patient.id}")
                    
            except Exception as e:
                print(f"[EHR] Warning: Could not create encounter: {e}")
                # Continue with assessment even if EHR fails
    
    def finalize_assessment(self):
        """
        Complete assessment and save to EHR
        
        Returns:
            Final assessment summary
        """
        # Generate summary (existing logic)
        summary = self._generate_summary()
        
        # EHR Integration: Save assessment
        if self.ehr_enabled and self.ehr_patient_id and self.ehr_encounter_id:
            try:
                # Prepare symptom data
                symptom_data = {
                    "chief_complaint": self.chief_complaint,
                    "onset": self.collected_info.get("onset"),
                    "location": self.collected_info.get("location"),
                    "duration": self.collected_info.get("duration"),
                    "character": self.collected_info.get("character"),
                    "severity": self.collected_info.get("severity"),
                    "differential_diagnoses": self.differential_diagnoses,
                    "urgency": self.urgency_level
                }
                
                # Save symptom assessment
                self.ehr_client.save_symptom_assessment(
                    patient_id=self.ehr_patient_id,
                    encounter_id=self.ehr_encounter_id,
                    symptom_data=symptom_data
                )
                
                # Save consultation summary
                self.ehr_client.save_consultation_summary(
                    patient_id=self.ehr_patient_id,
                    encounter_id=self.ehr_encounter_id,
                    summary_text=summary
                )
                
                # Close encounter
                self.ehr_client.close_encounter(self.ehr_encounter_id)
                
                print(f"[EHR] Assessment saved to SystmOne")
                
            except Exception as e:
                print(f"[EHR] Warning: Could not save to EHR: {e}")
                # Don't fail the assessment if EHR save fails
        
        return summary
```

### Flask Endpoint for EHR Operations

Add new endpoints to `container_rest.py`:

```python
# Add to llm-medical-container/container_rest.py

from ehr_integration import get_systmone_client
from nhs_auth import NHSAuthClient

# Initialize NHS auth client
nhs_auth = NHSAuthClient()

@app.route("/ehr/patient/<nhs_number>", methods=["GET"])
def get_ehr_patient(nhs_number):
    """
    Retrieve patient from SystmOne by NHS Number
    
    Query params:
        - access_token: NHS CIS2 access token (or use header)
    
    Returns:
        FHIR Patient resource
    """
    try:
        # Get access token from header or query param
        access_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not access_token:
            access_token = request.args.get("access_token")
        
        if not access_token:
            return jsonify({"error": "Missing access token"}), 401
        
        # Validate NHS Number
        from ehr_integration import validate_nhs_number
        if not validate_nhs_number(nhs_number):
            return jsonify({"error": "Invalid NHS Number"}), 400
        
        # Get patient
        ehr_client = get_systmone_client(auth_client=nhs_auth)
        patient = ehr_client.get_patient(nhs_number)
        
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
        
        return jsonify(patient.dict()), 200
        
    except Exception as e:
        print(f"[EHR API] Error retrieving patient: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ehr/start-consultation", methods=["POST"])
def start_ehr_consultation():
    """
    Start EHR-integrated consultation
    
    Request body:
        {
            "nhs_number": "9434765870",
            "chief_complaint": "chest pain",
            "practitioner_id": "Practitioner/123",
            "session_id": "session_123"
        }
    
    Returns:
        {
            "encounter_id": "Encounter/456",
            "patient_id": "Patient/789",
            "status": "started"
        }
    """
    try:
        data = request.get_json()
        
        nhs_number = data.get("nhs_number")
        chief_complaint = data.get("chief_complaint")
        practitioner_id = data.get("practitioner_id")
        session_id = data.get("session_id")
        
        if not all([nhs_number, chief_complaint, practitioner_id]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get or create clinician session
        from clinician_mode import get_clinician_session
        session = get_clinician_session(session_id, llm_chat, llm_chat_simple)
        
        # Start assessment with EHR integration
        session.start_assessment(
            chief_complaint=chief_complaint,
            patient_nhs_number=nhs_number,
            practitioner_id=practitioner_id
        )
        
        return jsonify({
            "encounter_id": session.ehr_encounter_id,
            "patient_id": session.ehr_patient_id,
            "status": "started",
            "session_id": session_id
        }), 200
        
    except Exception as e:
        print(f"[EHR API] Error starting consultation: {e}")
        return jsonify({"error": str(e)}), 500
```

---

## Testing & Deployment

### Local Development & Testing

#### 1. **FHIR Test Servers**

Use public FHIR test servers for development:

**HAPI FHIR Test Server:**
- URL: `https://hapi.fhir.org/baseR4`
- No authentication required
- FHIR R4 compliant
- Public sandbox

**Usage:**
```python
# For testing only - replace with real SystmOne URL in production
test_client = SystmOneClient(
    fhir_server_url="https://hapi.fhir.org/baseR4",
    auth_client=None  # No auth for test server
)
```

#### 2. **Mock SystmOne Responses**

Create mock responses for unit testing:

```python
# tests/test_ehr_integration.py
import pytest
from unittest.mock import Mock, patch
from ehr_integration import SystmOneClient

@pytest.fixture
def mock_patient_response():
    """Mock FHIR Patient resource"""
    return {
        "resourceType": "Bundle",
        "total": 1,
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "123",
                    "identifier": [
                        {
                            "system": "https://fhir.nhs.uk/Id/nhs-number",
                            "value": "9434765870"
                        }
                    ],
                    "name": [
                        {
                            "family": "Smith",
                            "given": ["John"]
                        }
                    ],
                    "gender": "male",
                    "birthDate": "1980-01-01"
                }
            }
        ]
    }

def test_get_patient(mock_patient_response):
    """Test patient retrieval"""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_patient_response
        mock_get.return_value.raise_for_status = Mock()
        
        client = SystmOneClient(fhir_server_url="https://test.com/fhir")
        patient = client.get_patient("9434765870")
        
        assert patient is not None
        assert patient.id == "123"
        assert patient.identifier[0].value == "9434765870"
```

#### 3. **Integration Testing Checklist**

- [ ] Test patient retrieval by NHS Number
- [ ] Test encounter creation
- [ ] Test observation creation (symptoms)
- [ ] Test consultation summary save
- [ ] Test encounter closure
- [ ] Test error handling (network failures, auth failures)
- [ ] Test data validation (NHS Number, SNOMED codes)
- [ ] Test concurrent sessions

### Production Deployment

#### 1. **NHS Digital Onboarding Process**

**Steps to go live with NHS integration:**

1. **Register with NHS Digital API Platform**
   - Visit: https://digital.nhs.uk/developer
   - Create developer account
   - Apply for API access

2. **Complete Clinical Safety Assessment**
   - Appoint Clinical Safety Officer
   - Complete DCB0129 clinical risk assessment
   - Submit Clinical Safety Case

3. **Technical Assurance**
   - Demonstrate FHIR conformance
   - Pass security assessment
   - Verify NHS Identity integration
   - Load testing and performance validation

4. **Information Governance**
   - Complete NHS DSPT (Data Security and Protection Toolkit)
   - Sign data sharing agreement (DSA)
   - Demonstrate GDPR compliance

5. **Pilot Deployment**
   - Deploy to NHS test environment
   - Conduct user acceptance testing (UAT) with NHS trust
   - Gather clinical feedback

6. **Production Deployment**
   - Obtain production credentials
   - Deploy to production environment
   - Monitor and support

**Timeline:** Typically 6-12 months

#### 2. **Environment Configuration**

```bash
# config.env for production

# SystmOne FHIR API
SYSTMONE_FHIR_URL=https://api.systmone.nhs.uk/fhir  # Example - verify with TPP

# NHS CIS2 Authentication
NHS_CLIENT_ID=your_client_id_from_nhs_digital
NHS_CLIENT_SECRET=your_client_secret_from_nhs_digital
NHS_REDIRECT_URI=https://your-app.nhs.uk/callback

# Feature flags
EHR_INTEGRATION_ENABLED=true

# Security
TOKEN_ENCRYPTION_KEY=your_secure_encryption_key_from_kms

# Logging
LOG_LEVEL=INFO
LOG_EHR_REQUESTS=true
```

#### 3. **Monitoring & Audit Logging**

Implement comprehensive audit logging (NHS requirement):

```python
# ehr_integration.py - Add audit logging

import logging
import json
from datetime import datetime

class EHRAuditLogger:
    """
    Audit logger for NHS compliance
    Logs all EHR access and modifications
    """
    
    def __init__(self, log_file="/var/log/aura/ehr_audit.log"):
        self.logger = logging.getLogger("ehr_audit")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_access(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        success: bool,
        details: dict = None
    ):
        """
        Log EHR access event
        
        Args:
            user_id: NHS user identifier (smartcard UUID or email)
            action: Action performed (read, create, update, delete)
            resource_type: FHIR resource type (Patient, Encounter, etc.)
            resource_id: FHIR resource ID
            success: Whether action succeeded
            details: Additional context
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "success": success,
            "details": details or {}
        }
        
        self.logger.info(json.dumps(audit_entry))

# Usage in SystmOneClient
audit_logger = EHRAuditLogger()

def get_patient(self, nhs_number: str, user_id: str) -> Optional[Patient]:
    """Retrieve patient with audit logging"""
    try:
        # ... existing code ...
        
        audit_logger.log_access(
            user_id=user_id,
            action="read",
            resource_type="Patient",
            resource_id=patient.id,
            success=True,
            details={"nhs_number": nhs_number}
        )
        
        return patient
        
    except Exception as e:
        audit_logger.log_access(
            user_id=user_id,
            action="read",
            resource_type="Patient",
            resource_id="unknown",
            success=False,
            details={"error": str(e), "nhs_number": nhs_number}
        )
        raise
```

#### 4. **Error Handling & Resilience**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class SystmOneClient:
    """Enhanced with retry logic and circuit breaker"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get_patient(self, nhs_number: str) -> Optional[Patient]:
        """
        Retrieve patient with automatic retry on transient failures
        
        Retries up to 3 times with exponential backoff
        """
        # ... existing implementation ...
        pass
    
    def health_check(self) -> bool:
        """
        Check if SystmOne FHIR API is available
        
        Returns:
            True if API is healthy
        """
        try:
            response = self.session.get(
                f"{self.fhir_server_url}/metadata",
                headers=self._get_headers(),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
```

---

## Compliance Checklist

### Pre-Deployment Checklist

#### Clinical Safety
- [ ] Clinical Safety Officer appointed
- [ ] Clinical risk assessment (DCB0129) completed
- [ ] Hazard log created and maintained
- [ ] Clinical Safety Case documented
- [ ] Incident reporting process established

#### Information Governance
- [ ] NHS DSPT completed
- [ ] Data sharing agreement signed
- [ ] GDPR compliance documented
- [ ] Privacy impact assessment (DPIA) completed
- [ ] Patient consent mechanism implemented

#### Technical Standards
- [ ] FHIR UK Core conformance validated
- [ ] SNOMED CT license obtained
- [ ] NHS Number validation implemented
- [ ] Audit logging implemented
- [ ] Error handling and resilience tested

#### Security
- [ ] NHS CIS2 authentication integrated
- [ ] Token encryption implemented
- [ ] Role-based access control (RBAC) implemented
- [ ] Secure credential storage (KMS)
- [ ] Penetration testing completed

#### Testing
- [ ] Unit tests passed (>80% coverage)
- [ ] Integration tests with FHIR test server passed
- [ ] User acceptance testing (UAT) completed
- [ ] Performance testing completed
- [ ] Clinical validation completed

#### Documentation
- [ ] Integration architecture documented
- [ ] API documentation created
- [ ] User guide for clinicians created
- [ ] Operational runbook created
- [ ] Disaster recovery plan documented

---

## Additional Resources

### Official NHS Documentation

1. **NHS Digital Developer Portal**
   - https://digital.nhs.uk/developer
   - API catalog, documentation, and registration

2. **FHIR UK Core**
   - https://simplifier.net/HL7FHIRUKCoreR4
   - UK-specific FHIR profiles and implementation guides

3. **GP Connect**
   - https://digital.nhs.uk/services/gp-connect
   - Access GP records across systems

4. **NHS CIS2 (Care Identity Service)**
   - https://digital.nhs.uk/services/care-identity-service
   - Authentication and identity management

5. **NHS Data Security and Protection Toolkit**
   - https://www.dsptoolkit.nhs.uk/
   - Self-assessment and compliance

### Clinical Coding Resources

1. **SNOMED CT UK Edition**
   - https://termbrowser.nhs.uk/
   - Browse and search SNOMED CT codes

2. **TRUD (Technology Reference Data Update Distribution)**
   - https://isd.digital.nhs.uk/trud
   - Download SNOMED CT, dm+d, and other terminologies

3. **NHS Data Dictionary**
   - https://www.datadictionary.nhs.uk/
   - Definitions and standards

### Python Libraries

```bash
# requirements_ehr.txt

# FHIR client
fhir.resources==6.5.0

# Authentication
authlib==1.2.1

# HTTP client
requests==2.31.0

# Retry logic
tenacity==8.2.3

# Encryption
cryptography==41.0.5

# Testing
pytest==7.4.3
pytest-mock==3.12.0
```

### SystmOne / TPP Contacts

1. **TPP (The Phoenix Partnership) - SystmOne Vendor**
   - Website: https://tpp-uk.com/
   - Integration team: integrations@tpp-uk.com (verify current contact)
   - Technical support: support@tpp-uk.com

2. **NHS Digital API Support**
   - Email: api.management@nhs.net
   - Support portal: https://digital.nhs.uk/developer/support

---

## Summary & Next Steps

### Integration Summary

Integrating Aura Medical Chatbot with SystmOne involves:

1. ✅ **FHIR API Integration** - Modern, standards-based approach
2. ✅ **NHS Authentication** - NHS CIS2 (OAuth 2.0)
3. ✅ **Clinical Coding** - SNOMED CT for symptoms and diagnoses
4. ✅ **Compliance** - GDPR, clinical safety, IG toolkit
5. ✅ **Bidirectional Data Flow**:
   - **Read**: Patient demographics, medical history
   - **Write**: Symptom assessments, consultation summaries

### Recommended Next Steps

#### Phase 1: Preparation (Weeks 1-4)
1. Contact TPP for SystmOne API access
2. Register with NHS Digital API Platform
3. Appoint Clinical Safety Officer
4. Begin DCB0129 clinical risk assessment

#### Phase 2: Development (Weeks 5-12)
1. Implement FHIR client (`ehr_integration.py`)
2. Integrate NHS CIS2 authentication
3. Build unit and integration tests
4. Develop audit logging system

#### Phase 3: Testing (Weeks 13-20)
1. Test with FHIR test servers
2. Conduct clinical validation
3. User acceptance testing with NHS trust
4. Security and penetration testing

#### Phase 4: Compliance (Weeks 21-28)
1. Complete NHS DSPT
2. Finalize Clinical Safety Case
3. Sign data sharing agreements
4. Obtain SNOMED CT license

#### Phase 5: Pilot (Weeks 29-36)
1. Deploy to NHS test environment
2. Pilot with single NHS practice/trust
3. Gather feedback and iterate
4. Monitor performance and safety

#### Phase 6: Production (Week 37+)
1. Obtain production credentials
2. Production deployment
3. Ongoing monitoring and support
4. Continuous improvement

---

**Questions or Need Help?**

- NHS Digital API Support: api.management@nhs.net
- TPP Integration Team: integrations@tpp-uk.com
- Clinical Safety Standards: safetyguidance@nhsx.nhs.uk (verify current contact)

---

**Document Version:** 1.0  
**Last Updated:** October 21, 2025  
**Author:** Aura Medical Team

