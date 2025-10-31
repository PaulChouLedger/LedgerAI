"""
EHR Integration Example - SystmOne FHIR Integration
Demonstrates how to integrate Aura Medical Chatbot with SystmOne EHR

This is a working example that can be tested with HAPI FHIR test server
and adapted for production SystmOne deployment.
"""

import requests
from typing import Dict, Optional
from datetime import datetime
import logging

# FHIR libraries
from fhir.resources.patient import Patient
from fhir.resources.encounter import Encounter, EncounterParticipant
from fhir.resources.reference import Reference
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.period import Period
from fhir.resources.observation import Observation
from fhir.resources.documentreference import DocumentReference, DocumentReferenceContent
from fhir.resources.attachment import Attachment
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFHIRClient:
    """
    Simplified FHIR client for SystmOne integration
    
    For testing: Uses HAPI FHIR public test server
    For production: Replace with SystmOne FHIR endpoint
    """
    
    def __init__(self, fhir_base_url: str = "https://hapi.fhir.org/baseR4"):
        """
        Initialize FHIR client
        
        Args:
            fhir_base_url: Base URL of FHIR server
                - Test: "https://hapi.fhir.org/baseR4"
                - Prod: "https://api.systmone.nhs.uk/fhir" (verify with TPP)
        """
        self.base_url = fhir_base_url.rstrip('/')
        self.session = requests.Session()
        logger.info(f"[FHIR] Initialized client: {self.base_url}")
    
    def _get_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Build request headers"""
        headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        }
        
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        
        return headers
    
    def search_patient(self, identifier: str) -> Optional[Patient]:
        """
        Search for patient by identifier (e.g., NHS Number)
        
        Args:
            identifier: NHS Number or other patient identifier
        
        Returns:
            Patient resource or None
        """
        try:
            url = f"{self.base_url}/Patient"
            params = {"identifier": identifier}
            
            logger.info(f"[FHIR] Searching for patient: {identifier}")
            
            response = self.session.get(
                url,
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            bundle = response.json()
            
            if bundle.get('total', 0) == 0:
                logger.warning(f"[FHIR] Patient not found: {identifier}")
                return None
            
            # Get first matching patient
            patient_data = bundle['entry'][0]['resource']
            patient = Patient(**patient_data)
            
            logger.info(f"[FHIR] Found patient: {patient.id}")
            return patient
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[FHIR] Error searching patient: {e}")
            return None
    
    def create_encounter(
        self,
        patient_id: str,
        encounter_type: str = "virtual"
    ) -> Optional[Encounter]:
        """
        Create a new encounter (clinical consultation)
        
        Args:
            patient_id: FHIR Patient ID
            encounter_type: Type of encounter
        
        Returns:
            Created Encounter or None
        """
        try:
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
                                display="Telemedicine consultation"
                            )
                        ]
                    )
                ],
                subject=Reference(reference=f"Patient/{patient_id}"),
                period=Period(start=datetime.utcnow().isoformat())
            )
            
            logger.info(f"[FHIR] Creating encounter for patient: {patient_id}")
            
            response = self.session.post(
                f"{self.base_url}/Encounter",
                json=encounter.dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            created = Encounter(**response.json())
            logger.info(f"[FHIR] Created encounter: {created.id}")
            
            return created
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[FHIR] Error creating encounter: {e}")
            return None
    
    def create_observation(
        self,
        patient_id: str,
        encounter_id: str,
        code: str,
        display: str,
        value: str
    ) -> Optional[Observation]:
        """
        Create clinical observation (symptom, vital sign, etc.)
        
        Args:
            patient_id: FHIR Patient ID
            encounter_id: FHIR Encounter ID
            code: SNOMED CT code
            display: Human-readable description
            value: Observation value
        
        Returns:
            Created Observation or None
        """
        try:
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
                            code=code,
                            display=display
                        )
                    ]
                ),
                subject=Reference(reference=f"Patient/{patient_id}"),
                encounter=Reference(reference=f"Encounter/{encounter_id}"),
                effectiveDateTime=datetime.utcnow().isoformat(),
                valueString=value
            )
            
            logger.info(f"[FHIR] Creating observation: {display}")
            
            response = self.session.post(
                f"{self.base_url}/Observation",
                json=observation.dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            created = Observation(**response.json())
            logger.info(f"[FHIR] Created observation: {created.id}")
            
            return created
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[FHIR] Error creating observation: {e}")
            return None
    
    def create_document(
        self,
        patient_id: str,
        encounter_id: str,
        title: str,
        content: str
    ) -> Optional[DocumentReference]:
        """
        Create clinical document (e.g., consultation summary)
        
        Args:
            patient_id: FHIR Patient ID
            encounter_id: FHIR Encounter ID
            title: Document title
            content: Document content (plain text)
        
        Returns:
            Created DocumentReference or None
        """
        try:
            # Encode content as base64
            content_bytes = content.encode('utf-8')
            content_base64 = base64.b64encode(content_bytes).decode('utf-8')
            
            document = DocumentReference(
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
                description=title,
                content=[
                    DocumentReferenceContent(
                        attachment=Attachment(
                            contentType="text/plain",
                            data=content_base64,
                            title=title
                        )
                    )
                ]
            )
            
            logger.info(f"[FHIR] Creating document: {title}")
            
            response = self.session.post(
                f"{self.base_url}/DocumentReference",
                json=document.dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            created = DocumentReference(**response.json())
            logger.info(f"[FHIR] Created document: {created.id}")
            
            return created
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[FHIR] Error creating document: {e}")
            return None
    
    def close_encounter(self, encounter_id: str) -> bool:
        """
        Mark encounter as finished
        
        Args:
            encounter_id: FHIR Encounter ID
        
        Returns:
            True if successful
        """
        try:
            # Retrieve encounter
            response = self.session.get(
                f"{self.base_url}/Encounter/{encounter_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            encounter = Encounter(**response.json())
            
            # Update status
            encounter.status = "finished"
            if encounter.period:
                encounter.period.end = datetime.utcnow().isoformat()
            
            logger.info(f"[FHIR] Closing encounter: {encounter_id}")
            
            # Update encounter
            response = self.session.put(
                f"{self.base_url}/Encounter/{encounter_id}",
                json=encounter.dict(),
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            logger.info(f"[FHIR] Encounter closed: {encounter_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[FHIR] Error closing encounter: {e}")
            return False


def validate_nhs_number(nhs_number: str) -> bool:
    """
    Validate NHS Number using Modulus 11 algorithm
    
    Args:
        nhs_number: NHS Number (10 digits)
    
    Returns:
        True if valid
    """
    # Remove spaces
    nhs_str = str(nhs_number).replace(" ", "").replace("-", "")
    
    # Must be 10 digits
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
    
    if check_digit == 11:
        check_digit = 0
    
    if check_digit == 10:
        return False
    
    return check_digit == int(nhs_str[9])


# ============================================================================
# EXAMPLE USAGE: Complete Consultation Workflow
# ============================================================================

def example_consultation_workflow():
    """
    Example: Complete consultation workflow with SystmOne integration
    
    This demonstrates:
    1. Find patient
    2. Create encounter
    3. Save symptom observations
    4. Save consultation summary
    5. Close encounter
    """
    
    print("\n" + "="*70)
    print("EXAMPLE: Aura Medical Chatbot → SystmOne Integration")
    print("="*70 + "\n")
    
    # Initialize FHIR client (using HAPI test server)
    # For production, replace with: SimpleFHIRClient("https://api.systmone.nhs.uk/fhir")
    client = SimpleFHIRClient("https://hapi.fhir.org/baseR4")
    
    # Step 1: Create or find patient (for testing, create a test patient)
    print("Step 1: Creating test patient...")
    
    test_patient_data = {
        "resourceType": "Patient",
        "identifier": [
            {
                "system": "https://fhir.nhs.uk/Id/nhs-number",
                "value": "9434765870"  # Valid test NHS Number
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
    
    # Create patient on test server
    response = client.session.post(
        f"{client.base_url}/Patient",
        json=test_patient_data,
        headers=client._get_headers()
    )
    
    if response.status_code == 201:
        patient = Patient(**response.json())
        print(f"✅ Created test patient: {patient.id}\n")
    else:
        print(f"❌ Failed to create patient: {response.status_code}")
        return
    
    # Step 2: Create encounter (consultation session)
    print("Step 2: Creating encounter (consultation)...")
    encounter = client.create_encounter(patient_id=patient.id)
    
    if not encounter:
        print("❌ Failed to create encounter")
        return
    
    print(f"✅ Created encounter: {encounter.id}\n")
    
    # Step 3: Save symptom observations (from Aura chatbot)
    print("Step 3: Saving symptom observations...")
    
    # Example: Patient reported chest pain
    symptoms = [
        {
            "code": "29857009",
            "display": "Chest pain",
            "value": "Severity: 8/10, Character: crushing, Duration: 2 hours"
        },
        {
            "code": "267036007",
            "display": "Dyspnea",
            "value": "Shortness of breath, onset with chest pain"
        }
    ]
    
    for symptom in symptoms:
        obs = client.create_observation(
            patient_id=patient.id,
            encounter_id=encounter.id,
            code=symptom["code"],
            display=symptom["display"],
            value=symptom["value"]
        )
        
        if obs:
            print(f"✅ Saved: {symptom['display']}")
        else:
            print(f"❌ Failed to save: {symptom['display']}")
    
    print()
    
    # Step 4: Save consultation summary (from Aura AI)
    print("Step 4: Saving consultation summary...")
    
    summary = """
AURA MEDICAL AI - CONSULTATION SUMMARY
======================================

Patient: John Smith
NHS Number: 9434765870
Date: 2025-10-21

CHIEF COMPLAINT:
Chest pain

HISTORY OF PRESENT ILLNESS:
Patient reports crushing chest pain in center of chest, onset 2 hours ago.
Pain severity 8/10, radiates to left arm. Associated with shortness of breath
and diaphoresis. No relief with rest.

OLDCARTS ASSESSMENT:
- Onset: 2 hours ago, sudden
- Location: Center of chest
- Duration: Continuous
- Character: Crushing, pressure-like
- Aggravating factors: Movement
- Relieving factors: None
- Timing: Constant
- Severity: 8/10

DIFFERENTIAL DIAGNOSIS:
1. Acute Coronary Syndrome (ACS) - HIGH PROBABILITY
2. Pulmonary Embolism (PE)
3. Aortic Dissection

URGENCY ASSESSMENT: EMERGENCY

RECOMMENDATION:
Immediate emergency department evaluation recommended.
Call 999 or proceed to nearest A&E immediately.
This presentation is concerning for acute myocardial infarction.

Generated by: Aura Medical AI Assistant
Reviewed by: [Supervising Clinician Name]
    """
    
    doc = client.create_document(
        patient_id=patient.id,
        encounter_id=encounter.id,
        title="Aura AI Consultation Summary - Chest Pain Assessment",
        content=summary
    )
    
    if doc:
        print(f"✅ Saved consultation summary: {doc.id}\n")
    else:
        print("❌ Failed to save consultation summary\n")
    
    # Step 5: Close encounter
    print("Step 5: Closing encounter...")
    
    if client.close_encounter(encounter.id):
        print(f"✅ Encounter closed: {encounter.id}\n")
    else:
        print(f"❌ Failed to close encounter\n")
    
    # Summary
    print("="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nPatient ID: {patient.id}")
    print(f"Encounter ID: {encounter.id}")
    print(f"Observations Created: {len(symptoms)}")
    print(f"Document ID: {doc.id if doc else 'N/A'}")
    print("\nAll data has been saved to the FHIR server.")
    print("In production, this would appear in SystmOne for clinician review.")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run example workflow
    example_consultation_workflow()
    
    # Test NHS Number validation
    print("\n" + "="*70)
    print("NHS NUMBER VALIDATION EXAMPLES")
    print("="*70 + "\n")
    
    test_numbers = [
        ("9434765870", True),   # Valid
        ("1234567890", False),  # Invalid
        ("943 476 5870", True), # Valid with spaces
        ("12345", False)        # Too short
    ]
    
    for nhs_num, expected in test_numbers:
        is_valid = validate_nhs_number(nhs_num)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        expected_status = "✅" if is_valid == expected else "❌"
        print(f"{expected_status} NHS Number: {nhs_num:15s} → {status}")
    
    print("\n" + "="*70 + "\n")

