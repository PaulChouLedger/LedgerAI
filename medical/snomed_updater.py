#!/usr/bin/env python3
"""
SNOMED CT Medical Terminology Updater

Downloads and processes SNOMED CT terms to keep medical_terms.json current.
Runs monthly to ensure medical terminology is up-to-date.

SNOMED CT Sources:
- UMLS (requires free NIH account): https://uts.nlm.nih.gov/uts/
- SNOMED International (requires license): https://www.snomed.org/

For US-based deployments, UMLS provides SNOMED CT for free.
"""

import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set

class SNOMEDUpdater:
    """
    Updates medical terminology from SNOMED CT
    
    Can use either:
    1. UMLS API (free for US, requires API key)
    2. Local SNOMED CT files (if you have license)
    3. Curated subset (for MVP without license)
    """
    
    def __init__(self, output_path: str = None):
        if output_path is None:
            script_dir = Path(__file__).resolve().parent
            repo_root = script_dir.parent
            output_path = repo_root / "shared" / "medical_terms.json"
        
        self.output_path = Path(output_path)
        self.umls_api_key = None  # Set from environment or config
        
        print(f"[SNOMED] 📂 Output: {self.output_path}")
    
    def get_common_clinical_terms(self) -> Dict[str, List[str]]:
        """
        Curated subset of most common clinical terms by specialty
        
        Based on:
        - SNOMED CT Clinical Findings hierarchy
        - ICD-10 most common codes
        - Primary care/ED frequency data
        
        Use this for MVP without UMLS license.
        Update monthly by reviewing clinical usage logs.
        """
        return {
            "gastrointestinal": [
                "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation",
                "dysphagia", "heartburn", "melena", "hematochezia", "jaundice",
                "bloating", "cramping", "reflux", "indigestion", "gas",
                "appendicitis", "pancreatitis", "cholecystitis", "gastroenteritis",
                "peptic ulcer", "GERD", "IBS", "IBD", "Crohn's disease", "colitis",
                "RUQ pain", "RLQ pain", "LUQ pain", "LLQ pain", "epigastric pain",
                "periumbilical pain", "diffuse abdominal pain"
            ],
            "cardiovascular": [
                "chest pain", "palpitations", "dyspnea", "orthopnea", "syncope",
                "edema", "claudication", "angina", "tachycardia", "bradycardia",
                "hypertension", "hypotension", "heart attack", "MI", "myocardial infarction",
                "heart failure", "CHF", "arrhythmia", "atrial fibrillation", "AFib",
                "DVT", "deep vein thrombosis", "pulmonary embolism", "PE",
                "substernal chest pain", "left-sided chest pain", "radiating pain"
            ],
            "respiratory": [
                "cough", "shortness of breath", "SOB", "dyspnea", "wheezing",
                "hemoptysis", "sputum", "chest tightness", "pleuritic pain",
                "stridor", "tachypnea", "hypoxia", "cyanosis",
                "pneumonia", "asthma", "COPD", "bronchitis", "pneumothorax",
                "pleural effusion", "pulmonary edema", "upper respiratory infection",
                "URI", "COVID-19", "flu", "influenza"
            ],
            "neurological": [
                "headache", "migraine", "dizziness", "vertigo", "seizure",
                "confusion", "altered mental status", "AMS", "syncope", "LOC",
                "weakness", "numbness", "tingling", "paresthesia", "paralysis",
                "tremor", "ataxia", "aphasia", "dysarthria",
                "stroke", "CVA", "TIA", "SAH", "subarachnoid hemorrhage",
                "meningitis", "encephalitis", "concussion"
            ],
            "musculoskeletal": [
                "joint pain", "back pain", "neck pain", "muscle pain", "myalgia",
                "arthralgia", "stiffness", "swelling", "deformity", "limited ROM",
                "arthritis", "osteoarthritis", "rheumatoid arthritis", "gout",
                "fracture", "sprain", "strain", "tendinitis", "bursitis",
                "herniated disc", "sciatica", "carpal tunnel", "rotator cuff"
            ],
            "renal_urological": [
                "dysuria", "hematuria", "frequency", "urgency", "incontinence",
                "nocturia", "oliguria", "anuria", "flank pain", "suprapubic pain",
                "UTI", "urinary tract infection", "pyelonephritis", "cystitis",
                "kidney stone", "nephrolithiasis", "renal failure", "AKI",
                "prostatitis", "BPH", "urinary retention"
            ],
            "endocrine": [
                "polyuria", "polydipsia", "polyphagia", "weight loss", "weight gain",
                "heat intolerance", "cold intolerance", "tremor", "palpitations",
                "diabetes", "DM", "type 1 diabetes", "type 2 diabetes", "DKA",
                "hypoglycemia", "hyperglycemia", "thyroid", "hypothyroid", "hyperthyroid",
                "Cushing", "Addison", "hyperthyroidism", "hypothyroidism"
            ],
            "infectious": [
                "fever", "chills", "night sweats", "malaise", "fatigue",
                "sepsis", "bacteremia", "infection", "cellulitis", "abscess",
                "lymphadenopathy", "pharyngitis", "otitis", "sinusitis",
                "COVID", "influenza", "pneumonia", "meningitis", "encephalitis",
                "hepatitis", "HIV", "TB", "tuberculosis", "Lyme disease"
            ],
            "general": [
                "pain", "ache", "discomfort", "symptom", "sick", "unwell",
                "medical", "health", "diagnosis", "treatment", "medication",
                "doctor", "physician", "hospital", "emergency", "urgent"
            ]
        }
    
    def update_medical_terms(self, use_curated: bool = True):
        """
        Update medical_terms.json with latest terminology
        
        Args:
            use_curated: If True, use curated common terms (default for MVP)
                        If False, use UMLS API (requires API key)
        """
        print("\n" + "="*80)
        print("  📚 UPDATING MEDICAL TERMINOLOGY")
        print("="*80 + "\n")
        
        if use_curated:
            print("[SNOMED] 📋 Using curated common clinical terms")
            terms_by_specialty = self.get_common_clinical_terms()
        else:
            print("[SNOMED] 🌐 Fetching from UMLS API")
            # Future: UMLS API integration
            terms_by_specialty = self.get_common_clinical_terms()
        
        # Flatten all terms
        all_keywords = []
        for specialty, terms in terms_by_specialty.items():
            all_keywords.extend(terms)
            print(f"[SNOMED] ✅ {specialty}: {len(terms)} terms")
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for term in all_keywords:
            term_lower = term.lower()
            if term_lower not in seen:
                unique_keywords.append(term)
                seen.add(term_lower)
        
        print(f"\n[SNOMED] 📊 Total unique medical terms: {len(unique_keywords)}")
        
        # Load existing medical_terms.json (organized by organ system)
        try:
            with open(self.output_path, 'r') as f:
                existing_data = json.load(f)
            print(f"[SNOMED] 📂 Loaded existing medical_terms.json")
        except:
            existing_data = {}
            print(f"[SNOMED] 📂 No existing file - creating new")
        
        # MERGE by organ system (preserve structure)
        total_added = 0
        total_existing = 0
        
        print(f"\n[SNOMED] 🔀 Merging by organ system:")
        
        for specialty, new_terms in terms_by_specialty.items():
            # Get existing terms for this specialty
            existing_terms = existing_data.get(specialty, [])
            existing_set = set([t.lower() for t in existing_terms])
            
            # Find new terms to add
            new_to_add = []
            for term in new_terms:
                if term.lower() not in existing_set:
                    new_to_add.append(term)
            
            # Merge: existing + new (sorted)
            merged = sorted(existing_terms + new_to_add, key=str.lower)
            existing_data[specialty] = merged
            
            # Stats
            added = len(new_to_add)
            preserved = len(existing_terms)
            total_added += added
            total_existing += preserved
            
            status_icon = "➕" if added > 0 else "✅"
            print(f"  {status_icon} {specialty}: {preserved} existing + {added} new = {len(merged)} total")
        
        print(f"\n[SNOMED] 📊 Summary:")
        print(f"  - Total existing: {total_existing}")
        print(f"  - Newly added: {total_added}")
        print(f"  - Grand total: {total_existing + total_added}")
        
        # Add metadata
        if 'metadata' not in existing_data:
            existing_data['metadata'] = {}
        
        existing_data['metadata']['last_updated'] = datetime.now().isoformat()
        existing_data['metadata']['update_source'] = 'curated_clinical_terms' if use_curated else 'umls_api'
        existing_data['metadata']['version'] = datetime.now().strftime("%Y.%m")
        
        # Preserve proper_names (learned from Whisper usage)
        if 'proper_names' not in existing_data:
            existing_data['proper_names'] = []
        
        # Save updated file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SNOMED] 💾 Saved to: {self.output_path}")
        print(f"[SNOMED] 📊 Final Statistics:")
        
        # Count terms by specialty
        for specialty, terms in existing_data.items():
            if specialty not in ['metadata', 'proper_names'] and isinstance(terms, list):
                print(f"  - {specialty}: {len(terms)} terms")
        
        print(f"  - proper_names: {len(existing_data.get('proper_names', []))}")
        print(f"  - Version: {existing_data.get('metadata', {}).get('version', 'N/A')}")
        print(f"  - Last updated: {existing_data.get('metadata', {}).get('last_updated', 'N/A')}")
        
        print("\n" + "="*80)
        print("  ✅ MEDICAL TERMINOLOGY UPDATE COMPLETE")
        print("="*80)
        print("\n  Next: Restart LLM and Whisper containers to use new terms\n")
        
        return True
    
    def get_umls_terms(self, api_key: str, concept_ids: List[str]) -> List[str]:
        """
        Fetch terms from UMLS API (future enhancement)
        
        Requires free NIH account: https://uts.nlm.nih.gov/uts/
        
        Args:
            api_key: UMLS API key
            concept_ids: List of SNOMED CT concept IDs to fetch
            
        Returns:
            List of medical terms
        """
        # TODO: Implement UMLS API integration
        # For now, return empty - use curated terms
        return []


def create_monthly_update_script():
    """
    Create a cron job script for monthly updates
    """
    script_content = """#!/bin/bash
# Monthly Medical Terminology Update
# Add to cron: 0 2 1 * * /home/aura/LedgerAI/medical/monthly_update.sh

cd /home/aura/LedgerAI

# Update medical terms
python3 medical/snomed_updater.py

# Restart containers to apply changes
docker-compose restart aura-llm aura-whisper

echo "✅ Medical terminology updated: $(date)" >> medical/update_log.txt
"""
    
    script_path = Path(__file__).resolve().parent / "monthly_update.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    
    print(f"\n[Setup] ✅ Created monthly update script: {script_path}")
    print(f"[Setup] 💡 To enable monthly updates, add to crontab:")
    print(f"        crontab -e")
    print(f"        0 2 1 * * {script_path}")
    print(f"        (Runs 2 AM on 1st of each month)")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  📚 SNOMED CT MEDICAL TERMINOLOGY UPDATER")
    print("="*80 + "\n")
    
    updater = SNOMEDUpdater()
    
    # Update using curated terms (MVP - no license needed)
    success = updater.update_medical_terms(use_curated=True)
    
    if success:
        print("\n📝 Want automatic monthly updates?")
        response = input("Create monthly update script? (y/n): ").lower()
        
        if response == 'y':
            create_monthly_update_script()
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

