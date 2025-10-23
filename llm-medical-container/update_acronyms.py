#!/usr/bin/env python3
"""
Script to update all medical guidelines to spell out acronyms
"""

import os
import json
import re
from pathlib import Path

# Define acronym mappings
ACRONYM_MAPPINGS = {
    # Anatomical quadrants
    'LLQ': 'left lower quadrant (LLQ)',
    'RLQ': 'right lower quadrant (RLQ)', 
    'RUQ': 'right upper quadrant (RUQ)',
    'LUQ': 'left upper quadrant (LUQ)',
    
    # Medical conditions
    'GERD': 'gastroesophageal reflux disease (GERD)',
    'IBS': 'irritable bowel syndrome (IBS)',
    'IBD': 'inflammatory bowel disease (IBD)',
    'UTI': 'urinary tract infection (UTI)',
    'PID': 'pelvic inflammatory disease (PID)',
    'COPD': 'chronic obstructive pulmonary disease (COPD)',
    'CHF': 'congestive heart failure (CHF)',
    'MI': 'myocardial infarction (MI)',
    'STEMI': 'ST-elevation myocardial infarction (STEMI)',
    'NSTEMI': 'non-ST-elevation myocardial infarction (NSTEMI)',
    'PE': 'pulmonary embolism (PE)',
    'DVT': 'deep vein thrombosis (DVT)',
    'CVA': 'cerebrovascular accident (CVA)',
    'TIA': 'transient ischemic attack (TIA)',
    'HTN': 'hypertension (HTN)',
    'DM': 'diabetes mellitus (DM)',
    'T2DM': 'type 2 diabetes mellitus (T2DM)',
    'T1DM': 'type 1 diabetes mellitus (T1DM)',
    'CKD': 'chronic kidney disease (CKD)',
    'ESRD': 'end-stage renal disease (ESRD)',
    'AKI': 'acute kidney injury (AKI)',
    
    # Cardiac conditions
    'AF': 'atrial fibrillation (AF)',
    'AFlutter': 'atrial flutter (AFlutter)',
    'VT': 'ventricular tachycardia (VT)',
    'VF': 'ventricular fibrillation (VF)',
    'SVT': 'supraventricular tachycardia (SVT)',
    'PSVT': 'paroxysmal supraventricular tachycardia (PSVT)',
    'AVNRT': 'atrioventricular nodal reentrant tachycardia (AVNRT)',
    'AVRT': 'atrioventricular reentrant tachycardia (AVRT)',
    'WPW': 'Wolff-Parkinson-White syndrome (WPW)',
    'LQTS': 'long QT syndrome (LQTS)',
    'HOCM': 'hypertrophic obstructive cardiomyopathy (HOCM)',
    'DCM': 'dilated cardiomyopathy (DCM)',
    'RCM': 'restrictive cardiomyopathy (RCM)',
    'HCM': 'hypertrophic cardiomyopathy (HCM)',
    
    # Cardiac valves
    'AS': 'aortic stenosis (AS)',
    'AR': 'aortic regurgitation (AR)',
    'MS': 'mitral stenosis (MS)',
    'MR': 'mitral regurgitation (MR)',
    'TR': 'tricuspid regurgitation (TR)',
    'TS': 'tricuspid stenosis (TS)',
    'PDA': 'patent ductus arteriosus (PDA)',
    
    # Cardiac conduction
    'LBBB': 'left bundle branch block (LBBB)',
    'RBBB': 'right bundle branch block (RBBB)',
    'AVB': 'atrioventricular block (AVB)',
    
    # Cardiac vessels
    'LAD': 'left anterior descending artery (LAD)',
    'RCA': 'right coronary artery (RCA)',
    'LCx': 'left circumflex artery (LCx)',
    
    # Vital signs
    'SBP': 'systolic blood pressure (SBP)',
    'DBP': 'diastolic blood pressure (DBP)',
    'HR': 'heart rate (HR)',
    'BP': 'blood pressure (BP)',
    'RR': 'respiratory rate (RR)',
    'O2': 'oxygen (O2)',
    'SaO2': 'arterial oxygen saturation (SaO2)',
    'SpO2': 'pulse oximetry oxygen saturation (SpO2)',
    
    # Diagnostic tests
    'EKG': 'electrocardiogram (EKG)',
    'ECG': 'electrocardiogram (ECG)',
    'CT': 'computed tomography (CT)',
    'MRI': 'magnetic resonance imaging (MRI)',
    'US': 'ultrasound (US)',
    'CXR': 'chest X-ray (CXR)',
    'ABG': 'arterial blood gas (ABG)',
    'CBC': 'complete blood count (CBC)',
    'BMP': 'basic metabolic panel (BMP)',
    'CMP': 'comprehensive metabolic panel (CMP)',
    'UA': 'urinalysis (UA)',
    
    # Other medical terms
    'UC': 'ulcerative colitis (UC)',
    'CD': 'Crohn\'s disease (CD)',
    'ACS': 'acute coronary syndrome (ACS)',
    'OMI': 'occlusion myocardial infarction (OMI)',
    'NOMI': 'non-occlusive mesenteric ischemia (NOMI)',
    'PLV': 'pulmonary venous pressure (PLV)',
    'TOF': 'tetralogy of Fallot (TOF)',
    'TGA': 'transposition of great arteries (TGA)',
    'DORV': 'double outlet right ventricle (DORV)',
    'HLHS': 'hypoplastic left heart syndrome (HLHS)',
    'TAPVR': 'total anomalous pulmonary venous return (TAPVR)',
    'PAPVR': 'partial anomalous pulmonary venous return (PAPVR)',
    'VSD': 'ventricular septal defect (VSD)',
    'ASD': 'atrial septal defect (ASD)',
    'COA': 'coarctation of aorta (COA)',
    'IAA': 'interrupted aortic arch (IAA)',
    'DAA': 'double aortic arch (DAA)',
    'RAA': 'right aortic arch (RAA)',
    'LAA': 'left aortic arch (LAA)',
    'SAA': 'single aortic arch (SAA)',
    'AAA': 'abdominal aortic aneurysm (AAA)',
    'TAA': 'thoracic aortic aneurysm (TAA)',
    'DTA': 'descending thoracic aorta (DTA)',
    'ATA': 'ascending thoracic aorta (ATA)',
    'PTA': 'posterior tibial artery (PTA)',
    'STA': 'superior temporal artery (STA)',
    'LTA': 'left temporal artery (LTA)',
    'RTA': 'right temporal artery (RTA)',
    'BTA': 'basilar temporal artery (BTA)',
    'CTA': 'computed tomography angiography (CTA)',
    'ETA': 'external temporal artery (ETA)',
    'FTA': 'frontal temporal artery (FTA)',
    'GTA': 'greater temporal artery (GTA)',
    'HTA': 'hepatic temporal artery (HTA)',
    'ITA': 'internal temporal artery (ITA)',
    'JTA': 'jugular temporal artery (JTA)',
    'KTA': 'kidney temporal artery (KTA)',
    'MTA': 'middle temporal artery (MTA)',
    'NTA': 'nasal temporal artery (NTA)',
    'OTA': 'occipital temporal artery (OTA)',
    'QTA': 'quadrate temporal artery (QTA)',
    'STA': 'superior temporal artery (STA)',
    'TTA': 'transverse temporal artery (TTA)',
    'UTA': 'upper temporal artery (UTA)',
    'VTA': 'ventral temporal artery (VTA)',
    'WTA': 'wrist temporal artery (WTA)',
    'XTA': 'xiphoid temporal artery (XTA)',
    'YTA': 'yolk temporal artery (YTA)',
    'ZTA': 'zygomatic temporal artery (ZTA)'
}

def update_file(file_path):
    """Update a single JSON file to spell out acronyms"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply acronym replacements (avoid double expansions)
        for acronym, expansion in ACRONYM_MAPPINGS.items():
            # Use word boundaries to avoid partial matches
            # Also avoid expanding if already expanded (contains parentheses)
            pattern = r'\b' + re.escape(acronym) + r'\b(?!\s*\([^)]*' + re.escape(acronym) + r'[^)]*\))'
            content = re.sub(pattern, expansion, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all guideline files"""
    guidelines_dir = Path("medical/guidelines")
    
    if not guidelines_dir.exists():
        print(f"Guidelines directory not found: {guidelines_dir}")
        return
    
    updated_count = 0
    total_count = 0
    
    # Process all JSON files in the guidelines directory
    for json_file in guidelines_dir.rglob("*.json"):
        total_count += 1
        if update_file(json_file):
            updated_count += 1
    
    print(f"\nSummary:")
    print(f"Total files processed: {total_count}")
    print(f"Files updated: {updated_count}")
    print(f"Files unchanged: {total_count - updated_count}")

if __name__ == "__main__":
    main()
