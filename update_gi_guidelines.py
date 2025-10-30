#!/usr/bin/env python3
"""
Script to update all organ system guidelines with new patient-friendly structure
"""

import json
import os
from pathlib import Path

# Comprehensive patient-friendly mappings for all organ systems
PATIENT_FRIENDLY_MAPPINGS = {
    # Location terms - GI/Abdominal
    'right upper quadrant': 'top right side near your ribs',
    'left upper quadrant': 'top left side near your ribs', 
    'right lower quadrant': 'lower right side near your groin',
    'left lower quadrant': 'lower left side near your groin',
    'periumbilical': 'around your belly button',
    'epigastric': 'upper middle part of your belly',
    'suprapubic': 'lower part of your belly',
    'right subcostal': 'right side under your ribs',
    'left subcostal': 'left side under your ribs',
    'right flank': 'right side of your back',
    'left flank': 'left side of your back',
    'midline': 'down the middle of your belly',
    'bilateral': 'on both sides',
    'radiates to right shoulder': 'spreads to your right shoulder',
    'radiates to left shoulder': 'spreads to your left shoulder',
    'radiates to scapula': 'spreads to your shoulder blade',
    'radiates to back': 'spreads to your back',
    
    # Location terms - Cardiovascular
    'chest': 'chest area',
    'left chest': 'left side of your chest',
    'right chest': 'right side of your chest',
    'central chest': 'middle of your chest',
    'chest pain': 'chest pain',
    'chest pressure': 'pressure in your chest',
    'chest tightness': 'tightness in your chest',
    'chest heaviness': 'heaviness in your chest',
    'radiates to left arm': 'spreads to your left arm',
    'radiates to jaw': 'spreads to your jaw',
    'radiates to neck': 'spreads to your neck',
    'radiates to shoulder': 'spreads to your shoulder',
    
    # Location terms - Neurological
    'head': 'head',
    'headache': 'head pain',
    'migraine': 'severe head pain',
    'temple': 'side of your head',
    'forehead': 'front of your head',
    'back of head': 'back of your head',
    'neck': 'neck area',
    'spine': 'spine or back',
    'radiates to arm': 'spreads to your arm',
    'radiates to leg': 'spreads to your leg',
    
    # Location terms - Musculoskeletal
    'back': 'back area',
    'lower back': 'lower part of your back',
    'upper back': 'upper part of your back',
    'shoulder': 'shoulder area',
    'knee': 'knee area',
    'ankle': 'ankle area',
    'wrist': 'wrist area',
    'elbow': 'elbow area',
    'hip': 'hip area',
    'joint': 'joint area',
    'muscle': 'muscle area',
    
    # Location terms - Pulmonary
    'lungs': 'lung area',
    'chest wall': 'chest wall',
    'pleural': 'around your lungs',
    'respiratory': 'breathing area',
    
    # Location terms - Genitourinary
    'pelvic': 'pelvic area',
    'groin': 'groin area',
    'genital': 'genital area',
    'bladder': 'bladder area',
    'kidney': 'kidney area',
    'flank': 'side of your back',
    
    # Character terms
    'sharp': 'sharp or stabbing',
    'dull': 'dull or achy',
    'burning': 'burning sensation',
    'cramping': 'crampy or cramping',
    'crampy': 'crampy or cramping',
    'throbbing': 'throbbing or pulsing',
    'stabbing': 'sharp or stabbing',
    'aching': 'achy or sore',
    'pressure': 'pressure or heaviness',
    'colicky': 'comes and goes in waves',
    'severe': 'very bad or intense',
    'moderate': 'somewhat bad',
    'mild': 'not too bad',
    'crushing': 'crushing or squeezing',
    'squeezing': 'squeezing or crushing',
    'tight': 'tight or constricting',
    'constricting': 'tight or constricting',
    'tearing': 'tearing or ripping',
    'ripping': 'tearing or ripping',
    
    # Onset terms
    'sudden': 'all at once or very quickly',
    'gradual': 'slowly over time',
    'acute': 'sudden and severe',
    'chronic': 'ongoing for a long time',
    'insidious': 'slowly and gradually',
    'rapid': 'very quickly',
    
    # Duration terms
    'constant': 'continuous without stopping',
    'intermittent': 'comes and goes',
    'episodic': 'happens in episodes',
    'persistent': 'keeps going',
    'hours': 'for several hours',
    'minutes': 'just a few minutes',
    'days': 'for several days',
    'weeks': 'for several weeks',
    'months': 'for several months',
    'years': 'for several years',
    'brief': 'for a short time',
    'prolonged': 'for a long time',
    
    # Aggravating terms
    'fatty foods': 'greasy or fatty foods',
    'deep inspiration': 'taking deep breaths',
    'movement': 'moving around',
    'eating': 'eating food',
    'bending': 'bending over',
    'coughing': 'coughing',
    'sneezing': 'sneezing',
    'straining': 'pushing or straining',
    'exercise': 'physical activity',
    'stress': 'stress or anxiety',
    'cold weather': 'cold weather',
    'heat': 'hot weather',
    'lying flat': 'lying flat on your back',
    'sitting': 'sitting down',
    'standing': 'standing up',
    'walking': 'walking around',
    'exertion': 'physical exertion',
    'emotional stress': 'emotional stress',
    
    # Relieving terms
    'rest': 'lying still or resting',
    'heat': 'applying heat',
    'cold': 'applying cold',
    'pain medication': 'pain medicine',
    'avoiding fatty foods': 'staying away from greasy foods',
    'position changes': 'changing how you sit or lie',
    'sitting up': 'sitting up',
    'leaning forward': 'leaning forward',
    'nitroglycerin': 'heart medicine',
    'oxygen': 'breathing oxygen',
    'ice': 'applying ice',
    'massage': 'massaging the area',
    'stretching': 'stretching',
    
    # Timing terms
    'after meals': 'after eating',
    'before meals': 'before eating',
    'at night': 'during the night',
    'in the morning': 'in the morning',
    'progressive': 'gets worse over time',
    'unrelated to food': 'not connected to eating',
    'at rest': 'when resting',
    'with activity': 'during activity',
    'during sleep': 'while sleeping',
    'upon waking': 'when you wake up',
    'periodic': 'happens regularly',
    'irregular': 'happens irregularly',
    
    # Severity terms (numbers)
    '1': '1 out of 10',
    '2': '2 out of 10', 
    '3': '3 out of 10',
    '4': '4 out of 10',
    '5': '5 out of 10',
    '6': '6 out of 10',
    '7': '7 out of 10',
    '8': '8 out of 10',
    '9': '9 out of 10',
    '10': '10 out of 10',
    
    # Additional common terms
    'nausea': 'feeling sick to your stomach',
    'vomiting': 'throwing up',
    'diarrhea': 'loose or watery bowel movements',
    'constipation': 'hard to have bowel movements',
    'fever': 'high temperature',
    'chills': 'feeling cold and shivering',
    'sweating': 'sweating a lot',
    'dizziness': 'feeling dizzy or lightheaded',
    'fainting': 'passing out or fainting',
    'shortness of breath': 'trouble breathing',
    'difficulty breathing': 'hard to breathe',
    'wheezing': 'whistling sound when breathing',
    'cough': 'coughing',
    'sputum': 'mucus or phlegm',
    'blood': 'blood',
    'bleeding': 'bleeding',
    'swelling': 'swelling or puffiness',
    'redness': 'redness',
    'warmth': 'feeling warm',
    'numbness': 'feeling numb',
    'tingling': 'feeling tingly or pins and needles',
    'weakness': 'feeling weak',
    'fatigue': 'feeling very tired',
    'malaise': 'feeling unwell',
    'anorexia': 'not feeling hungry',
    'weight loss': 'losing weight',
    'weight gain': 'gaining weight',
}

def convert_term_to_patient_friendly(term):
    """Convert a medical term to patient-friendly language"""
    term_lower = term.lower()
    return PATIENT_FRIENDLY_MAPPINGS.get(term_lower, term)

def update_structured_oldcarts(structured_oldcarts):
    """Update structured OLDCARTS with patient-friendly terms"""
    for element, data in structured_oldcarts.items():
        if isinstance(data, dict) and 'includes' in data:
            new_includes = []
            for term in data['includes']:
                if isinstance(term, str):
                    new_includes.append({
                        'medical': term,
                        'patient_friendly': convert_term_to_patient_friendly(term)
                    })
                else:
                    new_includes.append(term)  # Already converted
            data['includes'] = new_includes
            
        if isinstance(data, dict) and 'excludes' in data:
            new_excludes = []
            for term in data['excludes']:
                if isinstance(term, str):
                    new_excludes.append({
                        'medical': term,
                        'patient_friendly': convert_term_to_patient_friendly(term)
                    })
                else:
                    new_excludes.append(term)  # Already converted
            data['excludes'] = new_excludes
    
    return structured_oldcarts

def update_guideline_file(file_path):
    """Update a single guideline file"""
    print(f"Updating {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            guideline = json.load(f)
        
        # Check if already updated
        if 'structured_oldcarts' in guideline.get('key_features', {}):
            structured = guideline['key_features']['structured_oldcarts']
            # Check if already has patient_friendly format
            if structured and any(isinstance(term, dict) for term in structured.get('onset', {}).get('includes', [])):
                print(f"  Already updated: {file_path}")
                return True
        
        # Update structured_oldcarts
        if 'structured_oldcarts' in guideline.get('key_features', {}):
            guideline['key_features']['structured_oldcarts'] = update_structured_oldcarts(
                guideline['key_features']['structured_oldcarts']
            )
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(guideline, f, indent=2)
            
            print(f"  ✅ Updated: {file_path}")
            return True
        else:
            print(f"  ⚠️ No structured_oldcarts found: {file_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error updating {file_path}: {e}")
        return False

def main():
    """Update all organ system guidelines"""
    guidelines_dir = Path("llm-medical-container/medical/guidelines")
    
    if not guidelines_dir.exists():
        print(f"Directory not found: {guidelines_dir}")
        return
    
    # Define all organ systems to update
    organ_systems = [
        "CARDIO", "DERM", "GU", "GYN", "MSK", "NEURO", "PULMONARY", "RENAL"
    ]
    
    total_updated = 0
    total_files = 0
    
    for system in organ_systems:
        system_dir = guidelines_dir / system
        if not system_dir.exists():
            print(f"⚠️ Directory not found: {system_dir}")
            continue
        
        print(f"\n🔄 Updating {system} guidelines...")
        system_updated = 0
        system_total = 0
        
        for json_file in system_dir.glob("*.json"):
            system_total += 1
            total_files += 1
            if update_guideline_file(json_file):
                system_updated += 1
                total_updated += 1
        
        print(f"  {system}: {system_updated}/{system_total} files updated")
    
    print(f"\n📊 Overall Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successfully updated: {total_updated}")
    print(f"  Failed: {total_files - total_updated}")
    print(f"  Success rate: {(total_updated/total_files*100):.1f}%")

if __name__ == "__main__":
    main()
