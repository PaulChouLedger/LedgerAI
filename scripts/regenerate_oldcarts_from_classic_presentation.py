#!/usr/bin/env python3
"""
Parse classic_presentation text and regenerate structured_oldcarts with concise, targeted terms.

This script:
1. Parses classic_presentation by OLDCARTS sections
2. Extracts key medical terms from each section
3. Maps medical terms to patient-friendly equivalents
4. Creates concise structured_oldcarts with only the most important terms
"""

import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

GUIDELINES_ROOT = Path(__file__).parent.parent / 'llm-medical-container' / 'medical' / 'guidelines'

# Patient-friendly mappings for common medical terms
PATIENT_FRIENDLY_MAPPINGS = {
    # Onset
    'sudden': 'all at once or very quickly',
    'gradual': 'slowly over time',
    'acute': 'sudden and severe',
    'chronic': 'ongoing for a long time',
    'hours': 'within a few hours',
    'days': 'for several days',
    'weeks': 'for weeks',
    'months': 'for months',
    'years': 'for years',
    'minutes': 'just a few minutes',
    
    # Location
    'chest': 'your chest',
    'back': 'your back',
    'neck': 'your neck',
    'abdomen': 'your belly',
    'right upper quadrant': 'top right side near your ribs',
    'ruq': 'top right side near your ribs',
    'right subcostal': 'right side under your ribs',
    'right lower quadrant': 'lower right side around your groin',
    'rlq': 'lower right side around your groin',
    'left upper quadrant': 'top left side near your ribs',
    'left lower quadrant': 'lower left side around your groin',
    'llq': 'lower left side around your groin',
    'epigastric': 'upper middle part of your belly',
    'periumbilical': 'around your belly button',
    'midline': 'middle of your belly',
    'diffuse': 'all over your belly',
    'radiates to right shoulder': 'spreads to your right shoulder',
    'radiates to scapula': 'spreads to your shoulder blade',
    'radiates to back': 'spreads to your back',
    'radiates to jaw': 'spreads to your jaw',
    'radiates': 'spreads or moves',
    'unilateral': 'one side',
    'bilateral': 'both sides',
    'frontal': 'front of your head',
    'temporal': 'side of your head',
    'occipital': 'back of your head',
    'cervical': 'back of your neck',
    'thoracic': 'your upper or mid back',
    'lumbar': 'your lower back',
    'interscapular': 'between your shoulder blades',
    'extremity': 'your arm or leg',
    'upper extremity': 'your arm',
    'lower extremity': 'your leg',
    
    # Duration
    'constant': 'continuous without stopping',
    'intermittent': 'comes and goes',
    'episodic': 'happens in episodes',
    'persistent': 'keeps going',
    
    # Character
    'sharp': 'sharp or stabbing',
    'stabbing': 'sharp or stabbing',
    'dull': 'dull or achy',
    'aching': 'aching',
    'burning': 'burning sensation',
    'cramping': 'cramping',
    'crampy': 'crampy or cramping',
    'pressure': 'pressure-like',
    'crushing': 'crushing',
    'colicky': 'comes and goes in waves',
    'tearing': 'tearing',
    'ripping': 'ripping',
    'severe': 'very bad or intense',
    
    # Aggravating
    'movement': 'moving around',
    'moving': 'moving around',
    'deep breathing': 'taking deep breaths',
    'coughing': 'coughing',
    'eating': 'eating',
    'after eating': 'after eating',
    'fatty food': 'fatty or greasy foods',
    'fatty meal': 'fatty or greasy foods',
    'lying down': 'lying down',
    'bending': 'bending over',
    'after eating': 'after eating',
    'after meals': 'after eating',
    
    # Relieving
    'rest': 'resting',
    'lying still': 'lying still',
    'antacids': 'antacids',
    'sitting up': 'sitting up',
    'nothing': 'nothing helps',
    'nothing helps': 'nothing helps',
    
    # Timing
    'after meals': 'after eating',
    'at night': 'during the night',
    'morning': 'in the morning',
    
    # Severity
    'mild': 'not too bad',
    'moderate': 'somewhat bad',
    'severe': 'very bad or intense',
    
    # Associated
    'nausea': 'nausea',
    'vomiting': 'vomiting',
    'fever': 'fever',
    'diarrhea': 'diarrhea',
    'constipation': 'constipation',
}


def normalize_term(term: str) -> str:
    """Normalize term for matching"""
    return term.lower().strip().replace('(', '').replace(')', '').replace('/', ' or ')

def normalize_term_preserve_parens(term: str) -> str:
    """Normalize term but preserve parentheses for medical term formatting"""
    return term.strip()


def get_patient_friendly(medical_term: str) -> str:
    """Get patient-friendly term, or return medical term if no mapping exists"""
    normalized = normalize_term(medical_term)
    
    # Handle radiation phrases starting with "TO" or "RADIATES TO"
    original_term = medical_term.strip()
    if original_term.upper().startswith('TO ') or original_term.upper().startswith('RADIATES TO'):
        # Remove "TO" or "RADIATES TO" prefix
        clean_term = re.sub(r'^(to|radiates to)\s+', '', original_term, flags=re.IGNORECASE).strip()
        # Remove explanatory parentheses (not anatomical progression)
        clean_term = re.sub(r'\s*\([^)]*\)', '', clean_term).strip()
        # Normalize for matching
        clean_normalized = normalize_term(clean_term)
        
        # Map anatomical terms
        anatomical_map = {
            'shoulder': 'shoulder',
            'scapula': 'shoulder blade',
            'jaw': 'jaw',
            'arm': 'arm',
            'leg': 'leg',
            'groin': 'groin',
            'back': 'back',
            'chest': 'chest',
            'abdomen': 'abdomen'
        }
        
        # Handle "OR" alternatives
        if ' or ' in clean_normalized or '/ ' in clean_normalized:
            parts = re.split(r'\s+or\s+|/\s+', clean_normalized)
            friendly_parts = []
            for part in parts:
                part = part.strip()
                # Check for anatomical terms
                for anat_term, friendly_anat in anatomical_map.items():
                    if anat_term in part.lower():
                        # Check for side indicators (right/left)
                        if 'right' in part.lower():
                            friendly_parts.append(f"your right {friendly_anat}")
                        elif 'left' in part.lower():
                            friendly_parts.append(f"your left {friendly_anat}")
                        else:
                            friendly_parts.append(f"your {friendly_anat}")
                        break
            if friendly_parts:
                if len(friendly_parts) == 1:
                    return f"spreads to {friendly_parts[0]}"
                else:
                    return f"spreads to {' or '.join(friendly_parts)}"
        
        # Single anatomical term
        for anat_term, friendly_anat in anatomical_map.items():
            if anat_term in clean_normalized:
                # Check for side indicators
                if 'right' in clean_normalized:
                    return f"spreads to your right {friendly_anat}"
                elif 'left' in clean_normalized:
                    return f"spreads to your left {friendly_anat}"
                else:
                    return f"spreads to your {friendly_anat}"
    
    # Handle radiation phrases - extract anatomical progression from parentheses
    if '→' in normalized:
        # Look for anatomical progression patterns with arrows
        if 'chest' in normalized and 'back' in normalized and 'abdomen' in normalized:
            return 'spreads from your chest to your back and down your abdomen'
        elif 'chest' in normalized and 'back' in normalized:
            return 'spreads from your chest to your back'
        elif 'chest' in normalized and 'abdomen' in normalized:
            return 'spreads from your chest down to your abdomen'
        elif 'back' in normalized and 'abdomen' in normalized:
            return 'spreads from your back to your abdomen'
    
    # Handle radiation patterns with "radiates to" or "along" phrases and parentheses
    if 'radiates' in normalized or 'along' in normalized:
        # Check for parenthesized anatomical progression (with arrows = anatomical)
        paren_match = re.search(r'\(([^)]+)\)', medical_term)
        if paren_match:
            paren_content = paren_match.group(1)
            # If parentheses contain arrows, it's anatomical progression
            if '→' in paren_content:
                paren_normalized = normalize_term(paren_content)
                # Extract anatomical path from parentheses
                if 'chest' in paren_normalized and 'back' in paren_normalized and 'abdomen' in paren_normalized:
                    return 'spreads from your chest to your back and down your abdomen'
                elif 'chest' in paren_normalized and 'back' in paren_normalized:
                    return 'spreads from your chest to your back'
                elif 'chest' in paren_normalized and 'abdomen' in paren_normalized:
                    return 'spreads from your chest down to your abdomen'
                elif 'back' in paren_normalized and 'abdomen' in paren_normalized:
                    return 'spreads from your back to your abdomen'
            # Otherwise, it's explanatory - remove it and process main term
        
        # Fallback: Look for anatomical terms in the phrase
        anatomical_terms = ['chest', 'back', 'abdomen', 'shoulder', 'scapula', 'jaw', 'arm', 'leg', 'groin']
        found_terms = [term for term in anatomical_terms if term in normalized]
        if len(found_terms) >= 2:
            if 'chest' in found_terms and 'abdomen' in found_terms:
                return 'spreads from your chest down to your abdomen'
            elif 'chest' in found_terms and 'back' in found_terms:
                return 'spreads from your chest to your back'
            elif 'back' in found_terms:
                return 'spreads to your back'
            else:
                return f"spreads to your {found_terms[-1]}"
        elif len(found_terms) == 1:
            return f"spreads to your {found_terms[0]}"
        else:
            return 'spreads or moves'
    
    # Direct mapping
    if normalized in PATIENT_FRIENDLY_MAPPINGS:
        return PATIENT_FRIENDLY_MAPPINGS[normalized]
    
    # Partial matches for compound terms
    for key, friendly in PATIENT_FRIENDLY_MAPPINGS.items():
        if key in normalized or normalized in key:
            return friendly
    
    # Default: return as-is (will need manual review)
    return medical_term


def extract_oldcarts_section(text: str, section_name: str) -> str:
    """Extract a specific OLDCARTS section from classic_presentation"""
    section_tags = {
        'onset': 'ONSET:',
        'location': 'LOCATION:',
        'duration': 'DURATION:',
        'character': 'CHARACTER:',
        'aggravating': 'AGGRAVATING:',
        'relieving': 'RELIEVING:',
        'timing': 'TIMING:',
        'severity': 'SEVERITY:',
        'associated': 'ASSOCIATED SYMPTOMS:',
    }
    
    tag = section_tags.get(section_name.lower())
    if not tag:
        return ""
    
    # Find the section
    pattern = rf'{re.escape(tag)}(.*?)(?=\b(?:ONSET|LOCATION|DURATION|CHARACTER|AGGRAVATING|RELIEVING|TIMING|SEVERITY|ASSOCIATED SYMPTOMS|KEY POSITIVES|KEY NEGATIVES):|$)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_key_terms(text: str, section_name: str) -> List[str]:
    """Extract key medical terms from an OLDCARTS section - prioritize ALL CAPS terms"""
    section_text = extract_oldcarts_section(text, section_name)
    if not section_text:
        return []
    
    # For LOCATION, only extract from first sentence (before RADIATES or similar radiation statements)
    if section_name.lower() == 'location':
        # Split on sentence boundaries and take first sentence
        first_sentence = section_text.split('.')[0] if '.' in section_text else section_text
        # Remove any mention of radiation
        first_sentence = re.sub(r'\b[Rr]ADIATES?\s+.+', '', first_sentence)
        section_text = first_sentence.strip()
    
    terms = []
    seen = set()
    
    # PRIMARY METHOD: Extract ALL CAPS terms (key medical terminology emphasized in text)
    all_caps_pattern = r'\b([A-Z]{3,}(?:\s+[A-Z]{3,})*)\b'  # 3+ letter ALL CAPS words/phrases
    all_caps = re.findall(all_caps_pattern, section_text)
    for term in all_caps:
        # Include single ALL CAPS words of length 4+ (likely medical terms)
        words = term.split()
        if len(words) == 1 and len(term) >= 4:
            normalized = term.strip().lower()
            if normalized not in seen:
                terms.append(term.strip())
                seen.add(normalized)
        elif len(words) > 1:  # Multi-word ALL CAPS phrases are definitely medical terms
            normalized = term.strip().lower()
            if normalized not in seen:
                terms.append(term.strip())
                seen.add(normalized)
    
    # SECONDARY: Extract parenthetical abbreviations like (RUQ), (RLQ)
    abbreviations = re.findall(r'\(([A-Z]{2,})\)', section_text)
    for abbr in abbreviations:
        normalized = abbr.lower()
        if normalized not in seen:
            terms.append(abbr)
            seen.add(normalized)
    
    # TERTIARY: Extract simple lowercase medical terms (1-2 word phrases only)
    # Focus on common anatomical locations, qualities, etc.
    simple_medical_terms = []
    
    # Extract 1-2 word lowercase phrases
    simple_patterns = [
        r'\b(chest|back|neck|head|abdomen|arm|leg|shoulder|knee|hip|jaw|hand|foot|wrist|ankle)\b',
        r'\b(unilateral|bilateral|frontal|temporal|occipital|epigastric|periumbilical)\b',
        r'\b(sharp|dull|aching|burning|cramping|pressure|crushing|tearing|ripping)\b',
        r'\b(constant|intermittent|episodic|persistent|sudden|gradual|acute|chronic)\b',
        r'\b(sudden|gradual|acute|chronic)\b',
        r'\b(mild|moderate|severe)\b',
        r'\b(movement|eating|coughing|breathing|lying down|sitting up|rest)\b',
    ]
    
    for pattern in simple_patterns:
        matches = re.findall(pattern, section_text, re.IGNORECASE)
        for match in matches:
            normalized = match.lower().strip()
            if normalized not in seen and len(normalized) > 2:
                terms.append(match.strip())
                seen.add(normalized)
    
    # Prioritize ALL CAPS terms first, then others, remove duplicates
    prioritized = []
    seen_prioritized = set()
    
    # Add ALL CAPS terms first
    for term in terms:
        if term.isupper() or term.istitle():
            normalized = term.lower()
            if normalized not in seen_prioritized:
                prioritized.append(term)
                seen_prioritized.add(normalized)
    
    # Add remaining terms
    for term in terms:
        normalized = term.lower()
        if normalized not in seen_prioritized:
            prioritized.append(term)
            seen_prioritized.add(normalized)
    
    return prioritized[:10]  # Return top 10 most important terms


def extract_excludes(text: str, section_name: str) -> List[str]:
    """Extract exclusion terms from KEY NEGATIVES section"""
    key_negatives = extract_oldcarts_section(text, 'KEY NEGATIVES')
    if not key_negatives:
        return []
    
    excludes = []
    seen = set()
    
    # Extract terms after "rules out", "uncommon", "NOT", etc.
    patterns = [
        r'(rules?\s+out\s+[^,\.]+)',
        r'(not\s+[A-Z][^,\.]+)',
        r'(no\s+[A-Z][^,\.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, key_negatives, re.IGNORECASE)
        for match in matches:
            # Extract key terms from the phrase
            terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[a-z]+)*)\b', match)
            for term in terms:
                normalized = term.lower().strip()
                if normalized not in seen and len(normalized) > 3:
                    excludes.append(term.strip())
                    seen.add(normalized)
    
    return excludes[:3]  # Limit to 3 most important excludes


def extract_radiation_terms(text: str) -> List[str]:
    """Extract radiation phrases from LOCATION section (after RADIATES keyword)"""
    location_text = extract_oldcarts_section(text, 'location')
    if not location_text:
        return []
    
    # Extract everything after "RADIATES" or "RADIATE" - capture the full phrase
    radiation_pattern = r'\b[Rr]ADIATES?\s+(.+?)(?:\.|$)'
    match = re.search(radiation_pattern, location_text)
    if not match:
        return []
    
    radiation_phrase = match.group(1).strip()
    
    # Clean up trailing explanatory phrases like "as dissection extends"
    # But keep parenthesized anatomical progression
    radiation_phrase = re.sub(r'\s+(as|when|if|whenever).+?(?=\)|$)', '', radiation_phrase, flags=re.IGNORECASE)
    radiation_phrase = radiation_phrase.strip()
    
    return [radiation_phrase]


def regenerate_structured_oldcarts(classic_presentation: str) -> Dict:
    """Regenerate structured_oldcarts from classic_presentation"""
    structured = OrderedDict()  # Use OrderedDict to preserve insertion order
    
    oldcarts_elements = ['onset', 'location', 'duration', 'character', 
                        'aggravating', 'relieving', 'timing', 'severity', 'associated']
    
    for element in oldcarts_elements:
        terms = extract_key_terms(classic_presentation, element)
        
        includes = []
        for term in terms[:5]:  # Limit to top 5 terms per element
            medical = normalize_term(term)
            patient_friendly = get_patient_friendly(term)
            includes.append({
                'medical': medical,
                'patient_friendly': patient_friendly
            })
        
        # Extract excludes from KEY NEGATIVES
        excludes = []
        excludes_list = extract_excludes(classic_presentation, element)
        for term in excludes_list[:3]:  # Limit to 3 excludes
            medical = normalize_term(term)
            patient_friendly = get_patient_friendly(term)
            excludes.append({
                'medical': medical,
                'patient_friendly': patient_friendly
            })
        
        structured[element] = {
            'includes': includes,
            'excludes': excludes if excludes else []
        }
        
        # Special handling: add radiation section immediately after location
        if element == 'location':
            # Special handling for location anatomical_type
            location_text = extract_oldcarts_section(classic_presentation, 'location')
            if location_text:
                if 'right upper' in location_text.lower() or 'ruq' in location_text.lower():
                    structured['location']['anatomical_type'] = 'right_upper'
                elif 'right lower' in location_text.lower() or 'rlq' in location_text.lower():
                    structured['location']['anatomical_type'] = 'right_lower'
                elif 'left upper' in location_text.lower() or 'luq' in location_text.lower():
                    structured['location']['anatomical_type'] = 'left_upper'
                elif 'left lower' in location_text.lower() or 'llq' in location_text.lower():
                    structured['location']['anatomical_type'] = 'left_lower'
                elif 'diffuse' in location_text.lower() or 'periumbilical' in location_text.lower():
                    structured['location']['anatomical_type'] = 'midline'
            
            # Extract separate radiation section immediately after location
            radiation_terms = extract_radiation_terms(classic_presentation)
            if radiation_terms:
                includes = []
                for term in radiation_terms[:5]:  # Limit to top 5 radiation terms
                    medical = normalize_term_preserve_parens(term)  # Preserve parentheses for radiation terms
                    patient_friendly = get_patient_friendly(term)
                    includes.append({
                        'medical': medical,
                        'patient_friendly': patient_friendly
                    })
                structured['radiation'] = {
                    'includes': includes,
                    'excludes': []
                }
    
    return structured


def update_guideline_file(file_path: Path, dry_run: bool = True) -> bool:
    """Update a single guideline file with regenerated structured_oldcarts"""
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return False
    
    classic_presentation = content.get('data', {}).get('key_features', {}).get('classic_presentation', '')
    if not classic_presentation:
        classic_presentation = content.get('key_features', {}).get('classic_presentation', '')
    
    if not classic_presentation:
        print(f"[Skip] No classic_presentation in {file_path.name}")
        return False
    
    # Regenerate structured_oldcarts
    new_structured = regenerate_structured_oldcarts(classic_presentation)
    
    if not dry_run:
        # Update the file
        root = content.get('data', content)
        root.setdefault('key_features', {})['structured_oldcarts'] = new_structured
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"[Update] {file_path.relative_to(GUIDELINES_ROOT)}")
    else:
        print(f"[Dry-run] Would update {file_path.relative_to(GUIDELINES_ROOT)}")
        print(f"  New structured_oldcarts has {len(new_structured)} elements")
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Regenerate structured_oldcarts from classic_presentation')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no file changes)')
    parser.add_argument('--organ-system', type=str, help='Only process specific organ system (GI, CARDIO, etc.)')
    args = parser.parse_args()
    
    if not GUIDELINES_ROOT.exists():
        print(f"[Error] Guidelines path does not exist: {GUIDELINES_ROOT}")
        return
    
    total = 0
    updated = 0
    
    for root, dirs, files in os.walk(GUIDELINES_ROOT):
        # Filter by organ system if specified
        if args.organ_system:
            organ_dir = Path(root).name
            if organ_dir.upper() != args.organ_system.upper():
                continue
        
        for file in files:
            if file.endswith('.json'):
                total += 1
                file_path = Path(root) / file
                if update_guideline_file(file_path, dry_run=args.dry_run):
                    updated += 1
    
    mode = "Dry-run" if args.dry_run else "Updated"
    print(f"\n[{mode}] Scanned {total} files, processed {updated} files.")


if __name__ == '__main__':
    main()

