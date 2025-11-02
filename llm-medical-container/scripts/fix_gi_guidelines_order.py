#!/usr/bin/env python3
"""
Fix GI guidelines to match new OLDCARTS order:
1. Reorder structured_oldcarts: onset, location, timing, duration, character, aggravating, relieving, severity
2. Move timing terms (constant, episodic, intermittent) from duration/character to timing
3. Add proper duration terms (timeframes) and timing terms
4. Remove timing terms from duration/character sections
"""

import json
import os
from pathlib import Path

# Define the correct order
CORRECT_ORDER = ['onset', 'location', 'timing', 'duration', 'character', 'aggravating', 'relieving', 'severity']

# Terms that belong in TIMING (not duration or character)
TIMING_TERMS = {
    'constant': 'continuous without stopping',
    'intermittent': 'comes and goes',
    'episodic': 'happens in episodes',
    'continuous': 'continuous without stopping',
    'comes and goes': 'comes and goes'
}

# Common duration terms (timeframes)
DURATION_TERMS = {
    'seconds': 'for a few seconds',
    'minutes': 'for a few minutes',
    'hours': 'for several hours',
    'days': 'for a few days',
    'weeks': 'for a few weeks',
    'months': 'for a few months'
}

def extract_medical_term(term_obj):
    """Extract medical term from term object (can be dict or string)"""
    if isinstance(term_obj, dict):
        return term_obj.get('medical', '')
    elif isinstance(term_obj, str):
        return term_obj
    return ''

def extract_patient_friendly(term_obj):
    """Extract patient_friendly term from term object"""
    if isinstance(term_obj, dict):
        return term_obj.get('patient_friendly', term_obj.get('medical', ''))
    elif isinstance(term_obj, str):
        return term_obj
    return ''

def is_timing_term(term_obj):
    """Check if a term belongs in timing section"""
    medical = extract_medical_term(term_obj).lower()
    return medical in TIMING_TERMS

def fix_guideline(guideline_path):
    """Fix a single guideline file"""
    print(f"\n📝 Fixing {guideline_path.name}...")
    
    with open(guideline_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'key_features' not in data or 'structured_oldcarts' not in data['key_features']:
        print(f"  ⚠️  No structured_oldcarts found, skipping")
        return False
    
    structured = data['key_features']['structured_oldcarts']
    
    # Step 1: Collect all timing terms from duration and character
    timing_terms_collected = {}
    timing_terms_from_duration = []
    timing_terms_from_character = []
    
    # Check duration section
    if 'duration' in structured:
        includes = structured['duration'].get('includes', [])
        new_includes = []
        for term in includes:
            if is_timing_term(term):
                medical = extract_medical_term(term)
                patient_friendly = extract_patient_friendly(term)
                timing_terms_from_duration.append({
                    'medical': medical,
                    'patient_friendly': patient_friendly or TIMING_TERMS.get(medical.lower(), medical)
                })
            else:
                new_includes.append(term)
        structured['duration']['includes'] = new_includes
        if timing_terms_from_duration:
            print(f"  🔄 Moved {len(timing_terms_from_duration)} timing terms from duration")
    
    # Check character section
    if 'character' in structured:
        includes = structured['character'].get('includes', [])
        new_includes = []
        for term in includes:
            if is_timing_term(term):
                medical = extract_medical_term(term)
                patient_friendly = extract_patient_friendly(term)
                timing_terms_from_character.append({
                    'medical': medical,
                    'patient_friendly': patient_friendly or TIMING_TERMS.get(medical.lower(), medical)
                })
            else:
                new_includes.append(term)
        structured['character']['includes'] = new_includes
        if timing_terms_from_character:
            print(f"  🔄 Moved {len(timing_terms_from_character)} timing terms from character")
    
    # Step 2: Merge timing terms into timing section and simplify (remove redundancies)
    if 'timing' not in structured:
        structured['timing'] = {'includes': [], 'excludes': []}
    
    existing_timing_terms = set()
    for term in structured['timing'].get('includes', []):
        medical = extract_medical_term(term).lower()
        existing_timing_terms.add(medical)
    
    # Add timing terms from duration/character
    all_timing_terms = timing_terms_from_duration + timing_terms_from_character
    for term_dict in all_timing_terms:
        medical_lower = term_dict['medical'].lower()
        if medical_lower not in existing_timing_terms:
            structured['timing']['includes'].append(term_dict)
            existing_timing_terms.add(medical_lower)
    
    # Simplify timing section - keep only primary pattern from classic_presentation (synonyms handle variations)
    # "continuous" is redundant with "constant" (synonym covers it)
    # "comes and goes" is redundant with "intermittent" (synonym covers it)
    classic = data['key_features'].get('classic_presentation', '').upper()
    
    # Core essential terms (what synonyms map to)
    core_timing_terms = {
        'constant': 'continuous without stopping',
        'intermittent': 'comes and goes',
        'episodic': 'happens in episodes'
    }
    
    # Determine primary timing pattern from classic_presentation (keep only what's actually present)
    # Priority: Check TIMING section first, then general mentions
    primary_timing = None
    
    # Look for TIMING: section
    timing_section_start = classic.find('TIMING:')
    if timing_section_start != -1:
        timing_section = classic[timing_section_start:timing_section_start+100].upper()
        if 'CONSTANT' in timing_section:
            primary_timing = 'constant'
        elif 'EPISODIC' in timing_section:
            primary_timing = 'episodic'
        elif 'INTERMITTENT' in timing_section:
            primary_timing = 'intermittent'
    
    # Fallback: check general mentions if TIMING section didn't specify
    if not primary_timing:
        if 'TIMING:' in classic or 'TIMING' in classic:
            # Check what follows TIMING keyword
            if 'CONSTANT' in classic:
                primary_timing = 'constant'
            elif 'EPISODIC' in classic:
                primary_timing = 'episodic'
            elif 'INTERMITTENT' in classic:
                primary_timing = 'intermittent'
    
    # Build simplified timing - only keep primary pattern, skip redundant terms
    simplified_timing = []
    if primary_timing:
        simplified_timing.append({
            'medical': primary_timing,
            'patient_friendly': core_timing_terms[primary_timing]
        })
        print(f"  ✨ Simplified timing to primary pattern: {primary_timing}")
    else:
        # Fallback: keep only core terms that aren't redundant
        seen_core_terms = set()
        for term in structured['timing'].get('includes', []):
            medical = extract_medical_term(term).lower()
            
            # Skip redundant terms
            if medical == 'continuous':  # Redundant with 'constant'
                continue
            if medical == 'comes and goes':  # Redundant with 'intermittent'
                continue
            
            # Keep core terms
            if medical in core_timing_terms and medical not in seen_core_terms:
                simplified_timing.append({
                    'medical': medical,
                    'patient_friendly': core_timing_terms[medical]
                })
                seen_core_terms.add(medical)
    
    structured['timing']['includes'] = simplified_timing
    
    # Step 3: Ensure duration has timeframes (extract from classic_presentation if needed)
    if 'duration' not in structured:
        structured['duration'] = {'includes': [], 'excludes': []}
    
    # Check classic_presentation for duration/timeframe mentions
    classic = data['key_features'].get('classic_presentation', '').lower()
    existing_duration_terms = set(extract_medical_term(t).lower() for t in structured['duration'].get('includes', []))
    
    # Add duration terms based on classic_presentation
    if 'seconds' in classic or 'second' in classic:
        if 'seconds' not in existing_duration_terms and 'second' not in existing_duration_terms:
            structured['duration']['includes'].append({
                'medical': 'seconds',
                'patient_friendly': 'for a few seconds'
            })
    if 'minutes' in classic or 'minute' in classic:
        if 'minutes' not in existing_duration_terms and 'minute' not in existing_duration_terms:
            structured['duration']['includes'].append({
                'medical': 'minutes',
                'patient_friendly': 'for a few minutes'
            })
    if 'hours' in classic or 'hour' in classic:
        if 'hours' not in existing_duration_terms and 'hour' not in existing_duration_terms:
            structured['duration']['includes'].append({
                'medical': 'hours',
                'patient_friendly': 'for several hours'
            })
    if 'days' in classic or 'day' in classic:
        if 'days' not in existing_duration_terms and 'day' not in existing_duration_terms:
            structured['duration']['includes'].append({
                'medical': 'days',
                'patient_friendly': 'for a few days'
            })
    if 'weeks' in classic or 'week' in classic:
        if 'weeks' not in existing_duration_terms and 'week' not in existing_duration_terms:
            structured['duration']['includes'].append({
                'medical': 'weeks',
                'patient_friendly': 'for a few weeks'
            })
    
    # Step 4: Reorder structured_oldcarts
    new_structured = {}
    for key in CORRECT_ORDER:
        if key in structured:
            new_structured[key] = structured[key]
    
    # Add any remaining keys (like 'associated', 'radiation') at the end
    for key in structured:
        if key not in CORRECT_ORDER:
            new_structured[key] = structured[key]
    
    data['key_features']['structured_oldcarts'] = new_structured
    
    # Step 5: Save the file
    with open(guideline_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Fixed: reordered, moved timing terms, ensured proper duration/timing terms")
    return True

def main():
    """Main function"""
    script_dir = Path(__file__).parent
    gi_dir = script_dir.parent / 'medical' / 'guidelines' / 'GI'
    
    if not gi_dir.exists():
        print(f"❌ GI guidelines directory not found: {gi_dir}")
        return
    
    gi_files = sorted(gi_dir.glob('GI_*.json'))
    print(f"📚 Found {len(gi_files)} GI guideline files")
    
    fixed_count = 0
    for gi_file in gi_files:
        try:
            if fix_guideline(gi_file):
                fixed_count += 1
        except Exception as e:
            print(f"  ❌ Error fixing {gi_file.name}: {e}")
    
    print(f"\n✅ Fixed {fixed_count}/{len(gi_files)} guidelines")
    print("\n📋 Summary:")
    print("  - Reordered structured_oldcarts: onset, location, timing, duration, ...")
    print("  - Moved timing terms (constant, episodic, intermittent) from duration/character to timing")
    print("  - Added duration terms (timeframes) where appropriate")
    print("  - Ensured timing section has proper terms")

if __name__ == '__main__':
    main()

