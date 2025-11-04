#!/usr/bin/env python3
"""
Migration script to add urgency designations to OLDCARTS elements
and remove red_flags section from guideline JSON files.

This script:
1. Reads existing guideline JSON files
2. Adds urgency field to each item in structured_oldcarts includes
3. Migrates red_flags to emergent urgency in appropriate OLDCARTS elements
4. Removes the red_flags section
"""

import json
import os
import sys
from pathlib import Path

# Emergent keywords that match red flags (only these get urgency field)
EMERGENT_KEYWORDS = [
    'hypotension', 'shock', 'septic', 'confusion', 'encephalopathy', 
    'altered mental', 'severe sepsis', 'call 911', 'rigors', 
    'high fever', '>102', '>103', 'perforation', 'perforated',
    'pyrexia', 'jaundice', 'bacteremia', 'hepatic encephalopathy'
]

def is_emergent_from_red_flag(red_flag_text: str, medical_term: str) -> bool:
    """Check if medical term matches a red flag statement"""
    red_flag_lower = red_flag_text.lower()
    medical_term_lower = medical_term.lower()
    
    # Check if medical term appears in red flag text
    if medical_term_lower in red_flag_lower:
        return True
    
    # Check for keyword matches
    for keyword in EMERGENT_KEYWORDS:
        if keyword in medical_term_lower and keyword in red_flag_lower:
            return True
    
    return False

def migrate_red_flags_to_oldcarts(red_flags: list, structured_oldcarts: dict) -> dict:
    """Migrate red flags to appropriate OLDCARTS elements - only mark matching items as emergent"""
    if not red_flags:
        return structured_oldcarts
    
    # Go through each OLDCARTS element and check if items match red flags
    for element, data in structured_oldcarts.items():
        if isinstance(data, dict) and 'includes' in data:
            for item in data['includes']:
                if isinstance(item, dict):
                    medical_term = item.get('medical', '')
                    
                    # Check if this medical term appears in any red flag
                    for red_flag in red_flags:
                        if is_emergent_from_red_flag(red_flag, medical_term):
                            # Mark as emergent - only items matching red flags get this
                            item['urgency'] = 'emergent'
                            break
                    # If no match, item doesn't get urgency field (normal symptom)
    
    return structured_oldcarts

def add_urgency_to_items(structured_oldcarts: dict) -> dict:
    """Only add urgency field to items that match red flags (emergent only)"""
    # Don't add urgency to items that don't match red flags
    # Items without urgency field are normal symptoms
    # Only items explicitly matching red flags get "emergent" urgency
    return structured_oldcarts

def migrate_guideline_file(file_path: Path) -> bool:
    """Migrate a single guideline file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            guideline = json.load(f)
        
        # Check if already migrated (has urgency in items)
        has_urgency = False
        structured = guideline.get('key_features', {}).get('structured_oldcarts', {})
        for element, data in structured.items():
            if isinstance(data, dict) and 'includes' in data:
                for item in data['includes']:
                    if isinstance(item, dict) and 'urgency' in item:
                        has_urgency = True
                        break
                if has_urgency:
                    break
        
        if has_urgency:
            print(f"  ⏭️  Already migrated: {file_path.name}")
            return False
        
        # Get red flags before migration
        red_flags = guideline.get('red_flags', [])
        
        # Migrate red flags to structured_oldcarts
        if red_flags:
            structured = migrate_red_flags_to_oldcarts(red_flags, structured)
        
        # Add urgency to all items that don't have it
        structured = add_urgency_to_items(structured)
        
        # Update guideline
        guideline['key_features']['structured_oldcarts'] = structured
        
        # Remove red_flags section
        if 'red_flags' in guideline:
            del guideline['red_flags']
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(guideline, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Migrated: {file_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error migrating {file_path.name}: {e}")
        return False

def main():
    """Main migration function"""
    if len(sys.argv) > 1:
        guidelines_dir = Path(sys.argv[1])
    else:
        # Default to llm-medical-container/medical/guidelines
        script_dir = Path(__file__).parent
        guidelines_dir = script_dir.parent / 'llm-medical-container' / 'medical' / 'guidelines'
    
    if not guidelines_dir.exists():
        print(f"❌ Guidelines directory not found: {guidelines_dir}")
        sys.exit(1)
    
    print(f"📋 Migrating guidelines in: {guidelines_dir}")
    print("")
    
    migrated_count = 0
    skipped_count = 0
    
    # Find all JSON files in subdirectories
    for json_file in sorted(guidelines_dir.glob("**/*.json")):
        if json_file.name == 'README.md':
            continue
        
        if migrate_guideline_file(json_file):
            migrated_count += 1
        else:
            skipped_count += 1
    
    print("")
    print(f"✅ Migration complete: {migrated_count} migrated, {skipped_count} skipped")

if __name__ == "__main__":
    main()

