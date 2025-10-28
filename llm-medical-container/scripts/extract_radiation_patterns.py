#!/usr/bin/env python3
"""
Extract radiation patterns from classic_presentation and add to structured_oldcarts location.includes
"""

import json
import re
import os
from pathlib import Path

def extract_radiation_patterns(text):
    """Extract radiation patterns from classic presentation text"""
    radiation_patterns = []
    
    # Look for LOCATION section
    location_match = re.search(r'LOCATION:([^A-Z]+?)(?=\s+[A-Z]+:|$)', text, re.IGNORECASE | re.DOTALL)
    if location_match:
        location_text = location_match.group(1)
        
        # Find radiation patterns
        radiation_keywords = [
            r'RADIATES?\s+TO\s+([^.,]+?)(?:\.|,|$)',
            r'RADIA\s+(?:pattern|quality)',
            r'goes\s+to',
            r'travels?\s+to',
            r'spreads?\s+to',
        ]
        
        for pattern in radiation_keywords:
            matches = re.finditer(pattern, location_text, re.IGNORECASE)
            for match in matches:
                if 'radiates' in match.group(0).lower() or 'radiate' in match.group(0).lower():
                    # Extract what it radiates to
                    full_match = match.group(0)
                    if 'to' in full_match.lower():
                        # Extract everything after "to"
                        rad_target = re.search(r'to\s+(.+?)(?:\.|,|$)', full_match, re.IGNORECASE)
                        if rad_target:
                            target = rad_target.group(1).strip()
                            # Clean up
                            target = re.sub(r'\s+', ' ', target)
                            # Remove extra words like "in", "the", etc.
                            target = re.sub(r'^to\s+', '', target, flags=re.I)
                            radiation_patterns.append(f"radiates to {target}")
                    else:
                        # Just add "radiates"
                        radiation_patterns.append("radiates")
    
    return radiation_patterns

def process_guidelines():
    """Process all GI guidelines"""
    guidelines_dir = Path(__file__).parent.parent / "medical" / "guidelines" / "GI"
    
    updated_files = []
    
    for json_file in sorted(guidelines_dir.glob("*.json")):
        print(f"\n{'='*80}")
        print(f"Processing: {json_file.name}")
        
        with open(json_file, 'r') as f:
            guideline = json.load(f)
        
        # Get classic presentation
        classic_presentation = guideline.get('key_features', {}).get('classic_presentation', '')
        
        # Extract radiation patterns
        radiation_patterns = extract_radiation_patterns(classic_presentation)
        
        if radiation_patterns:
            print(f"  Found radiation patterns: {radiation_patterns}")
            
            # Get structured_oldcarts
            structured_oldcarts = guideline.get('key_features', {}).get('structured_oldcarts', {})
            
            if 'location' not in structured_oldcarts:
                structured_oldcarts['location'] = {'includes': [], 'excludes': []}
            
            location_includes = structured_oldcarts['location'].get('includes', [])
            
            # Check if radiation already present
            has_radiation = any('radiat' in str(item).lower() for item in location_includes)
            
            if not has_radiation:
                # Add radiation patterns
                for pattern in radiation_patterns:
                    if pattern not in location_includes:
                        location_includes.append(pattern)
                        print(f"  ✅ Added: '{pattern}'")
                
                # Save updated file
                with open(json_file, 'w') as f:
                    json.dump(guideline, f, indent=2)
                
                updated_files.append(json_file.name)
            else:
                print(f"  ⏭️  Radiation already present in location.includes")
        else:
            print(f"  ℹ️  No radiation patterns found")
    
    print(f"\n{'='*80}")
    print(f"✅ Processed {len(list(guidelines_dir.glob('*.json')))} files")
    if updated_files:
        print(f"✅ Updated {len(updated_files)} files: {', '.join(updated_files)}")
    else:
        print(f"ℹ️  No files needed updates")

if __name__ == "__main__":
    process_guidelines()

