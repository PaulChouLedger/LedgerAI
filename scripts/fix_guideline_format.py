#!/usr/bin/env python3
"""
Fix DERM and RENAL guideline format inconsistency
Moves structured_oldcarts from top-level to inside key_features
"""

import json
import os
from pathlib import Path

def fix_guideline_format(file_path):
    """Fix guideline format by moving structured_oldcarts inside key_features"""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if structured_oldcarts is at top level (incorrect format)
    if 'structured_oldcarts' in data and 'key_features' in data:
        if 'structured_oldcarts' not in data['key_features']:
            print(f"  🔧 Fixing format: {file_path.name}")
            
            # Move structured_oldcarts inside key_features
            structured_oldcarts = data.pop('structured_oldcarts')
            data['key_features']['structured_oldcarts'] = structured_oldcarts
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Fixed: {file_path.name}")
            return True
        else:
            print(f"  ⏭️ Already correct format: {file_path.name}")
            return False
    else:
        print(f"  ⚠️ Missing required fields: {file_path.name}")
        return False

def main():
    """Main function to fix guideline formats for all organ systems"""
    print("\n" + "="*80)
    print("  🔧 FIXING GUIDELINE FORMATS")
    print("="*80)
    
    guidelines_dir = Path("llm-medical-container/medical/guidelines")
    
    # Process all organ systems that might have format issues
    organ_systems = ["DERM", "RENAL", "GU", "GYN"]
    
    for organ_system in organ_systems:
        organ_dir = guidelines_dir / organ_system
        if organ_dir.exists():
            print(f"\n📚 Fixing {organ_system} guidelines...")
            organ_files = list(organ_dir.glob("*.json"))
            fixed_count = 0
            
            for file_path in organ_files:
                try:
                    if fix_guideline_format(file_path):
                        fixed_count += 1
                except Exception as e:
                    print(f"  ❌ Error fixing {file_path.name}: {e}")
            
            print(f"✅ Fixed {fixed_count}/{len(organ_files)} {organ_system} guidelines")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
