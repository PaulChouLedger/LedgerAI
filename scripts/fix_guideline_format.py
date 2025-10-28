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
    """Main function to fix DERM and RENAL guideline formats"""
    print("\n" + "="*80)
    print("  🔧 FIXING DERM AND RENAL GUIDELINE FORMATS")
    print("="*80)
    
    # Fix DERM guidelines
    derm_dir = Path("llm-medical-container/medical/guidelines/DERM")
    if derm_dir.exists():
        print(f"\n📚 Fixing DERM guidelines...")
        derm_files = list(derm_dir.glob("*.json"))
        fixed_count = 0
        
        for file_path in derm_files:
            try:
                if fix_guideline_format(file_path):
                    fixed_count += 1
            except Exception as e:
                print(f"  ❌ Error fixing {file_path.name}: {e}")
        
        print(f"✅ Fixed {fixed_count}/{len(derm_files)} DERM guidelines")
    
    # Fix RENAL guidelines
    renal_dir = Path("llm-medical-container/medical/guidelines/RENAL")
    if renal_dir.exists():
        print(f"\n📚 Fixing RENAL guidelines...")
        renal_files = list(renal_dir.glob("*.json"))
        fixed_count = 0
        
        for file_path in renal_files:
            try:
                if fix_guideline_format(file_path):
                    fixed_count += 1
            except Exception as e:
                print(f"  ❌ Error fixing {file_path.name}: {e}")
        
        print(f"✅ Fixed {fixed_count}/{len(renal_files)} RENAL guidelines")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
