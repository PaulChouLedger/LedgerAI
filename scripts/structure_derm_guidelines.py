#!/usr/bin/env python3
"""
DERM Guidelines Structuring Script
Adds structured_oldcarts data to all DERM guidelines
"""

import json
import os
from pathlib import Path

def add_structured_oldcarts_to_derm_file(file_path):
    """Add structured_oldcarts to a DERM guideline file"""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Skip if already has structured_oldcarts
    if 'structured_oldcarts' in data:
        print(f"  ⏭️ Already structured: {file_path.name}")
        return False
    
    # Extract classic_presentation
    classic_presentation = data['key_features']['classic_presentation'].lower()
    
    # Create structured_oldcarts based on classic_presentation
    structured_oldcarts = {
        "onset": {
            "includes": [],
            "excludes": []
        },
        "location": {
            "includes": [],
            "excludes": []
        },
        "duration": {
            "includes": [],
            "excludes": []
        },
        "character": {
            "includes": [],
            "excludes": []
        },
        "aggravating": {
            "includes": [],
            "excludes": []
        },
        "relieving": {
            "includes": [],
            "excludes": []
        },
        "timing": {
            "includes": [],
            "excludes": []
        },
        "severity": {
            "includes": [],
            "excludes": []
        }
    }
    
    # Parse onset
    if "sudden" in classic_presentation:
        structured_oldcarts["onset"]["includes"].extend(["sudden", "immediate"])
        structured_oldcarts["onset"]["excludes"].extend(["gradual", "chronic"])
    elif "gradual" in classic_presentation:
        structured_oldcarts["onset"]["includes"].extend(["gradual", "progressive"])
        structured_oldcarts["onset"]["excludes"].extend(["sudden", "immediate"])
    
    if "hours" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("hours")
    if "days" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("days")
    if "weeks" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("weeks")
    
    # Parse location
    if "anywhere" in classic_presentation or "generalized" in classic_presentation:
        structured_oldcarts["location"]["includes"].extend(["anywhere", "generalized"])
    if "face" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("face")
    if "hands" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("hands")
    if "trunk" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("trunk")
    if "extremities" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("extremities")
    
    # Parse duration
    if "chronic" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("chronic")
        structured_oldcarts["duration"]["excludes"].extend(["acute", "short-term"])
    if "acute" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("acute")
        structured_oldcarts["duration"]["excludes"].extend(["chronic", "long-term"])
    
    # Parse character
    if "erythematous" in classic_presentation or "red" in classic_presentation:
        structured_oldcarts["character"]["includes"].extend(["erythematous", "red"])
    if "pruritic" in classic_presentation or "itchy" in classic_presentation:
        structured_oldcarts["character"]["includes"].extend(["pruritic", "itchy"])
    if "raised" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("raised")
    if "warm" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("warm")
    if "tender" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("tender")
    if "swollen" in classic_presentation or "edematous" in classic_presentation:
        structured_oldcarts["character"]["includes"].extend(["swollen", "edematous"])
    
    # Parse aggravating
    if "scratching" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("scratching")
    if "heat" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("heat")
    if "pressure" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("pressure")
    if "stress" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("stress")
    
    # Parse relieving
    if "cool" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("cool compresses")
    if "antihistamines" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("antihistamines")
    if "steroids" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("topical steroids")
    if "avoiding" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("avoiding triggers")
    
    # Parse timing
    if "night" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("worse at night")
    if "progressive" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("progressive")
    if "intermittent" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("intermittent")
    
    # Parse severity
    if "moderate" in classic_presentation:
        structured_oldcarts["severity"]["includes"].append("moderate")
    if "severe" in classic_presentation:
        structured_oldcarts["severity"]["includes"].append("severe")
    if "mild" in classic_presentation:
        structured_oldcarts["severity"]["includes"].append("mild")
    
    # Add to data
    data['structured_oldcarts'] = structured_oldcarts
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Structured: {file_path.name}")
    return True

def main():
    """Main function to structure all DERM guidelines"""
    print("\n" + "="*80)
    print("  🔧 STRUCTURING DERM GUIDELINES")
    print("="*80)
    
    derm_dir = Path("llm-medical-container/medical/guidelines/DERM")
    
    if not derm_dir.exists():
        print(f"❌ DERM directory not found: {derm_dir}")
        return
    
    derm_files = list(derm_dir.glob("*.json"))
    print(f"\n📚 Found {len(derm_files)} DERM guideline files")
    
    structured_count = 0
    
    for file_path in derm_files:
        try:
            if add_structured_oldcarts_to_derm_file(file_path):
                structured_count += 1
        except Exception as e:
            print(f"  ❌ Error structuring {file_path.name}: {e}")
    
    print(f"\n✅ Structured {structured_count}/{len(derm_files)} DERM guidelines")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
