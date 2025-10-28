#!/usr/bin/env python3
"""
RENAL Guidelines Structuring Script
Adds structured_oldcarts data to all RENAL guidelines
"""

import json
import os
from pathlib import Path

def add_structured_oldcarts_to_renal_file(file_path):
    """Add structured_oldcarts to a RENAL guideline file"""
    
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
        structured_oldcarts["onset"]["includes"].extend(["sudden", "acute", "immediate"])
        structured_oldcarts["onset"]["excludes"].extend(["gradual", "chronic"])
    elif "gradual" in classic_presentation:
        structured_oldcarts["onset"]["includes"].extend(["gradual", "progressive"])
        structured_oldcarts["onset"]["excludes"].extend(["sudden", "immediate"])
    
    if "minutes" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("minutes")
    if "hours" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("hours")
    if "days" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("days")
    if "weeks" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("weeks")
    
    # Parse location
    if "kidney" in classic_presentation or "renal" in classic_presentation:
        structured_oldcarts["location"]["includes"].extend(["kidney", "renal"])
    if "flank" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("flank")
    if "bilateral" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("bilateral")
    if "unilateral" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("unilateral")
    if "back" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("back")
    
    # Parse duration
    if "constant" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("constant")
        structured_oldcarts["duration"]["excludes"].extend(["intermittent", "episodic"])
    if "intermittent" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("intermittent")
        structured_oldcarts["duration"]["excludes"].extend(["constant", "continuous"])
    if "persists" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("persistent")
    if "develops" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("progressive")
    
    # Parse character
    if "severe" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("severe")
    if "sharp" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("sharp")
    if "dull" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("dull")
    if "aching" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("aching")
    if "colicky" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("colicky")
    if "decreased urine output" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("decreased urine output")
    if "swelling" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("swelling")
    if "fluid retention" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("fluid retention")
    
    # Parse aggravating
    if "movement" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("movement")
    if "urination" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("urination")
    if "dehydration" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("dehydration")
    if "medications" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("medications")
    if "contrast" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("contrast agents")
    
    # Parse relieving
    if "antibiotics" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("antibiotics")
    if "pain medications" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("pain medications")
    if "fluid management" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("fluid management")
    if "dialysis" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("dialysis")
    
    # Parse timing
    if "constant" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("constant")
    if "worsening" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("progressively worsening")
    if "unrelenting" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("unrelenting")
    if "develops" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("progressive")
    
    # Parse severity
    if "7-9/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["7-9/10", "severe"])
    elif "6-9/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["6-9/10", "moderate to severe"])
    elif "8-10/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["8-10/10", "severe"])
    elif "moderate" in classic_presentation:
        structured_oldcarts["severity"]["includes"].append("moderate")
    elif "severe" in classic_presentation:
        structured_oldcarts["severity"]["includes"].append("severe")
    
    # Add to data
    data['structured_oldcarts'] = structured_oldcarts
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Structured: {file_path.name}")
    return True

def main():
    """Main function to structure all RENAL guidelines"""
    print("\n" + "="*80)
    print("  🔧 STRUCTURING RENAL GUIDELINES")
    print("="*80)
    
    renal_dir = Path("llm-medical-container/medical/guidelines/RENAL")
    
    if not renal_dir.exists():
        print(f"❌ RENAL directory not found: {renal_dir}")
        return
    
    renal_files = list(renal_dir.glob("*.json"))
    print(f"\n📚 Found {len(renal_files)} RENAL guideline files")
    
    structured_count = 0
    
    for file_path in renal_files:
        try:
            if add_structured_oldcarts_to_renal_file(file_path):
                structured_count += 1
        except Exception as e:
            print(f"  ❌ Error structuring {file_path.name}: {e}")
    
    print(f"\n✅ Structured {structured_count}/{len(renal_files)} RENAL guidelines")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
