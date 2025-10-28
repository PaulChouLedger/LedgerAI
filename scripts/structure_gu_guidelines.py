#!/usr/bin/env python3
"""
GU Guidelines Structuring Script
Adds structured_oldcarts data to all GU guidelines
"""

import json
import os
from pathlib import Path

def add_structured_oldcarts_to_gu_file(file_path):
    """Add structured_oldcarts to a GU guideline file"""
    
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
    
    # Parse location
    if "testicle" in classic_presentation or "testicular" in classic_presentation:
        structured_oldcarts["location"]["includes"].extend(["testicle", "testicular", "scrotal"])
    if "perineum" in classic_presentation or "perineal" in classic_presentation:
        structured_oldcarts["location"]["includes"].extend(["perineum", "perineal"])
    if "pelvic" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("pelvic")
    if "suprapubic" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("suprapubic")
    if "flank" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("flank")
    if "groin" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("groin")
    if "lower abdomen" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("lower abdomen")
    
    # Parse duration
    if "constant" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("constant")
        structured_oldcarts["duration"]["excludes"].extend(["intermittent", "episodic"])
    if "intermittent" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("intermittent")
        structured_oldcarts["duration"]["excludes"].extend(["constant", "continuous"])
    if "colicky" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("colicky")
    
    # Parse character
    if "severe" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("severe")
    if "sharp" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("sharp")
    if "dull" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("dull")
    if "aching" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("aching")
    if "stabbing" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("stabbing")
    if "cramping" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("cramping")
    
    # Parse aggravating
    if "urination" in classic_presentation or "dysuria" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].extend(["urination", "dysuria"])
    if "movement" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("movement")
    if "palpation" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("palpation")
    if "sitting" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("prolonged sitting")
    
    # Parse relieving
    if "nothing relieves" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("nothing relieves")
        structured_oldcarts["relieving"]["excludes"].extend(["position changes", "medications"])
    if "lying down" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("lying down")
    if "elevation" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("elevation")
    
    # Parse timing
    if "constant" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("constant")
    if "worsening" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("progressively worsening")
    if "waves" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("comes in waves")
    
    # Parse severity
    if "9-10/10" in classic_presentation or "10/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["9-10/10", "severe"])
    elif "8-10/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["8-10/10", "severe"])
    elif "6-8/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["6-8/10", "moderate to severe"])
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
    """Main function to structure all GU guidelines"""
    print("\n" + "="*80)
    print("  🔧 STRUCTURING GU GUIDELINES")
    print("="*80)
    
    gu_dir = Path("llm-medical-container/medical/guidelines/GU")
    
    if not gu_dir.exists():
        print(f"❌ GU directory not found: {gu_dir}")
        return
    
    gu_files = list(gu_dir.glob("*.json"))
    print(f"\n📚 Found {len(gu_files)} GU guideline files")
    
    structured_count = 0
    
    for file_path in gu_files:
        try:
            if add_structured_oldcarts_to_gu_file(file_path):
                structured_count += 1
        except Exception as e:
            print(f"  ❌ Error structuring {file_path.name}: {e}")
    
    print(f"\n✅ Structured {structured_count}/{len(gu_files)} GU guidelines")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
