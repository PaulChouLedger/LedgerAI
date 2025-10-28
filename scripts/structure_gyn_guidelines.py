#!/usr/bin/env python3
"""
GYN Guidelines Structuring Script
Adds structured_oldcarts data to all GYN guidelines
"""

import json
import os
from pathlib import Path

def add_structured_oldcarts_to_gyn_file(file_path):
    """Add structured_oldcarts to a GYN guideline file"""
    
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
    elif "subacute" in classic_presentation:
        structured_oldcarts["onset"]["includes"].extend(["subacute", "days to weeks"])
        structured_oldcarts["onset"]["excludes"].extend(["sudden", "chronic"])
    
    if "minutes" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("minutes")
    if "hours" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("hours")
    if "days" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("days")
    if "weeks" in classic_presentation:
        structured_oldcarts["onset"]["includes"].append("weeks")
    
    # Parse location
    if "pelvic" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("pelvic")
    if "lower abdominal" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("lower abdominal")
    if "unilateral" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("unilateral")
    if "bilateral" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("bilateral")
    if "adnexal" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("adnexal")
    if "vaginal" in classic_presentation:
        structured_oldcarts["location"]["includes"].append("vaginal")
    
    # Parse duration
    if "constant" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("constant")
        structured_oldcarts["duration"]["excludes"].extend(["intermittent", "episodic"])
    if "intermittent" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("intermittent")
        structured_oldcarts["duration"]["excludes"].extend(["constant", "continuous"])
    if "persists" in classic_presentation:
        structured_oldcarts["duration"]["includes"].append("persistent")
    
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
    if "crampy" in classic_presentation or "cramping" in classic_presentation:
        structured_oldcarts["character"]["includes"].extend(["crampy", "cramping"])
    if "twisting" in classic_presentation:
        structured_oldcarts["character"]["includes"].append("twisting")
    
    # Parse aggravating
    if "movement" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("movement")
    if "intercourse" in classic_presentation or "dyspareunia" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].extend(["intercourse", "dyspareunia"])
    if "physical activity" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("physical activity")
    if "position change" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("position changes")
    if "jarring" in classic_presentation:
        structured_oldcarts["aggravating"]["includes"].append("jarring movements")
    
    # Parse relieving
    if "nothing helps" in classic_presentation or "nothing relieves" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("nothing relieves")
        structured_oldcarts["relieving"]["excludes"].extend(["position changes", "medications"])
    if "rest" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("rest")
    if "antibiotics" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("antibiotics")
    if "surgical" in classic_presentation:
        structured_oldcarts["relieving"]["includes"].append("surgical intervention")
    
    # Parse timing
    if "constant" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("constant")
    if "worsening" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("progressively worsening")
    if "unrelenting" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("unrelenting")
    if "episodic" in classic_presentation:
        structured_oldcarts["timing"]["includes"].append("episodic")
    
    # Parse severity
    if "8-10/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["8-10/10", "very severe"])
    elif "9-10/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["9-10/10", "severe"])
    elif "5-8/10" in classic_presentation:
        structured_oldcarts["severity"]["includes"].extend(["5-8/10", "moderate to severe"])
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
    """Main function to structure all GYN guidelines"""
    print("\n" + "="*80)
    print("  🔧 STRUCTURING GYN GUIDELINES")
    print("="*80)
    
    gyn_dir = Path("llm-medical-container/medical/guidelines/GYN")
    
    if not gyn_dir.exists():
        print(f"❌ GYN directory not found: {gyn_dir}")
        return
    
    gyn_files = list(gyn_dir.glob("*.json"))
    print(f"\n📚 Found {len(gyn_files)} GYN guideline files")
    
    structured_count = 0
    
    for file_path in gyn_files:
        try:
            if add_structured_oldcarts_to_gyn_file(file_path):
                structured_count += 1
        except Exception as e:
            print(f"  ❌ Error structuring {file_path.name}: {e}")
    
    print(f"\n✅ Structured {structured_count}/{len(gyn_files)} GYN guidelines")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
