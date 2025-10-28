#!/usr/bin/env python3
"""
Auto-generate structured_oldcarts from guideline text using LLM
This avoids manual JSON expansion for new guidelines

Usage:
  python auto_generate_structured_oldcarts.py --guideline-file GI_NewCondition.json
"""

import json
import argparse
import sys
import os

def generate_structured_oldcarts_with_llm(classic_presentation: str, organ_system: str, llm_fn) -> dict:
    """
    Use LLM to extract includes/excludes for each OLDCARTS element
    
    Args:
        classic_presentation: The classic presentation text from guideline
        organ_system: Organ system (GI, CARDIO, etc.)
        llm_fn: LLM function to use for generation
        
    Returns:
        Dictionary with structured_oldcarts format
    """
    
    # Build comprehensive prompt
    system_msg = """You are a medical expert. Extract structured OLDCARTS data from medical guideline text.
    
Return ONLY valid JSON with this exact structure:
{
  "onset": {
    "includes": ["term1", "term2"],
    "excludes": ["term3", "term4"]
  },
  "location": {
    "includes": ["term1", "term2"],
    "excludes": ["term3", "term4"]
  },
  "duration": { ... },
  "character": { ... },
  "aggravating": { ... },
  "relieving": { ... },
  "timing": { ... },
  "severity": { ... }
}

Rules:
- Includes: Terms that indicate THIS condition (specific, diagnostic terms)
- Excludes: Terms that indicate OTHER conditions (ruling out differentials)
- Location: Include anatomical sites, exclude opposite sides
- Be specific: Use medical terms, not vague descriptions
- Minimum 2 includes per element
- Maximum 8 includes per element"""
    
    user_msg = f"""Extract structured OLDCARTS from this {organ_system} guideline:

{classic_presentation}

Return JSON only, no explanation."""
    
    # Call LLM
    try:
        response = llm_fn(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,  # Low temperature for consistent output
            max_tokens=2000
        )
        
        # Parse JSON from response
        response_text = response.strip()
        
        # Try to find JSON in response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            structured_oldcarts = json.loads(json_str)
            return structured_oldcarts
        else:
            print(f"⚠️ No valid JSON found in LLM response")
            return {}
            
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON from LLM response: {e}")
        return {}
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return {}


def update_guideline_file(guideline_path: str, structured_oldcarts: dict, dry_run: bool = True):
    """
    Update guideline JSON file with generated structured_oldcarts
    
    Args:
        guideline_path: Path to guideline JSON file
        structured_oldcarts: Generated structured data
        dry_run: If True, show what would be done without saving
    """
    
    try:
        # Load existing guideline
        with open(guideline_path, 'r') as f:
            guideline = json.load(f)
        
        # Update structured_oldcarts
        if 'key_features' not in guideline:
            guideline['key_features'] = {}
        
        guideline['key_features']['structured_oldcarts'] = structured_oldcarts
        
        if dry_run:
            print(f"\n📋 Would update: {guideline_path}")
            print(f"✅ Generated structured_oldcarts for {len(structured_oldcarts)} elements")
            print(json.dumps(structured_oldcarts, indent=2))
        else:
            # Save updated guideline
            with open(guideline_path, 'w') as f:
                json.dump(guideline, f, indent=2)
            print(f"✅ Updated: {guideline_path}")
            
    except Exception as e:
        print(f"❌ Error updating guideline: {e}")


def main():
    parser = argparse.ArgumentParser(description="Auto-generate structured_oldcarts from guideline text")
    parser.add_argument('--guideline-file', required=True, help='Path to guideline JSON file')
    parser.add_argument('--llm-provider', choices=['openai', 'anthropic', 'ollama'], default='ollama', help='LLM provider')
    parser.add_argument('--apply', action='store_true', help='Actually update the file (default is dry-run)')
    
    args = parser.parse_args()
    
    # Load guideline
    if not os.path.exists(args.guideline_file):
        print(f"❌ File not found: {args.guideline_file}")
        sys.exit(1)
    
    with open(args.guideline_file, 'r') as f:
        guideline = json.load(f)
    
    # Extract needed info
    classic_presentation = guideline.get('key_features', {}).get('classic_presentation', '')
    if not classic_presentation:
        print(f"❌ No classic_presentation found in guideline")
        sys.exit(1)
    
    # Extract organ system from file path
    path_parts = args.guideline_file.split(os.sep)
    if len(path_parts) >= 2 and path_parts[-2] in ['GI', 'CARDIO', 'NEURO', 'MSK', 'DERM', 'GU', 'GYN', 'RENAL', 'ENDOCRINE', 'PULMONARY']:
        organ_system = path_parts[-2]
    else:
        organ_system = 'GI'  # Default
    
    print(f"📋 Processing: {guideline.get('condition', 'Unknown')}")
    print(f"🏥 Organ System: {organ_system}")
    print(f"📄 Classic Presentation: {classic_presentation[:200]}...")
    
    # TODO: Initialize LLM function based on provider
    # For now, use a placeholder that returns empty dict
    def mock_llm_fn(*args, **kwargs):
        return '{}'  # Empty JSON for now
    
    llm_fn = mock_llm_fn
    
    print(f"\n🤖 Generating structured_oldcarts with LLM...")
    structured_oldcarts = generate_structured_oldcarts_with_llm(
        classic_presentation, 
        organ_system, 
        llm_fn
    )
    
    if structured_oldcarts:
        update_guideline_file(args.guideline_file, structured_oldcarts, dry_run=not args.apply)
    else:
        print("❌ Failed to generate structured_oldcarts")


if __name__ == '__main__':
    main()

