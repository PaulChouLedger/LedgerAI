#!/usr/bin/env python3
"""
Fix Training Dataset: Ensure All Evidence is Verbatim
This script:
1. Checks all evidence quotes in the dataset
2. Verifies they match verbatim from context
3. Fixes non-verbatim evidence
4. Adds examples emphasizing strict verbatim extraction
"""

import json
import re
from typing import List, Dict, Any, Tuple

SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- Read entire descriptions/chunks completely - titles may appear later in the text.
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate."""

def find_verbatim_match(evidence: str, context: str) -> Tuple[bool, str]:
    """
    Find if evidence exists verbatim in context.
    Returns (is_verbatim, corrected_evidence)
    """
    evidence_clean = evidence.strip()
    
    # Try exact match (case-insensitive)
    if evidence_clean.lower() in context.lower():
        # Find the exact case version
        idx = context.lower().find(evidence_clean.lower())
        if idx != -1:
            exact_match = context[idx:idx+len(evidence_clean)]
            return True, exact_match
    
    # Try to find longest matching substring
    words = evidence_clean.split()
    if len(words) < 3:
        return False, evidence_clean
    
    # Try progressively shorter phrases
    for length in range(len(words), 2, -1):
        for start in range(len(words) - length + 1):
            phrase = ' '.join(words[start:start+length])
            if phrase.lower() in context.lower():
                idx = context.lower().find(phrase.lower())
                if idx != -1:
                    # Try to expand to full sentence or meaningful phrase
                    # Look for sentence boundaries
                    start_idx = max(0, idx - 50)
                    end_idx = min(len(context), idx + len(phrase) + 50)
                    expanded = context[start_idx:end_idx]
                    
                    # Try to find a complete phrase containing our match
                    if phrase in expanded:
                        # Find sentence boundaries
                        sentence_start = expanded.rfind('.', 0, expanded.find(phrase))
                        sentence_end = expanded.find('.', expanded.find(phrase) + len(phrase))
                        
                        if sentence_start != -1 and sentence_end != -1:
                            full_phrase = expanded[sentence_start+1:sentence_end].strip()
                            if len(full_phrase) > len(phrase) and len(full_phrase) < 200:
                                return True, full_phrase
                        elif sentence_end != -1:
                            full_phrase = expanded[:sentence_end].strip()
                            if len(full_phrase) > len(phrase) and len(full_phrase) < 200:
                                return True, full_phrase
                    
                    return True, phrase
    
    return False, evidence_clean

def fix_example_evidence(example: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fix evidence in a single example to ensure verbatim matching.
    Returns (fixed_example, warnings)
    """
    warnings = []
    user_content = example['messages'][1]['content']
    assistant_content = example['messages'][2]['content']
    
    # Extract context
    context = user_content.split('Question:')[0].replace('Knowledge context:', '').strip()
    
    # Extract evidence quotes
    evidence_pattern = r'- Evidence:\s*"([^"]+)"'
    evidences = re.findall(evidence_pattern, assistant_content)
    
    if not evidences:
        return example, warnings
    
    # Fix each evidence
    fixed_content = assistant_content
    for evidence in evidences:
        is_verbatim, corrected = find_verbatim_match(evidence, context)
        
        if not is_verbatim:
            warnings.append(f"Could not find verbatim match for: '{evidence[:50]}...'")
            # Try to find closest match
            words = evidence.split()
            if len(words) >= 3:
                # Try first 3 words
                phrase = ' '.join(words[:3])
                if phrase.lower() in context.lower():
                    idx = context.lower().find(phrase.lower())
                    if idx != -1:
                        # Extract surrounding text
                        start = max(0, idx - 20)
                        end = min(len(context), idx + 100)
                        corrected = context[start:end].strip()
                        # Clean up
                        if corrected.startswith('.'):
                            corrected = corrected[1:].strip()
                        if len(corrected) > 150:
                            # Find sentence end
                            period = corrected.find('.', 50)
                            if period != -1:
                                corrected = corrected[:period+1]
        
        # Replace in content
        old_evidence = f'- Evidence: "{evidence}"'
        new_evidence = f'- Evidence: "{corrected}"'
        fixed_content = fixed_content.replace(old_evidence, new_evidence, 1)
    
    # Update example
    fixed_example = example.copy()
    fixed_example['messages'][2]['content'] = fixed_content
    
    return fixed_example, warnings

def add_verbatim_emphasis_examples() -> List[Dict[str, Any]]:
    """
    Add examples that emphasize strict verbatim evidence extraction.
    """
    examples = []
    
    # Example 1: Co-founder with exact verbatim evidence
    example1 = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": """Knowledge context: John Smith is the CEO and Co-Founder of TechCorp. He has been leading the company since 2015. Sarah Johnson is the Co-Founder and Chief Technology Officer of TechCorp. She joined in 2016. Michael Brown is the Head of Engineering at TechCorp. He is not a co-founder.
---
Question: Who are the co-founders of TechCorp?"""
            },
            {
                "role": "assistant",
                "content": """REASONING:
- Item: John Smith
  - Evidence: "John Smith is the CEO and Co-Founder of TechCorp"
  - Action: [KEEP]
- Item: Sarah Johnson
  - Evidence: "Sarah Johnson is the Co-Founder and Chief Technology Officer of TechCorp"
  - Action: [KEEP]
- Item: Michael Brown
  - Evidence: "Michael Brown is the Head of Engineering at TechCorp"
  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder).
- End of scan.

FINAL ANSWER:
The co-founders of TechCorp are John Smith and Sarah Johnson."""
            }
        ]
    }
    examples.append(example1)
    
    # Example 2: Role query with exact verbatim
    example2 = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": """Knowledge context: David Martinez serves as the Chief Financial Officer of DataFlow Inc. He has been CFO since 2018. Robert Kim is the Chief Technology Officer at DataFlow Inc. Mark Williams is the CEO of DataFlow Inc.
---
Question: Who is the CFO of DataFlow Inc?"""
            },
            {
                "role": "assistant",
                "content": """REASONING:
- Item: David Martinez
  - Evidence: "David Martinez serves as the Chief Financial Officer of DataFlow Inc"
  - Action: [KEEP]
- Item: Robert Kim
  - Evidence: "Robert Kim is the Chief Technology Officer at DataFlow Inc"
  - Action: [DISCARD] (Reason: CTO, not CFO).
- Item: Mark Williams
  - Evidence: "Mark Williams is the CEO of DataFlow Inc"
  - Action: [DISCARD] (Reason: CEO, not CFO).
- End of scan.

FINAL ANSWER:
The CFO of DataFlow Inc is David Martinez."""
            }
        ]
    }
    examples.append(example2)
    
    # Example 3: Multi-chunk co-founder extraction
    example3 = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": """Knowledge context: Paul Anderson is a visionary leader. As Co-Founder and CEO of InnovateLab, he drives innovation. He previously worked at Google.
---
Jennifer Lee is the Co-Founder and Chief Product Officer of InnovateLab. She has 15 years of experience in product development.
---
Thomas Wilson leads the engineering team at InnovateLab as Head of Engineering. He joined in 2020.
---
Question: Who are the co-founders of InnovateLab?"""
            },
            {
                "role": "assistant",
                "content": """REASONING:
- Item: Paul Anderson
  - Evidence: "As Co-Founder and CEO of InnovateLab"
  - Action: [KEEP]
- Item: Jennifer Lee
  - Evidence: "Jennifer Lee is the Co-Founder and Chief Product Officer of InnovateLab"
  - Action: [KEEP]
- Item: Thomas Wilson
  - Evidence: "Thomas Wilson leads the engineering team at InnovateLab as Head of Engineering"
  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder).
- End of scan.

FINAL ANSWER:
The co-founders of InnovateLab are Paul Anderson and Jennifer Lee."""
            }
        ]
    }
    examples.append(example3)
    
    return examples

def main():
    print("=" * 80)
    print("FIXING TRAINING DATASET: VERBATIM EVIDENCE EXTRACTION")
    print("=" * 80)
    
    # Load dataset
    with open('rag_cot_training_dataset.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Original Dataset:")
    print(f"   Total examples: {len(data)}")
    
    # Fix examples
    fixed_data = []
    total_warnings = []
    
    print(f"\n🔧 Fixing evidence in examples...")
    for i, ex in enumerate(data):
        fixed_ex, warnings = fix_example_evidence(ex)
        fixed_data.append(fixed_ex)
        if warnings:
            total_warnings.extend([(i, w) for w in warnings])
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(data)} examples...")
    
    print(f"   ✅ Fixed {len(data)} examples")
    
    if total_warnings:
        print(f"\n⚠️  Warnings: {len(total_warnings)} examples had evidence that couldn't be fully matched")
        print(f"   Sample warnings:")
        for idx, warning in total_warnings[:5]:
            print(f"      Example {idx}: {warning}")
    
    # Add verbatim emphasis examples
    print(f"\n➕ Adding verbatim emphasis examples...")
    verbatim_examples = add_verbatim_emphasis_examples()
    fixed_data = verbatim_examples + fixed_data  # Add at beginning for priority
    print(f"   ✅ Added {len(verbatim_examples)} verbatim emphasis examples")
    
    # Save fixed dataset
    output_file = 'rag_cot_training_dataset_fixed.json'
    with open(output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"\n✅ Fixed dataset saved to: {output_file}")
    print(f"   Total examples: {len(fixed_data)}")
    print(f"   New examples added: {len(verbatim_examples)}")
    
    # Verify fixes
    print(f"\n🔍 Verifying fixes...")
    verbatim_count = 0
    total_evidence = 0
    
    for ex in fixed_data[:20]:  # Check first 20
        user_content = ex['messages'][1]['content']
        assistant_content = ex['messages'][2]['content']
        
        context = user_content.split('Question:')[0].replace('Knowledge context:', '').strip()
        evidences = re.findall(r'- Evidence:\s*"([^"]+)"', assistant_content)
        
        total_evidence += len(evidences)
        for evidence in evidences:
            if evidence.strip().lower() in context.lower():
                verbatim_count += 1
    
    if total_evidence > 0:
        rate = (verbatim_count / total_evidence) * 100
        print(f"   Verbatim rate in first 20 examples: {rate:.1f}%")
        if rate > 90:
            print(f"   ✅ Good verbatim rate!")
        else:
            print(f"   ⚠️  Still needs improvement")
    
    print(f"\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print(f"1. Review {output_file}")
    print(f"2. Manually verify evidence in examples with warnings")
    print(f"3. Replace original dataset: mv {output_file} rag_cot_training_dataset.json")
    print(f"4. Retrain the model with fixed dataset")

if __name__ == "__main__":
    main()
