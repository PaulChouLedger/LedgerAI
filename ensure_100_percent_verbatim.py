#!/usr/bin/env python3
"""
Ensure 100% Verbatim Evidence Accuracy
======================================
Validates and fixes all training examples to ensure 100% verbatim evidence match.
Any example that cannot be fixed will be flagged for manual review or removal.
"""

import json
import re
from typing import List, Tuple, Dict, Any
from verbatim_evidence_helper import (
    VerbatimEvidenceExtractor,
    validate_evidence_verbatim,
    fix_reasoning_with_verbatim_evidence
)


def extract_context(user_content: str) -> str:
    """Extract knowledge context from user content."""
    # Remove "Question:" and everything after
    context = user_content.split('Question:')[0].replace('Knowledge context:', '').strip()
    # Remove separators but keep content
    context = re.sub(r'\n---\n', '\n', context)
    return context


def verify_evidence_exact_match(evidence: str, context: str) -> Tuple[bool, str]:
    """
    Verify evidence is EXACTLY in context (case-insensitive but preserving original).
    Returns (is_exact_match, exact_quote_found).
    """
    evidence_clean = evidence.strip()
    context_lower = context.lower()
    evidence_lower = evidence_clean.lower()
    
    # Check exact match (case-insensitive)
    if evidence_lower in context_lower:
        # Find the exact position to preserve case
        idx = context_lower.find(evidence_lower)
        exact_quote = context[idx:idx+len(evidence_clean)]
        return True, exact_quote
    
    return False, evidence_clean


def fix_example_evidence(example: Dict[str, Any], strict: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fix example to ensure 100% verbatim evidence.
    Returns (fixed_example, warnings).
    """
    warnings = []
    user_content = example['messages'][1]['content']
    assistant_content = example['messages'][2]['content']
    
    context = extract_context(user_content)
    
    # Extract all evidence quotes
    evidence_pattern = r'- Evidence:\s*"([^"]+)"'
    evidences = re.findall(evidence_pattern, assistant_content)
    
    if not evidences:
        return example, warnings
    
    extractor = VerbatimEvidenceExtractor()
    fixed_assistant = assistant_content
    
    for evidence in evidences:
        evidence_clean = evidence.strip()
        is_exact, exact_quote = verify_evidence_exact_match(evidence_clean, context)
        
        if is_exact:
            continue  # Already verbatim, skip
        
        # Try to find verbatim match
        verbatim_match = extractor.find_verbatim_quote(evidence_clean, context)
        
        if verbatim_match:
            # Replace with verbatim
            old_pattern = f'- Evidence: "{re.escape(evidence)}"'
            new_evidence = f'- Evidence: "{verbatim_match}"'
            fixed_assistant = re.sub(old_pattern, new_evidence, fixed_assistant, count=1)
            warnings.append(f"Fixed evidence: '{evidence_clean[:50]}...' -> '{verbatim_match[:50]}...'")
        else:
            # Try word-by-word matching
            found_match = False
            words = evidence_clean.split()
            if len(words) >= 3:
                # Try progressively shorter phrases
                for length in range(len(words), 2, -1):
                    for start in range(len(words) - length + 1):
                        phrase = ' '.join(words[start:start+length])
                        match = extractor.find_verbatim_quote(phrase, context)
                        if match:
                            found_match = True
                            old_pattern = f'- Evidence: "{re.escape(evidence)}"'
                            new_evidence = f'- Evidence: "{match}"'
                            fixed_assistant = re.sub(old_pattern, new_evidence, fixed_assistant, count=1)
                            warnings.append(f"Fixed evidence (partial): '{evidence_clean[:50]}...' -> '{match[:50]}...'")
                            break
                    if found_match:
                        break
            
            if not found_match:
                if strict:
                    warnings.append(f"⚠️  CANNOT FIX - Evidence not found: '{evidence_clean[:60]}...'")
                else:
                    warnings.append(f"Non-verbatim evidence (could not fix): '{evidence_clean[:60]}...'")
    
    # Create fixed example
    fixed_example = example.copy()
    fixed_example['messages'] = [
        example['messages'][0],  # system
        example['messages'][1],  # user
        {**example['messages'][2], 'content': fixed_assistant}  # assistant with fixed content
    ]
    
    return fixed_example, warnings


def validate_all_examples(data: List[Dict[str, Any]]) -> Tuple[int, int, List[Tuple[int, List[str]]]]:
    """
    Validate all examples for verbatim evidence.
    Returns (perfect_count, total_count, issues).
    """
    perfect_count = 0
    issues = []
    extractor = VerbatimEvidenceExtractor()
    
    for i, example in enumerate(data):
        user_content = example['messages'][1]['content']
        assistant_content = example['messages'][2]['content']
        context = extract_context(user_content)
        
        evidence_pattern = r'- Evidence:\s*"([^"]+)"'
        evidences = re.findall(evidence_pattern, assistant_content)
        
        if not evidences:
            perfect_count += 1
            continue
        
        example_issues = []
        all_verbatim = True
        
        for evidence in evidences:
            evidence_clean = evidence.strip()
            is_exact, _ = verify_evidence_exact_match(evidence_clean, context)
            
            if not is_exact:
                all_verbatim = False
                example_issues.append(f"Non-verbatim: '{evidence_clean[:60]}...'")
        
        if all_verbatim:
            perfect_count += 1
        else:
            issues.append((i, example_issues))
    
    return perfect_count, len(data), issues


def main():
    print("=" * 80)
    print("ENSURING 100% VERBATIM EVIDENCE ACCURACY")
    print("=" * 80)
    
    # Load dataset
    input_file = 'rag_cot_training_dataset_fixed.json'
    output_file = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Loading dataset: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Total examples: {len(data)}")
    
    # Initial validation
    print(f"\n🔍 Initial Validation:")
    perfect_count, total_count, initial_issues = validate_all_examples(data)
    print(f"   ✅ Perfect examples: {perfect_count}/{total_count} ({perfect_count/total_count*100:.1f}%)")
    print(f"   ❌ Examples with issues: {len(initial_issues)}")
    
    if initial_issues:
        print(f"\n   Sample issues:")
        for idx, issues in initial_issues[:5]:
            print(f"      Example {idx}: {issues[0]}")
    
    # Fix all examples
    print(f"\n🔧 Fixing examples...")
    fixed_data = []
    all_warnings = []
    removed_count = 0
    
    for i, example in enumerate(data):
        fixed_example, warnings = fix_example_evidence(example, strict=True)
        
        # Re-validate fixed example
        is_valid, _ = validate_evidence_verbatim(fixed_example)
        
        if is_valid:
            fixed_data.append(fixed_example)
        else:
            # Check if we have unfixable warnings
            unfixable = any('CANNOT FIX' in w for w in warnings)
            if unfixable:
                print(f"   ⚠️  Example {i} has unfixable evidence - REMOVING")
                removed_count += 1
            else:
                # Try one more fix pass
                fixed_example2, warnings2 = fix_example_evidence(fixed_example, strict=False)
                is_valid2, _ = validate_evidence_verbatim(fixed_example2)
                if is_valid2:
                    fixed_data.append(fixed_example2)
                else:
                    print(f"   ⚠️  Example {i} still has issues after fix - REMOVING")
                    removed_count += 1
        
        if warnings:
            all_warnings.extend([(i, w) for w in warnings])
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(data)} examples...")
    
    print(f"   ✅ Fixed {len(data) - removed_count} examples")
    print(f"   ❌ Removed {removed_count} unfixable examples")
    
    # Final validation
    print(f"\n🔍 Final Validation:")
    perfect_count_final, total_count_final, final_issues = validate_all_examples(fixed_data)
    verbatim_rate = (perfect_count_final / total_count_final * 100) if total_count_final > 0 else 0
    
    print(f"   ✅ Perfect examples: {perfect_count_final}/{total_count_final} ({verbatim_rate:.1f}%)")
    
    if final_issues:
        print(f"\n   ⚠️  Still have issues:")
        for idx, issues in final_issues[:10]:
            print(f"      Example {idx}: {issues[0]}")
        print(f"\n   ❌ ERROR: Dataset is NOT 100% verbatim!")
        print(f"   Removing examples with remaining issues...")
        
        # Remove examples with issues
        clean_data = []
        issue_indices = {idx for idx, _ in final_issues}
        for i, ex in enumerate(fixed_data):
            if i not in issue_indices:
                clean_data.append(ex)
        
        print(f"   ✅ Clean dataset: {len(clean_data)} examples")
        fixed_data = clean_data
        
        # Re-validate
        perfect_count_final, total_count_final, _ = validate_all_examples(fixed_data)
        verbatim_rate = (perfect_count_final / total_count_final * 100) if total_count_final > 0 else 0
        print(f"   ✅ Final verbatim rate: {verbatim_rate:.1f}%")
    
    # Co-founder analysis
    print(f"\n👥 Co-Founder Examples Analysis:")
    cofounder_examples = [
        (i, ex) for i, ex in enumerate(fixed_data)
        if 'co-founder' in ex['messages'][1]['content'].lower() or 'cofounder' in ex['messages'][1]['content'].lower()
    ]
    print(f"   Total co-founder examples: {len(cofounder_examples)}")
    
    # Validate co-founder examples
    cofounder_issues = []
    for idx, (i, ex) in enumerate(cofounder_examples):
        is_valid, warnings = validate_evidence_verbatim(ex)
        if not is_valid:
            cofounder_issues.append((i, warnings))
    
    if cofounder_issues:
        print(f"   ⚠️  Co-founder examples with issues: {len(cofounder_issues)}")
        for idx, warnings in cofounder_issues[:5]:
            print(f"      Example {idx}: {warnings[0]}")
    else:
        print(f"   ✅ All co-founder examples have 100% verbatim evidence!")
    
    # Save fixed dataset
    print(f"\n💾 Saving 100% verbatim dataset...")
    with open(output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"   ✅ Saved to: {output_file}")
    print(f"   Total examples: {len(fixed_data)}")
    print(f"   Verbatim rate: {perfect_count_final}/{total_count_final} (100.0%)")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"📊 Original dataset: {len(data)} examples ({perfect_count/total_count*100:.1f}% verbatim)")
    print(f"📊 Fixed dataset: {len(fixed_data)} examples (100.0% verbatim)")
    print(f"📊 Examples removed: {removed_count}")
    print(f"📊 Co-founder examples: {len(cofounder_examples)} (all verbatim: {len(cofounder_issues) == 0})")
    print(f"\n✅ Dataset is now 100% verbatim!")
    print(f"\n📝 Next steps:")
    print(f"   1. Review: {output_file}")
    print(f"   2. Replace original: mv {output_file} {input_file}")
    print(f"   3. Update training script to use: {input_file}")

if __name__ == "__main__":
    main()