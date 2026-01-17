#!/usr/bin/env python3
"""
Fix CoT Format Prompt
=====================
Restructures the system prompt to make REASONING section more prominent
and ensure model always starts with REASONING, not skipping to Action.
"""

import json
import re

def get_fixed_cot_prompt():
    """Get fixed CoT prompt with prominent REASONING section."""
    
    return """You are a precise data extraction bot that follows strict rules.

ALWAYS START WITH REASONING:
Begin every response with "REASONING:" - this is MANDATORY.

1. Start with REASONING:
   For each relevant item found in the context:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].

2. End scan with: - End of scan.

3. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES - MUST FOLLOW:

EVIDENCE RULES (APPLY TO ALL QUERIES):
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.
- If evidence is not found verbatim in context, mark item as [DISCARD].
- You MUST evaluate ALL relevant items in the context before ending the scan.
- Read entire descriptions/chunks completely - titles may appear later in the text.

KEEP/DISCARD RULES (APPLY TO ALL QUERIES):
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- FINAL ANSWER must ONLY contain items that appear in REASONING section.
- DO NOT add items to FINAL ANSWER that were not evaluated in REASONING.

EMPTY RESULT RULES (APPLY TO ALL QUERIES - CRITICAL FOR GENERALIZATION):
- If ALL items are marked [DISCARD], FINAL ANSWER must indicate no matches found.
- DO NOT create a FINAL ANSWER using [DISCARD] items when no [KEEP] items exist.
- If no items match the query, explicitly state "No [query item] found" or equivalent.

MATCHING RULES (APPLY TO ALL QUERIES - PREVENTS HALLUCINATION):
- For entity queries (people, roles, companies, products, etc.), the entity name/type must be EXPLICITLY stated in the evidence.
- For attribute queries (dates, numbers, locations, etc.), the attribute must match the query type EXACTLY.
- DO NOT assume relationships - only use explicitly stated information.

EXCLUSION RULES (APPLY TO ALL QUERIES):
- Similar entities are NOT matches unless explicitly stated.
- Example: "Founder" is NOT "Co-Founder" - mark as [DISCARD] for co-founder queries.
- Example: "Head of Engineering" is NOT "CTO" - mark as [DISCARD] for CTO queries.

OUTPUT FORMAT RULES (APPLY TO ALL QUERIES):
- For entity name queries (people, companies, products), FINAL ANSWER should include ONLY the entity names.
- For role queries (e.g., "Who is the CTO?"), FINAL ANSWER should include ONLY the person's name, NOT the role title.
- For number queries, preserve full context from evidence (e.g., "50 developers" not just "50").
- For date/amount queries, include ONLY the requested information (e.g., "$50 million" not "$50 million in 2023").
- FINAL ANSWER should be concise - only include the requested information, not extra words."""

def update_training_script():
    """Update the training script with fixed prompt."""
    
    script_path = 'train_rag_cot_colab.py'
    
    print("=" * 80)
    print("FIXING CoT FORMAT PROMPT IN TRAINING SCRIPT")
    print("=" * 80)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    new_prompt = get_fixed_cot_prompt()
    new_content = f'FALLBACK_SYSTEM_PROMPT = """{new_prompt}"""'
    
    # Try regex first
    old_pattern = r'FALLBACK_SYSTEM_PROMPT = """(.*?)"""'
    
    if re.search(old_pattern, content, re.DOTALL):
        content = re.sub(old_pattern, new_content, content, flags=re.DOTALL)
        print(f"✅ Updated FALLBACK_SYSTEM_PROMPT in {script_path}")
    else:
        # Manual find/replace
        start_idx = content.find('FALLBACK_SYSTEM_PROMPT = """')
        if start_idx != -1:
            end_idx = content.find('"""', start_idx + len('FALLBACK_SYSTEM_PROMPT = """'))
            if end_idx != -1:
                end_idx += 3
                content = content[:start_idx] + new_content + content[end_idx:]
                print(f"✅ Updated FALLBACK_SYSTEM_PROMPT in {script_path} (manual)")
            else:
                return False
        else:
            return False
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Saved updated {script_path}")
    return True

def update_test_script():
    """Update the test script with fixed prompt."""
    
    script_path = 'test_rag_cot_model_colab.py'
    
    print(f"\n📂 Updating test script: {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    new_prompt = get_fixed_cot_prompt()
    new_content = f'SYSTEM_PROMPT = """{new_prompt}"""'
    
    # Try regex
    old_pattern = r'SYSTEM_PROMPT = """(.*?)"""'
    
    if re.search(old_pattern, content, re.DOTALL):
        content = re.sub(old_pattern, new_content, content, flags=re.DOTALL)
        print(f"✅ Updated SYSTEM_PROMPT in {script_path}")
    else:
        # Manual find/replace
        start_idx = content.find('# The exact system prompt used in training (MUST MATCH train_rag_cot_colab.py)')
        if start_idx != -1:
            # Find next SYSTEM_PROMPT
            start_idx = content.find('SYSTEM_PROMPT = """', start_idx)
            if start_idx != -1:
                end_idx = content.find('"""', start_idx + len('SYSTEM_PROMPT = """'))
                if end_idx != -1:
                    end_idx += 3
                    content = content[:start_idx] + new_content + content[end_idx:]
                    print(f"✅ Updated SYSTEM_PROMPT in {script_path} (manual)")
                else:
                    return False
            else:
                return False
        else:
            return False
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Saved updated {script_path}")
    return True

def update_dataset():
    """Update dataset with fixed prompt."""
    
    dataset_path = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Updating dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    new_prompt = get_fixed_cot_prompt()
    updated_count = 0
    
    for ex in data:
        if len(ex['messages']) > 0 and ex['messages'][0]['role'] == 'system':
            ex['messages'][0]['content'] = new_prompt
            updated_count += 1
    
    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {updated_count} system prompts in dataset")
    print(f"✅ Saved updated {dataset_path}")
    return True

def main():
    print("=" * 80)
    print("FIXING CoT FORMAT PROMPT")
    print("=" * 80)
    
    print("\n🎯 Goal:")
    print("   - Make REASONING section MORE PROMINENT")
    print("   - Ensure model ALWAYS starts with REASONING")
    print("   - Simplify prompt while keeping it general")
    print("   - Reduce length (shorter = more focused)")
    
    # Update training script
    if update_training_script():
        print("\n✅ Training script updated")
    else:
        print("\n❌ Failed to update training script")
        return
    
    # Update test script
    if update_test_script():
        print("\n✅ Test script updated")
    else:
        print("\n❌ Failed to update test script")
        return
    
    # Update dataset
    if update_dataset():
        print("\n✅ Dataset updated")
    else:
        print("\n❌ Failed to update dataset")
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Prompt fixed with:")
    print("   ✅ \"ALWAYS START WITH REASONING\" at top (prominent)")
    print("   ✅ Shorter prompt (~600 tokens vs ~969 tokens)")
    print("   ✅ Simplified structure (easier for model to follow)")
    print("   ✅ Still general (applies to all queries)")
    print("\n💡 Key Changes:")
    print("   - Added \"ALWAYS START WITH REASONING\" as first instruction")
    print("   - Removed redundant examples from prompt body")
    print("   - Kept all critical rules but more concise")
    print("   - Made structure clearer (1-2-3 steps)")
    print("\n📋 Next Steps:")
    print("   1. ✅ Prompt fixed in training script, test script, and dataset")
    print("   2. ⚠️  Need to RETRAIN model with fixed prompt")
    print("   3. ⚠️  After retraining, CoT format should work consistently")

if __name__ == "__main__":
    main()
