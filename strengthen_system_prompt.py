#!/usr/bin/env python3
"""
Strengthen System Prompt for Better Generalization
==================================================
Updates the system prompt in training script to be more rule-based,
preventing hallucination and improving generalization to new queries.
"""

import re

def get_strengthened_system_prompt():
    """Get strengthened system prompt with explicit rules."""
    
    return """You are a precise data extraction bot that follows strict rules.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES - MUST FOLLOW:

EVIDENCE RULES:
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.
- If evidence is not found verbatim in context, mark item as [DISCARD].
- You MUST evaluate ALL relevant items in the context before ending the scan.
- Read entire descriptions/chunks completely - titles may appear later in the text.

KEEP/DISCARD RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- FINAL ANSWER must ONLY contain items that appear in REASONING section.
- DO NOT add items to FINAL ANSWER that were not evaluated in REASONING.

EMPTY RESULT RULES (CRITICAL FOR GENERALIZATION):
- If ALL items are marked [DISCARD], FINAL ANSWER must be "No [query item] found" or empty.
- DO NOT create a FINAL ANSWER using [DISCARD] items when no [KEEP] items exist.
- If no items match the query, explicitly state "No [query item] found" in FINAL ANSWER.
- Example: If query is "co-founders" and all are [DISCARD], say "No co-founders found".

CO-FOUNDER SPECIFIC RULES (APPLY TO ALL QUERIES):
- "Co-Founder" must be EXPLICITLY stated in the evidence - the word "Co-Founder" must appear.
- CEO, CTO, CFO, COO, President, Vice President, Head of X, Director, Manager are NOT co-founders unless the word "Co-Founder" appears in their title.
- "Founder" (singular) is NOT the same as "Co-Founder" - mark Founder as [DISCARD] for co-founder queries.
- If someone is "CEO and Co-Founder", mark as [KEEP] because "Co-Founder" is explicitly stated.
- If someone is "CEO" only (no "Co-Founder"), mark as [DISCARD] for co-founder queries.
- "Established by" or "founded by" does NOT mean co-founders unless explicitly stated as "co-founders".

ROLE QUERY RULES:
- For role queries (e.g., "Who is the CTO?"), FINAL ANSWER should include ONLY the person's name, NOT the role title.
- Example: FINAL ANSWER should be "Sarah Johnson" not "The CTO is Sarah Johnson".

NUMBER QUERY RULES:
- For number queries, preserve full context from evidence (e.g., "50 developers" not just "50" or "50 employees").
- Use the exact wording from evidence when possible.

DATE/AMOUNT QUERY RULES:
- For date/amount queries, include ONLY the requested information (e.g., "$50 million" not "$50 million in 2023").
- Remove extra context like years unless specifically requested.

LOCATION QUERY RULES:
- For location queries, include ONLY the requested location type (e.g., "headquarters" not all offices).
- If query asks for "headquarters", do NOT include other office locations."""

def update_training_script():
    """Update the training script with strengthened prompt."""
    
    script_path = 'train_rag_cot_colab.py'
    
    print("=" * 80)
    print("STRENGTHENING SYSTEM PROMPT IN TRAINING SCRIPT")
    print("=" * 80)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find the FALLBACK_SYSTEM_PROMPT section (multi-line)
    old_pattern = r'FALLBACK_SYSTEM_PROMPT = """(.*?)"""'
    
    new_prompt = get_strengthened_system_prompt()
    
    new_content = f'FALLBACK_SYSTEM_PROMPT = """{new_prompt}"""'
    
    if re.search(old_pattern, content, re.DOTALL):
        content = re.sub(old_pattern, new_content, content, flags=re.DOTALL)
        print(f"✅ Updated FALLBACK_SYSTEM_PROMPT in {script_path}")
    else:
        print(f"⚠️  Could not find FALLBACK_SYSTEM_PROMPT pattern in {script_path}")
        # Try to find it manually
        if 'FALLBACK_SYSTEM_PROMPT' in content:
            # Find the start and end
            start_idx = content.find('FALLBACK_SYSTEM_PROMPT = """')
            if start_idx != -1:
                # Find the closing """
                end_idx = content.find('"""', start_idx + len('FALLBACK_SYSTEM_PROMPT = """'))
                if end_idx != -1:
                    end_idx += 3  # Include the closing """
                    # Replace
                    content = content[:start_idx] + new_content + content[end_idx:]
                    print(f"✅ Updated FALLBACK_SYSTEM_PROMPT in {script_path} (manual)")
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

def update_dataset_system_prompts():
    """Update system prompts in dataset to match strengthened version."""
    
    dataset_path = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Updating system prompts in dataset: {dataset_path}")
    
    import json
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    new_prompt = get_strengthened_system_prompt()
    updated_count = 0
    
    for ex in data:
        if len(ex['messages']) > 0 and ex['messages'][0]['role'] == 'system':
            old_prompt = ex['messages'][0]['content']
            if old_prompt != new_prompt:
                ex['messages'][0]['content'] = new_prompt
                updated_count += 1
    
    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {updated_count} system prompts in dataset")
    print(f"✅ Saved updated {dataset_path}")
    return True

def main():
    print("=" * 80)
    print("STRENGTHENING SYSTEM PROMPT FOR BETTER GENERALIZATION")
    print("=" * 80)
    
    print("\n🎯 Goal:")
    print("   - Make prompt more rule-based, less pattern-based")
    print("   - Add explicit rules for empty results")
    print("   - Strengthen co-founder discrimination rules")
    print("   - Improve generalization to new queries")
    
    # Update training script
    if update_training_script():
        print("\n✅ Training script updated")
    else:
        print("\n❌ Failed to update training script")
        return
    
    # Update dataset
    if update_dataset_system_prompts():
        print("\n✅ Dataset updated")
    else:
        print("\n❌ Failed to update dataset")
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ System prompt strengthened with:")
    print("   ✅ Explicit empty result rules (prevent hallucination)")
    print("   ✅ Stronger co-founder discrimination rules")
    print("   ✅ Role/Number/Date/Location query rules")
    print("   ✅ Evidence verbatim requirements")
    print("\n💡 Benefits:")
    print("   - Model learns to APPLY RULES, not just memorize patterns")
    print("   - Better generalization to new queries and RAG chunks")
    print("   - Prevents hallucination when no matches found")
    print("   - Clearer distinction between roles and co-founder status")

if __name__ == "__main__":
    main()
