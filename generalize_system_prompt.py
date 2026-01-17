#!/usr/bin/env python3
"""
Generalize System Prompt for ALL Query Types
============================================
Updates the system prompt to be generalizable to ANY query type,
not just co-founders. Works for people, roles, locations, dates,
numbers, products, and any other entity/attribute extraction.
"""

import re
import json

def get_generalized_system_prompt():
    """Get generalized system prompt that works for ALL query types."""
    
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
- Example: If query is "co-founders" and all are [DISCARD], say "No co-founders found".
- Example: If query is "products" and all are [DISCARD], say "No products found".
- Example: If query is "locations" and all are [DISCARD], say "No locations found".

MATCHING RULES (APPLY TO ALL QUERIES - PREVENTS HALLUCINATION):
- For entity queries (people, roles, companies, products, etc.), the entity name/type must be EXPLICITLY stated in the evidence.
- For attribute queries (dates, numbers, locations, etc.), the attribute must match the query type EXACTLY.
- DO NOT assume relationships - only use explicitly stated information.
- Example: For "co-founders" query, the word "co-founder" or "cofounder" must appear in evidence.
- Example: For "CTO" query, "Chief Technology Officer" or "CTO" must appear in evidence.
- Example: For "revenue" query, "revenue" must appear in evidence (not "funding" or "expenses").
- Example: For "headquarters" query, "headquarters" must appear (not just "offices").

EXCLUSION RULES (APPLY TO ALL QUERIES):
- Similar entities are NOT matches unless explicitly stated.
- Example: "Founder" is NOT "Co-Founder" - mark as [DISCARD] for co-founder queries.
- Example: "Head of Engineering" is NOT "CTO" - mark as [DISCARD] for CTO queries.
- Example: "Funding" is NOT "Revenue" - mark as [DISCARD] for revenue queries.
- Example: "Offices" are NOT "Headquarters" - mark as [DISCARD] for headquarters queries.

OUTPUT FORMAT RULES (APPLY TO ALL QUERIES):
- For entity name queries (people, companies, products), FINAL ANSWER should include ONLY the entity names, not descriptions or roles.
- For role queries (e.g., "Who is the CTO?"), FINAL ANSWER should include ONLY the person's name, NOT the role title.
- For number queries, preserve full context from evidence (e.g., "50 developers" not just "50" or "50 employees").
- For date/amount queries, include ONLY the requested information (e.g., "$50 million" not "$50 million in 2023").
- For location queries, include ONLY the requested location type (e.g., "headquarters" not all offices).
- For list queries (products, languages, locations), include ALL matching items from [KEEP], separated appropriately.
- FINAL ANSWER should be concise - only include the requested information, not extra words."""

def update_training_script():
    """Update the training script with generalized prompt."""
    
    script_path = 'train_rag_cot_colab.py'
    
    print("=" * 80)
    print("GENERALIZING SYSTEM PROMPT IN TRAINING SCRIPT")
    print("=" * 80)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find the FALLBACK_SYSTEM_PROMPT section (multi-line)
    new_prompt = get_generalized_system_prompt()
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
                print(f"⚠️  Could not find closing \"\"\"")
                return False
        else:
            print(f"⚠️  Could not find FALLBACK_SYSTEM_PROMPT")
            return False
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Saved updated {script_path}")
    return True

def update_dataset_system_prompts():
    """Update system prompts in dataset to match generalized version."""
    
    dataset_path = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Updating system prompts in dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    new_prompt = get_generalized_system_prompt()
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
    print("GENERALIZING SYSTEM PROMPT FOR ALL QUERY TYPES")
    print("=" * 80)
    
    print("\n🎯 Goal:")
    print("   - Make prompt work for ANY query type (not just co-founders)")
    print("   - Apply rules generally: people, roles, locations, dates, numbers, products, etc.")
    print("   - Maintain strict evidence/empty result rules")
    print("   - Ensure generalization to new query types")
    
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
    print("✅ System prompt generalized with:")
    print("   ✅ Evidence rules (apply to all queries)")
    print("   ✅ Keep/Discard rules (apply to all queries)")
    print("   ✅ Empty result rules (apply to all queries)")
    print("   ✅ Matching rules (explicit entity/attribute matching)")
    print("   ✅ Exclusion rules (similar entities are not matches)")
    print("   ✅ Output format rules (entity/role/number/date/location queries)")
    print("\n💡 Benefits:")
    print("   - Works for ANY query type, not just co-founders")
    print("   - Generalizable to new query types without retraining")
    print("   - Prevents hallucination through explicit matching rules")
    print("   - Clear output format rules for different query types")
    print("\n📊 Query Types Supported:")
    print("   ✅ People/Entity queries (co-founders, team members, etc.)")
    print("   ✅ Role queries (CTO, CEO, etc.)")
    print("   ✅ Location queries (headquarters, offices, etc.)")
    print("   ✅ Date queries (establishment, launch dates, etc.)")
    print("   ✅ Number queries (team size, revenue, etc.)")
    print("   ✅ Product queries (products, services, etc.)")
    print("   ✅ List queries (languages, locations, etc.)")
    print("   ✅ ANY other entity/attribute extraction query")

if __name__ == "__main__":
    main()
