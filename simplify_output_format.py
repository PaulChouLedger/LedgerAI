#!/usr/bin/env python3
"""
Simplify Output Format Rules to Generic Principles
====================================================
Remove query-type-specific rules (role queries, revenue, list queries)
and replace with generic principles that apply to ALL queries.
"""

import json

def load_dataset():
    """Load the training dataset."""
    with open('rag_cot_training_dataset_100percent.json', 'r') as f:
        return json.load(f)

def save_dataset(data):
    """Save the training dataset."""
    with open('rag_cot_training_dataset_100percent.json', 'w') as f:
        json.dump(data, f, indent=2)

def get_system_prompt():
    """Get the simplified, generic system prompt."""
    return """You are a precise data extraction bot.

ALWAYS START WITH REASONING:
Begin every response with "REASONING:" - this is MANDATORY.

1. REASONING: For each relevant item found in the context:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].

2. End scan with: - End of scan.

3. FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES (APPLY TO ALL QUERIES):

EVIDENCE:
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.
- You MUST evaluate ALL relevant items in the context before ending the scan.
- Read through the ENTIRE context completely - do NOT stop scanning early.
- Scan systematically through all chunks, paragraphs, and sections.
- In complex contexts with many entities, scan ALL entities before ending.
- Entities may appear late in the context - continue scanning until the very end.
- Do NOT end scan until you have checked EVERY relevant item in the context.

KEEP/DISCARD:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.

MATCHING (PREVENTS HALLUCINATION - STRICT VERBATIM RULE):
- Query term MUST appear verbatim in evidence for [KEEP].
- If query term appears verbatim in evidence → [KEEP] (regardless of other roles/info mentioned).
- If query term does NOT appear verbatim in evidence → [DISCARD] (NO exceptions, NO inference, NO assumptions).
- Similar roles/titles are NOT matches unless query term appears verbatim (e.g., "Business Development Lead" ≠ "co-founder", "Ambassador" ≠ "co-founder", "CTO" ≠ "co-founder").
- DO NOT infer or assume relationships - only use explicitly stated information.
- DO NOT use context clues - only verbatim presence of query term matters.

EMPTY RESULTS:
- If ALL items are marked [DISCARD], FINAL ANSWER must indicate no matches found.

OUTPUT FORMAT:
- FINAL ANSWER must include ONLY the information explicitly requested in the query - nothing more, nothing less.
- Include ONLY what is requested - exclude extra words, role titles, dates, or any context not explicitly requested.
- If query asks for a list, include ALL matching items found in the context (do not omit any).
- Preserve verbatim information from evidence - do NOT paraphrase (e.g., if evidence says "50 developers", do NOT change to "50 employees")."""

def main():
    print("=" * 80)
    print("SIMPLIFYING OUTPUT FORMAT RULES TO GENERIC PRINCIPLES")
    print("=" * 80)
    
    print("\n📝 CHANGES:")
    print("   ❌ REMOVED query-type-specific rules:")
    print("      - 'For role queries, include ONLY the person's name'")
    print("      - 'If query asks for revenue amount, include ONLY the amount'")
    print("      - 'For list queries, include ALL matching items'")
    print()
    print("   ✅ REPLACED with generic principles:")
    print("      - 'FINAL ANSWER must include ONLY the information explicitly requested'")
    print("      - 'Include ONLY what is requested - exclude extra words, role titles, dates'")
    print("      - 'If query asks for a list, include ALL matching items'")
    print()
    print("   💡 BENEFIT: Model learns to reason through ANY query type")
    print("              instead of memorizing specific query patterns")
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"\n📊 Current dataset size: {original_count} examples")
    
    # Update all system prompts
    updated_prompt = get_system_prompt()
    updated_count = 0
    for ex in data:
        if len(ex['messages']) > 0 and ex['messages'][0]['role'] == 'system':
            ex['messages'][0]['content'] = updated_prompt
            updated_count += 1
    
    # Save dataset
    save_dataset(data)
    
    print(f"\n✅ Dataset updated!")
    print(f"   - Updated: {updated_count} system prompts with generic rules")
    print(f"   - Total examples: {original_count} (unchanged)")
    
    print(f"\n📋 NEW GENERIC RULES:")
    print(f"   ✅ 'FINAL ANSWER must include ONLY the information explicitly requested'")
    print(f"   ✅ 'Include ONLY what is requested - exclude extra words, role titles, dates'")
    print(f"   ✅ 'If query asks for a list, include ALL matching items'")
    print(f"   ✅ 'Preserve verbatim information from evidence - do NOT paraphrase'")
    
    print(f"\n🎯 EXPECTED BENEFIT:")
    print(f"   - Model learns to reason through ANY query type")
    print(f"   - Not dependent on specific query patterns")
    print(f"   - Better generalization to new query types")

if __name__ == "__main__":
    main()
