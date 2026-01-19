#!/usr/bin/env python3
"""
Fix Final Regressions
=====================
Addressing:
1. Headquarters Query: 0% (regression) - Reasoning format inconsistency + incorrect [KEEP] + empty final answer
2. Role Query - CTO: 75% - Includes "The CTO" (output format)
3. Team Size: 50% - Paraphrasing "developers" → "employees" (verbatim preservation)
4. Revenue: 83.33% - Includes "2023" (output format)

Need stronger examples that:
- Enforce consistent reasoning format (bullet format, not inline)
- Show correct verbatim matching for headquarters (only evidence WITH "headquarters")
- Show final answer format: ONLY requested information
- Preserve verbatim from evidence (no paraphrasing)
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
    """Get the current system prompt."""
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
- FINAL ANSWER must include ALL items marked [KEEP] - do NOT omit any [KEEP] items.
- If query asks for a list, include ALL matching items found in the context (do not omit any).
- Preserve verbatim information from evidence - do NOT paraphrase (e.g., if evidence says "50 developers", do NOT change to "50 employees").
- For queries asking "Who is the [ROLE]?", include ONLY the person's name, not the role title or company name.
- For queries asking for amounts/numbers, include ONLY the amount/number, not dates, years, or other context."""

def create_fix_examples():
    """Create examples that fix the final regressions."""
    
    examples = [
        # 1. Headquarters Query - Strict verbatim matching (fix regression)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: TechCorp was founded in 2015 in San Francisco. The company moved to New York in 2020. The headquarters is now located at 123 Tech Street, New York, NY. The company also has offices in London and Tokyo.\n---\nQuestion: Where is TechCorp's headquarters located?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: San Francisco\n  - Evidence: \"TechCorp was founded in 2015 in San Francisco\"\n  - Action: [DISCARD] (Reason: Query term \"headquarters\" does NOT appear verbatim in evidence. Evidence says \"founded in San Francisco\", not \"headquarters\".)\n- Item: New York\n  - Evidence: \"The company moved to New York in 2020\"\n  - Action: [DISCARD] (Reason: Query term \"headquarters\" does NOT appear verbatim in evidence. Evidence says \"moved to New York\", not \"headquarters\".)\n- Item: 123 Tech Street, New York, NY\n  - Evidence: \"The headquarters is now located at 123 Tech Street, New York, NY\"\n  - Action: [KEEP] (Reason: Query term \"headquarters\" appears verbatim in evidence.)\n- Item: London\n  - Evidence: \"The company also has offices in London\"\n  - Action: [DISCARD] (Reason: Query term \"headquarters\" does NOT appear verbatim in evidence. Evidence says \"offices in London\", not \"headquarters\".)\n- Item: Tokyo\n  - Evidence: \"The company also has offices in Tokyo\"\n  - Action: [DISCARD] (Reason: Query term \"headquarters\" does NOT appear verbatim in evidence. Evidence says \"offices in Tokyo\", not \"headquarters\".)\n- End of scan.\n\nFINAL ANSWER:\n123 Tech Street, New York, NY."
                }
            ]
        },
        # 2. Role Query - Name ONLY (fix "The CTO" output format)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Sarah Johnson is the Chief Technology Officer at DataFlow. She leads the engineering team and oversees all technical decisions. Mark Williams is Head of Engineering. Robert Kim is Chief Financial Officer.\n---\nQuestion: Who is the CTO of DataFlow?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is the Chief Technology Officer at DataFlow\"\n  - Action: [KEEP] (Reason: Query term \"CTO\" appears verbatim as \"Chief Technology Officer\" in evidence.)\n- Item: Mark Williams\n  - Evidence: \"Mark Williams is Head of Engineering\"\n  - Action: [DISCARD] (Reason: Query term \"CTO\" does NOT appear verbatim in evidence.)\n- Item: Robert Kim\n  - Evidence: \"Robert Kim is Chief Financial Officer\"\n  - Action: [DISCARD] (Reason: Query term \"CTO\" does NOT appear verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nSarah Johnson."
                }
            ]
        },
        # 3. Team Size - Preserve verbatim "developers" (no paraphrasing)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The engineering team consists of 50 developers. The marketing team has 20 members. The sales team includes 30 people. Total employees: 100.\n---\nQuestion: How many employees are in the engineering team?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 50 developers\n  - Evidence: \"The engineering team consists of 50 developers\"\n  - Action: [KEEP] (Reason: Evidence explicitly states engineering team size.)\n- Item: 20 members\n  - Evidence: \"The marketing team has 20 members\"\n  - Action: [DISCARD] (Reason: Marketing team, not engineering team.)\n- Item: 30 people\n  - Evidence: \"The sales team includes 30 people\"\n  - Action: [DISCARD] (Reason: Sales team, not engineering team.)\n- Item: 100\n  - Evidence: \"Total employees: 100\"\n  - Action: [DISCARD] (Reason: Total employees, not engineering team.)\n- End of scan.\n\nFINAL ANSWER:\nThe engineering team consists of 50 developers."
                }
            ]
        },
        # 4. Revenue - Amount ONLY (no year "2023")
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The annual revenue for 2023 was $50 million. The company raised $10 million in Series A funding in 2022. Operating expenses totaled $35 million in 2023.\n---\nQuestion: What is the annual revenue?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: $50 million\n  - Evidence: \"The annual revenue for 2023 was $50 million\"\n  - Action: [KEEP] (Reason: Evidence explicitly states annual revenue amount.)\n- Item: $10 million\n  - Evidence: \"The company raised $10 million in Series A funding in 2022\"\n  - Action: [DISCARD] (Reason: Funding, not revenue.)\n- Item: $35 million\n  - Evidence: \"Operating expenses totaled $35 million in 2023\"\n  - Action: [DISCARD] (Reason: Expenses, not revenue.)\n- End of scan.\n\nFINAL ANSWER:\n$50 million."
                }
            ]
        }
    ]
    
    return examples

def main():
    print("=" * 80)
    print("FIXING FINAL REGRESSIONS")
    print("=" * 80)
    
    print("\n🎉 MAJOR SUCCESS:")
    print("   ✅ LedgerAI: 100% (found ALL 4 co-founders including Bob Carella!)")
    print()
    print("📊 CURRENT ISSUES:")
    print("   1. Headquarters Query: 0% (regression)")
    print("      - Reasoning format inconsistency")
    print("      - Incorrect [KEEP] for New York")
    print("      - Empty final answer")
    print("   2. Role Query - CTO: 75% - Includes 'The CTO'")
    print("   3. Team Size: 50% - Paraphrasing 'developers' → 'employees'")
    print("   4. Revenue: 83.33% - Includes '2023'")
    print()
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"📊 Current dataset size: {original_count} examples")
    
    # Create fix examples
    fix_examples = create_fix_examples()
    
    print(f"\n📝 Adding {len(fix_examples)} fix examples:")
    print(f"   1. Headquarters: Strict verbatim matching + consistent format")
    print(f"      → Only evidence WITH 'headquarters' → [KEEP]")
    print(f"      → Consistent bullet format (not inline)")
    print(f"   2. Role Query: Name ONLY 'Sarah Johnson' (no 'The CTO')")
    print(f"   3. Team Size: Preserve verbatim '50 developers' (no paraphrasing)")
    print(f"   4. Revenue: Amount ONLY '$50 million' (no '2023')")
    print()
    print(f"   ✅ These teach PRINCIPLES:")
    print(f"      - Consistent reasoning format (bullet format)")
    print(f"      - Strict verbatim matching (query term MUST be in evidence)")
    print(f"      - Final answer: ONLY requested information")
    print(f"      - Preserve verbatim from evidence (no paraphrasing)")
    
    # Add to the very beginning (highest priority)
    data = fix_examples + data
    
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
    print(f"   - Original: {original_count} examples")
    print(f"   - Added: {len(fix_examples)} fix examples")
    print(f"   - Updated: {updated_count} system prompts")
    print(f"   - New total: {len(data)} examples")
    
    print(f"\n🎯 Expected Improvements After Retraining:")
    print(f"   - Headquarters Query: 0% → 100% (strict verbatim + consistent format)")
    print(f"   - Role Query - CTO: 75% → 100% (name only)")
    print(f"   - Team Size: 50% → 100% (preserve verbatim)")
    print(f"   - Revenue: 83.33% → 100% (amount only)")
    print(f"   - Overall: 85.26% → 98%+ (target)")

if __name__ == "__main__":
    main()
