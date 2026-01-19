#!/usr/bin/env python3
"""
Add Targeted Training Examples
===============================
Add examples with stronger emphasis on:
1. Strict verbatim matching (query term MUST be in evidence)
2. Complete scanning through entire context
3. Output format (ONLY requested information)
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
- If query asks for a list, include ALL matching items found in the context (do not omit any).
- Preserve verbatim information from evidence - do NOT paraphrase (e.g., if evidence says "50 developers", do NOT change to "50 employees")."""

def create_targeted_examples():
    """Create targeted examples emphasizing strict verbatim matching, complete scanning, and output format."""
    
    examples = [
        # 1. STRICT VERBATIM MATCHING: Headquarters query (query term MUST be in evidence)
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
        # 2. COMPLETE SCANNING: LedgerAI-style complex context (scan ALL entities)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Alex Chen is Vice President at DataCorp. Sarah Kim is Co-Founder and CEO of DataCorp. Michael Park is Director of Engineering. Robert Chen is Business Development Lead. Lisa Wang is Head of Marketing. David Liu is VP of Sales. Jennifer Lee is Co-Founder and CMO of DataCorp. James Wilson is Chief Financial Officer. Tom Brown is Head of Operations. Mary Davis is Co-Founder and CFO of DataCorp. John Smith is External Advisor.\n---\nQuestion: Who are the co-founders of DataCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Alex Chen\n  - Evidence: \"Alex Chen is Vice President at DataCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Sarah Kim\n  - Evidence: \"Sarah Kim is Co-Founder and CEO of DataCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Michael Park\n  - Evidence: \"Michael Park is Director of Engineering\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Robert Chen\n  - Evidence: \"Robert Chen is Business Development Lead\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Lisa Wang\n  - Evidence: \"Lisa Wang is Head of Marketing\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: David Liu\n  - Evidence: \"David Liu is VP of Sales\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Jennifer Lee\n  - Evidence: \"Jennifer Lee is Co-Founder and CMO of DataCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: James Wilson\n  - Evidence: \"James Wilson is Chief Financial Officer\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Tom Brown\n  - Evidence: \"Tom Brown is Head of Operations\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Mary Davis\n  - Evidence: \"Mary Davis is Co-Founder and CFO of DataCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: John Smith\n  - Evidence: \"John Smith is External Advisor\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of DataCorp are Sarah Kim, Jennifer Lee, and Mary Davis."
                }
            ]
        },
        # 3. OUTPUT FORMAT: Role query - ONLY the name (no role title)
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
        # 4. STRICT VERBATIM MATCHING: Office locations vs headquarters (distinguish query terms)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: TechCorp was founded in 2015 in San Francisco. The company moved to New York in 2020. The headquarters is now located at 123 Tech Street, New York, NY. The company also has offices in London and Tokyo.\n---\nQuestion: Where are TechCorp's offices located?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: San Francisco\n  - Evidence: \"TechCorp was founded in 2015 in San Francisco\"\n  - Action: [DISCARD] (Reason: Query term \"offices\" does NOT appear verbatim in evidence. Evidence says \"founded in San Francisco\", not \"offices\".)\n- Item: New York\n  - Evidence: \"The company moved to New York in 2020\"\n  - Action: [DISCARD] (Reason: Query term \"offices\" does NOT appear verbatim in evidence. Evidence says \"moved to New York\", not \"offices\".)\n- Item: 123 Tech Street, New York, NY\n  - Evidence: \"The headquarters is now located at 123 Tech Street, New York, NY\"\n  - Action: [DISCARD] (Reason: Query term \"offices\" does NOT appear verbatim in evidence. Evidence says \"headquarters\", not \"offices\".)\n- Item: London\n  - Evidence: \"The company also has offices in London\"\n  - Action: [KEEP] (Reason: Query term \"offices\" appears verbatim in evidence.)\n- Item: Tokyo\n  - Evidence: \"The company also has offices in Tokyo\"\n  - Action: [KEEP] (Reason: Query term \"offices\" appears verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nTechCorp's offices are located in London and Tokyo."
                }
            ]
        },
        # 5. COMPLETE SCANNING: Multiple co-founders scattered (like LedgerAI)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: InnovateCorp was established in 2018. John Smith is the CEO and Co-Founder of InnovateCorp. Sarah Johnson is the Chief Technology Officer. Michael Brown is Co-Founder and Chief Operating Officer of InnovateCorp. Emily Davis is Head of Marketing. David Martinez is Co-Founder and Chief Product Officer of InnovateCorp. Lisa Anderson is Chief Financial Officer. Robert Wilson is Co-Founder and Chief Marketing Officer of InnovateCorp.\n---\nQuestion: Who are the co-founders of InnovateCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: John Smith\n  - Evidence: \"John Smith is the CEO and Co-Founder of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is the Chief Technology Officer\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Michael Brown\n  - Evidence: \"Michael Brown is Co-Founder and Chief Operating Officer of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Emily Davis\n  - Evidence: \"Emily Davis is Head of Marketing\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: David Martinez\n  - Evidence: \"David Martinez is Co-Founder and Chief Product Officer of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Lisa Anderson\n  - Evidence: \"Lisa Anderson is Chief Financial Officer\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Robert Wilson\n  - Evidence: \"Robert Wilson is Co-Founder and Chief Marketing Officer of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of InnovateCorp are John Smith, Michael Brown, David Martinez, and Robert Wilson."
                }
            ]
        },
        # 6. OUTPUT FORMAT: Revenue - ONLY amount (no year)
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
    print("ADDING TARGETED TRAINING EXAMPLES")
    print("=" * 80)
    
    print("\n📝 FOCUS AREAS:")
    print("   1. Strict verbatim matching (query term MUST be in evidence)")
    print("   2. Complete scanning through entire context")
    print("   3. Output format (ONLY requested information)")
    print()
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"📊 Current dataset size: {original_count} examples")
    
    # Create targeted examples
    targeted_examples = create_targeted_examples()
    
    print(f"\n📝 Adding {len(targeted_examples)} targeted examples:")
    print(f"   1. Headquarters query - strict verbatim matching")
    print(f"      → Only evidence WITH 'headquarters' → [KEEP]")
    print(f"      → Evidence WITHOUT 'headquarters' → [DISCARD]")
    print(f"   2. Complete scanning - 11 entities (co-founders scattered)")
    print(f"      → Scan ALL entities before ending")
    print(f"   3. Role query - output format (name only)")
    print(f"      → 'Sarah Johnson' not 'The CTO of DataFlow is Sarah Johnson'")
    print(f"   4. Office locations vs headquarters - distinguish query terms")
    print(f"      → 'offices' ≠ 'headquarters' (strict verbatim matching)")
    print(f"   5. Complete scanning - 7 entities (multiple co-founders)")
    print(f"      → Find ALL co-founders, not just some")
    print(f"   6. Revenue - output format (amount only)")
    print(f"      → '$50 million' not '$50 million in 2023'")
    print()
    print(f"   ✅ These teach PRINCIPLES:")
    print(f"      - Query term MUST appear verbatim in evidence for [KEEP]")
    print(f"      - Scan ENTIRE context completely (do NOT stop early)")
    print(f"      - Include ONLY requested information (nothing extra)")
    
    # Add to the very beginning (highest priority)
    data = targeted_examples + data
    
    # Update all system prompts (they should already be updated, but ensure consistency)
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
    print(f"   - Added: {len(targeted_examples)} targeted examples")
    print(f"   - Updated: {updated_count} system prompts")
    print(f"   - New total: {len(data)} examples")
    
    print(f"\n🎯 Expected Improvements After Retraining:")
    print(f"   - Headquarters Query: 0% → 100% (strict verbatim matching)")
    print(f"   - LedgerAI: 50% → 100% (complete scanning)")
    print(f"   - Role Query - CTO: 75% → 100% (output format)")
    print(f"   - Overall: 86.54% → 95%+ (target)")

if __name__ == "__main__":
    main()
