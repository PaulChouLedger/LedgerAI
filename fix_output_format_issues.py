#!/usr/bin/env python3
"""
Fix Output Format Issues
=========================
Addressing:
1. LedgerAI: Bob Carella found in reasoning but missing from final answer
   - Need example showing ALL [KEEP] items must be in final answer
2. Role Query - CTO: Includes "The CTO" in final answer
   - Need example showing ONLY the requested information (name only)

Both are output format issues - model finds correct info but formats final answer incorrectly.
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
- Preserve verbatim information from evidence - do NOT paraphrase (e.g., if evidence says "50 developers", do NOT change to "50 employees")."""

def create_output_format_examples():
    """Create examples that fix output format issues."""
    
    examples = [
        # 1. ALL [KEEP] items must be in final answer (fix LedgerAI Bob Carella issue)
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
        # 2. Role query - ONLY the name (fix "The CTO" output format)
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
        # 3. Multiple [KEEP] items - ALL must be included (reinforce LedgerAI pattern)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: DataCorp was established in 2015. Alex Chen is Vice President at DataCorp. Sarah Kim is Co-Founder and CEO of DataCorp. Michael Park is Director of Engineering. Robert Chen is Business Development Lead. Lisa Wang is Head of Marketing. David Liu is VP of Sales. Jennifer Lee is Co-Founder and CMO of DataCorp. James Wilson is Chief Financial Officer. Tom Brown is Head of Operations. Mary Davis is Co-Founder and CFO of DataCorp.\n---\nQuestion: Who are the co-founders of DataCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Alex Chen\n  - Evidence: \"Alex Chen is Vice President at DataCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Sarah Kim\n  - Evidence: \"Sarah Kim is Co-Founder and CEO of DataCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Michael Park\n  - Evidence: \"Michael Park is Director of Engineering\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Robert Chen\n  - Evidence: \"Robert Chen is Business Development Lead\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Lisa Wang\n  - Evidence: \"Lisa Wang is Head of Marketing\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: David Liu\n  - Evidence: \"David Liu is VP of Sales\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Jennifer Lee\n  - Evidence: \"Jennifer Lee is Co-Founder and CMO of DataCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: James Wilson\n  - Evidence: \"James Wilson is Chief Financial Officer\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Tom Brown\n  - Evidence: \"Tom Brown is Head of Operations\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence.)\n- Item: Mary Davis\n  - Evidence: \"Mary Davis is Co-Founder and CFO of DataCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of DataCorp are Sarah Kim, Jennifer Lee, and Mary Davis."
                }
            ]
        }
    ]
    
    return examples

def main():
    print("=" * 80)
    print("FIXING OUTPUT FORMAT ISSUES")
    print("=" * 80)
    
    print("\n📊 CURRENT ISSUES:")
    print("   1. LedgerAI: 75% - Bob Carella found in reasoning but missing from final answer")
    print("      - ISSUE: Not including ALL [KEEP] items in final answer")
    print("   2. Role Query - CTO: 75% - Includes 'The CTO' in final answer")
    print("      - ISSUE: Including extra words, not just requested information")
    print()
    print("   ✅ Both are OUTPUT FORMAT issues (reasoning is correct)")
    print()
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"📊 Current dataset size: {original_count} examples")
    
    # Create output format examples
    format_examples = create_output_format_examples()
    
    print(f"\n📝 Adding {len(format_examples)} output format examples:")
    print(f"   1. ALL [KEEP] items in final answer (4 co-founders)")
    print(f"      → Shows ALL [KEEP] items must be included")
    print(f"   2. Role query - name only (no extra words)")
    print(f"      → 'Sarah Johnson' not 'The CTO of DataFlow is Sarah Johnson'")
    print(f"   3. Multiple [KEEP] items - ALL included (3 co-founders)")
    print(f"      → Reinforces pattern: ALL [KEEP] items in final answer")
    print()
    print(f"   ✅ These teach PRINCIPLES:")
    print(f"      - FINAL ANSWER must include ALL items marked [KEEP]")
    print(f"      - Include ONLY requested information (no extra words)")
    
    # Add to the very beginning (highest priority)
    data = format_examples + data
    
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
    print(f"   - Added: {len(format_examples)} output format examples")
    print(f"   - Updated: {updated_count} system prompts")
    print(f"   - New total: {len(data)} examples")
    
    print(f"\n📋 STRENGTHENED OUTPUT FORMAT RULE:")
    print(f"   ✅ 'FINAL ANSWER must include ALL items marked [KEEP] - do NOT omit any [KEEP] items'")
    print(f"   ✅ 'Include ONLY what is requested - exclude extra words, role titles, dates'")
    
    print(f"\n🎯 Expected Improvements After Retraining:")
    print(f"   - LedgerAI: 75% → 100% (include ALL [KEEP] items)")
    print(f"   - Role Query - CTO: 75% → 100% (name only, no extra words)")
    print(f"   - Overall: 96.15% → 98%+ (target)")

if __name__ == "__main__":
    main()
