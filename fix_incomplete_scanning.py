#!/usr/bin/env python3
"""
Fix Incomplete Scanning in Complex Contexts
============================================
LedgerAI test shows incomplete scanning - missing Paul Chou and Bob Carella.
The verbatim rule is working correctly (Peter/Liam correctly excluded),
but the model is not scanning the entire context completely.

Need to reinforce complete scanning, especially for complex/messy contexts
with multiple chunks, long descriptions, and entities scattered throughout.
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
- FINAL ANSWER should include ONLY the requested information, not extra words or role titles."""

def create_complete_scanning_examples():
    """
    Create examples that enforce complete scanning in complex contexts.
    Use scenarios with multiple entities scattered across chunks/paragraphs.
    """
    
    examples = [
        # Complex context: Multiple co-founders scattered across chunks
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Alex Chen is Vice President of Product at InnovateCorp. Sarah Kim is Co-Founder and CEO of InnovateCorp, driving the company's vision and strategy. Michael Park is Co-Founder and CTO of InnovateCorp, leading all technical development. Robert Chen is Business Development Lead at InnovateCorp. Lisa Wang is Co-Founder and COO of InnovateCorp, overseeing daily operations. David Liu is Director of Engineering at InnovateCorp. Jennifer Lee is Co-Founder and CMO of InnovateCorp, managing all marketing initiatives.\n---\nQuestion: Who are the co-founders of InnovateCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Alex Chen\n  - Evidence: \"Alex Chen is Vice President of Product at InnovateCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Vice President\", not \"co-founder\".)\n- Item: Sarah Kim\n  - Evidence: \"Sarah Kim is Co-Founder and CEO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Michael Park\n  - Evidence: \"Michael Park is Co-Founder and CTO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Robert Chen\n  - Evidence: \"Robert Chen is Business Development Lead at InnovateCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Business Development Lead\", not \"co-founder\".)\n- Item: Lisa Wang\n  - Evidence: \"Lisa Wang is Co-Founder and COO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: David Liu\n  - Evidence: \"David Liu is Director of Engineering at InnovateCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Director of Engineering\", not \"co-founder\".)\n- Item: Jennifer Lee\n  - Evidence: \"Jennifer Lee is Co-Founder and CMO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of InnovateCorp are Sarah Kim, Michael Park, Lisa Wang, and Jennifer Lee."
                }
            ]
        },
        # Complex context: Multiple people, co-founders in middle of list
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: James Wilson serves as Chief Executive Officer of DataSystems. Sarah Kim is Co-Founder and CEO of DataSystems. Michael Park is Director of Engineering. Robert Chen is Co-Founder and CTO of DataSystems. Lisa Wang is Head of Marketing. David Liu is Co-Founder and COO of DataSystems. Jennifer Lee is VP of Sales.\n---\nQuestion: Who are the co-founders of DataSystems?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: James Wilson\n  - Evidence: \"James Wilson serves as Chief Executive Officer of DataSystems\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"CEO\", not \"co-founder\".)\n- Item: Sarah Kim\n  - Evidence: \"Sarah Kim is Co-Founder and CEO of DataSystems\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Michael Park\n  - Evidence: \"Michael Park is Director of Engineering\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Director of Engineering\", not \"co-founder\".)\n- Item: Robert Chen\n  - Evidence: \"Robert Chen is Co-Founder and CTO of DataSystems\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Lisa Wang\n  - Evidence: \"Lisa Wang is Head of Marketing\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Head of Marketing\", not \"co-founder\".)\n- Item: David Liu\n  - Evidence: \"David Liu is Co-Founder and COO of DataSystems\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Jennifer Lee\n  - Evidence: \"Jennifer Lee is VP of Sales\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"VP of Sales\", not \"co-founder\".)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of DataSystems are Sarah Kim, Robert Chen, and David Liu."
                }
            ]
        },
        # Complex context: Long descriptions, co-founders embedded in text
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: TechCorp was established in 2018. John Smith is the CEO and Co-Founder of TechCorp, leading the company's strategic direction. Sarah Johnson is the Chief Technology Officer at TechCorp, overseeing all technical operations. Michael Brown is Co-Founder and Chief Operating Officer of TechCorp, managing daily operations. Emily Davis is Head of Marketing at TechCorp. David Martinez is Co-Founder and Chief Product Officer of TechCorp, driving product innovation. Lisa Anderson is Chief Financial Officer.\n---\nQuestion: Who are the co-founders of TechCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: John Smith\n  - Evidence: \"John Smith is the CEO and Co-Founder of TechCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is the Chief Technology Officer at TechCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"CTO\", not \"co-founder\".)\n- Item: Michael Brown\n  - Evidence: \"Michael Brown is Co-Founder and Chief Operating Officer of TechCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Emily Davis\n  - Evidence: \"Emily Davis is Head of Marketing at TechCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Head of Marketing\", not \"co-founder\".)\n- Item: David Martinez\n  - Evidence: \"David Martinez is Co-Founder and Chief Product Officer of TechCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence.)\n- Item: Lisa Anderson\n  - Evidence: \"Lisa Anderson is Chief Financial Officer\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"CFO\", not \"co-founder\".)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechCorp are John Smith, Michael Brown, and David Martinez."
                }
            ]
        },
        # Role Query - Name only (fix "The CTO" output format issue)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Sarah Johnson is the Chief Technology Officer at DataFlow. She leads the engineering team and oversees all technical decisions.\n---\nQuestion: Who is the CTO of DataFlow?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is the Chief Technology Officer at DataFlow\"\n  - Action: [KEEP] (Reason: Query term \"CTO\" appears verbatim as \"Chief Technology Officer\" in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nSarah Johnson."
                }
            ]
        }
    ]
    
    return examples

def main():
    print("=" * 80)
    print("FIXING INCOMPLETE SCANNING IN COMPLEX CONTEXTS")
    print("=" * 80)
    
    print("\n💡 KEY INSIGHT:")
    print("   Verbatim rule is WORKING correctly (Peter/Liam correctly excluded)")
    print("   But model is INCOMPLETE SCANNING (missing Paul Chou, Bob Carella)")
    print()
    print("   ❌ PROBLEM: Model stops scanning early in complex contexts")
    print("   ✅ SOLUTION: Reinforce complete scanning through ENTIRE context")
    print()
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"📊 Current dataset size: {original_count} examples")
    
    # Create complete scanning examples
    scanning_examples = create_complete_scanning_examples()
    
    print(f"\n📝 Adding {len(scanning_examples)} complete scanning examples:")
    print(f"   1. Complex context: 7 people, co-founders scattered throughout")
    print(f"   2. Complex context: Co-founders in middle of list")
    print(f"   3. Complex context: Long descriptions, co-founders embedded")
    print(f"   4. Role Query: Name only (fix 'The CTO' output format)")
    print()
    print(f"   ✅ These teach PRINCIPLE: 'Scan ENTIRE context completely'")
    print(f"   ✅ Use DIFFERENT scenarios than test cases (generalization)")
    
    # Add to the very beginning (highest priority)
    data = scanning_examples + data
    
    # Update all system prompts with strengthened scanning rule
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
    print(f"   - Added: {len(scanning_examples)} complete scanning examples")
    print(f"   - Updated: {updated_count} system prompts with strengthened scanning rule")
    print(f"   - New total: {len(data)} examples")
    
    print(f"\n📋 STRENGTHENED SCANNING RULE:")
    print(f"   ✅ 'You MUST evaluate ALL relevant items before ending the scan'")
    print(f"   ✅ 'Read through the ENTIRE context completely'")
    print(f"   ✅ 'Scan systematically through all chunks, paragraphs, sections'")
    print(f"   ✅ 'In complex contexts with many entities, scan ALL entities'")
    
    print(f"\n🎯 Expected Improvements:")
    print(f"   - LedgerAI: 50% → 100% (find ALL 4 co-founders via complete scanning)")
    print(f"   - Role Query - CTO: 75% → 100% (name only, not 'The CTO')")
    print(f"   - Overall: 94.23% → 97%+ (target)")

if __name__ == "__main__":
    main()
