#!/usr/bin/env python3
"""
Strengthen Verbatim Matching Principle
=======================================
The model should learn to apply the verbatim rule to ANY query, not memorize specific cases.
If query term appears verbatim in evidence → [KEEP], regardless of other roles/info mentioned.

This addresses the Bob Carella issue: Evidence says "Co-Founder" → should be [KEEP], period.
The verbatim rule already covers this - we just need to reinforce it with diverse examples.
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
    """Get the current system prompt - verbatim rule should be strong enough."""
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

KEEP/DISCARD:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.

MATCHING (PREVENTS HALLUCINATION):
- Entity/attribute must be EXPLICITLY stated in evidence - the query term must appear.
- If query term appears verbatim in evidence → [KEEP] (regardless of other roles/info mentioned).
- If query term does NOT appear verbatim in evidence → [DISCARD].
- Similar entities are NOT matches unless explicitly stated (e.g., "Founder" ≠ "Co-Founder").
- DO NOT assume relationships - only use explicitly stated information.

EMPTY RESULTS:
- If ALL items are marked [DISCARD], FINAL ANSWER must indicate no matches found.

OUTPUT FORMAT:
- FINAL ANSWER should include ONLY the requested information, not extra words or role titles."""

def create_verbatim_principle_examples():
    """
    Create diverse examples that reinforce the verbatim principle.
    Use DIFFERENT scenarios than test cases to ensure generalization.
    Key principle: If query term appears in evidence → [KEEP], period.
    """
    
    examples = [
        # Principle: If "Co-Founder" appears in evidence → [KEEP] (even with other roles)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Sarah Kim is Co-Founder and CEO of InnovateCorp. Michael Park is Co-Founder and CTO of InnovateCorp. Lisa Wang is Co-Founder and COO of InnovateCorp. Robert Chen is Vice President of Product at InnovateCorp. David Liu is Director of Engineering at InnovateCorp.\n---\nQuestion: Who are the co-founders of InnovateCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Sarah Kim\n  - Evidence: \"Sarah Kim is Co-Founder and CEO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence. Having additional role \"CEO\" does NOT change this - if query term appears, mark [KEEP].)\n- Item: Michael Park\n  - Evidence: \"Michael Park is Co-Founder and CTO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence. Having additional role \"CTO\" does NOT change this.)\n- Item: Lisa Wang\n  - Evidence: \"Lisa Wang is Co-Founder and COO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Query term \"co-founder\" appears verbatim in evidence. Having additional role \"COO\" does NOT change this.)\n- Item: Robert Chen\n  - Evidence: \"Robert Chen is Vice President of Product at InnovateCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Vice President\", not \"co-founder\".)\n- Item: David Liu\n  - Evidence: \"David Liu is Director of Engineering at InnovateCorp\"\n  - Action: [DISCARD] (Reason: Query term \"co-founder\" does NOT appear verbatim in evidence. Evidence says \"Director of Engineering\", not \"co-founder\".)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of InnovateCorp are Sarah Kim, Michael Park, and Lisa Wang."
                }
            ]
        },
        # Principle: If "headquarters" appears in evidence → [KEEP] (even with other locations)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The corporate headquarters is located at 500 Innovation Drive, Austin, Texas. The company has regional offices in Seattle and Boston. The Seattle office opened in 2018. The Boston office opened in 2020.\n---\nQuestion: Where is the headquarters located?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 500 Innovation Drive, Austin, Texas\n  - Evidence: \"The corporate headquarters is located at 500 Innovation Drive, Austin, Texas\"\n  - Action: [KEEP] (Reason: Query term \"headquarters\" appears verbatim in evidence. If query term appears, mark [KEEP].)\n- Item: Seattle\n  - Evidence: \"The company has regional offices in Seattle\"\n  - Action: [DISCARD] (Reason: Query term \"headquarters\" does NOT appear verbatim in evidence. Evidence says \"offices\", not \"headquarters\".)\n- Item: Boston\n  - Evidence: \"The company has regional offices in Boston\"\n  - Action: [DISCARD] (Reason: Query term \"headquarters\" does NOT appear verbatim in evidence. Evidence says \"offices\", not \"headquarters\".)\n- End of scan.\n\nFINAL ANSWER:\nThe headquarters is located at 500 Innovation Drive, Austin, Texas."
                }
            ]
        },
        # Principle: If "revenue" appears in evidence → [KEEP] (even with year mentioned)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The annual revenue for 2023 was $75 million. The company raised $20 million in Series A funding in 2022. Operating expenses totaled $45 million in 2023.\n---\nQuestion: What was the annual revenue?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: $75 million\n  - Evidence: \"The annual revenue for 2023 was $75 million\"\n  - Action: [KEEP] (Reason: Query term \"revenue\" appears verbatim in evidence. If query term appears, mark [KEEP]. The year \"2023\" is additional context but does NOT change the [KEEP] decision - the amount is what matches the query.)\n- Item: $20 million\n  - Evidence: \"The company raised $20 million in Series A funding in 2022\"\n  - Action: [DISCARD] (Reason: Query term \"revenue\" does NOT appear verbatim in evidence. Evidence says \"funding\", not \"revenue\".)\n- Item: $45 million\n  - Evidence: \"Operating expenses totaled $45 million in 2023\"\n  - Action: [DISCARD] (Reason: Query term \"revenue\" does NOT appear verbatim in evidence. Evidence says \"expenses\", not \"revenue\".)\n- End of scan.\n\nFINAL ANSWER:\nThe annual revenue was $75 million."
                }
            ]
        },
        # Principle: If "programming language" appears in evidence → [KEEP] (not documentation languages)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: DataSystems supports Python, JavaScript, and Go programming languages. The API is available in REST and GraphQL formats. Documentation is available in English, Spanish, and French.\n---\nQuestion: What programming languages does DataSystems support?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Python\n  - Evidence: \"DataSystems supports Python, JavaScript, and Go programming languages\"\n  - Action: [KEEP] (Reason: Query term \"programming language\" appears verbatim in evidence. If query term appears, mark [KEEP].)\n- Item: JavaScript\n  - Evidence: \"DataSystems supports Python, JavaScript, and Go programming languages\"\n  - Action: [KEEP] (Reason: Query term \"programming language\" appears verbatim in evidence.)\n- Item: Go\n  - Evidence: \"DataSystems supports Python, JavaScript, and Go programming languages\"\n  - Action: [KEEP] (Reason: Query term \"programming language\" appears verbatim in evidence.)\n- Item: REST\n  - Evidence: \"The API is available in REST and GraphQL formats\"\n  - Action: [DISCARD] (Reason: Query term \"programming language\" does NOT appear verbatim in evidence. Evidence says \"API formats\", not \"programming languages\".)\n- Item: GraphQL\n  - Evidence: \"The API is available in REST and GraphQL formats\"\n  - Action: [DISCARD] (Reason: Query term \"programming language\" does NOT appear verbatim in evidence. Evidence says \"API formats\", not \"programming languages\".)\n- Item: English\n  - Evidence: \"Documentation is available in English, Spanish, and French\"\n  - Action: [DISCARD] (Reason: Query term \"programming language\" does NOT appear verbatim in evidence. Evidence says \"Documentation\" languages, not \"programming\" languages.)\n- Item: Spanish\n  - Evidence: \"Documentation is available in English, Spanish, and French\"\n  - Action: [DISCARD] (Reason: Query term \"programming language\" does NOT appear verbatim in evidence. Evidence says \"Documentation\" languages, not \"programming\" languages.)\n- Item: French\n  - Evidence: \"Documentation is available in English, Spanish, and French\"\n  - Action: [DISCARD] (Reason: Query term \"programming language\" does NOT appear verbatim in evidence. Evidence says \"Documentation\" languages, not \"programming\" languages.)\n- End of scan.\n\nFINAL ANSWER:\nDataSystems supports Python, JavaScript, and Go."
                }
            ]
        }
    ]
    
    return examples

def main():
    print("=" * 80)
    print("STRENGTHENING VERBATIM MATCHING PRINCIPLE")
    print("=" * 80)
    
    print("\n💡 KEY INSIGHT:")
    print("   The verbatim rule should handle ALL cases, not specific examples.")
    print("   If query term appears verbatim in evidence → [KEEP], period.")
    print("   Having additional roles/info does NOT change the [KEEP] decision.")
    print()
    print("❌ WRONG APPROACH (Overfitting):")
    print("   - Add specific example for Bob Carella")
    print("   - Model might memorize this case, not learn the principle")
    print()
    print("✅ RIGHT APPROACH (Generalization):")
    print("   - Strengthen verbatim rule in prompt")
    print("   - Add diverse examples showing the principle")
    print("   - Model learns: 'If query term appears → [KEEP]' for ANY case")
    print()
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"📊 Current dataset size: {original_count} examples")
    
    # Create principle-based examples (NOT specific test case fixes)
    principle_examples = create_verbatim_principle_examples()
    
    print(f"\n📝 Adding {len(principle_examples)} principle-based examples:")
    print(f"   These examples use DIFFERENT scenarios to reinforce the verbatim rule:")
    print(f"   1. Co-Founder with other roles → [KEEP] (InnovateCorp, different company)")
    print(f"   2. Headquarters with other locations → [KEEP] (Austin, different location)")
    print(f"   3. Revenue with year → [KEEP] (amount matches query, year is context)")
    print(f"   4. Programming languages → [KEEP] (not documentation languages)")
    print()
    print(f"   ✅ These teach PRINCIPLES, not fix specific test cases")
    print(f"   ✅ Model will apply rule to ANY query, including Bob Carella case")
    
    # Remove the Bob Carella-specific fix if it exists (overfitting)
    # Find and remove examples that match test cases exactly
    data = [ex for ex in data if not (
        "Bob Carella is Co-Founder and Chief Financial Officer of LedgerAI" in str(ex) and
        "Paul Chou is the CEO and Co-Founder of LedgerAI" in str(ex) and
        "David Lara is Co-Founder and Chief Operating Officer of LedgerAI" in str(ex) and
        "Jorge Guinovart is Co-Founder and Chief Marketing Officer of LedgerAI" in str(ex)
    )]
    
    removed_count = original_count - len(data)
    if removed_count > 0:
        print(f"\n🧹 Removed {removed_count} overfitting example(s) (exact test case matches)")
    
    # Add principle-based examples to the beginning
    data = principle_examples + data
    
    # Save dataset
    save_dataset(data)
    
    print(f"\n✅ Dataset updated!")
    print(f"   - Original: {original_count} examples")
    print(f"   - After cleanup: {len(data) - len(principle_examples)} examples")
    print(f"   - Added: {len(principle_examples)} principle-based examples")
    print(f"   - New total: {len(data)} examples")
    
    print(f"\n📋 PRINCIPLE REINFORCED:")
    print(f"   ✅ If query term appears verbatim in evidence → [KEEP]")
    print(f"   ✅ Additional roles/info do NOT change [KEEP]/[DISCARD] decision")
    print(f"   ✅ Applies to ANY query: co-founders, headquarters, revenue, languages, etc.")
    print()
    print(f"💡 RESULT:")
    print(f"   Model will learn the VERBATIM PRINCIPLE, not memorize specific cases.")
    print(f"   Will handle Bob Carella case (and ANY similar case) correctly because:")
    print(f"   - Evidence says 'Co-Founder' → query term 'co-founder' appears → [KEEP]")
    print(f"   - Having 'Chief Financial Officer' as additional role doesn't matter")

if __name__ == "__main__":
    main()
