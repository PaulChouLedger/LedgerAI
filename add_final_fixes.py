#!/usr/bin/env python3
"""
Add Final Fixes for Remaining Issues
===================================
Addresses:
1. Incomplete scanning (missing Jorge Guinovart)
2. Reasoning errors (CTO marked [KEEP] instead of [DISCARD])
3. FINAL ANSWER violations (including DISCARD items)
"""

import json

def create_final_fix_examples():
    """Create examples to fix remaining issues."""
    
    enhanced_system = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- FINAL ANSWER must ONLY contain items that appear in REASONING section.
- DO NOT add items to FINAL ANSWER that were not evaluated in REASONING.
- Read entire descriptions/chunks completely - titles may appear later in the text.
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.
- You MUST evaluate ALL relevant items in the context before ending the scan.
- "Founder" is NOT the same as "Co-Founder" - mark Founder as [DISCARD] for co-founder queries.
- CTO, Head of Engineering, and other roles are NOT co-founders unless explicitly stated.
- If ALL items are marked [DISCARD], FINAL ANSWER must state "No [query items] found" or similar."""

    examples = [
        # Example 1: Complete scanning with 4+ co-founders (like LedgerAI)
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Paul Chou is the Co-Founder and CEO of LedgerAI. He has 20 years of experience.\n---\nBob Carella is the Co-Founder and Chief Operating Officer of LedgerAI. He previously worked at Binance.US.\n---\nDavid Lara is the Co-Founder and Chief Marketing Officer of LedgerAI. He has extensive marketing experience.\n---\nJorge Guinovart is the Co-Founder and Chief Technology Officer of LedgerAI. He founded the company in 2020.\n---\nPeter Moeller is the Business Development Lead at LedgerAI. He is not a co-founder.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Paul Chou\n  - Evidence: \"Paul Chou is the Co-Founder and CEO of LedgerAI\"\n  - Action: [KEEP]\n- Item: Bob Carella\n  - Evidence: \"Bob Carella is the Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: David Lara\n  - Evidence: \"David Lara is the Co-Founder and Chief Marketing Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Jorge Guinovart\n  - Evidence: \"Jorge Guinovart is the Co-Founder and Chief Technology Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Peter Moeller\n  - Evidence: \"Peter Moeller is the Business Development Lead at LedgerAI\"\n  - Action: [DISCARD] (Reason: Business Development Lead, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of LedgerAI are Paul Chou, Bob Carella, David Lara, and Jorge Guinovart."
                }
            ]
        },
        
        # Example 2: CTO marked [DISCARD] for co-founder query (single co-founder)
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Jennifer Park is the President and Co-Founder of CloudScale Technologies. She has been with the company since 2018. Mark Thompson is the Chief Technology Officer at CloudScale Technologies. He joined in 2019. He is not a co-founder.\n---\nQuestion: Who are the co-founders of CloudScale Technologies?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Jennifer Park\n  - Evidence: \"Jennifer Park is the President and Co-Founder of CloudScale Technologies\"\n  - Action: [KEEP]\n- Item: Mark Thompson\n  - Evidence: \"Mark Thompson is the Chief Technology Officer at CloudScale Technologies\"\n  - Action: [DISCARD] (Reason: CTO, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of CloudScale Technologies are Jennifer Park."
                }
            ]
        },
        
        # Example 3: No co-founders - FINAL ANSWER must NOT include Founder
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: James Wilson is the Founder and CEO of Acme Corporation. He started the company alone in 2010. Maria Garcia is the Chief Technology Officer at Acme Corporation. She joined in 2012. Thomas Lee is the Head of Sales at Acme Corporation.\n---\nQuestion: Who are the co-founders of Acme Corporation?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: James Wilson\n  - Evidence: \"James Wilson is the Founder and CEO of Acme Corporation\"\n  - Action: [DISCARD] (Reason: Founder, not Co-Founder. No co-founders mentioned in context).\n- Item: Maria Garcia\n  - Evidence: \"Maria Garcia is the Chief Technology Officer at Acme Corporation\"\n  - Action: [DISCARD] (Reason: CTO, not co-founder).\n- Item: Thomas Lee\n  - Evidence: \"Thomas Lee is the Head of Sales at Acme Corporation\"\n  - Action: [DISCARD] (Reason: Head of Sales, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nNo co-founders were found for Acme Corporation. The context mentions a Founder (James Wilson) but no co-founders."
                }
            ]
        },
        
        # Example 4: Another no co-founders example
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Robert Chen is the President of GlobalTech Solutions. He has been with the company since 2015. Lisa Anderson is the Chief Operating Officer at GlobalTech Solutions. She joined in 2016. David Kim is the Head of Engineering at GlobalTech Solutions.\n---\nQuestion: Who are the co-founders of GlobalTech Solutions?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Robert Chen\n  - Evidence: \"Robert Chen is the President of GlobalTech Solutions\"\n  - Action: [DISCARD] (Reason: President, not co-founder. No co-founders mentioned in context).\n- Item: Lisa Anderson\n  - Evidence: \"Lisa Anderson is the Chief Operating Officer at GlobalTech Solutions\"\n  - Action: [DISCARD] (Reason: COO, not co-founder).\n- Item: David Kim\n  - Evidence: \"David Kim is the Head of Engineering at GlobalTech Solutions\"\n  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nNo co-founders were found for GlobalTech Solutions. The context does not mention any co-founders."
                }
            ]
        },
        
        # Example 5: CTO query - FINAL ANSWER should only be the name
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Sarah Johnson is the Chief Technology Officer at DataFlow Inc. She has been CTO since 2018. Mark Williams is the Head of Engineering at DataFlow Inc. Robert Kim is the Chief Financial Officer at DataFlow Inc.\n---\nQuestion: Who is the CTO of DataFlow Inc?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is the Chief Technology Officer at DataFlow Inc\"\n  - Action: [KEEP]\n- Item: Mark Williams\n  - Evidence: \"Mark Williams is the Head of Engineering at DataFlow Inc\"\n  - Action: [DISCARD] (Reason: Head of Engineering, not CTO).\n- Item: Robert Kim\n  - Evidence: \"Robert Kim is the Chief Financial Officer at DataFlow Inc\"\n  - Action: [DISCARD] (Reason: CFO, not CTO).\n- End of scan.\n\nFINAL ANSWER:\nThe CTO of DataFlow Inc is Sarah Johnson."
                }
            ]
        },
        
        # Example 6: Another 4+ co-founders example (complete scanning)
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Alex Rodriguez is the Co-Founder and CEO of TechVentures. He started the company in 2015.\n---\nJessica Chen is the Co-Founder and Chief Product Officer of TechVentures. She joined in 2015.\n---\nKevin Lee is the Co-Founder and Chief Technology Officer of TechVentures. He joined in 2015.\n---\nRachel Kim is the Co-Founder and Chief Financial Officer of TechVentures. She joined in 2015.\n---\nMichael Brown is the Head of Engineering at TechVentures. He is not a co-founder.\n---\nQuestion: Who are the co-founders of TechVentures?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Alex Rodriguez\n  - Evidence: \"Alex Rodriguez is the Co-Founder and CEO of TechVentures\"\n  - Action: [KEEP]\n- Item: Jessica Chen\n  - Evidence: \"Jessica Chen is the Co-Founder and Chief Product Officer of TechVentures\"\n  - Action: [KEEP]\n- Item: Kevin Lee\n  - Evidence: \"Kevin Lee is the Co-Founder and Chief Technology Officer of TechVentures\"\n  - Action: [KEEP]\n- Item: Rachel Kim\n  - Evidence: \"Rachel Kim is the Co-Founder and Chief Financial Officer of TechVentures\"\n  - Action: [KEEP]\n- Item: Michael Brown\n  - Evidence: \"Michael Brown is the Head of Engineering at TechVentures\"\n  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechVentures are Alex Rodriguez, Jessica Chen, Kevin Lee, and Rachel Kim."
                }
            ]
        }
    ]
    
    return examples


def main():
    print("=" * 80)
    print("ADDING FINAL FIXES FOR REMAINING ISSUES")
    print("=" * 80)
    
    # Load current dataset
    input_file = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Loading dataset: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Current examples: {len(data)}")
    
    # Add final fix examples at the beginning (for maximum priority)
    print(f"\n➕ Adding final fix examples...")
    final_fix_examples = create_final_fix_examples()
    print(f"   ✅ Adding {len(final_fix_examples)} final fix examples")
    
    # Add at very beginning for maximum priority
    data = final_fix_examples + data
    
    print(f"\n📊 Final dataset: {len(data)} examples")
    
    # Save updated dataset
    print(f"\n💾 Saving updated dataset...")
    with open(input_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✅ Saved to: {input_file}")
    
    # Verify verbatim accuracy
    print(f"\n🔍 Verifying verbatim accuracy...")
    import re
    perfect = 0
    for ex in data:
        user_content = ex['messages'][1]['content']
        assistant_content = ex['messages'][2]['content']
        context = user_content.split('Question:')[0].replace('Knowledge context:', '').strip()
        context = re.sub(r'\n---\n', '\n', context).lower()
        
        evidence_pattern = r'- Evidence:\s*"([^"]+)"'
        evidences = re.findall(evidence_pattern, assistant_content)
        
        all_verbatim = True
        for evidence in evidences:
            if evidence.strip().lower() not in context:
                all_verbatim = False
                break
        if all_verbatim:
            perfect += 1
    
    verbatim_rate = (perfect / len(data) * 100) if data else 0
    print(f"   ✅ Verbatim rate: {perfect}/{len(data)} ({verbatim_rate:.1f}%)")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"📊 Original: {len(data) - len(final_fix_examples)} examples")
    print(f"📊 Added: {len(final_fix_examples)} final fix examples")
    print(f"📊 Final: {len(data)} examples")
    print(f"📊 Verbatim rate: {verbatim_rate:.1f}%")
    print(f"\n✅ Examples added to address:")
    print(f"   ✅ Complete scanning (4+ co-founders, like LedgerAI)")
    print(f"   ✅ CTO marked [DISCARD] for co-founder queries")
    print(f"   ✅ No co-founders - FINAL ANSWER excludes all DISCARD items")
    print(f"   ✅ CTO query - FINAL ANSWER only includes name")

if __name__ == "__main__":
    main()