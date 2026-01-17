#!/usr/bin/env python3
"""
Fix Reasoning and Hallucination Issues
======================================
Adds examples to fix:
1. Incomplete reasoning (not evaluating all items)
2. Hallucinations in FINAL ANSWER (items not in reasoning)
3. Reasoning errors (marking wrong items [KEEP]/[DISCARD])
4. Inconsistent FINAL ANSWER (not following reasoning)
"""

import json

def create_strict_examples():
    """Create examples with strict rules enforcement."""
    
    # Enhanced system prompt with stronger emphasis
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
- CTO, Head of Engineering, and other roles are NOT co-founders unless explicitly stated."""

    examples = [
        # Example 1: Complete scanning with multiple people
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Paul Chou is the Co-Founder and Chief Technology Officer of LedgerAI. He has 20 years of experience.\n---\nBob Carella is the Co-Founder and Chief Operating Officer of LedgerAI. He previously worked at Binance.US.\n---\nDavid Lara is the Co-Founder and Chief Marketing Officer of LedgerAI. He has extensive marketing experience.\n---\nJorge Guinovart is the Co-Founder and CEO of LedgerAI. He founded the company in 2020.\n---\nPeter Moeller is the Business Development Lead at LedgerAI. He is not a co-founder.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Paul Chou\n  - Evidence: \"Paul Chou is the Co-Founder and Chief Technology Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Bob Carella\n  - Evidence: \"Bob Carella is the Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: David Lara\n  - Evidence: \"David Lara is the Co-Founder and Chief Marketing Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Jorge Guinovart\n  - Evidence: \"Jorge Guinovart is the Co-Founder and CEO of LedgerAI\"\n  - Action: [KEEP]\n- Item: Peter Moeller\n  - Evidence: \"Peter Moeller is the Business Development Lead at LedgerAI\"\n  - Action: [DISCARD] (Reason: Business Development Lead, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of LedgerAI are Paul Chou, Bob Carella, David Lara, and Jorge Guinovart."
                }
            ]
        },
        
        # Example 2: Founder vs Co-Founder discrimination
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
        
        # Example 3: CTO vs Co-Founder (single co-founder)
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
        
        # Example 4: Complete scanning - all items evaluated
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Alex Rodriguez is the Co-Founder and CEO of TechVentures. He started the company in 2015.\n---\nJessica Chen is the Co-Founder and Chief Product Officer of TechVentures. She joined in 2015.\n---\nKevin Lee is the Head of Engineering at TechVentures. He is not a co-founder.\n---\nRachel Kim is the Chief Financial Officer at TechVentures. She is not a co-founder.\n---\nQuestion: Who are the co-founders of TechVentures?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Alex Rodriguez\n  - Evidence: \"Alex Rodriguez is the Co-Founder and CEO of TechVentures\"\n  - Action: [KEEP]\n- Item: Jessica Chen\n  - Evidence: \"Jessica Chen is the Co-Founder and Chief Product Officer of TechVentures\"\n  - Action: [KEEP]\n- Item: Kevin Lee\n  - Evidence: \"Kevin Lee is the Head of Engineering at TechVentures\"\n  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder).\n- Item: Rachel Kim\n  - Evidence: \"Rachel Kim is the Chief Financial Officer at TechVentures\"\n  - Action: [DISCARD] (Reason: CFO, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechVentures are Alex Rodriguez and Jessica Chen."
                }
            ]
        },
        
        # Example 5: Team size query - proper discrimination
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The engineering team consists of 50 developers. The marketing team has 20 members. The sales team includes 30 people.\n---\nQuestion: How many people are in the engineering team?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 50 developers\n  - Evidence: \"The engineering team consists of 50 developers\"\n  - Action: [KEEP]\n- Item: 20 marketing members\n  - Evidence: \"The marketing team has 20 members\"\n  - Action: [DISCARD] (Reason: Marketing team, not engineering team).\n- Item: 30 sales people\n  - Evidence: \"The sales team includes 30 people\"\n  - Action: [DISCARD] (Reason: Sales team, not engineering team).\n- End of scan.\n\nFINAL ANSWER:\nThe engineering team consists of 50 developers."
                }
            ]
        },
        
        # Example 6: Products query - only main products, not services
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics. The company also provides consulting services to enterprise customers.\n---\nQuestion: What products does CloudScale offer?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: CloudScale Compute\n  - Evidence: \"CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics\"\n  - Action: [KEEP]\n- Item: CloudScale Storage\n  - Evidence: \"CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics\"\n  - Action: [KEEP]\n- Item: CloudScale Analytics\n  - Evidence: \"CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics\"\n  - Action: [KEEP]\n- Item: Consulting services\n  - Evidence: \"The company also provides consulting services to enterprise customers\"\n  - Action: [DISCARD] (Reason: Consulting is a service, not a product).\n- End of scan.\n\nFINAL ANSWER:\nCloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics."
                }
            ]
        }
    ]
    
    return examples


def main():
    print("=" * 80)
    print("FIXING REASONING AND HALLUCINATION ISSUES")
    print("=" * 80)
    
    # Load current dataset
    input_file = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Loading dataset: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Current examples: {len(data)}")
    
    # Add strict examples at the beginning (for priority)
    print(f"\n➕ Adding strict enforcement examples...")
    strict_examples = create_strict_examples()
    print(f"   ✅ Adding {len(strict_examples)} strict examples")
    
    # Add at beginning for priority
    data = strict_examples + data
    
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
    print(f"📊 Original: {len(data) - len(strict_examples)} examples")
    print(f"📊 Added: {len(strict_examples)} strict enforcement examples")
    print(f"📊 Final: {len(data)} examples")
    print(f"📊 Verbatim rate: {verbatim_rate:.1f}%")
    print(f"\n✅ Examples added to address:")
    print(f"   ✅ Incomplete reasoning (complete scanning)")
    print(f"   ✅ Hallucinations (FINAL ANSWER = only REASONING items)")
    print(f"   ✅ Reasoning errors (Founder ≠ Co-Founder, CTO ≠ Co-Founder)")
    print(f"   ✅ Inconsistent FINAL ANSWER (strict adherence to [KEEP] only)")

if __name__ == "__main__":
    main()