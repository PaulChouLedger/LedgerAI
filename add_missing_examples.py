#!/usr/bin/env python3
"""
Add Missing Training Examples
=============================
Adds critical missing examples to improve model performance:
1. "No co-founders" scenarios (currently 0 examples)
2. Single co-founder scenarios (currently only 1)
3. Strong role discrimination examples
"""

import json
import random

def create_no_cofounders_example():
    """Create example where no co-founders exist."""
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise data extraction bot.\n1. Start with REASONING:\n2. Scan the context carefully for information relevant to the query.\n3. For each relevant item found, write:\n   - Item: [What you found]\n   - Evidence: \"[Verbatim quote from context]\"\n   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n4. End scan with: - End of scan.\n5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.\n\nCRITICAL RULES:\n- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.\n- FINAL ANSWER must ONLY include items marked [KEEP].\n- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.\n- Read entire descriptions/chunks completely - titles may appear later in the text.\n- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.\n- If no co-founders are mentioned in the context, FINAL ANSWER must state that no co-founders were found."
                },
                {
                    "role": "user",
                    "content": "Knowledge context: James Wilson is the Founder and CEO of Acme Corporation. He started the company in 2010. Maria Garcia is the Chief Technology Officer at Acme Corporation. She joined in 2012. Thomas Lee is the Head of Sales at Acme Corporation.\n---\nQuestion: Who are the co-founders of Acme Corporation?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: James Wilson\n  - Evidence: \"James Wilson is the Founder and CEO of Acme Corporation\"\n  - Action: [DISCARD] (Reason: Founder, not co-founder. No co-founders mentioned).\n- Item: Maria Garcia\n  - Evidence: \"Maria Garcia is the Chief Technology Officer at Acme Corporation\"\n  - Action: [DISCARD] (Reason: CTO, not co-founder).\n- Item: Thomas Lee\n  - Evidence: \"Thomas Lee is the Head of Sales at Acme Corporation\"\n  - Action: [DISCARD] (Reason: Head of Sales, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nNo co-founders were found for Acme Corporation. The context mentions a Founder (James Wilson) but no co-founders."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise data extraction bot.\n1. Start with REASONING:\n2. Scan the context carefully for information relevant to the query.\n3. For each relevant item found, write:\n   - Item: [What you found]\n   - Evidence: \"[Verbatim quote from context]\"\n   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n4. End scan with: - End of scan.\n5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.\n\nCRITICAL RULES:\n- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.\n- FINAL ANSWER must ONLY include items marked [KEEP].\n- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.\n- Read entire descriptions/chunks completely - titles may appear later in the text.\n- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.\n- If no co-founders are mentioned in the context, FINAL ANSWER must state that no co-founders were found."
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
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise data extraction bot.\n1. Start with REASONING:\n2. Scan the context carefully for information relevant to the query.\n3. For each relevant item found, write:\n   - Item: [What you found]\n   - Evidence: \"[Verbatim quote from context]\"\n   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n4. End scan with: - End of scan.\n5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.\n\nCRITICAL RULES:\n- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.\n- FINAL ANSWER must ONLY include items marked [KEEP].\n- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.\n- Read entire descriptions/chunks completely - titles may appear later in the text.\n- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.\n- If no co-founders are mentioned in the context, FINAL ANSWER must state that no co-founders were found."
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Sarah Martinez is the CEO of DataFlow Inc. She has been leading the company since 2018. Michael Brown is the Chief Technology Officer at DataFlow Inc. He joined in 2019. Emily Davis is the Head of Marketing at DataFlow Inc.\n---\nQuestion: Who are the co-founders of DataFlow Inc?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Sarah Martinez\n  - Evidence: \"Sarah Martinez is the CEO of DataFlow Inc\"\n  - Action: [DISCARD] (Reason: CEO, not co-founder. No co-founders mentioned).\n- Item: Michael Brown\n  - Evidence: \"Michael Brown is the Chief Technology Officer at DataFlow Inc\"\n  - Action: [DISCARD] (Reason: CTO, not co-founder).\n- Item: Emily Davis\n  - Evidence: \"Emily Davis is the Head of Marketing at DataFlow Inc\"\n  - Action: [DISCARD] (Reason: Head of Marketing, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nNo co-founders were found for DataFlow Inc. The context does not mention any co-founders."
                }
            ]
        }
    ]
    return examples


def create_single_cofounder_examples():
    """Create examples with single co-founder (with role discrimination)."""
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise data extraction bot.\n1. Start with REASONING:\n2. Scan the context carefully for information relevant to the query.\n3. For each relevant item found, write:\n   - Item: [What you found]\n   - Evidence: \"[Verbatim quote from context]\"\n   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n4. End scan with: - End of scan.\n5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.\n\nCRITICAL RULES:\n- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.\n- FINAL ANSWER must ONLY include items marked [KEEP].\n- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.\n- Read entire descriptions/chunks completely - titles may appear later in the text.\n- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate."
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
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise data extraction bot.\n1. Start with REASONING:\n2. Scan the context carefully for information relevant to the query.\n3. For each relevant item found, write:\n   - Item: [What you found]\n   - Evidence: \"[Verbatim quote from context]\"\n   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n4. End scan with: - End of scan.\n5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.\n\nCRITICAL RULES:\n- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.\n- FINAL ANSWER must ONLY include items marked [KEEP].\n- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.\n- Read entire descriptions/chunks completely - titles may appear later in the text.\n- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate."
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Alex Rodriguez is the Co-Founder and CEO of InnovateLabs. He started the company in 2015. Jessica Chen is the Head of Engineering at InnovateLabs. She joined in 2016. She is not a co-founder.\n---\nQuestion: Who are the co-founders of InnovateLabs?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Alex Rodriguez\n  - Evidence: \"Alex Rodriguez is the Co-Founder and CEO of InnovateLabs\"\n  - Action: [KEEP]\n- Item: Jessica Chen\n  - Evidence: \"Jessica Chen is the Head of Engineering at InnovateLabs\"\n  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of InnovateLabs are Alex Rodriguez."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise data extraction bot.\n1. Start with REASONING:\n2. Scan the context carefully for information relevant to the query.\n3. For each relevant item found, write:\n   - Item: [What you found]\n   - Evidence: \"[Verbatim quote from context]\"\n   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n4. End scan with: - End of scan.\n5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.\n\nCRITICAL RULES:\n- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.\n- FINAL ANSWER must ONLY include items marked [KEEP].\n- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.\n- Read entire descriptions/chunks completely - titles may appear later in the text.\n- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate."
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Kevin Lee is the Co-Founder and Chief Product Officer of TechVentures. He has been with the company since 2017. Rachel Kim is the Chief Technology Officer at TechVentures. She joined in 2018. She is not a co-founder.\n---\nQuestion: Who are the co-founders of TechVentures?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Kevin Lee\n  - Evidence: \"Kevin Lee is the Co-Founder and Chief Product Officer of TechVentures\"\n  - Action: [KEEP]\n- Item: Rachel Kim\n  - Evidence: \"Rachel Kim is the Chief Technology Officer at TechVentures\"\n  - Action: [DISCARD] (Reason: CTO, not co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechVentures are Kevin Lee."
                }
            ]
        }
    ]
    return examples


def main():
    print("=" * 80)
    print("ADDING MISSING TRAINING EXAMPLES")
    print("=" * 80)
    
    # Load current dataset
    input_file = 'rag_cot_training_dataset_100percent.json'
    output_file = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Loading dataset: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Current examples: {len(data)}")
    
    # Add missing examples
    print(f"\n➕ Adding missing examples...")
    
    # 1. No co-founders examples (critical - currently 0)
    no_cofounder_examples = create_no_cofounders_example()
    print(f"   ✅ Adding {len(no_cofounder_examples)} 'no co-founders' examples")
    data.extend(no_cofounder_examples)
    
    # 2. Single co-founder examples (currently only 1)
    single_cofounder_examples = create_single_cofounder_examples()
    print(f"   ✅ Adding {len(single_cofounder_examples)} single co-founder examples")
    data.extend(single_cofounder_examples)
    
    print(f"\n📊 Final dataset: {len(data)} examples")
    print(f"   - Added: {len(no_cofounder_examples) + len(single_cofounder_examples)} examples")
    
    # Save updated dataset
    print(f"\n💾 Saving updated dataset...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✅ Saved to: {output_file}")
    
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
    
    if verbatim_rate == 100.0:
        print(f"\n✅ Dataset is still 100% verbatim!")
    else:
        print(f"\n⚠️  Warning: Some examples may not be verbatim")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"📊 Original: {len(data) - len(no_cofounder_examples) - len(single_cofounder_examples)} examples")
    print(f"📊 Added: {len(no_cofounder_examples) + len(single_cofounder_examples)} examples")
    print(f"   - No co-founders: {len(no_cofounder_examples)}")
    print(f"   - Single co-founder: {len(single_cofounder_examples)}")
    print(f"📊 Final: {len(data)} examples")
    print(f"📊 Verbatim rate: {verbatim_rate:.1f}%")
    print(f"\n✅ Ready for training!")

if __name__ == "__main__":
    main()