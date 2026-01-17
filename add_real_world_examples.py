#!/usr/bin/env python3
"""
Add Real-World Examples
======================
Adds the specific real-world queries mentioned:
1. Where did David Lara go to school?
2. What do you know about the ledger token?
3. What are the benefits of localized?
"""

import json

def create_real_world_examples():
    """Create real-world examples with verbatim evidence."""
    
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
- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- FINAL ANSWER must ONLY contain items that appear in REASONING section.
- DO NOT add items to FINAL ANSWER that were not evaluated in REASONING.
- Read entire descriptions/chunks completely - titles may appear later in the text.
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.
- You MUST evaluate ALL relevant items in the context before ending the scan.
- "Founder" is NOT the same as "Co-Founder" - mark Founder as [DISCARD] for co-founder queries.
- CTO, Head of Engineering, and other roles are NOT co-founders unless explicitly stated.
- FINAL ANSWER should be concise - only include the requested information, not extra words."""

    examples = [
        # Example 1: Where did David Lara go to school?
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership.\n---\nQuestion: Where did David Lara go to school?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: University of Washington\n  - Evidence: \"David holds an MS in Material Science and Engineering from the University of Washington\"\n  - Action: [KEEP]\n- Item: University of Texas\n  - Evidence: \"and a Master's in Public Affairs from the University of Texas\"\n  - Action: [KEEP]\n- End of scan.\n\nFINAL ANSWER:\nDavid Lara attended the University of Washington, where he earned an MS in Material Science and Engineering, and the University of Texas, where he earned a Master's in Public Affairs."
                }
            ]
        },
        
        # Example 2: What do you know about the ledger token?
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: LedgerAI is revolutionizing the way enterprises access AI-powered business intelligence by introducing a tokenized revenue model that aligns economic incentives with the adoption and growth of our ecosystem. Unlike traditional SaaS subscription models or one-time software licensing fees, LedgerAI's approach ensures a dynamic and self-sustaining AI economy powered by $LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration. How It Works: The Future of AI Monetization 1. Tokenized AI Access – Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure. This creates a protected economy where $LEDGER tokens fuels platform adoption while maintaining liquidity. 2. AI Compute Marketplace – Instead of static software fees, enterprises allocate $LEDGER tokens toward AI processing power, dynamically scaling their usage based on real-time business needs – similar to how companies purchase AWS credits for cloud computing. 3. Hardware & Software Integration – The LedgerAI ecosystem extends beyond software, with dedicated hardware (including AuraVision's Hammerhead) requiring $LEDGER tokens for activation, secure processing, and on-premises AI acceleration. Revolutionary, not merely Evolutionary Unlike traditional business models where users pay subscription fees, LedgerAI introduces a decentralized AI monetization structure, benefitting and incentivizing LedgerAI Quantum Corporation, token holders, clients, and the broader AI and digital currency communities. Built-In Demand Creation – Every business adopting AuraVision purchases and holds $LEDGER tokens to access AuraVision services, creating organic demand for the token. Market Stabilization & Treasury Management – LedgerAI can also sell collected tokens back to the market in a strategic and controlled manner to maintain liquidity, fund further development and innovation, and expand ecosystem partnerships without relying on external capital raises. A Self-Sustaining AI Economy LedgerAI's model is self-sustaining. Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem, while mechanisms such as buybacks and revenue sharing drive long-term sustainability and growth. LedgerAI's strategy offers a forward-looking approach beyond outdated pay-as-you-go AI models, licensing fees, or static SaaS subscriptions. The $LEDGER token system will provide LedgerAI with a competitive advantage and allow LedgerAI to continue to scale globally while maintaining token-based value accrual. By aligning incentives across enterprises, investors, and token holders, LedgerAI is pioneering and revolutionizing the future of AI monetization, where access to intelligence is not only a service but an integrated economic model that ensures long-term positive value for all participants.\n---\nQuestion: What do you know about the ledger token?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: $LEDGER is an ERC-20 token\n  - Evidence: \"$LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration\"\n  - Action: [KEEP]\n- Item: Tokenized AI Access\n  - Evidence: \"Tokenized AI Access – Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure\"\n  - Action: [KEEP]\n- Item: AI Compute Marketplace\n  - Evidence: \"AI Compute Marketplace – Instead of static software fees, enterprises allocate $LEDGER tokens toward AI processing power, dynamically scaling their usage based on real-time business needs\"\n  - Action: [KEEP]\n- Item: Hardware Integration\n  - Evidence: \"Hardware & Software Integration – The LedgerAI ecosystem extends beyond software, with dedicated hardware (including AuraVision's Hammerhead) requiring $LEDGER tokens for activation, secure processing, and on-premises AI acceleration\"\n  - Action: [KEEP]\n- Item: Self-Sustaining Economy\n  - Evidence: \"LedgerAI's model is self-sustaining. Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem\"\n  - Action: [KEEP]\n- End of scan.\n\nFINAL ANSWER:\nThe $LEDGER token is an ERC-20 token that powers LedgerAI's tokenized revenue model. It enables tokenized AI access, where businesses purchase $LEDGER tokens to access compute power, AI services, and infrastructure. The token supports an AI compute marketplace for dynamic scaling of AI processing power, and integrates with hardware like AuraVision's Hammerhead for activation and secure processing. The $LEDGER token system creates a self-sustaining AI economy where clients use tokens to access AuraVision services, with mechanisms like buybacks and revenue sharing driving long-term sustainability."
                }
            ]
        },
        
        # Example 3: What are the benefits of localized?
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions. In response, LedgerAI has developed a powerful security protocol that leverages local AI processing power with the Hammerhead local hardware device that integrates blockchain-backed encryption with decentralized intelligence. On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors. Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information. The Shift to Local AI & Decentralized Intelligence: Compliance, Privacy & Security Measures As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions.\n---\nQuestion: What are the benefits of localized?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: On-Premises AI Processing\n  - Evidence: \"On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors\"\n  - Action: [KEEP]\n- Item: Decentralized AI Infrastructure\n  - Evidence: \"Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss\"\n  - Action: [KEEP]\n- Item: Self-Destruct & Recovery Mechanism\n  - Evidence: \"Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information\"\n  - Action: [KEEP]\n- Item: Compliance and Privacy\n  - Evidence: \"As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions\"\n  - Action: [KEEP]\n- End of scan.\n\nFINAL ANSWER:\nThe benefits of localized AI include: On-Premises AI Processing that ensures sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors. Decentralized AI Infrastructure that leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism that allows secure data recovery through blockchain encryption if hardware is lost or compromised. Additionally, localized AI addresses compliance and privacy concerns as data privacy regulations tighten and security threats increase."
                }
            ]
        }
    ]
    
    return examples


def main():
    print("=" * 80)
    print("ADDING REAL-WORLD EXAMPLES")
    print("=" * 80)
    
    # Load current dataset
    input_file = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Loading dataset: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Current examples: {len(data)}")
    
    # Add real-world examples at the beginning (high priority)
    print(f"\n➕ Adding real-world examples...")
    real_world_examples = create_real_world_examples()
    print(f"   ✅ Adding {len(real_world_examples)} real-world examples")
    
    # Add at beginning for priority
    data = real_world_examples + data
    
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
    issues = []
    for i, ex in enumerate(data[:len(real_world_examples)]):  # Check only new examples
        user_content = ex['messages'][1]['content']
        assistant_content = ex['messages'][2]['content']
        context = user_content.split('Question:')[0].replace('Knowledge context:', '').strip()
        context = re.sub(r'\n---\n', '\n', context).lower()
        
        evidence_pattern = r'- Evidence:\s*"([^"]+)"'
        evidences = re.findall(evidence_pattern, assistant_content)
        
        all_verbatim = True
        for evidence in evidences:
            evidence_clean = evidence.strip().lower()
            if evidence_clean not in context:
                all_verbatim = False
                issues.append(f"Example {i}: '{evidence[:60]}...' not found verbatim")
        
        if all_verbatim:
            perfect += 1
    
    verbatim_rate = (perfect / len(real_world_examples) * 100) if real_world_examples else 0
    print(f"   ✅ Verbatim rate for new examples: {perfect}/{len(real_world_examples)} ({verbatim_rate:.1f}%)")
    
    if issues:
        print(f"\n   ⚠️  Issues found:")
        for issue in issues[:5]:
            print(f"      {issue}")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"📊 Original: {len(data) - len(real_world_examples)} examples")
    print(f"📊 Added: {len(real_world_examples)} real-world examples")
    print(f"   - Where did David Lara go to school?")
    print(f"   - What do you know about the ledger token?")
    print(f"   - What are the benefits of localized?")
    print(f"📊 Final: {len(data)} examples")
    print(f"📊 Verbatim rate: {verbatim_rate:.1f}%")
    print(f"\n✅ Real-world examples added!")

if __name__ == "__main__":
    main()