#!/usr/bin/env python3
"""
Add Diverse Real-World Queries
==============================
Adds real-world queries with messy, production-like RAG chunks:
- Multiple people in same chunk
- Long, complex descriptions
- Mixed information types
- Real formatting (not synthetic)
"""

import json

def create_diverse_real_world_examples():
    """Create diverse real-world examples with messy chunks."""
    
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
- FINAL ANSWER should be concise - only include the requested information, not extra words.
- For role queries, FINAL ANSWER should include the person's name, not the role title.
- For number queries, include full context (e.g., "50 developers" not just "50").
- Programming languages are NOT the same as natural languages (English, French, Spanish)."""

    examples = [
        # Example 1: Education query with messy chunk (multiple people, long description)
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform.\n---\nQuestion: Where did David Lara go to school?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: University of Washington\n  - Evidence: \"David holds an MS in Material Science and Engineering from the University of Washington\"\n  - Action: [KEEP]\n- Item: University of Texas\n  - Evidence: \"and a Master's in Public Affairs from the University of Texas\"\n  - Action: [KEEP]\n- Item: Bob's education\n  - Evidence: \"he serves as an Adjunct Professor at Drew University\"\n  - Action: [DISCARD] (Reason: Different person - Bob, not David Lara).\n- Item: Jorge Guinovart\n  - Evidence: \"Jorge Guinovart is a visionary leader\"\n  - Action: [DISCARD] (Reason: Different person - Jorge, not David Lara).\n- End of scan.\n\nFINAL ANSWER:\nDavid Lara attended the University of Washington, where he earned an MS in Material Science and Engineering, and the University of Texas, where he earned a Master's in Public Affairs."
                }
            ]
        },
        
        # Example 2: Token query with complex, mixed information
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Ensures that all AI-driven decisions are traceable and auditable, allowing businesses to understand how insights are generated. Bias Mitigation Algorithms – Reduces the risk of data-driven bias by continuously refining AI models to reflect fair, transparent, and ethical decision-making. LedgerAI is revolutionizing the way enterprises access AI-powered business intelligence by introducing a tokenized revenue model that aligns economic incentives with the adoption and growth of our ecosystem. Unlike traditional SaaS subscription models or one-time software licensing fees, LedgerAI's approach ensures a dynamic and self-sustaining AI economy powered by $LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration. How It Works: The Future of AI Monetization 1. Tokenized AI Access – Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure. This creates a protected economy where $LEDGER tokens fuels platform adoption while maintaining liquidity. 2. AI Compute Marketplace – Instead of static software fees, enterprises allocate $LEDGER tokens toward AI processing power, dynamically scaling their usage based on real-time business needs – similar to how companies purchase AWS credits for cloud computing. 3. Hardware & Software Integration – The LedgerAI ecosystem extends beyond software, with dedicated hardware (including AuraVision's Hammerhead) requiring $LEDGER tokens for activation, secure processing, and on-premises AI acceleration. Revolutionary, not merely Evolutionary Unlike traditional business models where users pay subscription fees, LedgerAI introduces a decentralized AI monetization structure, benefitting and incentivizing LedgerAI Quantum Corporation, token holders, clients, and the broader AI and digital currency communities. Built-In Demand Creation – Every business adopting AuraVision purchases and holds $LEDGER tokens to access AuraVision services, creating organic demand for the token. Market Stabilization & Treasury Management – LedgerAI can also sell collected tokens back to the market in a strategic and controlled manner to maintain liquidity, fund further development and innovation, and expand ecosystem partnerships without relying on external capital raises. A Self-Sustaining AI Economy LedgerAI's model is self-sustaining. Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem, while mechanisms such as buybacks and revenue sharing drive long-term sustainability and growth.\n---\nQuestion: What do you know about the ledger token?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: $LEDGER is an ERC-20 token\n  - Evidence: \"$LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration\"\n  - Action: [KEEP]\n- Item: Tokenized AI Access\n  - Evidence: \"Tokenized AI Access – Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure\"\n  - Action: [KEEP]\n- Item: AI Compute Marketplace\n  - Evidence: \"AI Compute Marketplace – Instead of static software fees, enterprises allocate $LEDGER tokens toward AI processing power, dynamically scaling their usage based on real-time business needs\"\n  - Action: [KEEP]\n- Item: Hardware Integration\n  - Evidence: \"Hardware & Software Integration – The LedgerAI ecosystem extends beyond software, with dedicated hardware (including AuraVision's Hammerhead) requiring $LEDGER tokens for activation, secure processing, and on-premises AI acceleration\"\n  - Action: [KEEP]\n- Item: Self-Sustaining Economy\n  - Evidence: \"LedgerAI's model is self-sustaining. Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem\"\n  - Action: [KEEP]\n- Item: Bias Mitigation Algorithms\n  - Evidence: \"Bias Mitigation Algorithms – Reduces the risk of data-driven bias\"\n  - Action: [DISCARD] (Reason: Not about the token, about AI features).\n- End of scan.\n\nFINAL ANSWER:\nThe $LEDGER token is an ERC-20 token that powers LedgerAI's tokenized revenue model. It enables tokenized AI access, where businesses purchase $LEDGER tokens to access compute power, AI services, and infrastructure. The token supports an AI compute marketplace for dynamic scaling of AI processing power, and integrates with hardware like AuraVision's Hammerhead for activation and secure processing. The $LEDGER token system creates a self-sustaining AI economy where clients use tokens to access AuraVision services."
                }
            ]
        },
        
        # Example 3: Benefits query with complex, mixed information
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions. In response, LedgerAI has developed a powerful security protocol that leverages local AI processing power with the Hammerhead local hardware device that integrates blockchain-backed encryption with decentralized intelligence. On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors. Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information. The Shift to Local AI & Decentralized Intelligence: Compliance, Privacy & Security Measures As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions. Outdated Business Intelligence Approaches Many enterprises continue to rely on legacy reporting systems, static spreadsheets, and manual reconciliation processes, which are inherently slow and reactive as opposed to predictive. These outdated systems and processes result in: Delayed decision-making – Slow data processing which negatively impacts market positioning and weakens a company's competitive advantage. Reactive governance models – Without the benefit of cutting-edge predictive information, organizations struggle to anticipate risks and opportunities, responding only after challenges arise.\n---\nQuestion: What are the benefits of localized?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: On-Premises AI Processing\n  - Evidence: \"On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors\"\n  - Action: [KEEP]\n- Item: Decentralized AI Infrastructure\n  - Evidence: \"Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss\"\n  - Action: [KEEP]\n- Item: Self-Destruct & Recovery Mechanism\n  - Evidence: \"Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information\"\n  - Action: [KEEP]\n- Item: Compliance and Privacy\n  - Evidence: \"As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions\"\n  - Action: [KEEP]\n- Item: Delayed decision-making\n  - Evidence: \"Delayed decision-making – Slow data processing which negatively impacts market positioning\"\n  - Action: [DISCARD] (Reason: This is a problem with outdated systems, not a benefit of localized).\n- Item: Reactive governance models\n  - Evidence: \"Reactive governance models – Without the benefit of cutting-edge predictive information\"\n  - Action: [DISCARD] (Reason: This is a problem with outdated systems, not a benefit of localized).\n- End of scan.\n\nFINAL ANSWER:\nThe benefits of localized AI include: On-Premises AI Processing that ensures sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors. Decentralized AI Infrastructure that leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism that allows secure data recovery through blockchain encryption if hardware is lost or compromised. Additionally, localized AI addresses compliance and privacy concerns as data privacy regulations tighten and security threats increase."
                }
            ]
        },
        
        # Example 4: Experience/background query with messy chunk
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion.\n---\nQuestion: What is Will Specht's background?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 20 years of experience\n  - Evidence: \"Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development\"\n  - Action: [KEEP]\n- Item: Previous companies\n  - Evidence: \"With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens\"\n  - Action: [KEEP]\n- Item: University of Delaware\n  - Evidence: \"A University of Delaware engineering graduate\"\n  - Action: [KEEP]\n- Item: Current role\n  - Evidence: \"leading LedgerAI's cutting-edge engineering efforts as Head of Engineering\"\n  - Action: [KEEP]\n- Item: Jorge Guinovart\n  - Evidence: \"Jorge Guinovart is a visionary leader\"\n  - Action: [DISCARD] (Reason: Different person - Jorge, not Will Specht).\n- End of scan.\n\nFINAL ANSWER:\nWill Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development. He has an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, where he built and scaled complex systems. He is a University of Delaware engineering graduate and currently leads LedgerAI's cutting-edge engineering efforts as Head of Engineering."
                }
            ]
        },
        
        # Example 5: Company information query with mixed chunks
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: LedgerAI is revolutionizing the way enterprises access AI-powered business intelligence by introducing a tokenized revenue model that aligns economic incentives with the adoption and growth of our ecosystem. Unlike traditional SaaS subscription models or one-time software licensing fees, LedgerAI's approach ensures a dynamic and self-sustaining AI economy powered by $LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration. By integrating AI analytics, blockchain transparency, and decentralized intelligence, LedgerAI delivers unparalleled insights that empower organizations to remain agile, compliant, and competitive. AuraVision transforms financial governance, risk assessment, and strategic planning from burdensome, manual processes into automated, intelligent, and continuously evolving systems. The Ledger ERC-20 Token ($LEDGER) extends these capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across a variety of industries and establishing a viable and scalable ecosystem for enterprise intelligence. As businesses require greater transparency, security, and efficiency, AuraVision is ready now to aggressively meet those needs with its future-proof and innovative AI solution. For partnerships, inquiries, or more information, please contact info@ledgerai.co.\n---\nQuestion: What does LedgerAI do?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: AI-powered business intelligence\n  - Evidence: \"LedgerAI is revolutionizing the way enterprises access AI-powered business intelligence\"\n  - Action: [KEEP]\n- Item: Tokenized revenue model\n  - Evidence: \"by introducing a tokenized revenue model that aligns economic incentives with the adoption and growth of our ecosystem\"\n  - Action: [KEEP]\n- Item: AI analytics and blockchain\n  - Evidence: \"By integrating AI analytics, blockchain transparency, and decentralized intelligence, LedgerAI delivers unparalleled insights\"\n  - Action: [KEEP]\n- Item: AuraVision platform\n  - Evidence: \"AuraVision transforms financial governance, risk assessment, and strategic planning from burdensome, manual processes into automated, intelligent, and continuously evolving systems\"\n  - Action: [KEEP]\n- Item: Contact information\n  - Evidence: \"For partnerships, inquiries, or more information, please contact info@ledgerai.co\"\n  - Action: [DISCARD] (Reason: Contact info, not what the company does).\n- End of scan.\n\nFINAL ANSWER:\nLedgerAI revolutionizes how enterprises access AI-powered business intelligence through a tokenized revenue model. The company integrates AI analytics, blockchain transparency, and decentralized intelligence to deliver insights that help organizations remain agile, compliant, and competitive. LedgerAI's AuraVision platform transforms financial governance, risk assessment, and strategic planning from manual processes into automated, intelligent systems."
                }
            ]
        },
        
        # Example 6: Role query with multiple people in chunk
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion.\n---\nQuestion: Who is the CEO of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Paul Chou\n  - Evidence: \"As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence\"\n  - Action: [KEEP]\n- Item: David Lara\n  - Evidence: \"As Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [DISCARD] (Reason: COO, not CEO).\n- Item: Jorge Guinovart\n  - Evidence: \"As Co-Founder and Chief Marketing Officer of LedgerAI\"\n  - Action: [DISCARD] (Reason: CMO, not CEO).\n- End of scan.\n\nFINAL ANSWER:\nThe CEO of LedgerAI is Paul Chou."
                }
            ]
        },
        
        # Example 7: Product features query with complex chunk
        {
            "messages": [
                {
                    "role": "system",
                    "content": enhanced_system
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Ensures that all AI-driven decisions are traceable and auditable, allowing businesses to understand how insights are generated. Bias Mitigation Algorithms – Reduces the risk of data-driven bias by continuously refining AI models to reflect fair, transparent, and ethical decision-making. AuraVision transforms financial governance, risk assessment, and strategic planning from burdensome, manual processes into automated, intelligent, and continuously evolving systems. The Ledger ERC-20 Token ($LEDGER) extends these capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across a variety of industries and establishing a viable and scalable ecosystem for enterprise intelligence. By integrating AI analytics, blockchain transparency, and decentralized intelligence, LedgerAI delivers unparalleled insights that empower organizations to remain agile, compliant, and competitive.\n---\nQuestion: What are the key features of AuraVision?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Traceable and auditable decisions\n  - Evidence: \"Ensures that all AI-driven decisions are traceable and auditable, allowing businesses to understand how insights are generated\"\n  - Action: [KEEP]\n- Item: Bias Mitigation Algorithms\n  - Evidence: \"Bias Mitigation Algorithms – Reduces the risk of data-driven bias by continuously refining AI models to reflect fair, transparent, and ethical decision-making\"\n  - Action: [KEEP]\n- Item: Automated financial governance\n  - Evidence: \"AuraVision transforms financial governance, risk assessment, and strategic planning from burdensome, manual processes into automated, intelligent, and continuously evolving systems\"\n  - Action: [KEEP]\n- Item: AI analytics and blockchain\n  - Evidence: \"By integrating AI analytics, blockchain transparency, and decentralized intelligence, LedgerAI delivers unparalleled insights\"\n  - Action: [KEEP]\n- Item: Tokenized access\n  - Evidence: \"The Ledger ERC-20 Token ($LEDGER) extends these capabilities by delivering tokenized access to AI-driven insights\"\n  - Action: [KEEP]\n- End of scan.\n\nFINAL ANSWER:\nAuraVision's key features include: traceable and auditable AI-driven decisions, bias mitigation algorithms that reduce data-driven bias, automated financial governance and risk assessment that transforms manual processes into intelligent systems, integration of AI analytics and blockchain transparency for unparalleled insights, and tokenized access through the $LEDGER token."
                }
            ]
        }
    ]
    
    return examples


def main():
    print("=" * 80)
    print("ADDING DIVERSE REAL-WORLD QUERIES")
    print("=" * 80)
    
    # Load current dataset
    input_file = 'rag_cot_training_dataset_100percent.json'
    
    print(f"\n📂 Loading dataset: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Current examples: {len(data)}")
    
    # Add diverse real-world examples
    print(f"\n➕ Adding diverse real-world examples...")
    diverse_examples = create_diverse_real_world_examples()
    print(f"   ✅ Adding {len(diverse_examples)} diverse real-world examples")
    
    # Add at beginning for priority (after the 90%+ examples)
    # Insert after first 6 examples (90%+ final examples)
    data = data[:6] + diverse_examples + data[6:]
    
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
    for i, ex in enumerate(diverse_examples):
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
    
    verbatim_rate = (perfect / len(diverse_examples) * 100) if diverse_examples else 0
    print(f"   ✅ Verbatim rate for new examples: {perfect}/{len(diverse_examples)} ({verbatim_rate:.1f}%)")
    
    if issues:
        print(f"\n   ⚠️  Issues found:")
        for issue in issues[:5]:
            print(f"      {issue}")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"📊 Original: {len(data) - len(diverse_examples)} examples")
    print(f"📊 Added: {len(diverse_examples)} diverse real-world examples")
    print(f"📊 Final: {len(data)} examples")
    print(f"📊 Verbatim rate: {verbatim_rate:.1f}%")
    print(f"\n✅ Examples added:")
    print(f"   ✅ Education queries (messy chunks with multiple people)")
    print(f"   ✅ Token/product queries (complex, mixed information)")
    print(f"   ✅ Benefits queries (long descriptions)")
    print(f"   ✅ Background/experience queries")
    print(f"   ✅ Company information queries")
    print(f"   ✅ Role queries (multiple people in chunk)")
    print(f"   ✅ Product features queries")
    print(f"\n💡 Benefits:")
    print(f"   ✅ Handles real-world, messy RAG chunks")
    print(f"   ✅ Better production readiness")
    print(f"   ✅ More diverse query types")
    print(f"   ✅ Realistic training scenarios")

if __name__ == "__main__":
    main()