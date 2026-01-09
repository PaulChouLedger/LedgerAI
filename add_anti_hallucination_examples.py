#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Anti-Hallucination Training Examples
Emphasizes EXACT extraction from context - no invented names or information
"""

import json
import random

# System prompt emphasizing exact extraction
COT_SYSTEM_PROMPT = """You are a precise data extraction bot.
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
- Read entire descriptions/chunks completely - titles may appear later in the text.
- CRITICAL: FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.
- CRITICAL: Scan the ENTIRE context from start to finish - do not stop scanning early. Items may appear in any chunk.
- CRITICAL ANTI-HALLUCINATION: You MUST extract information EXACTLY as written in the context. NEVER invent, guess, or create names, titles, or information. ONLY use information that is EXPLICITLY stated in the context. If a name is not in the context, you CANNOT use it."""

# Examples emphasizing exact extraction from context (LedgerAI-like scenarios)
ANTI_HALLUCINATION_EXAMPLES = [
    {
        "context": """has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. As Co-Founder of Soler Salva LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures LedgerAI's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing LedgerAI's position as a trailblazer in AI-powered business intelligence. Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at Scarinci Hollenbeck, Attorneys at Law, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that LedgerAI continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-powered business intelligence. Liam Hugill is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as LedgerAI's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that LedgerAI's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying LedgerAI's brand, expanding its reach, and solidifying trust among investors and supporters.
---
Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering.
---
into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. Bob Carella is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.""",
        "query": "Who are the co-founders of LedgerAI?",
        "reasoning": """REASONING:
- Item: Albert Soler
- Evidence: "As External Counsel & Advisor"
- Action: [DISCARD] (Reason: External Counsel, not co-founder).
- Item: Peter Moeller
- Evidence: "serving as Business Development Lead at LedgerAI"
- Action: [DISCARD] (Reason: Business Development Lead, not co-founder).
- Item: David Lara
- Evidence: "As Co-Founder and Chief Operating Officer of LedgerAI"
- Action: [KEEP]
- Item: Jorge Guinovart
- Evidence: "As Co-Founder and Chief Marketing Officer of LedgerAI"
- Action: [KEEP]
- Item: Will Specht
- Evidence: "leading LedgerAI's cutting-edge engineering efforts as Head of Engineering"
- Action: [DISCARD] (Reason: Head of Engineering, not co-founder).
- Item: Paul Chou
- Evidence: "As CEO and Co-Founder of LedgerAI"
- Action: [KEEP]
- Item: Bob Carella
- Evidence: "As Co-Founder and Chief Financial Officer of LedgerAI"
- Action: [KEEP]
- End of scan.""",
        "answer": "The co-founders of LedgerAI are David Lara, Jorge Guinovart, Paul Chou, and Bob Carella."
    },
]

def create_training_examples():
    """Create training examples emphasizing exact extraction"""
    examples = []
    
    for scenario in ANTI_HALLUCINATION_EXAMPLES:
        user_content = f"Knowledge context: {scenario['context']}\n---\nQuestion: {scenario['query']}"
        assistant_content = f"{scenario['reasoning']}\n\nFINAL ANSWER:\n{scenario['answer']}"
        
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": COT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        })
    
    return examples

if __name__ == "__main__":
    print("=" * 80)
    print("Adding Anti-Hallucination Examples")
    print("=" * 80)
    print()
    
    # Load existing dataset
    try:
        with open("rag_cot_training_dataset.json", 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"✅ Loaded {len(existing_data)} existing examples")
    except FileNotFoundError:
        print("❌ Error: rag_cot_training_dataset.json not found!")
        exit(1)
    
    # Create new examples
    new_examples = create_training_examples()
    print(f"✅ Created {len(new_examples)} new anti-hallucination examples")
    print()
    
    # Add to existing dataset
    existing_data.extend(new_examples)
    
    # Shuffle to mix examples
    random.shuffle(existing_data)
    
    # Save updated dataset
    output_file = "rag_cot_training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added {len(new_examples)} examples to dataset")
    print(f"✅ Total examples: {len(existing_data)}")
    print(f"✅ Saved to: {output_file}")
    print()
    print("📋 New examples emphasize:")
    print("   - EXACT extraction from context (verbatim quotes)")
    print("   - NO invented names or information")
    print("   - ONLY use information EXPLICITLY stated in context")
    print("   - LedgerAI test scenario with exact names from context")
    print()
    print("=" * 80)
