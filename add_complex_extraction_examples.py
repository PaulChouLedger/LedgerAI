#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Complex Multi-Chunk Extraction Examples to Training Dataset
Addresses issues with:
- Long, multi-chunk contexts with noise
- Name truncation
- Contradictory DISCARD logic
- Missing co-founders in complex contexts
"""

import json
import random

# System prompt (same as training)
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
- Read entire descriptions/chunks completely - titles may appear later in the text."""

# Complex examples that address the extraction issues
COMPLEX_EXAMPLES = [
    # Example 1: Long multi-chunk with noise (similar to actual LedgerAI test)
    {
        "context": """has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that QuantumTech's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning QuantumTech at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Smith is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to QuantumTech as External Counsel & Advisor. As Co-Founder of Smith Legal LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures QuantumTech's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing QuantumTech's position as a trailblazer in AI-powered business intelligence. Peter Johnson is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at QuantumTech. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at TechCorp, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that QuantumTech continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-driven business intelligence. Liam Anderson is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as QuantumTech's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that QuantumTech's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying QuantumTech's brand, expanding its reach, and solidifying trust among investors and supporters.
---
Payroll & Stock Administration at Binance.US and Sprinklr, Bob Martinez managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Kim is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of QuantumTech, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving QuantumTech's mission to redefine enterprise intelligence and governance at a global scale. Jorge Rodriguez is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of QuantumTech, he is spearheading global adoption, brand strategy, and market expansion, ensuring QuantumTech becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Johnson is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading QuantumTech's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that QuantumTech's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning QuantumTech at the forefront of enterprise AI solutions.
---
into enterprises worldwide. Paul Chen is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of QuantumTech, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. Bob Martinez is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of QuantumTech, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.""",
        "query": "Who are the co-founders of QuantumTech?",
        "expected": ["Paul Chen", "Bob Martinez", "David Kim", "Jorge Rodriguez"],
        "not_expected": ["Albert Smith", "Will Johnson", "Peter Johnson", "Liam Anderson"],
        "reasoning": """REASONING:
- Item: Albert Smith
- Evidence: "As Co-Founder of Smith Legal LLP"
- Action: [DISCARD] (Reason: Co-founder of Smith Legal LLP, not QuantumTech).
- Item: Peter Johnson
- Evidence: "serving as Business Development Lead at QuantumTech"
- Action: [DISCARD] (Reason: Business Development Lead, not co-founder).
- Item: Liam Anderson
- Evidence: "being a natural fit as QuantumTech's Ambassador of Influence and Engagement"
- Action: [DISCARD] (Reason: Ambassador, not co-founder).
- Item: Bob Martinez
- Evidence: "As Co-Founder and Chief Financial Officer of QuantumTech"
- Action: [KEEP]
- Item: David Kim
- Evidence: "As Co-Founder and Chief Operating Officer of QuantumTech"
- Action: [KEEP]
- Item: Jorge Rodriguez
- Evidence: "As Co-Founder and Chief Marketing Officer of QuantumTech"
- Action: [KEEP]
- Item: Will Johnson
- Evidence: "leading QuantumTech's cutting-edge engineering efforts as Head of Engineering"
- Action: [DISCARD] (Reason: Head of Engineering, not co-founder).
- Item: Paul Chen
- Evidence: "As CEO and Co-Founder of QuantumTech"
- Action: [KEEP]
- End of scan.""",
        "answer": "The co-founders of QuantumTech are Paul Chen, Bob Martinez, David Kim, and Jorge Rodriguez."
    },
    
    # Example 2: Complex context with name variations and full names
    {
        "context": """TechFlow Innovations is a leading AI company. Sarah Miller is the Head of Engineering at TechFlow Innovations. She previously worked at Google and Microsoft. Mark Johnson serves as an advisor to TechFlow Innovations, providing strategic guidance on technology trends.
---
Alex Chen is a technology leader with extensive experience. As Co-Founder and Chief Technology Officer of TechFlow Innovations, he drives the company's technical vision. He graduated from Stanford University with a degree in Computer Science. Lisa Wang is a financial expert specializing in enterprise finance. As Co-Founder and Chief Financial Officer of TechFlow Innovations, she architects the company's financial strategy and manages investor relations.
---
Tom Brown is a strategic leader with 20 years of experience in business development. As CEO and Co-Founder of TechFlow Innovations, he leads the company's strategic vision and drives growth. He has previously worked at major tech companies including Amazon and Apple. Emily Davis is the Head of Marketing at TechFlow Innovations, responsible for brand strategy and customer acquisition.""",
        "query": "Who are the co-founders of TechFlow Innovations?",
        "expected": ["Alex Chen", "Lisa Wang", "Tom Brown"],
        "not_expected": ["Sarah Miller", "Mark Johnson", "Emily Davis"],
        "reasoning": """REASONING:
- Item: Sarah Miller
- Evidence: "Head of Engineering at TechFlow Innovations"
- Action: [DISCARD] (Reason: Head of Engineering, not co-founder).
- Item: Mark Johnson
- Evidence: "serves as an advisor to TechFlow Innovations"
- Action: [DISCARD] (Reason: Advisor, not co-founder).
- Item: Alex Chen
- Evidence: "As Co-Founder and Chief Technology Officer of TechFlow Innovations"
- Action: [KEEP]
- Item: Lisa Wang
- Evidence: "As Co-Founder and Chief Financial Officer of TechFlow Innovations"
- Action: [KEEP]
- Item: Tom Brown
- Evidence: "As CEO and Co-Founder of TechFlow Innovations"
- Action: [KEEP]
- Item: Emily Davis
- Evidence: "Head of Marketing at TechFlow Innovations"
- Action: [DISCARD] (Reason: Head of Marketing, not co-founder).
- End of scan.""",
        "answer": "The co-founders of TechFlow Innovations are Alex Chen, Lisa Wang, and Tom Brown."
    },
    
    # Example 3: Very long descriptions with co-founder title appearing later
    {
        "context": """DataFlow Systems is revolutionizing enterprise data management. Michael Thompson is a technology visionary with over 25 years of experience in software engineering, data infrastructure, and cloud computing. He has worked at major technology companies including IBM, Oracle, and Salesforce, where he led teams building enterprise-scale data platforms. Michael's expertise spans distributed systems, real-time data processing, and machine learning infrastructure. As Co-Founder and Chief Technology Officer of DataFlow Systems, he drives the company's technical strategy and oversees product development. His leadership has been instrumental in building DataFlow's core platform, which processes billions of data points daily for enterprise clients worldwide.
---
Jennifer Park is a financial strategist and operations expert with deep experience in scaling technology companies. She has held senior finance roles at several high-growth startups, managing everything from fundraising to financial planning and analysis. Jennifer's background includes an MBA from Wharton and previous experience at McKinsey & Company, where she advised Fortune 500 companies on digital transformation. As Co-Founder and Chief Financial Officer of DataFlow Systems, she architects the company's financial strategy, manages investor relations, and ensures sustainable growth. Her strategic financial planning has enabled DataFlow to scale from startup to a company serving hundreds of enterprise clients.
---
Robert Lee is a business development leader with extensive experience in enterprise sales and partnerships. He has built sales organizations from the ground up at multiple technology companies, consistently exceeding revenue targets and establishing strategic partnerships with major corporations. Robert's expertise includes go-to-market strategy, enterprise account management, and channel partnerships. As CEO and Co-Founder of DataFlow Systems, he leads the company's strategic vision, drives business development, and manages key customer relationships. Under his leadership, DataFlow has secured partnerships with major technology companies and expanded into new markets.
---
Susan White is the Head of Product at DataFlow Systems, responsible for product strategy and roadmap. James Wilson serves as an External Advisor, providing guidance on technology trends.""",
        "query": "Who are the co-founders of DataFlow Systems?",
        "expected": ["Michael Thompson", "Jennifer Park", "Robert Lee"],
        "not_expected": ["Susan White", "James Wilson"],
        "reasoning": """REASONING:
- Item: Michael Thompson
- Evidence: "As Co-Founder and Chief Technology Officer of DataFlow Systems"
- Action: [KEEP]
- Item: Jennifer Park
- Evidence: "As Co-Founder and Chief Financial Officer of DataFlow Systems"
- Action: [KEEP]
- Item: Robert Lee
- Evidence: "As CEO and Co-Founder of DataFlow Systems"
- Action: [KEEP]
- Item: Susan White
- Evidence: "Head of Product at DataFlow Systems"
- Action: [DISCARD] (Reason: Head of Product, not co-founder).
- Item: James Wilson
- Evidence: "serves as an External Advisor"
- Action: [DISCARD] (Reason: External Advisor, not co-founder).
- End of scan.""",
        "answer": "The co-founders of DataFlow Systems are Michael Thompson, Jennifer Park, and Robert Lee."
    },
    
    # Example 4: Multiple chunks with many non-co-founders (stress test)
    {
        "context": """CloudScale Technologies is a cloud infrastructure company. Maria Garcia is the Head of Engineering at CloudScale Technologies. She has 15 years of experience in distributed systems. Thomas Anderson is the Chief Technology Officer at CloudScale Technologies, leading the technical team.
---
David Kim is a strategic leader in cloud computing and infrastructure. As Co-Founder and Chief Operating Officer of CloudScale Technologies, he leads operations and strategic initiatives. He previously worked at AWS and Microsoft Azure, where he helped build large-scale cloud platforms. Sarah Johnson is the Head of Sales at CloudScale Technologies, responsible for enterprise customer acquisition.
---
Christopher Brown is a technology entrepreneur with extensive experience in cloud infrastructure. As CEO and Co-Founder of CloudScale Technologies, he drives the company's vision and strategic direction. He has founded multiple successful technology companies and has deep expertise in cloud architecture and scaling infrastructure. Lisa Martinez is the Chief Marketing Officer at CloudScale Technologies, leading marketing and brand strategy.
---
Michael Chen is a financial expert specializing in technology finance. As Co-Founder and Chief Financial Officer of CloudScale Technologies, he manages the company's financial strategy, fundraising, and investor relations. He has an MBA from Harvard and previously worked at Goldman Sachs, where he advised technology companies on financial strategy. Robert Wilson is an External Advisor to CloudScale Technologies, providing strategic guidance.""",
        "query": "Who are the co-founders of CloudScale Technologies?",
        "expected": ["David Kim", "Christopher Brown", "Michael Chen"],
        "not_expected": ["Maria Garcia", "Thomas Anderson", "Sarah Johnson", "Lisa Martinez", "Robert Wilson"],
        "reasoning": """REASONING:
- Item: Maria Garcia
- Evidence: "Head of Engineering at CloudScale Technologies"
- Action: [DISCARD] (Reason: Head of Engineering, not co-founder).
- Item: Thomas Anderson
- Evidence: "Chief Technology Officer at CloudScale Technologies"
- Action: [DISCARD] (Reason: CTO, not co-founder).
- Item: David Kim
- Evidence: "As Co-Founder and Chief Operating Officer of CloudScale Technologies"
- Action: [KEEP]
- Item: Sarah Johnson
- Evidence: "Head of Sales at CloudScale Technologies"
- Action: [DISCARD] (Reason: Head of Sales, not co-founder).
- Item: Christopher Brown
- Evidence: "As CEO and Co-Founder of CloudScale Technologies"
- Action: [KEEP]
- Item: Lisa Martinez
- Evidence: "Chief Marketing Officer at CloudScale Technologies"
- Action: [DISCARD] (Reason: CMO, not co-founder).
- Item: Michael Chen
- Evidence: "As Co-Founder and Chief Financial Officer of CloudScale Technologies"
- Action: [KEEP]
- Item: Robert Wilson
- Evidence: "External Advisor to CloudScale Technologies"
- Action: [DISCARD] (Reason: External Advisor, not co-founder).
- End of scan.""",
        "answer": "The co-founders of CloudScale Technologies are David Kim, Christopher Brown, and Michael Chen."
    },
    
    # Example 5: Very long single chunk with co-founder title appearing late (Late-Title Pattern)
    {
        "context": """InnovateAI Solutions is transforming how businesses use artificial intelligence. James Wilson is a technology leader with extensive experience in AI and machine learning. He has worked at major technology companies including Google, where he led AI research teams, and Microsoft, where he developed enterprise AI solutions. James has a PhD in Computer Science from MIT and has published numerous papers on machine learning and neural networks. His expertise spans deep learning, natural language processing, and computer vision. James has been instrumental in building InnovateAI's core AI platform, which powers intelligent automation for hundreds of enterprise clients. As Co-Founder and Chief Technology Officer of InnovateAI Solutions, he drives the company's technical vision and oversees product development. His leadership has been critical to InnovateAI's success in the competitive AI market.
---
Patricia Martinez is a financial strategist with deep experience in scaling technology companies. She has an MBA from Stanford and previously worked at venture capital firms, where she evaluated and invested in AI startups. Patricia's expertise includes financial planning, fundraising, and investor relations. She has helped InnovateAI raise multiple funding rounds and establish relationships with key investors. As Co-Founder and Chief Financial Officer of InnovateAI Solutions, she architects the company's financial strategy and ensures sustainable growth. Her strategic financial management has enabled InnovateAI to scale rapidly while maintaining financial discipline.
---
Daniel Lee is a business development expert with a track record of building successful technology companies. He has founded and scaled multiple startups, consistently achieving strong revenue growth and market expansion. Daniel's expertise includes go-to-market strategy, enterprise sales, and strategic partnerships. He has established key partnerships for InnovateAI with major technology companies and enterprise clients. As CEO and Co-Founder of InnovateAI Solutions, he leads the company's strategic vision and drives business development. Under his leadership, InnovateAI has expanded into new markets and secured partnerships with Fortune 500 companies.
---
Amanda White is the Head of Product at InnovateAI Solutions, responsible for product strategy. Kevin Brown serves as an External Advisor, providing guidance on AI trends.""",
        "query": "Who are the co-founders of InnovateAI Solutions?",
        "expected": ["James Wilson", "Patricia Martinez", "Daniel Lee"],
        "not_expected": ["Amanda White", "Kevin Brown"],
        "reasoning": """REASONING:
- Item: James Wilson
- Evidence: "As Co-Founder and Chief Technology Officer of InnovateAI Solutions"
- Action: [KEEP]
- Item: Patricia Martinez
- Evidence: "As Co-Founder and Chief Financial Officer of InnovateAI Solutions"
- Action: [KEEP]
- Item: Daniel Lee
- Evidence: "As CEO and Co-Founder of InnovateAI Solutions"
- Action: [KEEP]
- Item: Amanda White
- Evidence: "Head of Product at InnovateAI Solutions"
- Action: [DISCARD] (Reason: Head of Product, not co-founder).
- Item: Kevin Brown
- Evidence: "serves as an External Advisor"
- Action: [DISCARD] (Reason: External Advisor, not co-founder).
- End of scan.""",
        "answer": "The co-founders of InnovateAI Solutions are James Wilson, Patricia Martinez, and Daniel Lee."
    },
]

def create_training_examples():
    """Create training examples from complex scenarios"""
    examples = []
    
    for scenario in COMPLEX_EXAMPLES:
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
    print("Adding Complex Multi-Chunk Extraction Examples")
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
    
    # Create new complex examples
    new_examples = create_training_examples()
    print(f"✅ Created {len(new_examples)} new complex examples")
    print()
    
    # Add to existing dataset
    existing_data.extend(new_examples)
    
    # Shuffle to mix examples
    random.shuffle(existing_data)
    
    # Save updated dataset
    output_file = "rag_cot_training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added {len(new_examples)} complex examples to dataset")
    print(f"✅ Total examples: {len(existing_data)}")
    print(f"✅ Saved to: {output_file}")
    print()
    print("📋 New examples address:")
    print("   - Long multi-chunk contexts with noise")
    print("   - Full name handling (no truncation)")
    print("   - Clear KEEP logic for co-founders")
    print("   - Late-Title Pattern (co-founder title appears later in long descriptions)")
    print("   - Multiple non-co-founders mixed in")
    print()
    print("=" * 80)
