#!/usr/bin/env python3
"""
Add real-world RAG examples to training dataset.

These are actual production queries with real RAG chunks that the model
encounters in practice. This will improve generalization to real-world scenarios.
"""

import json
import re

# Enhanced system prompt (same as in strengthen_cot_rag_logic.py)
ENHANCED_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER - this is ABSOLUTE and MANDATORY.
- FINAL ANSWER must ONLY include items marked [KEEP] - no exceptions.
- If you mark an item [DISCARD] in reasoning, it is FORBIDDEN to mention it in FINAL ANSWER.
- Before writing FINAL ANSWER, verify: ONLY include items that were marked [KEEP] in REASONING.
- Read entire descriptions/chunks completely - titles may appear later in the text.
- SCAN ALL CHUNKS COMPLETELY - read each chunk from start to finish before moving to next.
- IMPORTANT: Identify ALL relevant items in REASONING before making any decisions. Do NOT stop after finding some matches - scan completely and list EVERY item that could be relevant.
- Extract Evidence from the EXACT ITEM MENTION that matches the query - find where the item is explicitly mentioned in relation to what you're looking for, not from descriptions or unrelated text.
- IMPORTANT: Do NOT include headers (like 'AURA VISION...'), page numbers, or metadata in Evidence. Extract ONLY the exact quote that shows the role or information.

CO-FOUNDER IDENTIFICATION RULES (CRITICAL):
- If evidence contains "Co-Founder" OR "co-founder" OR "Co-Founder and [Role]", the person IS a co-founder.
- If evidence says "As Co-Founder and Chief Operating Officer", this person IS a co-founder - mark [KEEP].
- If evidence says "As Co-Founder and [Any Role]", this person IS a co-founder - mark [KEEP].
- If evidence says "CEO and Co-Founder", this person IS a co-founder - mark [KEEP].
- If evidence does NOT mention "Co-Founder" or "co-founder", the person is NOT a co-founder (unless explicitly stated as "founder").
- Example: "As Co-Founder and Chief Operating Officer" → [KEEP] (co-founder confirmed).
- Example: "serves as Chief Operating Officer" → [DISCARD] (no co-founder mention).

FINAL ANSWER COMPLETENESS (CRITICAL):
- FINAL ANSWER must include EVERY item marked [KEEP] in REASONING.
- Count how many items you marked [KEEP] - ALL of them must appear in FINAL ANSWER.
- If you marked 4 items [KEEP], FINAL ANSWER must include all 4 items.
- Before writing FINAL ANSWER, list all [KEEP] items and ensure each one is included.
- If an item is marked [KEEP] in REASONING but missing from FINAL ANSWER, your answer is incomplete."""

def clean_chunk(chunk):
    """Clean up chunk text, removing extra whitespace."""
    # Remove leading/trailing whitespace and normalize newlines
    chunk = chunk.strip()
    # Replace multiple spaces with single space
    chunk = re.sub(r' +', ' ', chunk)
    # Replace multiple newlines with single newline
    chunk = re.sub(r'\n+', '\n', chunk)
    return chunk

def create_example(query, chunks, expected_reasoning, expected_answer):
    """Create a training example from query and chunks."""
    # Combine chunks with separator
    combined_context = "\n---\n".join([clean_chunk(chunk) for chunk in chunks])
    
    # Format user message
    user_content = f"Knowledge context: {combined_context}\n---\nQuestion: {query}"
    
    return {
        "messages": [
            {
                "role": "system",
                "content": ENHANCED_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": expected_reasoning + "\n\nFINAL ANSWER:\n" + expected_answer
            }
        ]
    }

def add_real_world_examples(dataset):
    """Add real-world RAG examples to dataset."""
    
    new_examples = []
    
    # Example 1: Who are the co-founders of LedgerAI? (Real-world)
    chunks1 = [
        """Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions.""",
        """into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. A recognized thought leader, Paul has been featured on TED Talks and major global conferences for over a decade, solidifying his role as a pioneer at the forefront of AI, blockchain, and financial innovation. Bob Carella is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.""",
        """has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. As Co-Founder of Soler Salva LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures LedgerAI's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing LedgerAI's position as a trailblazer in AI-powered business intelligence. Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at Scarinci Hollenbeck, Attorneys at Law, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that LedgerAI continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-driven business intelligence. Liam Hugill is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as LedgerAI's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that LedgerAI's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying LedgerAI's brand, expanding its reach, and solidifying trust among investors and supporters. Hailing from the United Kingdom, Liam's background as a top-performing salesman at Marlwood Financial honed his skills in strategic communication, relationship management, and high-impact messaging—all of which he now channels into building a strong and engaged global community for LedgerAI."""
    ]
    
    reasoning1 = """REASONING:
- Item: Bob Carella
  - Evidence: "As Co-Founder and Chief Financial Officer of LedgerAI"
  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).
- Item: David Lara
  - Evidence: "As Co-Founder and Chief Operating Officer of LedgerAI"
  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder. Ignore header 'AURA VISION...').
- Item: Jorge Guinovart
  - Evidence: "As Co-Founder and Chief Marketing Officer of LedgerAI"
  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).
- Item: Will Specht
  - Evidence: "leading LedgerAI's cutting-edge engineering efforts as Head of Engineering"
  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only Head of Engineering role).
- Item: Paul Chou
  - Evidence: "As CEO and Co-Founder of LedgerAI"
  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).
- Item: Albert Soler
  - Evidence: "bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor"
  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only External Counsel & Advisor role).
- Item: Peter Moeller
  - Evidence: "serving as Business Development Lead at LedgerAI"
  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only Business Development Lead role).
- Item: Liam Hugill
  - Evidence: "being a natural fit as LedgerAI's Ambassador of Influence and Engagement"
  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only Ambassador role).
- End of scan."""
    
    answer1 = "The co-founders of LedgerAI are Paul Chou, Bob Carella, David Lara, and Jorge Guinovart."
    
    new_examples.append(create_example(
        "Who are the co-founders of LedgerAI?",
        chunks1,
        reasoning1,
        answer1
    ))
    
    # Example 2: Do you know who David Lara is? (Real-world)
    chunks2 = [
        """Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions."""
    ]
    
    reasoning2 = """REASONING:
- Item: David Lara
  - Evidence: "David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making."
  - Action: [KEEP] (Reason: Information about David Lara, his role at LedgerAI, and his background).
- End of scan."""
    
    answer2 = "David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI. He has extensive experience in both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas."
    
    new_examples.append(create_example(
        "Do you know who David Lara is?",
        chunks2,
        reasoning2,
        answer2
    ))
    
    # Example 3: Who's the CFO of Ledger AI? (Real-world)
    chunks3 = [
        """expanding its reach, and solidifying trust among investors and supporters. Hailing from the United Kingdom, Liam's background as a top-performing salesman at Marlwood Financial honed his skills in strategic communication, relationship management, and high-impact messaging—all of which he now channels into building a strong and engaged global community for LedgerAI. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 25 13. LEDGERAI Quantum Corporation Notice and Disclaimer NOTICE: This Disclaimer references Confidential Information which may be presented By LedgerAI Quantum Corporation (the "Company") to people interested in learning more about the Company and its activities as set forth in the "LedgerAI Quantum Corporation Introduces our Flagship Product AuraVision and the Future of AI-Driven Solutions," which is CONFIDENTIAL and shall be referred to herein as the "Confidential Presentation." This Confidential Presentation is being delivered to you by the Company and is for INFORMATION PURPOSES ONLY. It is provided to you solely in your capacity of having requested information about the Company. Any reproduction or distribution of this Confidential Presentation, in whole or in part, or the disclosure of its contents, without the prior written consent of the Company is prohibited. By viewing, reading, or accepting possession of this Confidential Presentation whether in hard copy or electronic form, each recipient, and its partners, directors, officers, employees, attorneys, agents, and representatives (collectively, "Recipient"), agrees: (i) to maintain the confidentiality of all information contained in this Confidential Presentation and not already in the public domain; and, (ii) to, within three days, return or destroy all copies of this Confidential Presentation or portions thereof in the Recipient's possession following the request by the Company for the return or destruction of such copies. This Confidential Presentation and any oral statements made in connection with this Confidential Presentation neither constitute an offer to sell nor a solicitation of an offer to buy any securities, or the solicitation of any proxy, vote, consent, or approval in any jurisdiction, nor shall there be any transaction in any jurisdiction in which the offer, solicitation, or sale would be unlawful under the laws of such jurisdiction. This Confidential Presentation is not intended for distribution to, or use by, any person in any jurisdiction where such distribution or use would be contrary to local law or regulation. Listed sources of definitions as well as listed sources of information are for convenience and reference only and do not imply – and nor may you infer – that anything contained herein is related to or governed by, through, or under such sources, nor may you rely upon such sourcing or definitions or the decisions which might flow therefrom. This Confidential Presentation may contain confidential information or "Trade Secrets" of the Company, with "trade secrets" defined in the Defend Trade Secrets Act 2016, 18 U.S.C. § 1839(3). Your acceptance to review this Confidential Presentation means that you will not reveal such Trade Secrets and other Confidential Information as is covered by any accompanying or other Non-Disclosure Agreement.""",
        """Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions.""",
        """has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. As Co-Founder of Soler Salva LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures LedgerAI's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing LedgerAI's position as a trailblazer in AI-powered business intelligence. Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at Scarinci Hollenbeck, Attorneys at Law, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that LedgerAI continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-driven business intelligence. Liam Hugill is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as LedgerAI's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that LedgerAI's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying LedgerAI's brand, expanding its reach, and solidifying trust among investors and supporters. Hailing from the United Kingdom, Liam's background as a top-performing salesman at Marlwood Financial honed his skills in strategic communication, relationship management, and high-impact messaging—all of which he now channels into building a strong and engaged global community for LedgerAI."""
    ]
    
    # Need to search for CFO information - looking for Bob Carella
    reasoning3 = """REASONING:
- Item: Bob Carella
  - Evidence: "Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders."
  - Action: [KEEP] (Reason: Mention of Bob and financial operations, but need to find explicit CFO title). Scanning further...
- Item: Bob Carella
  - Evidence: "As Co-Founder and Chief Financial Officer of LedgerAI"
  - Action: [KEEP] (Reason: Explicit mention of "Chief Financial Officer of LedgerAI").
- End of scan."""
    
    answer3 = "The CFO of LedgerAI is Bob Carella. He is Co-Founder and Chief Financial Officer of LedgerAI, where he architects the company's financial strategy, tokenomics, and investment framework. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs."
    
    new_examples.append(create_example(
        "Who's the CFO of Ledger AI?",
        chunks3,
        reasoning3,
        answer3
    ))
    
    # Example 4: Where did David Lara go to school? (Real-world)
    chunks4 = [
        """Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions."""
    ]
    
    reasoning4 = """REASONING:
- Item: David Lara education
  - Evidence: "David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas"
  - Action: [KEEP] (Reason: Explicit mention of David Lara's educational background).
- End of scan."""
    
    answer4 = "David Lara holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas."
    
    new_examples.append(create_example(
        "Where did David Lara go to school?",
        chunks4,
        reasoning4,
        answer4
    ))
    
    # Example 5: What do you know about the ledger token (Real-world)
    chunks5 = [
        """Ensures that all AI-driven decisions are traceable and auditable, allowing businesses to understand how insights are generated. Bias Mitigation Algorithms – Reduces the risk of data-driven bias by continuously refining AI models to reflect fair, transparent, and ethical decision-making. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 16 8. Differentiated Revenue Model Tokenized AI Access & Sustainable Growth LedgerAI is revolutionizing the way enterprises access AI-powered business intelligence by introducing a tokenized revenue model that aligns economic incentives with the adoption and growth of our ecosystem. Unlike traditional SaaS subscription models or one-time software licensing fees, LedgerAI's approach ensures a dynamic and self-sustaining AI economy powered by $LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration. How It Works: The Future of AI Monetization 1. Tokenized AI Access – Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure. This creates a protected economy where $LEDGER tokens fuels platform adoption while maintaining liquidity. 2. AI Compute Marketplace – Instead of static software fees, enterprises allocate $LEDGER tokens toward AI processing power, dynamically scaling their usage based on real-time business needs – similar to how companies purchase AWS credits for cloud computing. 3. Hardware & Software Integration – The LedgerAI ecosystem extends beyond software, with dedicated hardware (including AuraVision's Hammerhead) requiring $LEDGER tokens for activation, secure processing, and on-premises AI acceleration. Revolutionary, not merely Evolutionary Unlike traditional business models where users pay subscription fees, LedgerAI introduces a decentralized AI monetization structure, benefitting and incentivizing LedgerAI Quantum Corporation, token holders, clients, and the broader AI and digital currency communities. Built-In Demand Creation – Every business adopting AuraVision purchases and holds $LEDGER tokens to access AuraVision services, creating organic demand for the token. Market Stabilization & Treasury Management – LedgerAI can also sell collected tokens back to the market in a strategic and controlled manner to maintain liquidity, fund further development and innovation, and expand ecosystem partnerships without relying on external capital raises. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 17 A Self-Sustaining AI Economy LedgerAI's model is self-sustaining. Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem, while mechanisms such as buybacks and revenue sharing drive long-term sustainability and growth. LedgerAI's strategy offers a forward-looking approach beyond outdated pay-as-you-go AI models, licensing fees, or static SaaS subscriptions. The $LEDGER token system will provide LedgerAI with a competitive advantage and allow LedgerAI to continue to scale globally while maintaining token-based value accrual. By aligning incentives across enterprises, investors, and token holders, LedgerAI is pioneering and revolutionizing the future of AI monetization, where access to intelligence is not only a service but an integrated economic model that ensures long-term positive value for all participants.""",
        """can no longer be bound by outdated, reactive decision-making models. Instead, they must harness the power and innovations of AuraVision to anticipate challenges, identify opportunities, and execute data-driven strategies in real time, which provides an unparalleled and significant competitive advantage to businesses, entrepreneurs, government entities, and organizations across the globe. By integrating AI analytics, blockchain transparency, and decentralized intelligence, LedgerAI delivers unparalleled insights that empower organizations to remain agile, compliant, and competitive. AuraVision transforms financial governance, risk assessment, and strategic planning from burdensome, manual processes into automated, intelligent, and continuously evolving systems. The Ledger ERC-20 Token ($LEDGER) extends these capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across a variety of industries and establishing a viable and scalable ecosystem for enterprise intelligence. As businesses require greater transparency, security, and efficiency, AuraVision is ready now to aggressively meet those needs with its future-proof and innovative AI solution. Beyond AI—A Movement Toward Intelligent Governance LedgerAI is much more than an AI company – LedgerAI offers a movement toward a more intelligent and capable real-time and intuitive business ecosystem. By bridging the gap between AI, blockchain, and real-time analytics, LedgerAI is redefining and reinventing enterprise intelligence while setting the global benchmark and global standard for how businesses, government organizations, and entrepreneurs interact with and utilize, significant amounts of critical data. As AI continues to develop and rapidly expand its role in governance, financial operations, and decision-making, organizations that embrace AuraVision will be poised to become leaders in their respective industries. LedgerAI's innovation, software, hardware, and ongoing Research and Development activities, as well as its new method of data compilation and analysis is paving the way for the next era of AI-driven governance, strategy, and enterprise intelligence. For partnerships, inquiries, or more information, please contact info@ledgerai.co. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 22 12. LedgerAI Founders & Advisors The co-founders of LedgerAI bring an unmatched depth of experience across AI, blockchain, finance, and enterprise technology, forging a team built to redefine the future of intelligent business solutions. With expertise spanning market-making, regulatory compliance, high-frequency trading, and AI-powered decision intelligence, this elite group has engineered global financial systems, launched cutting-edge platforms, and pioneered AI-driven innovation at scale. Their diverse backgrounds – spanning fintech, enterprise AI, government, and digital assets – position LedgerAI at the forefront of AI-driven business solutions, ensuring a multifaceted approach to governance, strategy, and automation. Backed by a dynamic team of experts across engineering, legal, business development, and marketing, LedgerAI is not only shaping the future of AI-powered intelligence but also ensuring its seamless integration into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations.""",
        """expanding its reach, and solidifying trust among investors and supporters. Hailing from the United Kingdom, Liam's background as a top-performing salesman at Marlwood Financial honed his skills in strategic communication, relationship management, and high-impact messaging—all of which he now channels into building a strong and engaged global community for LedgerAI."""
    ]
    
    reasoning5 = """REASONING:
- Item: $LEDGER token information
  - Evidence: "$LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration"
  - Action: [KEEP] (Reason: Information about the ledger token).
- Item: Token usage
  - Evidence: "Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure"
  - Action: [KEEP] (Reason: How the token is used).
- Item: Token economy
  - Evidence: "Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem, while mechanisms such as buybacks and revenue sharing drive long-term sustainability and growth"
  - Action: [KEEP] (Reason: Token economy model).
- Item: The Ledger ERC-20 Token
  - Evidence: "The Ledger ERC-20 Token ($LEDGER) extends these capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across a variety of industries and establishing a viable and scalable ecosystem for enterprise intelligence"
  - Action: [KEEP] (Reason: Additional information about the token).
- End of scan."""
    
    answer5 = "The Ledger token ($LEDGER) is an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration. It powers LedgerAI's tokenized revenue model, allowing businesses to purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure. Clients use $LEDGER tokens to access AuraVision services, and the token system includes mechanisms such as buybacks and revenue sharing to drive long-term sustainability and growth. The $LEDGER token extends LedgerAI's capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across industries and establishing a viable and scalable ecosystem for enterprise intelligence."
    
    new_examples.append(create_example(
        "What do you know about the ledger token?",
        chunks5,
        reasoning5,
        answer5
    ))
    
    # Example 6: What are the benefits of localized? (Real-world - note: query says "localized" but context talks about "local AI")
    chunks6 = [
        """is spread across a variety of disparate systems, increasing exposure to breaches, compliance violations, and regulatory scrutiny. Without a unified, real-time, and AI-driven approach, organizations are forced to make decisions with limited visibility and outdated information, increasing their risk exposure and diminishing their ability to respond proactively to market shifts. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 7 Outdated Business Intelligence Approaches Many enterprises continue to rely on legacy reporting systems, static spreadsheets, and manual reconciliation processes, which are inherently slow and reactive as opposed to predictive. These outdated systems and processes result in: Delayed decision-making – Slow data processing which negatively impacts market positioning and weakens a company's competitive advantage. Reactive governance models – Without the benefit of cutting-edge predictive information, organizations struggle to anticipate risks and opportunities, responding only after challenges arise. Lack of predictive insights – Businesses fail to benefit from AI-driven forecasting and automation to optimize strategies, mitigate risks, and drive operational efficiency. The Shift to Local AI & Decentralized Intelligence: Compliance, Privacy & Security Measures As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions. In response, LedgerAI has developed a powerful security protocol that leverages local AI processing power with the Hammerhead local hardware device that integrates blockchain-backed encryption with decentralized intelligence. On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors. Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 8 3. The Evolution of AI-Driven Decision-Making and AuraVision's AI-Powered Strategic Advisor From Reactive to Proactive Intelligence Traditional business intelligence and data analytics systems have been reactive, analyzing past data to generate reports and insights after key events have already occurred. This retrospective approach limits an organization's ability to anticipate challenges, identify opportunities and risks, and act with immediacy and agility. AI-driven decision-making revolutionizes the C-Suite and management process, enabling businesses to transition from reactive analysis to cutting-edge predictive and autonomous intelligence. AuraVision is the future made manifest today, delivering real-time intelligence, combining local AI processing, blockchain security, and decentralized intelligence to reveal actionable insights the moment they are needed. The Rise of Local AI and Decentralized Intelligence Businesses continue to generate and rely upon unprecedented volumes of data. Relying on cloud-based AI solutions presents several notable and important drawbacks and limitations, including latency issues, security risks, and compliance challenges."""
    ]
    
    reasoning6 = """REASONING:
- Item: Local AI benefits
  - Evidence: "On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors"
  - Action: [KEEP] (Reason: Benefits of localized/local AI processing - data privacy and security).
- Item: Decentralized AI benefits
  - Evidence: "Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss"
  - Action: [KEEP] (Reason: Benefits of decentralized/localized AI - security and compliance).
- Item: Data recovery benefits
  - Evidence: "Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information"
  - Action: [KEEP] (Reason: Benefits of localized system - data recovery and security).
- Item: Local AI advantages over cloud
  - Evidence: "Relying on cloud-based AI solutions presents several notable and important drawbacks and limitations, including latency issues, security risks, and compliance challenges"
  - Action: [KEEP] (Reason: Implies benefits of local AI - avoiding cloud drawbacks).
- End of scan."""
    
    answer6 = "The benefits of localized (local AI) processing include: On-Premises AI Processing – AuraVision operates locally on secure business hardware, ensuring that sensitive data never leaves the organization's premises and eliminating reliance on centralized data processors. Decentralized AI Infrastructure – Leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism – If a device is lost, stolen, or compromised, it can self-destruct while allowing businesses to securely recover data through blockchain encryption, ensuring continuity without exposing sensitive information. Local AI also avoids the drawbacks of cloud-based solutions, including latency issues, security risks, and compliance challenges."
    
    new_examples.append(create_example(
        "What are the benefits of localized?",
        chunks6,
        reasoning6,
        answer6
    ))
    
    # Add all new examples to the beginning of the dataset (high priority - real-world examples)
    dataset[:0] = new_examples
    return len(new_examples)

def main():
    """Main function to add real-world RAG examples."""
    print("=" * 80)
    print("Adding Real-World RAG Examples to Training Dataset")
    print("=" * 80)
    print()
    
    # Load dataset
    input_file = "rag_cot_training_dataset.json"
    print(f"📂 Loading dataset: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"   Found {len(dataset)} existing examples")
    print()
    
    # Add real-world examples
    print("➕ Adding real-world RAG examples...")
    added = add_real_world_examples(dataset)
    print(f"   Added {added} new real-world examples")
    print()
    
    # Save updated dataset
    output_file = "rag_cot_training_dataset.json"
    print(f"💾 Saving updated dataset: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Dataset updated: {len(dataset)} total examples")
    print()
    
    print("=" * 80)
    print("✅ Real-World Examples Added!")
    print("=" * 80)
    print()
    print("Key additions:")
    print("  ✓ Co-founders query (real-world chunks with headers)")
    print("  ✓ Individual person query (David Lara)")
    print("  ✓ Role-specific query (CFO)")
    print("  ✓ Educational background query")
    print("  ✓ Token information query")
    print("  ✓ Benefits/features query (localized AI)")
    print()
    print("All examples include:")
    print("  ✓ Enhanced system prompt with co-founder rules")
    print("  ✓ Real-world RAG chunks with headers/metadata")
    print("  ✓ Complete reasoning showing evidence extraction")
    print("  ✓ Comprehensive FINAL ANSWER with all [KEEP] items")
    print()

if __name__ == "__main__":
    main()
