#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG CoT Model Test Suite (Colab Version)
Evaluates the fine-tuned Qwen 2.5 model's ability to perform RAG with Chain of Thought
"""

import os
import torch
import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter

# Try to import unsloth first
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pass

# Try to import llama_cpp for GGUF models
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "outputs_rag_cot"  # Path to LoRA adapters or merged model (fallback)
GGUF_MODEL_DIR = "gguf_model_rag_cot"  # Path to GGUF quantized model directory
MAX_SEQ_LENGTH = 4096

# The exact array-based system prompt used in training (matches rag_cot_training_dataset.json)
SYSTEM_PROMPT = """You are a data extraction bot. 

STEP 1: REASONING
- Scan EVERY chunk from start to finish.
- For each item found:
  Item: [name]
  Evidence: "[quote]"
  Action: [KEEP] or [DISCARD]
- **CRITICAL**: Do NOT stop after finding some matches. Scan ALL chunks until you reach the absolute end of the context.
- List Arrays: [KEEP_ARRAY]: [...] | [DISCARD_ARRAY]: [...]

STEP 2: FINAL ANSWER
- Use ONLY items from [KEEP_ARRAY].
- If empty, state "No items found."

RULES:
1. [DISCARD] items are FORBIDDEN in answer.
2. Complete the scan entirely before writing arrays."""

# ============================================================================
# Test Scenarios
# ============================================================================

def get_test_scenarios():
    """Returns a list of test scenarios for evaluation"""
    return [
        {
            "name": "LedgerAI Co-Founders (Real-World, 4 co-founders)",
            "query": "Who are the co-founders of LedgerAI?",
            "context": """Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions.
---
into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. A recognized thought leader, Paul has been featured on TED Talks and major global conferences for over a decade, solidifying his role as a pioneer at the forefront of AI, blockchain, and financial innovation. Bob Carella is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.
---
has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. As Co-Founder of Soler Salva LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures LedgerAI's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing LedgerAI's position as a trailblazer in AI-powered business intelligence. Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at Scarinci Hollenbeck, Attorneys at Law, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that LedgerAI continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-powered business intelligence. Liam Hugill is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as LedgerAI's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that LedgerAI's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying LedgerAI's brand, expanding its reach, and solidifying trust among investors and supporters. Hailing from the United Kingdom, Liam's background as a top-performing salesman at Marlwood Financial honed his skills in strategic communication, relationship management, and high-impact messaging—all of which he now channels into building a strong and engaged global community for LedgerAI.""",
            "expected": ["Paul Chou", "Bob Carella", "David Lara", "Jorge Guinovart"],
            "not_expected": ["Albert Soler", "Will Specht", "Peter Moeller", "Liam Hugill"],
            "answer_type": "person"
        },
        {
            "name": "David Lara - Individual Person Query (Real-World)",
            "query": "Do you know who David Lara is?",
            "context": """Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions.""",
            "expected": ["David Lara"],
            "not_expected": [],
            "answer_type": "person"
        },
        {
            "name": "CFO Query - Multi-Chunk Scanning (Real-World)",
            "query": "Who's the CFO of Ledger AI?",
            "context": """expanding its reach, and solidifying trust among investors and supporters. Hailing from the United Kingdom, Liam's background as a top-performing salesman at Marlwood Financial honed his skills in strategic communication, relationship management, and high-impact messaging—all of which he now channels into building a strong and engaged global community for LedgerAI. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 25 13. LEDGERAI Quantum Corporation Notice and Disclaimer NOTICE: This Disclaimer references Confidential Information which may be presented By LedgerAI Quantum Corporation (the "Company") to people interested in learning more about the Company and its activities as set forth in the "LedgerAI Quantum Corporation Introduces our Flagship Product AuraVision and the Future of AI-Driven Solutions," which is CONFIDENTIAL and shall be referred to herein as the "Confidential Presentation." This Confidential Presentation is being delivered to you by the Company and is for INFORMATION PURPOSES ONLY. It is provided to you solely in your capacity of having requested information about the Company. Any reproduction or distribution of this Confidential Presentation, in whole or in part, or the disclosure of its contents, without the prior written consent of the Company is prohibited. By viewing, reading, or accepting possession of this Confidential Presentation whether in hard copy or electronic form, each recipient, and its partners, directors, officers, employees, attorneys, agents, and representatives (collectively, "Recipient"), agrees: (i) to maintain the confidentiality of all information contained in this Confidential Presentation and not already in the public domain; and, (ii) to, within three days, return or destroy all copies of this Confidential Presentation or portions thereof in the Recipient's possession following the request by the Company for the return or destruction of such copies. This Confidential Presentation and any oral statements made in connection with this Confidential Presentation neither constitute an offer to sell nor a solicitation of an offer to buy any securities, or the solicitation of any proxy, vote, consent, or approval in any jurisdiction, nor shall there be any transaction in any jurisdiction in which the offer, solicitation, or sale would be unlawful under the laws of such jurisdiction. This Confidential Presentation is not intended for distribution to, or use by, any person in any jurisdiction where such distribution or use would be contrary to local law or regulation. Listed sources of definitions as well as listed sources of information are for convenience and reference only and do not imply – and nor may you infer – that anything contained herein is related to or governed by, through, or under such sources, nor may you rely upon such sourcing or definitions or the decisions which might flow therefrom. This Confidential Presentation may contain confidential information or "Trade Secrets" of the Company, with "trade secrets" defined in the Defend Trade Secrets Act 2016, 18 U.S.C. § 1839(3). Your acceptance to review this Confidential Presentation means that you will not reveal such Trade Secrets and other Confidential Information as is covered by any accompanying or other Non-Disclosure Agreement.
---
Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions.
---
into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. A recognized thought leader, Paul has been featured on TED Talks and major global conferences for over a decade, solidifying his role as a pioneer at the forefront of AI, blockchain, and financial innovation. Bob Carella is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.
---
has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. As Co-Founder of Soler Salva LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures LedgerAI's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing LedgerAI's position as a trailblazer in AI-powered business intelligence. Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at Scarinci Hollenbeck, Attorneys at Law, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that LedgerAI continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-powered business intelligence. Liam Hugill is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as LedgerAI's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that LedgerAI's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying LedgerAI's brand, expanding its reach, and solidifying trust among investors and supporters. Hailing from the United Kingdom, Liam's background as a top-performing salesman at Marlwood Financial honed his skills in strategic communication, relationship management, and high-impact messaging—all of which he now channels into building a strong and engaged global community for LedgerAI.""",
            "expected": ["Bob Carella"],
            "not_expected": [],
            "answer_type": "person"
        },
        {
            "name": "David Lara Education (Real-World)",
            "query": "Where did David Lara go to school?",
            "context": """Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions.""",
            "expected": ["University of Washington", "University of Texas"],
            "not_expected": [],
            "answer_type": "list"
        },
        {
            "name": "Ledger Token Information (Real-World)",
            "query": "What do you know about the ledger token",
            "context": """Ensures that all AI-driven decisions are traceable and auditable, allowing businesses to understand how insights are generated. Bias Mitigation Algorithms – Reduces the risk of data-driven bias by continuously refining AI models to reflect fair, transparent, and ethical decision-making. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 16 8. Differentiated Revenue Model Tokenized AI Access & Sustainable Growth LedgerAI is revolutionizing the way enterprises access AI-powered business intelligence by introducing a tokenized revenue model that aligns economic incentives with the adoption and growth of our ecosystem. Unlike traditional SaaS subscription models or one-time software licensing fees, LedgerAI's approach ensures a dynamic and self-sustaining AI economy powered by $LEDGER, an ERC-20 token designed for frictionless access to computing power, AI-driven insights, and hardware integration. How It Works: The Future of AI Monetization 1. Tokenized AI Access – Businesses purchase $LEDGER tokens either directly from LedgerAI or on the open market to access compute power, AI services, and infrastructure. This creates a protected economy where $LEDGER tokens fuels platform adoption while maintaining liquidity. 2. AI Compute Marketplace – Instead of static software fees, enterprises allocate $LEDGER tokens toward AI processing power, dynamically scaling their usage based on real-time business needs – similar to how companies purchase AWS credits for cloud computing. 3. Hardware & Software Integration – The LedgerAI ecosystem extends beyond software, with dedicated hardware (including AuraVision's Hammerhead) requiring $LEDGER tokens for activation, secure processing, and on-premises AI acceleration. Revolutionary, not merely Evolutionary Unlike traditional business models where users pay subscription fees, LedgerAI introduces a decentralized AI monetization structure, benefitting and incentivizing LedgerAI Quantum Corporation, token holders, clients, and the broader AI and digital currency communities. Built-In Demand Creation – Every business adopting AuraVision purchases and holds $LEDGER tokens to access AuraVision services, creating organic demand for the token. Market Stabilization & Treasury Management – LedgerAI can also sell collected tokens back to the market in a strategic and controlled manner to maintain liquidity, fund further development and innovation, and expand ecosystem partnerships without relying on external capital raises. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 17 A Self-Sustaining AI Economy LedgerAI's model is self-sustaining. Clients use the $LEDGER tokens to access AuraVision services and features fueling the ecosystem, while mechanisms such as buybacks and revenue sharing drive long-term sustainability and growth. LedgerAI's strategy offers a forward-looking approach beyond outdated pay-as-you-go AI models, licensing fees, or static SaaS subscriptions. The $LEDGER token system will provide LedgerAI with a competitive advantage and allow LedgerAI to continue to scale globally while maintaining token-based value accrual. By aligning incentives across enterprises, investors, and token holders, LedgerAI is pioneering and revolutionizing the future of AI monetization, where access to intelligence is not only a service but an integrated economic model that ensures long-term positive value for all participants.
---
can no longer be bound by outdated, reactive decision-making models. Instead, they must harness the power and innovations of AuraVision to anticipate challenges, identify opportunities, and execute data-driven strategies in real time, which provides an unparalleled and significant competitive advantage to businesses, entrepreneurs, government entities, and organizations across the globe. By integrating AI analytics, blockchain transparency, and decentralized intelligence, LedgerAI delivers unparalleled insights that empower organizations to remain agile, compliant, and competitive. AuraVision transforms financial governance, risk assessment, and strategic planning from burdensome, manual processes into automated, intelligent, and continuously evolving systems. The Ledger ERC-20 Token ($LEDGER) extends these capabilities by delivering tokenized access to AI-driven insights, enabling seamless integration across a variety of industries and establishing a viable and scalable ecosystem for enterprise intelligence. As businesses require greater transparency, security, and efficiency, AuraVision is ready now to aggressively meet those needs with its future-proof and innovative AI solution. Beyond AI—A Movement Toward Intelligent Governance LedgerAI is much more than an AI company – LedgerAI offers a movement toward a more intelligent and capable real-time and intuitive business ecosystem. By bridging the gap between AI, blockchain, and real-time analytics, LedgerAI is redefining and reinventing enterprise intelligence while setting the global benchmark and global standard for how businesses, government organizations, and entrepreneurs interact with and utilize, significant amounts of critical data. As AI continues to develop and rapidly expand its role in governance, financial operations, and decision-making, organizations that embrace AuraVision will be poised to become leaders in their respective industries. LedgerAI's innovation, software, hardware, and ongoing Research and Development activities, as well as its new method of data compilation and analysis is paving the way for the next era of AI-driven governance, strategy, and enterprise intelligence. For partnerships, inquiries, or more information, please contact info@ledgerai.co. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 22 12. LedgerAI Founders & Advisors The co-founders of LedgerAI bring an unmatched depth of experience across AI, blockchain, finance, and enterprise technology, forging a team built to redefine the future of intelligent business solutions. With expertise spanning market-making, regulatory compliance, high-frequency trading, and AI-powered decision intelligence, this elite group has engineered global financial systems, launched cutting-edge platforms, and pioneered AI-driven innovation at scale. Their diverse backgrounds – spanning fintech, enterprise AI, government, and digital assets – position LedgerAI at the forefront of AI-driven business solutions, ensuring a multifaceted approach to governance, strategy, and automation. Backed by a dynamic team of experts across engineering, legal, business development, and marketing, LedgerAI is not only shaping the future of AI-powered intelligence but also ensuring its seamless integration into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations.""",
            "expected": ["$LEDGER", "ERC-20 token", "tokenized access", "AI-driven insights"],
            "not_expected": [],
            "answer_type": "list"
        },
        {
            "name": "Benefits of Localized AI (Real-World)",
            "query": "What are the benefits of localized?",
            "context": """is spread across a variety of disparate systems, increasing exposure to breaches, compliance violations, and regulatory scrutiny. Without a unified, real-time, and AI-driven approach, organizations are forced to make decisions with limited visibility and outdated information, increasing their risk exposure and diminishing their ability to respond proactively to market shifts. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 7 Outdated Business Intelligence Approaches Many enterprises continue to rely on legacy reporting systems, static spreadsheets, and manual reconciliation processes, which are inherently slow and reactive as opposed to predictive. These outdated systems and processes result in: Delayed decision-making – Slow data processing which negatively impacts market positioning and weakens a company's competitive advantage. Reactive governance models – Without the benefit of cutting-edge predictive information, organizations struggle to anticipate risks and opportunities, responding only after challenges arise. Lack of predictive insights – Businesses fail to benefit from AI-driven forecasting and automation to optimize strategies, mitigate risks, and drive operational efficiency. The Shift to Local AI & Decentralized Intelligence: Compliance, Privacy & Security Measures As data privacy regulations tighten and security threats become even more insidious, organizations can no longer afford to rely solely on cloud-dependent AI solutions. In response, LedgerAI has developed a powerful security protocol that leverages local AI processing power with the Hammerhead local hardware device that integrates blockchain-backed encryption with decentralized intelligence. On-Premises AI Processing – AuraVision operates within the Aura Network, running locally on secure business hardware. This ensures that sensitive data never leaves the organization's premises, eliminating reliance on centralized data processors. Decentralized AI Infrastructure – AuraVision leverages blockchain encryption to enhance security, automate compliance, and protect against data loss. Self-Destruct & Recovery Mechanism – If an AuraVision hardware device is lost, stolen, or compromised, it can self-destruct, rendering it useless to unauthorized parties. Importantly, businesses can securely recover data through Hammerhead's blockchain encryption, ensuring continuity without exposing sensitive information. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 8 3. The Evolution of AI-Driven Decision-Making and AuraVision's AI-Powered Strategic Advisor From Reactive to Proactive Intelligence Traditional business intelligence and data analytics systems have been reactive, analyzing past data to generate reports and insights after key events have already occurred. This retrospective approach limits an organization's ability to anticipate challenges, identify opportunities and risks, and act with immediacy and agility. AI-driven decision-making revolutionizes the C-Suite and management process, enabling businesses to transition from reactive analysis to cutting-edge predictive and autonomous intelligence. AuraVision is the future made manifest today, delivering real-time intelligence, combining local AI processing, blockchain security, and decentralized intelligence to reveal actionable insights the moment they are needed. The Rise of Local AI and Decentralized Intelligence Businesses continue to generate and rely upon unprecedented volumes of data. Relying on cloud-based AI solutions presents several notable and important drawbacks and limitations, including latency issues, security risks, and compliance challenges.""",
            "expected": ["On-Premises AI Processing", "data never leaves premises", "blockchain encryption", "self-destruct recovery mechanism"],
            "not_expected": [],
            "answer_type": "list"
        },
        {
            "name": "TechCorp Co-Founders (3 co-founders)",
            "query": "Who are the co-founders of TechCorp?",
            "context": """John Smith is the CEO and Co-Founder of TechCorp. Sarah Johnson is the Chief Technology Officer at TechCorp. Michael Brown is the Co-Founder and Chief Operating Officer of TechCorp. Emily Davis is the Head of Marketing. Robert Wilson is an External Advisor. David Martinez is the Co-Founder and Chief Product Officer. Lisa Anderson is the Chief Financial Officer.""",
            "expected": ["John Smith", "Michael Brown", "David Martinez"],
            "not_expected": ["Sarah Johnson", "Emily Davis", "Robert Wilson", "Lisa Anderson"],
            "answer_type": "person"
        },
        {
            "name": "No Co-Founders Explicitly Stated",
            "query": "Who are the co-founders of Acme Corporation?",
            "context": """Acme Corporation was established in 2010 by a group of entrepreneurs. James Wilson serves as the Chief Executive Officer of Acme Corporation. Maria Garcia is the Chief Technology Officer of Acme Corporation. Thomas Lee is the Head of Sales.""",
            "expected": [],
            "not_expected": ["James Wilson", "Maria Garcia", "Thomas Lee"],
            "answer_type": "person"
        },
        {
            "name": "Single Co-Founder",
            "query": "Who are the co-founders of CloudScale Technologies?",
            "context": """Jennifer Park is the President and Co-Founder of CloudScale Technologies. Mark Thompson is the Chief Technology Officer at CloudScale Technologies.""",
            "expected": ["Jennifer Park"],
            "not_expected": ["Mark Thompson"],
            "answer_type": "person"
        },
        {
            "name": "Location Query - Headquarters",
            "query": "Where is TechCorp's headquarters located?",
            "context": """TechCorp was founded in 2015 in San Francisco. The company moved to New York in 2020. The headquarters is now located at 123 Tech Street, New York, NY. The company also has offices in London and Tokyo.""",
            "expected": ["123 Tech Street, New York, NY", "New York, NY"],
            "not_expected": ["San Francisco", "London", "Tokyo"],
            "answer_type": "location"
        },
        {
            "name": "Role Query - CTO",
            "query": "Who is the CTO of DataFlow?",
            "context": """Sarah Johnson is the Chief Technology Officer at DataFlow. She has been with the company since 2018. Mark Williams is the Head of Engineering. He joined in 2020. Robert Kim is the Chief Financial Officer.""",
            "expected": ["Sarah Johnson"],
            "not_expected": ["Mark Williams", "Robert Kim"],
            "answer_type": "person"
        },
        {
            "name": "Date Query - Establishment",
            "query": "When was TechCorp established?",
            "context": """TechCorp was established in 2015. The company moved to New York in 2020. Registration opens on February 1, 2024. The product launch event is scheduled for March 15, 2024.""",
            "expected": ["2015"],
            "not_expected": ["2020", "February 1, 2024", "March 15, 2024"],
            "answer_type": "date"
        },
        {
            "name": "Products Query",
            "query": "What products does CloudScale offer?",
            "context": """CloudScale offers three main products: CloudScale Compute, CloudScale Storage, and CloudScale Analytics. The company also provides consulting services. CloudScale Support is available for enterprise customers.""",
            "expected": ["CloudScale Compute", "CloudScale Storage", "CloudScale Analytics"],
            "not_expected": ["consulting services", "CloudScale Support"],
            "answer_type": "list"
        },
        {
            "name": "Team Size Query",
            "query": "How many employees are in the engineering team?",
            "context": """The engineering team consists of 50 developers. The marketing team has 20 members. The sales team includes 30 people. Total employees: 100.""",
            "expected": ["50", "50 developers"],
            "not_expected": ["20", "30", "100"],
            "answer_type": "number"
        },
        {
            "name": "Mission Statement Query",
            "query": "What is TechCorp's mission?",
            "context": """TechCorp's mission is to democratize AI technology. Their vision is to make AI accessible to everyone. The core values include innovation, transparency, and customer focus.""",
            "expected": ["democratize AI technology"],
            "not_expected": ["make AI accessible to everyone", "innovation", "transparency"],
            "answer_type": "text"
        },
        {
            "name": "Revenue Query",
            "query": "What was DataFlow's revenue in 2023?",
            "context": """The annual revenue for 2023 was $50 million. The company raised $10 million in Series A funding in 2022. Operating expenses totaled $35 million in 2023.""",
            "expected": ["$50 million", "50 million"],
            "not_expected": ["$10 million", "$35 million"],
            "answer_type": "number"
        },
        {
            "name": "Programming Languages Query",
            "query": "Which programming languages does CloudScale support?",
            "context": """CloudScale supports Python, JavaScript, and Go programming languages. The API is available in REST and GraphQL formats. Documentation is available in English, Spanish, and French.""",
            "expected": ["Python", "JavaScript", "Go"],
            "not_expected": ["REST", "GraphQL", "English", "Spanish", "French"],
            "answer_type": "list"
        },
        {
            "name": "Office Locations Query",
            "query": "Where are TechCorp's offices located?",
            "context": """TechCorp has offices in San Francisco, New York, London, and Tokyo. The San Francisco office is the headquarters. The New York office opened in 2019.""",
            "expected": ["San Francisco", "New York", "London", "Tokyo"],
            "not_expected": [],
            "answer_type": "list"
        }
    ]

# ============================================================================
# Evaluation Helpers
# ============================================================================

def check_cot_reasoning(response: str) -> Tuple[bool, List[str]]:
    """Checks if response contains Chain of Thought reasoning markers"""
    cot_indicators = [
        "REASONING:",
        "Item:",
        "Evidence:",
        "Action:",
        "Name:",
        "Role:",
        "[KEEP]",
        "[DISCARD]",
        "KEEP",
        "DISCARD",
        "[KEEP_ARRAY]",
        "[DISCARD_ARRAY]",
        "KEEP_ARRAY",
        "DISCARD_ARRAY",
        "Added to",
        "End of scan",
        "- End of scan.",
        "FINAL ANSWER:"
    ]
    
    response_upper = response.upper()
    found_indicators = [ind for ind in cot_indicators if ind.upper() in response_upper]
    
    # Also check for implicit reasoning (shows analysis, multiple items, evidence quotes)
    has_analysis = any([
        "Evidence:" in response or '"' in response,  # Has quoted evidence
        "Action:" in response or "[KEEP]" in response or "[DISCARD]" in response,  # Has action markers
        len([w for w in response.split() if w[0].isupper() and len(w) > 2]) >= 3  # Multiple capitalized items
    ])
    
    has_explicit_cot = len(found_indicators) >= 3
    has_implicit_reasoning = has_analysis and len(found_indicators) >= 1
    
    return has_explicit_cot or has_implicit_reasoning, found_indicators

def test_scenario(model, tokenizer, scenario, model_type='transformers'):
    """Test a single scenario - supports both GGUF (llama_cpp) and HuggingFace/Unsloth models"""
    print(f"\n{'='*80}")
    print(f"Testing: {scenario['name']}")
    print(f"{'='*80}")
    
    # Simple direct prompt relying on training
    user_prompt = f"Knowledge context: {scenario['context']}\n---\nQuestion: {scenario['query']}"
    
    # Format messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Generate response
    try:
        if model_type == 'gguf':
            # GGUF model using llama_cpp
            # Use create_chat_completion for proper chat format handling
            # This avoids early stopping issues with raw model() calls
            
            # Estimate input length from messages
            input_length = len(SYSTEM_PROMPT.split()) + len(user_prompt.split())
            print(f"   📏 Input length: ~{input_length} tokens (estimated)")
            
            # Generate with GGUF model using chat completion API
            # OPTIMIZED FOR LATENCY: Using n_ctx=16384 with max_tokens=8192 for fast processing
            # Input is ~2,365 tokens (for 4 co-founder example), so we can generate up to ~5,827 tokens safely
            # Typical output: ~800 tokens, but max_tokens=8192 provides 5x buffer for complex reasoning
            # This prevents truncation while optimizing for speed (smaller context = faster processing)
            GGUF_MAX_TOKENS = 4096  # Latency-optimized: typical ~800 tokens, but allow up to 4K for complex cases
            
            print(f"   📏 Max output tokens: {GGUF_MAX_TOKENS} (context window: 8192)")
            print(f"   📏 Available for generation: ~{8192 - int(input_length)} tokens (after input)")
            print(f"   ⚡ Latency-optimized: smaller context window = faster processing")
            
            # Generate with GGUF model using chat completion (handles chat format properly)
            # IMPORTANT: 
            # - Use create_chat_completion which properly handles Qwen chat format
            # - Don't use stop tokens to avoid premature stopping
            # - Let the model complete naturally or hit max_tokens limit
            # - Use very low temperature for deterministic, complete responses
            # Enhance system prompt to explicitly require FINAL ANSWER completion
            # This helps prevent early stopping before FINAL ANSWER is generated
            enhanced_system_prompt = SYSTEM_PROMPT + "\n\nCRITICAL: You MUST complete your response with a FINAL ANSWER section. Do NOT stop before generating FINAL ANSWER. Continue generating until you have provided the complete FINAL ANSWER based on all [KEEP] items from your reasoning."
            enhanced_messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                # Try create_chat_completion first (proper chat format handling)
                output = model.create_chat_completion(
                    messages=enhanced_messages,
                    max_tokens=GGUF_MAX_TOKENS,  # Maximum tokens for complete reasoning + FINAL ANSWER
                    temperature=0,  # Temperature=0 for fully deterministic inference (greedy decoding)
                    top_p=0.95,
                    repeat_penalty=1.2,
                    stop=None,  # Use None instead of [] - some llama_cpp versions handle this differently
                    stream=False
                )
                
                assistant_response = output['choices'][0]['message']['content'].strip()
            except TypeError:
                # Some llama_cpp versions might not accept stop=None, try without stop parameter
                try:
                    output = model.create_chat_completion(
                        messages=enhanced_messages,
                        max_tokens=GGUF_MAX_TOKENS,
                        temperature=0,  # Temperature=0 for fully deterministic inference
                        top_p=0.95,
                        repeat_penalty=1.2,
                        stream=False
                        # Don't pass stop parameter at all - let it use defaults
                    )
                    assistant_response = output['choices'][0]['message']['content'].strip()
                except Exception as e:
                    # Fallback to raw model() call if create_chat_completion fails
                    print(f"   ⚠️  Warning: create_chat_completion failed, using raw model() call: {e}")
                    # Format prompt for Qwen chat format manually
                    formatted_prompt = f"<|im_start|>system\n{enhanced_system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    output = model(
                        formatted_prompt,
                        max_tokens=GGUF_MAX_TOKENS,
                        temperature=0,  # Temperature=0 for fully deterministic inference
                        top_p=0.95,
                        repeat_penalty=1.2,
                        stop=None,  # Try None instead of []
                        echo=False
                    )
                    assistant_response = output['choices'][0]['text'].strip()
            except Exception as e:
                # Fallback to raw model() call if create_chat_completion fails
                print(f"   ⚠️  Warning: create_chat_completion failed, using raw model() call: {e}")
                import traceback
                traceback.print_exc()
                # Format prompt for Qwen chat format manually
                formatted_prompt = f"<|im_start|>system\n{enhanced_system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                output = model(
                    formatted_prompt,
                    max_tokens=GGUF_MAX_TOKENS,
                    temperature=0,  # Temperature=0 for fully deterministic inference
                    top_p=0.95,
                    repeat_penalty=1.2,
                    stop=None,  # Try None instead of []
                    echo=False
                )
                assistant_response = output['choices'][0]['text'].strip()
            
            # Check response completeness and truncation
            # Handle both create_chat_completion and raw model() response formats
            if 'usage' in output:
                response_tokens_used = output.get('usage', {}).get('completion_tokens', 0)
                response_finished = output['choices'][0].get('finish_reason', 'unknown')
            else:
                # Raw model() response format
                response_tokens_used = output.get('usage', {}).get('completion_tokens', len(assistant_response.split()))
                response_finished = output['choices'][0].get('finish_reason', 'unknown')
            
            print(f"   📊 Generation stats:")
            print(f"      - Tokens generated: {response_tokens_used}")
            print(f"      - Finish reason: {response_finished}")
            print(f"      - Response length: {len(assistant_response)} characters")
            
            # Check if response was truncated
            if response_finished == 'length':
                print(f"   ⚠️  WARNING: Response hit max_tokens limit ({GGUF_MAX_TOKENS}) - may be truncated")
                print(f"   💡 Consider increasing max_tokens if FINAL ANSWER is missing")
            elif response_finished in ['stop', 'eos']:
                print(f"   ✅ Response finished naturally (reason: {response_finished})")
            
            # Check for FINAL ANSWER section
            if "FINAL ANSWER" not in assistant_response and len(assistant_response) < 1000:
                print(f"   ⚠️  WARNING: Response seems incomplete (no FINAL ANSWER, short length)")
                if "End of scan" in assistant_response:
                    print(f"   ⚠️  WARNING: Response has 'End of scan' but missing 'FINAL ANSWER' - likely truncated")
            
        else:
            # HuggingFace/Unsloth model
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = tokenizer(
                    formatted_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                ).to(model.device)
            else:
                prompt_text = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
                inputs = tokenizer(
                    prompt_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=3000
                ).to(model.device)
            
            input_length = inputs['input_ids'].shape[1]
            print(f"   📏 Input length: {input_length} tokens")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0,  # Temperature=0 for fully deterministic inference
                    do_sample=False,  # Disable sampling for greedy decoding (temperature=0 requires do_sample=False)
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response from the chat format (remove system/user parts)
            assistant_marker = "assistant\n"
            if assistant_marker in response:
                assistant_response = response.split(assistant_marker)[-1].strip()
            else:
                # Fallback for non-chat templates
                assistant_response = response.strip()
        
        # Print full response for debugging (truncated display only)
        response_preview = assistant_response[:800] if len(assistant_response) > 800 else assistant_response
        print(f"\n📝 Model Response (first 800 chars):\n{response_preview}...")
        print(f"\n📏 Full response length: {len(assistant_response)} characters")
        if "FINAL ANSWER" in assistant_response:
            print(f"   ✅ FINAL ANSWER section found")
        else:
            print(f"   ⚠️  FINAL ANSWER section NOT found in response")
            # Try to find where it was cut off
            if "End of scan" in assistant_response:
                end_of_scan_idx = assistant_response.find("End of scan")
                print(f"   ⚠️  Last complete section ends at: 'End of scan' (char {end_of_scan_idx})")
                print(f"   💡 Response may be truncated - consider increasing max_tokens")
        
        # Check for CoT
        has_cot, indicators = check_cot_reasoning(assistant_response)
        cot_status = "✅ CoT reasoning detected" if has_cot else "⚠️  Explicit CoT reasoning NOT detected"
        print(f"\n🧠 CoT Reasoning Check:\n   {cot_status} (found indicators: {indicators})")
        
        
        # CLEAN RESPONSE: Extract FINAL ANSWER section (only text after FINAL ANSWER header counts)
        temp_response = assistant_response.strip()
        if temp_response.startswith('t'): temp_response = temp_response[1:].strip()
        
        clean_response = ""
        has_final_answer_section = False
        
        # Check for FINAL ANSWER marker (case-insensitive, with or without colon)
        if "FINAL ANSWER:" in temp_response:
            clean_response = temp_response.split("FINAL ANSWER:")[-1].strip()
            has_final_answer_section = True
        elif "Final Answer:" in temp_response:
            clean_response = temp_response.split("Final Answer:")[-1].strip()
            has_final_answer_section = True
        elif "FINAL ANSWER" in temp_response:
            # Check if FINAL ANSWER appears without colon (might be truncated)
            final_idx = temp_response.find("FINAL ANSWER")
            clean_response = temp_response[final_idx + len("FINAL ANSWER"):].strip()
            # Remove colon if present
            if clean_response.startswith(':'):
                clean_response = clean_response[1:].strip()
            has_final_answer_section = True
        elif "- End of scan." in temp_response or "End of scan." in temp_response:
            # If FINAL ANSWER is missing but End of scan is there, check after it
            end_marker = "- End of scan." if "- End of scan." in temp_response else "End of scan."
            text_after_scan = temp_response.split(end_marker)[-1].strip()
            # Only use if it's substantial (not just whitespace)
            if len(text_after_scan) > 10:
                clean_response = text_after_scan
                print(f"   ⚠️  FINAL ANSWER marker missing - using text after 'End of scan'")
            else:
                # FINAL ANSWER section was truncated - extract from reasoning as fallback
                print(f"   ⚠️  FINAL ANSWER section appears to be missing/truncated")
                print(f"   💡 Attempting to extract from REASONING section as fallback...")
                # Fall through to reasoning extraction
        else:
            # No FINAL ANSWER found - try to extract from reasoning
            print(f"   ⚠️  FINAL ANSWER section NOT FOUND in response")
            print(f"   💡 Response may be truncated - extracting from reasoning section...")
        
        # If clean_response is empty or very short, try to extract from reasoning
        if not clean_response or len(clean_response) < 10:
            # Extract items marked [KEEP] from reasoning as fallback
            keep_items = re.findall(r'- Item:\s*([^\n]+?)\s*- Action:\s*\[KEEP\]', assistant_response, re.IGNORECASE | re.MULTILINE)
            if keep_items:
                clean_response = ", ".join([item.strip() for item in keep_items])
                print(f"   ⚠️  Using fallback: Extracted {len(keep_items)} [KEEP] items from REASONING")
            
        # Remove all [KEEP] and [DISCARD] noise from the extracted answer
        clean_response = re.sub(r'\[(KEEP|DISCARD|Action|Result)\]', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'(?m)^- .*$', '', clean_response).strip() # Remove bulleted lines
        
        # Get expected values early (needed for extraction logic)
        expected = scenario.get('expected', [])
        if not isinstance(expected, list):
            expected = []
        not_expected = scenario.get('not_expected', [])
        if not isinstance(not_expected, list):
            not_expected = []
        
        # Extract items based on answer type
        answer_type = scenario.get('answer_type', 'person')  # Default to person/name extraction
        found_items = []
        
        if answer_type == 'person':
            # Extract person names
            name_patterns = [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][A-Za-z]+)+)\b',
            ]
            for pattern in name_patterns:
                matches = re.findall(pattern, clean_response)
                for match in matches:
                    if match not in found_items:
                        # Filter out non-name phrases
                        if not any(x in match.lower() for x in ["ledger", "corporation", "technologies", "systems", "innovations", "solutions", "chief", "officer", "founder", "manager", "lead", "ambassador"]):
                            found_items.append(match)
        elif answer_type == 'location':
            # Extract locations (addresses, cities, etc.)
            # First, try to find expected locations mentioned in the response
            for exp_loc in expected:
                exp_lower = exp_loc.lower()
                if exp_lower in clean_response.lower():
                    # Try to find exact match or close match
                    pattern = re.escape(exp_loc)
                    if re.search(pattern, clean_response, re.IGNORECASE):
                        if exp_loc not in found_items:
                            found_items.append(exp_loc)
            
            # Also extract using patterns
            location_patterns = [
                r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',  # Full address
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',  # City, State
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # City names (multi-word)
            ]
            for pattern in location_patterns:
                matches = re.findall(pattern, clean_response)
                for match in matches:
                    match_lower = match.lower()
                    # Filter out common non-location words
                    if not any(x in match_lower for x in ["techcorp", "cloudscale", "dataflow", "acme", "the", "and", "or"]):
                        if match not in found_items:
                            found_items.append(match)
        elif answer_type == 'date':
            # Extract dates (years, full dates)
            # First, try to find expected dates
            for exp_date in expected:
                if exp_date.lower() in clean_response.lower():
                    if exp_date not in found_items:
                        found_items.append(exp_date)
            
            # Also extract using patterns
            date_patterns = [
                r'\b(19|20)\d{2}\b',  # Years (captures full 4-digit year)
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',  # Full dates
            ]
            for pattern in date_patterns:
                matches = re.findall(pattern, clean_response)
                # For year pattern, reconstruct full year from groups
                if pattern == r'\b(19|20)\d{2}\b':
                    # Find all 4-digit years
                    year_matches = re.findall(r'\b(19|20)(\d{2})\b', clean_response)
                    for century, year in year_matches:
                        full_year = century + year
                        if full_year not in found_items:
                            found_items.append(full_year)
                else:
                    for match in matches:
                        if isinstance(match, tuple):
                            match = ' '.join(match)
                        if match not in found_items:
                            found_items.append(match)
        elif answer_type == 'number':
            # Extract numbers (with optional units)
            number_patterns = [
                r'\$\d+(?:,\d{3})*(?:\s+million|\s+billion)?',  # Money
                r'\b\d+(?:\s+developers?|\s+employees?|\s+members?)?\b',  # Counts with units
            ]
            for pattern in number_patterns:
                matches = re.findall(pattern, clean_response)
                found_items.extend(matches)
        elif answer_type == 'list':
            # Extract list items (product names, languages, locations, etc.)
            # Strategy: Look for the expected items in the response, plus any capitalized phrases
            # First, try to find expected items mentioned in the response
            for exp_item in expected:
                exp_lower = exp_item.lower()
                # Check if the expected item appears in the response (case-insensitive)
                if exp_lower in clean_response.lower():
                    # Try to find the exact phrase or close match
                    pattern = re.escape(exp_item)
                    if re.search(pattern, clean_response, re.IGNORECASE):
                        found_items.append(exp_item)
            
            # Also extract capitalized phrases that might be list items
            list_patterns = [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Multi-word capitalized phrases (e.g., "San Francisco", "CloudScale Storage")
                r'\b([A-Z][a-z]+)\b',  # Single capitalized words (e.g., "Python", "Go")
            ]
            for pattern in list_patterns:
                matches = re.findall(pattern, clean_response)
                for match in matches:
                    match_lower = match.lower()
                    # Filter out common words and already-found items
                    if not any(x in match_lower for x in ["the", "and", "or", "are", "is", "was", "were", "company", "corporation", "techcorp", "cloudscale", "dataflow", "acme"]):
                        if match not in found_items:
                            # Only add if it's a reasonable length
                            if len(match.split()) <= 4:
                                found_items.append(match)
        elif answer_type == 'text':
            # For text answers, extract key phrases (mission statements, etc.)
            # Just use the full clean response
            found_items = [clean_response]
        else:
            # Default: extract any capitalized phrases
            name_patterns = [r'\b([A-Z][a-z]+(?:\s+[A-Z][A-Za-z]+)+)\b']
            for pattern in name_patterns:
                matches = re.findall(pattern, clean_response)
                found_items.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        found_items = [x for x in found_items if not (x in seen or seen.add(x))]
        
        item_label = "Items" if answer_type != 'person' else "Names"
        print(f"\n{item_label} Found in Final Answer: {found_items}")
        print(f"   (Extracted from final answer only, excluding reasoning steps)")
        
        # Scoring (expected and not_expected already defined above)
        
        # Match found items to expected (case-insensitive, partial matching)
        correctly_found = []
        for exp in expected:
            for found in found_items:
                # Normalize for comparison
                exp_norm = exp.lower().strip()
                found_norm = found.lower().strip()
                if exp_norm in found_norm or found_norm in exp_norm:
                    correctly_found.append(exp)
                    break
        
        incorrectly_included = []
        for found in found_items:
            found_norm = found.lower().strip()  # Define here to avoid scoping issues
            is_expected = False
            for exp in expected:
                exp_norm = exp.lower().strip()
                if exp_norm in found_norm or found_norm in exp_norm:
                    is_expected = True
                    break
            if not is_expected:
                # Check if it's in not_expected list
                is_forbidden = False
                for forbidden in not_expected:
                    forbidden_norm = forbidden.lower().strip()
                    if forbidden_norm in found_norm or found_norm in forbidden_norm:
                        is_forbidden = True
                        break
                if is_forbidden or (not_expected and found_norm not in [e.lower() for e in expected]):
                    incorrectly_included.append(found)
        
        missing = [n for n in expected if n not in correctly_found]
        
        # ========================================================================
        # CRITICAL: Check for DISCARD violations and reasoning errors
        # ========================================================================
        
        # Extract DISCARD items from reasoning section
        reasoning_section = ""
        final_answer_section_for_check = clean_response if clean_response else assistant_response
        
        if "FINAL ANSWER:" in assistant_response:
            parts = assistant_response.split("FINAL ANSWER:", 1)
            reasoning_section = parts[0].strip()
            final_answer_section_for_check = parts[1].strip() if len(parts) > 1 else ""
        elif "Final Answer:" in assistant_response:
            parts = assistant_response.split("Final Answer:", 1)
            reasoning_section = parts[0].strip()
            final_answer_section_for_check = parts[1].strip() if len(parts) > 1 else ""
        else:
            # No FINAL ANSWER marker - entire response is reasoning (problem)
            reasoning_section = assistant_response
            final_answer_section_for_check = ""
        
        # Extract DISCARD items from reasoning using better pattern matching
        discard_items_from_reasoning = []
        lines = reasoning_section.split('\n')
        current_item = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Check for Item line
            if '- Item:' in line or 'Item:' in line:
                # Extract item name
                item_match = re.search(r'[-\s]*Item:\s*([^\n-]+?)(?:\s*[-]|\s*Evidence|\s*$)', line, re.IGNORECASE)
                if item_match:
                    current_item = item_match.group(1).strip()
                    # Clean up item (remove trailing punctuation)
                    current_item = re.sub(r'[.,;:]$', '', current_item).strip()
            # Check for Action line with DISCARD
            elif current_item and '[DISCARD]' in line and '[KEEP]' not in line:
                if current_item and current_item not in discard_items_from_reasoning:
                    discard_items_from_reasoning.append(current_item)
                current_item = None
            # Check for Action line with KEEP
            elif current_item and '[KEEP]' in line:
                current_item = None
            # Reset on end markers
            elif 'End of scan' in line or 'FINAL ANSWER' in line:
                current_item = None
        
        # Check if any DISCARD items appear in FINAL ANSWER (violation)
        discard_violations = []
        if final_answer_section_for_check and discard_items_from_reasoning:
            for discard_item in discard_items_from_reasoning:
                # Check if discard item (or key parts) appears in final answer
                item_lower = discard_item.lower()
                final_answer_lower = final_answer_section_for_check.lower()
                
                # Split item into parts for better matching
                item_parts = [p for p in item_lower.split() if len(p) > 2]  # Words > 2 chars
                
                if len(item_parts) >= 2:
                    # Multi-word item: check if significant parts appear together
                    # Look for both first and last significant word
                    first_part = item_parts[0]
                    last_part = item_parts[-1]
                    if first_part in final_answer_lower and last_part in final_answer_lower:
                        # Check if they appear close together (within 50 chars suggests same entity)
                        first_idx = final_answer_lower.find(first_part)
                        last_idx = final_answer_lower.find(last_part)
                        if first_idx != -1 and last_idx != -1 and abs(first_idx - last_idx) < 50:
                            # Additional check: make sure it's not part of an expected item
                            is_part_of_expected = False
                            for exp in expected:
                                if item_lower in exp.lower() or exp.lower() in item_lower:
                                    is_part_of_expected = True
                                    break
                            if not is_part_of_expected:
                                discard_violations.append(discard_item)
                elif len(item_parts) == 1 and len(item_parts[0]) > 3:
                    # Single word item - check if it appears as standalone word
                    if re.search(rf'\b{re.escape(item_parts[0])}\b', final_answer_section_for_check, re.IGNORECASE):
                        # But check it's not part of an expected item
                        is_part_of_expected = False
                        for exp in expected:
                            if item_parts[0] in exp.lower() or exp.lower() in item_parts[0]:
                                is_part_of_expected = True
                                break
                        if not is_part_of_expected and item_parts[0] not in ['the', 'and', 'or', 'are', 'is', 'was', 'were', 'no', 'not']:
                            discard_violations.append(discard_item)
        
        # Check for missing [KEEP]/[DISCARD] actions in reasoning
        has_action_markers = '[KEEP]' in reasoning_section or '[DISCARD]' in reasoning_section
        # Check for both singular "Item:" and plural "Items:" formats
        has_items = '- Item:' in reasoning_section or 'Item:' in reasoning_section or '- Items:' in reasoning_section or 'Items:' in reasoning_section
        # Also check for simplified format (list items without proper structure)
        has_simplified_format = ('- Items:' in reasoning_section or 'Items:' in reasoning_section) and not has_action_markers
        missing_actions = (has_items and not has_action_markers and "REASONING" in reasoning_section) or has_simplified_format
        
        # Check for reasoning errors (e.g., item marked as DISCARD when evidence shows it should be KEEP)
        reasoning_errors = []
        
        # General check: look for items marked as DISCARD where evidence contains query keywords
        # This catches cases like "CEO and Co-Founder" marked as DISCARD for co-founder queries
        if "co-founder" in scenario.get('query', '').lower() or "founder" in scenario.get('query', '').lower():
            # Look for items marked as DISCARD where evidence contains "Co-Founder"
            item_pattern = r'- Item:\s*([^\n-]+?)\s*-?\s*Evidence[^\n]*?([^\n]*)\s*-?\s*Action:\s*\[DISCARD\]'
            matches = re.findall(item_pattern, reasoning_section, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for item_name, evidence in matches:
                item_name = item_name.strip()
                evidence_text = evidence.strip()
                # Check if evidence contains "Co-Founder" or "co-founder" but action is DISCARD
                if ("Co-Founder" in evidence_text or "co-founder" in evidence_text or "Co-Founder" in evidence_text) and item_name:
                    # Additional check: make sure it's not a false positive (e.g., "not a co-founder")
                    if "not" not in evidence_text.lower()[:50] and "no" not in evidence_text.lower()[:50]:
                        reasoning_errors.append(f"{item_name} incorrectly marked as [DISCARD] despite evidence showing Co-Founder")
        
        # Specific check for Paul Chou (hardcoded for this common error)
        if "Paul Chou" in reasoning_section:
            # Check if Paul Chou was marked as DISCARD
            paul_discard_pattern = r'Paul Chou.*?\[DISCARD\]'
            if re.search(paul_discard_pattern, reasoning_section, re.IGNORECASE | re.DOTALL):
                # Check if evidence says he's Co-Founder
                paul_section_start = reasoning_section.find("Paul Chou")
                if paul_section_start != -1:
                    paul_section = reasoning_section[max(0, paul_section_start-100):paul_section_start+600]
                    # Check if evidence contains "Co-Founder" or "CEO and Co-Founder"
                    if ("Co-Founder" in paul_section or "co-founder" in paul_section) and "CEO" in paul_section:
                        # Additional check: make sure it's not a negative statement
                        paul_evidence = re.search(r'Evidence[^\n]*?([^\n]{50,300})', paul_section, re.IGNORECASE | re.DOTALL)
                        if paul_evidence:
                            evidence_text = paul_evidence.group(1)
                            if "CEO and Co-Founder" in evidence_text or ("CEO" in evidence_text and "Co-Founder" in evidence_text):
                                reasoning_errors.append("Paul Chou incorrectly marked as [DISCARD] despite being 'CEO and Co-Founder' - he IS a co-founder")
        
        # Calculate base score
        if not expected:
            score = 100.0 if not incorrectly_included and not discard_violations else 0.0
        else:
            # Weighted score: correct - 0.5 * incorrect
            points = len(correctly_found) / len(expected) * 100
            penalty = (len(incorrectly_included) / max(1, len(found_items))) * 50
            score = max(0, points - penalty)
        
        # Apply penalties for violations
        if discard_violations:
            score = max(0, score - (len(discard_violations) * 25))  # Penalty: -25% per violation
        if missing_actions:
            score = max(0, score - 35)  # Penalty: -35% for missing actions
        if reasoning_errors:
            score = max(0, score - (len(reasoning_errors) * 20))  # Penalty: -20% per reasoning error
        
        expected_label = "Expected" if answer_type == 'text' else f"Expected {answer_type.title()}"
        print(f"\n✅ {expected_label}: {expected}")
        if not_expected:
            print(f"❌ Should NOT Include: {not_expected}")
            
        print(f"\n📊 Results:")
        print(f"   ✅ Correctly Found: {correctly_found}")
        if missing:
            print(f"   ⚠️  Missing: {missing}")
        if incorrectly_included:
            print(f"   ❌ Incorrectly Included: {incorrectly_included}")
        
        # Report DISCARD violations
        if discard_violations:
            print(f"\n   ❌ DISCARD VIOLATION: {len(discard_violations)} [DISCARD] item(s) appear in FINAL ANSWER!")
            for violation in discard_violations[:5]:  # Show first 5
                print(f"      - '{violation}' was marked [DISCARD] in reasoning but appears in FINAL ANSWER")
            print(f"      💡 This is a critical error - model is not respecting DISCARD rules")
        
        # Report missing actions
        if missing_actions:
            print(f"\n   ❌ REASONING FORMAT ERROR: Reasoning has items but missing [KEEP]/[DISCARD] actions!")
            print(f"      - Items found in reasoning but no Action markers")
            print(f"      - This indicates the model is not following the expected reasoning format")
        
        # Report reasoning errors
        if reasoning_errors:
            print(f"\n   ❌ REASONING LOGIC ERROR: {len(reasoning_errors)} error(s) detected!")
            for error in reasoning_errors[:3]:  # Show first 3
                print(f"      - {error}")
            print(f"      💡 Model's reasoning is incorrect despite correct FINAL ANSWER")
        
        print(f"   📈 Score: {score:.2f}%")
        
        # Additional diagnostic for "No Co-Founders" test
        if len(expected) == 0 and "no" in scenario.get('query', '').lower() and ("co-founders" in scenario.get('query', '').lower() or "founder" in scenario.get('query', '').lower()):
            # This is a "no co-founders" test - FINAL ANSWER should say no co-founders, not list CEO/CTO
            not_expected_in_answer = [name for name in not_expected if name.lower() in final_answer_section_for_check.lower()]
            if not_expected_in_answer:
                print(f"\n   ❌ INCORRECT FINAL ANSWER: Query asks for co-founders (none exist), but FINAL ANSWER lists CEO/CTO!")
                print(f"      - FINAL ANSWER should state: 'No co-founders' or similar")
                print(f"      - Instead contains: {not_expected_in_answer}")
                print(f"      - FINAL ANSWER excerpt: {final_answer_section_for_check[:250]}...")
                score = max(0, score - 40)  # Heavy penalty for wrong answer structure
        
        # Additional diagnostic for benefits query
        if "benefits" in scenario.get('query', '').lower() and "localized" in scenario.get('query', '').lower():
            # Benefits query should extract benefits, not drawbacks
            if any(drawback in final_answer_section_for_check.lower() for drawback in ["delayed decision-making", "reactive governance", "lack of predictive"]):
                print(f"\n   ❌ INCORRECT CONTENT: Query asks for BENEFITS of localized AI but FINAL ANSWER contains DRAWBACKS!")
                print(f"      - Query: {scenario.get('query', '')}")
                print(f"      - FINAL ANSWER should list benefits (On-Premises, data never leaves, etc.)")
                print(f"      - Instead contains drawbacks (Delayed decision-making, Reactive, etc.)")
                print(f"      - FINAL ANSWER excerpt: {final_answer_section_for_check[:300]}...")
                score = max(0, score - 50)  # Heavy penalty for completely wrong content
        
        return {
            "name": scenario['name'],
            "score": score,
            "has_cot": has_cot,
            "correct": correctly_found,
            "missing": missing,
            "incorrect": incorrectly_included,
            "answer_type": answer_type,
            "discard_violations": discard_violations if 'discard_violations' in locals() else [],
            "missing_actions": missing_actions if 'missing_actions' in locals() else False,
            "reasoning_errors": reasoning_errors if 'reasoning_errors' in locals() else []
        }
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_tests():
    """Main function to load model and run tests"""
    print("=" * 80)
    print("RAG CoT Model Test Suite")
    print("=" * 80)
    
    model = None
    tokenizer = None
    model_type = None  # 'gguf', 'unsloth', or 'transformers'
    
    # First, try to load GGUF model from gguf_model_rag_cot directory
    print(f"\n================================================================================")
    print(f"Checking for GGUF Quantized Model")
    print(f"================================================================================")
    print(f"   Looking for GGUF directory: {GGUF_MODEL_DIR}")
    print(f"   llama-cpp-python available: {HAS_LLAMA_CPP}")
    print(f"   Current working directory: {os.getcwd()}")
    
    gguf_file = None
    
    # Step 1: Find GGUF directory (check multiple possible locations)
    actual_gguf_dir = None
    possible_paths = [
        GGUF_MODEL_DIR,
        "./gguf_model_rag_cot",
        "../gguf_model_rag_cot",
        "gguf_model_rag_cot/",
        os.path.join(os.getcwd(), "gguf_model_rag_cot"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            actual_gguf_dir = path
            print(f"   ✅ Found GGUF directory: {path}")
            break
    
    if not actual_gguf_dir:
        print(f"   ⚠️  GGUF directory not found: {GGUF_MODEL_DIR}")
        print(f"   💡 Checked: {possible_paths}")
        print(f"   💡 Make sure training completed and saved GGUF model to this directory")
    else:
        # Step 2: Check for llama_cpp availability
        if not HAS_LLAMA_CPP:
            print(f"\n   ⚠️  llama-cpp-python not available")
            print(f"   💡 Install with: pip install llama-cpp-python")
            print(f"   ⚠️  Cannot load GGUF format - will try HuggingFace/Unsloth format instead")
        else:
            print(f"   ✅ llama-cpp-python available")
            
            # Step 3: Look for GGUF files (including in subdirectories)
            try:
                gguf_files = []
                # Check root of directory
                if os.path.isdir(actual_gguf_dir):
                    for f in os.listdir(actual_gguf_dir):
                        if f.endswith(".gguf"):
                            gguf_files.append(os.path.join(actual_gguf_dir, f))
                    # Also check subdirectories
                    for root, dirs, files in os.walk(actual_gguf_dir):
                        for f in files:
                            if f.endswith(".gguf"):
                                full_path = os.path.join(root, f)
                                if full_path not in gguf_files:
                                    gguf_files.append(full_path)
                
                if gguf_files:
                    print(f"   ✅ Found {len(gguf_files)} GGUF file(s):")
                    for gf in gguf_files:
                        size_mb = os.path.getsize(gf) / (1024 * 1024)
                        print(f"      - {os.path.basename(gf)} ({size_mb:.2f} MB)")
                    
                    # Prefer file with "rag-cot" in name, then "q4"
                    preferred = [f for f in gguf_files if "rag-cot" in os.path.basename(f).lower()]
                    if not preferred:
                        preferred = [f for f in gguf_files if "q4" in os.path.basename(f).lower() or "q4_" in os.path.basename(f).lower()]
                    if preferred:
                        gguf_file = preferred[0]
                        print(f"   ✅ Selected (preferred): {os.path.basename(gguf_file)}")
                    else:
                        gguf_file = gguf_files[0]
                        print(f"   ✅ Selected: {os.path.basename(gguf_file)}")
                    
                    print(f"\n================================================================================")
                    print(f"Loading GGUF Quantized Model")
                    print(f"================================================================================")
                    print(f"   File: {os.path.basename(gguf_file)}")
                    print(f"   Path: {gguf_file}")
                    file_size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
                    print(f"   Size: {file_size_mb:.2f} MB")
                    print(f"   Loading with llama-cpp-python...")
                    
                    try:
                        # Optimized for latency while avoiding truncation
                        # Analysis: 4 co-founder example needs ~2,365 input + typical ~800 output = ~3,165 tokens
                        # Using n_ctx=16384: input (~2.4K) + max_output (4K) + buffer = 6,400 < 8,192 ✓
                        # This balances latency (smaller context = faster) with safety (won't truncate typical responses)
                        GGUF_N_CTX = 8192  # Optimized: 2.4K input + 4K max output + 1.8K buffer = fast but safe
                        model = Llama(
                            model_path=gguf_file,
                            n_ctx=GGUF_N_CTX,  # 8192 tokens: latency-optimized while avoiding truncation
                            n_gpu_layers=-1,  # Use GPU if available
                            n_threads=4,
                            chat_format="chatml",  # Explicitly set Qwen chat format (chatml) for proper chat handling
                            verbose=False
                        )
                        print(f"   ✅ Context window: {GGUF_N_CTX} tokens (latency-optimized: avoids truncation)")
                        print(f"   ✅ Chat format: chatml (Qwen format)")
                        print(f"   💡 Optimized for speed: smaller context = faster processing + lower memory")
                        model_type = 'gguf'
                        print(f"   ✅ GGUF model loaded successfully!")
                        print(f"   ✅ Model format: GGUF (Q4_K_M quantization)")
                    except Exception as e:
                        print(f"   ❌ Failed to load GGUF model: {e}")
                        import traceback
                        traceback.print_exc()
                        gguf_file = None
                        model = None
                else:
                    print(f"   ⚠️  No .gguf files found in {actual_gguf_dir}")
                    print(f"   💡 Checked root and all subdirectories")
                    print(f"   💡 Make sure the GGUF model was saved during training")
            except Exception as e:
                print(f"   ❌ Error reading GGUF directory: {e}")
                import traceback
                traceback.print_exc()
    
    # Fallback to Unsloth/HuggingFace format if GGUF not found or failed to load
    if model is None:
        print(f"\n================================================================================")
        if not os.path.exists(MODEL_PATH):
            print(f"\n❌ ERROR: Neither GGUF model nor HuggingFace model found!")
            print(f"   Checked:")
            print(f"   - GGUF: {GGUF_MODEL_DIR}")
            print(f"   - HuggingFace: {MODEL_PATH}")
            return
        
        print(f"\n================================================================================")
        print(f"Loading Trained Model (HuggingFace/Unsloth format)")
        print(f"================================================================================")
        print(f"✅ Found model at: {MODEL_PATH}")
        print(f"   Attempting to load model...")
        
        if HAS_UNSLOTH:
            # Load with explicit 4-bit quantization
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=MODEL_PATH,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    quantization_config=quantization_config,
                    load_in_4bit=True,  # Explicitly enable 4-bit quantization
                )
            except Exception as e:
                # Fallback to simpler loading if BitsAndBytesConfig fails
                print(f"   ⚠️  Warning: Could not use BitsAndBytesConfig, using standard 4-bit loading: {e}")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=MODEL_PATH,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    load_in_4bit=True,  # Explicitly enable 4-bit quantization
                )
            
            FastLanguageModel.for_inference(model)
            model_type = 'unsloth'
            
            # Verify quantization
            print(f"   ✅ Model loaded with 4-bit quantization enabled")
            # Check quantization status
            try:
                if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
                    print(f"   ✅ 4-bit quantization confirmed: {model.is_loaded_in_4bit}")
                elif hasattr(model, 'quantization_config') and model.quantization_config:
                    print(f"   ✅ Quantization config present: {type(model.quantization_config).__name__}")
                else:
                    # Try to check through model attributes
                    if hasattr(model, 'model') and hasattr(model.model, 'is_loaded_in_4bit'):
                        print(f"   ✅ 4-bit quantization confirmed: {model.model.is_loaded_in_4bit}")
                    else:
                        print(f"   ⚠️  Could not verify quantization status, but load_in_4bit=True was used")
            except Exception as e:
                print(f"   ⚠️  Could not check quantization status: {e}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model_type = 'transformers'
    
    if model is None:
        print(f"❌ ERROR: Failed to load model!")
        return
    
    print(f"✅ Model loaded successfully")
    print(f"   Model type: {model_type}")
    print(f"   Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Run tests
    scenarios = get_test_scenarios()
    results = []
    
    for scenario in scenarios:
        result = test_scenario(model, tokenizer, scenario, model_type=model_type)
        if result:
            results.append(result)
            
    # Summary
    print(f"\n{'='*80}")
    print(f"Test Summary")
    print(f"{'='*80}")
    
    total_score = 0
    cot_count = 0
    
    total_discard_violations = 0
    total_missing_actions = 0
    total_reasoning_errors = 0
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"   CoT Reasoning: {'✅' if r['has_cot'] else '❌'}")
        print(f"   Score: {r['score']:.2f}%")
        
        # Show detailed results
        if r['correct']:
            print(f"   ✅ Found: {r['correct']}")
        if r['missing']:
            print(f"   ⚠️  Missing: {r['missing']}")
        if r['incorrect']:
            print(f"   ❌ Incorrect: {r['incorrect']}")
        
        # Show violations and errors
        discard_viols = r.get('discard_violations', [])
        if discard_viols:
            total_discard_violations += len(discard_viols)
            print(f"   ⚠️  DISCARD Violation: {len(discard_viols)} [DISCARD] item(s) in FINAL ANSWER: {discard_viols[:3]}")
        
        if r.get('missing_actions', False):
            total_missing_actions += 1
            print(f"   ⚠️  Missing Actions: Reasoning has items but no [KEEP]/[DISCARD] actions")
        
        reasoning_errs = r.get('reasoning_errors', [])
        if reasoning_errs:
            total_reasoning_errors += len(reasoning_errs)
            print(f"   ⚠️  Reasoning Error: {len(reasoning_errs)} error(s) detected")
            
        total_score += r['score']
        if r['has_cot']:
            cot_count += 1
            
    avg_score = total_score / len(results) if results else 0
    cot_pct = (cot_count / len(results) * 100) if results else 0
    
    # Breakdown by query type
    type_scores = {}
    for r in results:
        qtype = r.get('answer_type', 'person')
        if qtype not in type_scores:
            type_scores[qtype] = []
        type_scores[qtype].append(r['score'])
    
    print(f"\n{'='*80}")
    print(f"Overall Results:")
    print(f"   Average Score: {avg_score:.2f}%")
    print(f"   CoT Reasoning: {cot_count}/{len(results)} ({cot_pct:.1f}%)")
    
    # Report violations and errors
    print(f"\n   Critical Issues:")
    if total_discard_violations > 0:
        print(f"   ❌ DISCARD Violations: {total_discard_violations} total (should be 0)")
        print(f"      - Items marked [DISCARD] in reasoning appear in FINAL ANSWER")
        print(f"      - This violates the core DISCARD rule")
    else:
        print(f"   ✅ DISCARD Violations: 0 (correct!)")
    
    if total_missing_actions > 0:
        print(f"   ❌ Missing Actions: {total_missing_actions} test(s) missing [KEEP]/[DISCARD] in reasoning")
        print(f"      - Model not following expected reasoning format")
    else:
        print(f"   ✅ Reasoning Format: All tests have [KEEP]/[DISCARD] actions")
    
    if total_reasoning_errors > 0:
        print(f"   ❌ Reasoning Errors: {total_reasoning_errors} logic error(s) detected")
        print(f"      - Model's reasoning is incorrect (e.g., marking Co-Founder as DISCARD)")
    else:
        print(f"   ✅ Reasoning Logic: No errors detected")
    
    if len(type_scores) > 1:
        print(f"\n   Breakdown by Query Type:")
        for qtype, scores in type_scores.items():
            avg_type_score = sum(scores) / len(scores)
            print(f"      {qtype.title()}: {avg_type_score:.2f}% ({len(scores)} tests)")
    print(f"{'='*80}")
    
    # Enhanced recommendations based on violations
    print(f"\n⚠️  Model performance summary:")
    print(f"   ✅ Accuracy: {'Good' if avg_score > 80 else 'Needs improvement'}")
    print(f"   ✅ CoT Reasoning: {'Present' if cot_pct > 75 else 'Missing'}")
    if total_discard_violations > 0:
        print(f"   ❌ DISCARD Enforcement: {total_discard_violations} violations (should be 0)")
        print(f"      💡 Model is not respecting DISCARD rules - needs retraining")
    if total_missing_actions > 0:
        print(f"   ❌ Reasoning Format: {total_missing_actions} test(s) missing [KEEP]/[DISCARD] actions")
        print(f"      💡 Model not following expected format - needs more training examples")
    if total_reasoning_errors > 0:
        print(f"   ❌ Reasoning Logic: {total_reasoning_errors} error(s) detected")
        print(f"      💡 Model's reasoning is incorrect - needs better training examples")
    
    print(f"\n💡 Recommendations:")
    if total_discard_violations > 0:
        print(f"   - Focus on DISCARD enforcement in training dataset")
        print(f"   - Add examples emphasizing: items marked [DISCARD] must NEVER appear in FINAL ANSWER")
        print(f"   - Verify system prompt includes explicit DISCARD rules")
    if total_missing_actions > 0:
        print(f"   - Add training examples showing proper [KEEP]/[DISCARD] action format")
        print(f"   - Ensure all reasoning examples have explicit Action markers")
    if total_reasoning_errors > 0:
        print(f"   - Add more explicit examples for complex cases (e.g., 'CEO and Co-Founder')")
        print(f"   - Emphasize: read complete descriptions - titles may appear later in text")
    if avg_score < 80:
        print(f"   - Increase training epochs (30-40) for better learning")
        print(f"   - Add more diverse training examples")
        print(f"   - Verify training dataset has no violations")

if __name__ == "__main__":
    run_tests()
