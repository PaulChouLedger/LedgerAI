#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Chunk Analysis Training Dataset Generator
Generates supervised fine-tuning dataset to teach LLM how to:
1. Read and analyze RAG chunks with scores
2. Evaluate relevance (HIGH/MEDIUM/LOW)
3. Extract only HIGH relevance information
4. Synthesize answers from multiple chunks
5. Handle various query types (factual, analytical, list, personal reflection)

Usage:
    python generate_rag_analysis_dataset.py
    Output: rag_analysis_dataset.json
"""

import json
import random
from typing import List, Dict, Any

# ============================================================================
# Query Templates and Examples
# ============================================================================

FACTUAL_QUERIES = [
    "who are the co-founders of {company}?",
    "what is the mission of {company}?",
    "who is the CEO of {company}?",
    "what products does {company} offer?",
    "when was {company} founded?",
    "what is {person}'s role at {company}?",
    "what are {person}'s qualifications?",
    "where is {company} headquartered?",
    "what is {company}'s business model?",
    "who are the key executives at {company}?",
]

ANALYTICAL_QUERIES = [
    "analyze the strategic direction of {company} based on the documents",
    "what are the key challenges mentioned in these documents?",
    "identify the main themes across these documents",
    "what patterns emerge from the leadership structure?",
    "how does {company} differentiate itself from competitors?",
    "what are the core values evident in these documents?",
    "analyze the growth trajectory described in these materials",
    "what are the key partnerships mentioned?",
    "identify recurring themes in the company's strategy",
    "what vision emerges from these documents?",
]

LIST_QUERIES = [
    "list all the co-founders of {company}",
    "what are all the products mentioned?",
    "name all the executives and their roles",
    "list all the partnerships mentioned",
    "what are all the key milestones?",
    "list all the departments or divisions",
    "name all the board members",
    "what are all the locations mentioned?",
    "list all the technologies used",
    "name all the advisors mentioned",
]

PERSONAL_REFLECTION_QUERIES = [
    "help me map the major turning points of my life and how they shaped my identity",
    "identify recurring patterns in my past writing that show how my thinking evolved",
    "extract themes from these letters and emails that show my motivations and blind spots",
    "highlight moments where I deferred judgment to others. What emotional states show up most often in those moments",
    "analyze relationships documented in these files. What were the power imbalances. What language did I use when I felt loyalty or doubt",
    "summarize examples across the documents where persuasion shaped my decisions",
    "create a detailed but neutral timeline from these filings and correspondence",
    "identify misunderstandings the public may have based on the documents I uploaded",
    "compare my personal writings with legal documents. Where do perspectives converge and diverge",
    "show me the moments in my journals where I recognized mistakes or began questioning my environment",
    "help me structure a chapter on accountability using these notes and letters",
    "what emotional themes show up most often when I write about regret or responsibility",
    "help me identify my core values today based on my private essays and planning documents",
    "what goals appear most consistently in the forward looking materials I uploaded",
    "draft an outline for a chapter on what I want my next decade to reflect",
    "analyze my writings about isolation. What words or metaphors do I use most often",
    "given these therapy notes and reflections, help me identify the arc of healing I seem to be describing",
    "what insights about forgiveness appear across these documents",
    "extract passages from these documents that connect my values to my philanthropic interests",
    "identify the language I use around {topic}. How can this become a metaphor in my writing",
    "organize these documents into chapters about {theme1}, {theme2}, and {theme3}",
]

# Sample chunk templates for different scenarios
SAMPLE_CHUNKS = {
    "co_founders": [
        {
            "text": "The company's quarterly earnings report showed strong growth in Q3. Revenue increased by 35% year-over-year, driven primarily by enterprise sales. John Smith is a renowned leader in AI, blockchain, and institutional finance. As CEO and Co-Founder of TechCorp, he is driving the development of AI-powered business intelligence. The board meeting minutes from last month discussed expansion into new markets. A graduate of Tech University with degrees in Mathematics and Electrical Engineering & Computer Science, John's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded FinanceExchange (2014–2020), the first U.S. federally regulated crypto derivatives exchange. The marketing department is planning a new campaign for the upcoming product launch.",
            "score": 0.85,
            "file": "company_info.pdf"
        },
        {
            "text": "Our customer satisfaction scores have improved significantly this quarter. Jane Doe is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management. As Co-Founder and Chief Operating Officer of TechCorp, she leads the execution of AI-powered intelligence solutions. The HR department announced new benefits packages for employees. She is also the CEO of Capital Advisory Group, focusing on AI technology and fintech investments. Jane holds an MS in Material Science and Engineering from State University and a Master's in Public Affairs from Public University. The office renovation project is scheduled to begin next month.",
            "score": 0.82,
            "file": "company_info.pdf"
        },
        {
            "text": "The annual technology conference will be held in City B next spring. Mike Johnson is a visionary leader at the intersection of AI, blockchain, and decentralized finance. As Co-Founder and Chief Marketing Officer of TechCorp, he is spearheading global adoption, brand strategy, and market expansion. Our partnership with several international firms has been finalized. In addition, as Founder and CEO of ProductCorp, he is pioneering AI integration within the metaverse. The legal team is reviewing new compliance requirements for international operations.",
            "score": 0.80,
            "file": "company_info.pdf"
        },
        {
            "text": "The quarterly budget review showed we're on track to meet financial targets. Sarah Williams is a driving force in finance, blockchain, and enterprise strategy. As Co-Founder and Chief Financial Officer of TechCorp, she architects the company's financial strategy, tokenomics, and investment framework. The IT department upgraded our server infrastructure last week. Previously, as Global Head of Payroll & Stock Administration at FinanceCorp and MarketingCorp, Sarah managed multi-billion-dollar payroll and equity programs. Employee training sessions on new software will begin next week.",
            "score": 0.78,
            "file": "company_info.pdf"
        },
        {
            "text": "The company picnic was a great success with over 200 employees attending. Tom Brown is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading TechCorp's cutting-edge engineering efforts as Head of Engineering. The cafeteria menu has been updated with healthier options. An Engineering University graduate, Tom has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems. Parking arrangements for the new office building have been finalized.",
            "score": 0.45,
            "file": "company_info.pdf"
        },
        {
            "text": "The holiday party planning committee met yesterday to discuss venue options. Lisa Davis is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to TechCorp as External Counsel & Advisor. The company newsletter highlighted recent achievements. As Co-Founder of Legal Partners LLP, she has led high-profile cases in entertainment, media, and corporate law. The office holiday decorations will be put up next week.",
            "score": 0.35,
            "file": "company_info.pdf"
        },
    ],
    "mission": [
        {
            "text": "The company's annual report was published last month, showing strong financial performance. TechCorp's mission is to redefine enterprise intelligence and governance at a global scale through AI-powered business intelligence solutions. The company integrates blockchain technology to transform governance, strategy, and financial operations. Our customer support team received excellent feedback in the latest survey. The board approved the new strategic plan for the next fiscal year.",
            "score": 0.90,
            "file": "company_info.pdf"
        },
        {
            "text": "The product development team is working on several new features. PlatformX is TechCorp's flagship product, providing seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning TechCorp at the forefront of enterprise AI solutions. The sales team exceeded their quarterly targets. Marketing materials for the new product line are being prepared.",
            "score": 0.65,
            "file": "product_info.pdf"
        },
        {
            "text": "The office building renovation project is progressing on schedule. The company focuses on AI-driven analytics, high-frequency data processing, and secure enterprise platforms that power intelligent digital ecosystems. Employee wellness programs have been expanded this year. The company picnic is scheduled for next month.",
            "score": 0.55,
            "file": "company_info.pdf"
        },
    ],
    "personal_turning_points": [
        {
            "text": "The weather was beautiful that spring, and I remember spending weekends hiking in the mountains. In 2015, I made the decision to leave my corporate job and start my own consulting practice. This marked a fundamental shift in how I viewed work-life balance and personal fulfillment. I had been reading a lot of philosophy books at the time, exploring different perspectives on meaning. The journal entry from that time shows I was struggling with questions of purpose and meaning. My apartment lease was up for renewal, and I was considering moving to a smaller place to reduce expenses.",
            "score": 0.88,
            "file": "journal_2015.pdf"
        },
        {
            "text": "I was working on a project at the office when I got the call. The death of my father in 2018 forced me to confront my own mortality and reassess my priorities. Letters from that period show a deep questioning of what truly matters in life. I began writing more frequently about legacy and impact. The funeral arrangements were difficult to coordinate, but family and friends were supportive. I spent a lot of time that year reflecting on our relationship and the conversations we never had.",
            "score": 0.85,
            "file": "personal_correspondence.pdf"
        },
        {
            "text": "The real estate market in City A was competitive, and it took months to find the right place. My move to City A in 2020 represented not just a geographic change but a complete reinvention of my professional identity. The emails from that transition period reveal both excitement and anxiety about starting over. I had to sell most of my furniture because the new apartment was smaller. The job market was challenging, but I was determined to make it work.",
            "score": 0.82,
            "file": "emails_2020.pdf"
        },
        {
            "text": "I had been seeing a therapist for about six months when we started discussing decision-making patterns. The therapy notes from 2019-2021 document a period of significant personal growth where I began to recognize patterns in my decision-making that were driven by fear rather than authentic desire. We worked through exercises to identify when I was making choices based on others' expectations versus my own values. The sessions were challenging but ultimately very helpful.",
            "score": 0.75,
            "file": "therapy_notes.pdf"
        },
        {
            "text": "I had been tracking my expenses carefully and noticed I was spending too much on dining out. Financial planning documents from 2017 show I was focused on early retirement, but by 2022, my planning documents reflect a shift toward building something meaningful rather than escaping work. I started investing more in education and skill development. The retirement savings plan I had set up was performing well, but my goals had changed.",
            "score": 0.70,
            "file": "financial_planning.pdf"
        },
        {
            "text": "The quarterly board meeting was scheduled for next Tuesday. Meeting minutes from board meetings show I was consistently advocating for more aggressive growth strategies, which contrasts with my personal writings about wanting more balance. The discussion about expanding into new markets was particularly heated. I found myself questioning whether rapid growth was really what I wanted for the company.",
            "score": 0.40,
            "file": "board_minutes.pdf"
        },
    ],
    "emotional_themes": [
        {
            "text": "I've been trying to read more fiction lately, and I just finished a novel about time travel. When writing about regret, I consistently use metaphors of weight and burden. Phrases like 'carrying the weight of that decision' and 'the burden of what could have been' appear repeatedly across journal entries from 2018-2022. I noticed this pattern when I was reviewing old entries for a writing project. The metaphors seem to come naturally when I'm processing difficult experiences. I'm planning to start a new exercise routine next month.",
            "score": 0.92,
            "file": "journals.pdf"
        },
        {
            "text": "I had a conversation with a colleague about leadership styles last week. Responsibility is described in terms of stewardship and care. I write about 'holding space for others' and 'being a steward of trust' rather than using language of obligation or duty. This language choice reflects how I want to approach my relationships and commitments. I've been reading about different philosophical approaches to responsibility. The weekend weather forecast looks promising for outdoor activities.",
            "score": 0.88,
            "file": "journals.pdf"
        },
        {
            "text": "The therapy session started with a discussion about sleep patterns and stress management. The therapy notes document that when discussing regret, I often become physically tense and use language that suggests I'm still processing these events rather than having resolved them. We explored techniques for managing the physical response to these memories. The therapist suggested journaling exercises to help process these feelings more fully.",
            "score": 0.85,
            "file": "therapy_notes.pdf"
        },
        {
            "text": "I received a letter from an old friend who moved to another country. Letters to friends show a pattern where I express responsibility through questions rather than statements - 'Did I do enough?' 'Could I have been more present?' This suggests ongoing self-reflection rather than closure. The questions seem to be a way of exploring my role in various situations. I'm planning to visit them next summer if travel restrictions allow.",
            "score": 0.80,
            "file": "personal_correspondence.pdf"
        },
        {
            "text": "The tax preparation documents are ready for review. Financial documents show careful planning and risk management, which contrasts with the emotional language used in personal writings about responsibility. The investment portfolio has been rebalanced according to the new strategy. I scheduled a meeting with my financial advisor to discuss retirement planning options.",
            "score": 0.35,
            "file": "financial_planning.pdf"
        },
    ],
    "co_founders_multi_company": [
        {
            "text": "The company's quarterly earnings report showed strong growth in Q3. Revenue increased by 35% year-over-year. John Smith is a renowned leader in AI, blockchain, and institutional finance. As CEO and Co-Founder of TechCorp, he is driving the development of AI-powered business intelligence. A graduate of Tech University, John's expertise spans high-frequency trading and AI-driven analytics. Previously, he co-founded FinanceExchange (2014–2020). Meanwhile, Alex Chen is the Co-Founder and CEO of DataSystems Inc., a company specializing in enterprise data analytics. Alex has over 15 years of experience in data science and machine learning. The board meeting minutes from last month discussed expansion into new markets.",
            "score": 0.85,
            "file": "company_info.pdf"
        },
        {
            "text": "Our customer satisfaction scores have improved significantly this quarter. Jane Doe is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management. As Co-Founder and Chief Operating Officer of TechCorp, she leads the execution of AI-powered intelligence solutions. She is also the CEO of Capital Advisory Group. Jane holds an MS from State University. On the other hand, Maria Rodriguez is Co-Founder and CTO of DataSystems Inc., where she has built the company's technical infrastructure from the ground up. Maria previously worked at several tech startups before founding DataSystems. The HR department announced new benefits packages for employees.",
            "score": 0.82,
            "file": "company_info.pdf"
        },
        {
            "text": "The annual technology conference will be held in City B next spring. Mike Johnson is a visionary leader at the intersection of AI, blockchain, and decentralized finance. As Co-Founder and Chief Marketing Officer of TechCorp, he is spearheading global adoption and market expansion. In addition, as Founder and CEO of ProductCorp, he is pioneering AI integration. Separately, David Kim serves as Co-Founder and Chief Product Officer of DataSystems Inc., leading product strategy and development. David has a background in software engineering and product design. Our partnership with several international firms has been finalized.",
            "score": 0.80,
            "file": "company_info.pdf"
        },
        {
            "text": "The quarterly budget review showed we're on track to meet financial targets. Sarah Williams is a driving force in finance, blockchain, and enterprise strategy. As Co-Founder and Chief Financial Officer of TechCorp, she architects the company's financial strategy and tokenomics. Previously, she managed multi-billion-dollar payroll programs at FinanceCorp. Additionally, Robert Taylor is Co-Founder and CFO of DataSystems Inc., overseeing all financial operations and investor relations. Robert brings extensive experience from his previous role at a major consulting firm. The IT department upgraded our server infrastructure last week.",
            "score": 0.78,
            "file": "company_info.pdf"
        },
        {
            "text": "The company picnic was a great success with over 200 employees attending. Tom Brown is a technological architect with over 20 years of experience, leading TechCorp's engineering efforts as Head of Engineering. An Engineering University graduate, Tom has spent two decades pioneering breakthrough technologies. In contrast, Jennifer Lee is Co-Founder and VP of Engineering at DataSystems Inc., where she manages a team of 50 engineers. Jennifer has a PhD in Computer Science and has published numerous papers on distributed systems. The cafeteria menu has been updated with healthier options.",
            "score": 0.45,
            "file": "company_info.pdf"
        },
        {
            "text": "The holiday party planning committee met yesterday to discuss venue options. Lisa Davis is a top-tier legal strategist, bringing expertise in litigation and business law to TechCorp as External Counsel & Advisor. As Co-Founder of Legal Partners LLP, she has led high-profile cases. Meanwhile, James Wilson serves as Co-Founder and General Counsel of DataSystems Inc., handling all legal matters and compliance. James has a law degree from a prestigious university and specializes in technology law. The office holiday decorations will be put up next week.",
            "score": 0.35,
            "file": "company_info.pdf"
        },
    ],
}

def generate_chunks_for_query(query_type: str, query: str) -> List[Dict[str, Any]]:
    """Generate relevant chunks for a query type"""
    if query_type == "co_founders":
        return SAMPLE_CHUNKS["co_founders"]
    elif query_type == "co_founders_multi_company":
        return SAMPLE_CHUNKS["co_founders_multi_company"]
    elif query_type == "mission":
        return SAMPLE_CHUNKS["mission"]
    elif query_type == "personal_turning_points":
        return SAMPLE_CHUNKS["personal_turning_points"]
    elif query_type == "emotional_themes":
        return SAMPLE_CHUNKS["emotional_themes"]
    else:
        # Generate generic chunks
        return [
            {
                "text": f"Relevant information about {query[:50]}... This contains key details that directly answer the question.",
                "score": 0.75,
                "file": "document.pdf"
            },
            {
                "text": f"Additional context related to {query[:50]}... This provides supporting information.",
                "score": 0.65,
                "file": "document.pdf"
            },
            {
                "text": f"Tangentially related information that mentions similar concepts but doesn't directly answer the question.",
                "score": 0.40,
                "file": "document.pdf"
            },
        ]

def create_expected_response(query: str, chunks: List[Dict[str, Any]], query_type: str) -> str:
    """Create expected response showing proper RAG analysis"""
    
    # Analyze chunks and extract HIGH relevance information
    # CRITICAL: Score >= 0.70 is HIGH relevance, NOT medium
    high_relevance_chunks = [c for c in chunks if c["score"] >= 0.70]
    medium_relevance_chunks = [c for c in chunks if 0.50 <= c["score"] < 0.70]
    low_relevance_chunks = [c for c in chunks if c["score"] < 0.50]
    
    response_parts = []
    
    # For factual/list queries, extract direct information
    if query_type in ["factual", "list"]:
        response_parts.append("ANALYSIS: Reading all chunks completely to extract relevant information.\n")
        response_parts.append(f"RELEVANCE EVALUATION:\n")
        # CRITICAL: Scores >= 0.70 are HIGH relevance. Scores 0.82, 0.80, 0.78 are all HIGH, not MEDIUM.
        response_parts.append(f"- HIGH relevance (score ≥0.70): {len(high_relevance_chunks)} chunks\n")
        response_parts.append(f"- MEDIUM relevance (0.50-0.69): {len(medium_relevance_chunks)} chunks\n")
        response_parts.append(f"- LOW relevance (score <0.50): {len(low_relevance_chunks)} chunks\n")
        # Add explicit note showing actual scores to emphasize they are HIGH
        if high_relevance_chunks:
            high_scores = [f"{c['score']:.2f}" for c in high_relevance_chunks]
            response_parts.append(f"\n✅ IMPORTANT: HIGH relevance chunks have scores: {', '.join(high_scores)}\n")
            response_parts.append(f"   All of these scores are ≥0.70, so they are ALL HIGH relevance, NOT MEDIUM.\n")
            response_parts.append(f"   For example: 0.85 ≥ 0.70 = HIGH, 0.82 ≥ 0.70 = HIGH, 0.80 ≥ 0.70 = HIGH, 0.78 ≥ 0.70 = HIGH\n")
            response_parts.append(f"   You must read ALL {len(high_relevance_chunks)} HIGH relevance chunks completely.\n")
        response_parts.append(f"\nEXTRACTING INFORMATION:\n")
        
        # Extract from high relevance chunks - show that we read the entire chunk but extract only relevant parts
        extracted_info = []
        chunk_extractions = []  # Track what was found in each chunk for co-founder queries
        
        for i, chunk in enumerate(high_relevance_chunks, 1):
            full_text = chunk['text']
            response_parts.append(f"Chunk {i} (score: {chunk['score']:.3f}): Read entire chunk ({len(full_text)} characters)\n")
            # Extract only the relevant information from the chunk
            if "co-founders" in query.lower() or "cofounders" in query.lower():
                # Extract only the co-founder information, ignoring other content
                if "Co-Founder" in full_text or "co-founder" in full_text:
                    # Extract the sentence(s) containing co-founder information
                    sentences = full_text.split('. ')
                    relevant_sentences = [s for s in sentences if "Co-Founder" in s or "co-founder" in s or "CEO" in s or "Chief" in s]
                    extracted_info.append('. '.join(relevant_sentences) + '.')
                    # Track what was found in this chunk for explicit listing
                    chunk_extractions.append(f"Chunk {i}: Found co-founder information")
                else:
                    extracted_info.append(full_text)
                    chunk_extractions.append(f"Chunk {i}: No co-founder information found")
            else:
                # For other queries, extract the most relevant sentence(s)
                sentences = full_text.split('. ')
                # Simple heuristic: take sentences that contain key terms from the query
                query_terms = set(query.lower().split())
                relevant_sentences = [s for s in sentences if any(term in s.lower() for term in query_terms if len(term) > 3)]
                if relevant_sentences:
                    extracted_info.append('. '.join(relevant_sentences[:2]) + '.')  # Take top 2 relevant sentences
                else:
                    extracted_info.append(full_text[:200] + '...')  # Fallback: first 200 chars
        
        response_parts.append(f"\nSYNTHESIS:\n")
        
        # Create answer based on query type
        if "co-founders" in query.lower() or "cofounders" in query.lower():
            # Extract names and roles - CRITICAL: Only extract co-founders of the company mentioned in the query
            query_company = None
            query_lower = query.lower()
            # Check for DataSystems first (more specific)
            if "datasystems inc" in query_lower or "datasystems" in query_lower:
                query_company = "DataSystems Inc."
            elif "techcorp" in query_lower:
                query_company = "TechCorp"
            else:
                # Default to TechCorp if not specified
                query_company = "TechCorp"
            
            names_found = []
            for chunk in high_relevance_chunks:
                text = chunk['text']
                # Read entire chunk and extract co-founders - check if chunk mentions the query company
                # Then find all co-founders of that company in the chunk
                if query_company in text:
                    # Extract all co-founders of the query company from this chunk
                    if query_company == "TechCorp":
                        # Check each person - they might be in different sentences
                        if "John Smith" in text and "TechCorp" in text and ("Co-Founder" in text or "CEO" in text):
                            # Verify John is a co-founder of TechCorp (not just mentioned)
                            john_context = text[max(0, text.find("John Smith")-50):text.find("John Smith")+200]
                            if "TechCorp" in john_context and ("Co-Founder" in john_context or ("CEO" in john_context and "Co-Founder" in text)):
                                names_found.append("John Smith - CEO and Co-Founder of TechCorp")
                        if "Jane Doe" in text and "TechCorp" in text and "Co-Founder" in text:
                            jane_context = text[max(0, text.find("Jane Doe")-50):text.find("Jane Doe")+200]
                            if "TechCorp" in jane_context and "Co-Founder" in jane_context:
                                names_found.append("Jane Doe - Co-Founder and Chief Operating Officer of TechCorp")
                        if "Mike Johnson" in text and "TechCorp" in text and "Co-Founder" in text:
                            mike_context = text[max(0, text.find("Mike Johnson")-50):text.find("Mike Johnson")+200]
                            if "TechCorp" in mike_context and "Co-Founder" in mike_context:
                                names_found.append("Mike Johnson - Co-Founder and Chief Marketing Officer of TechCorp")
                        if "Sarah Williams" in text and "TechCorp" in text and "Co-Founder" in text:
                            sarah_context = text[max(0, text.find("Sarah Williams")-50):text.find("Sarah Williams")+200]
                            if "TechCorp" in sarah_context and "Co-Founder" in sarah_context:
                                names_found.append("Sarah Williams - Co-Founder and Chief Financial Officer of TechCorp")
                    elif query_company == "DataSystems Inc.":
                        # Check each person for DataSystems
                        if "Alex Chen" in text and "DataSystems" in text and "Co-Founder" in text:
                            alex_context = text[max(0, text.find("Alex Chen")-50):text.find("Alex Chen")+200]
                            if "DataSystems" in alex_context and "Co-Founder" in alex_context:
                                names_found.append("Alex Chen - Co-Founder and CEO of DataSystems Inc.")
                        if "Maria Rodriguez" in text and "DataSystems" in text and "Co-Founder" in text:
                            maria_context = text[max(0, text.find("Maria Rodriguez")-50):text.find("Maria Rodriguez")+200]
                            if "DataSystems" in maria_context and "Co-Founder" in maria_context:
                                names_found.append("Maria Rodriguez - Co-Founder and CTO of DataSystems Inc.")
                        if "David Kim" in text and "DataSystems" in text and "Co-Founder" in text:
                            david_context = text[max(0, text.find("David Kim")-50):text.find("David Kim")+200]
                            if "DataSystems" in david_context and "Co-Founder" in david_context:
                                names_found.append("David Kim - Co-Founder and Chief Product Officer of DataSystems Inc.")
                        if "Robert Taylor" in text and "DataSystems" in text and "Co-Founder" in text:
                            robert_context = text[max(0, text.find("Robert Taylor")-50):text.find("Robert Taylor")+200]
                            if "DataSystems" in robert_context and "Co-Founder" in robert_context:
                                names_found.append("Robert Taylor - Co-Founder and CFO of DataSystems Inc.")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in names_found:
                if name not in seen:
                    seen.add(name)
                    unique_names.append(name)
            
            # Show step-by-step extraction to reinforce the pattern
            response_parts.append(f"Step-by-step extraction from HIGH relevance chunks:\n")
            response_parts.append(f"⚠️  IMPORTANT: Query asks for co-founders of {query_company}. Only extract co-founders of {query_company}, ignore co-founders of other companies.\n\n")
            for i, chunk in enumerate(high_relevance_chunks, 1):
                text = chunk['text']
                found_in_chunk = []
                # CRITICAL: Check if this chunk mentions the query company
                if query_company not in text:
                    response_parts.append(f"  Chunk {i}: Does not mention {query_company}, skipping (continuing to read remaining chunks...)\n")
                    continue
                
                if query_company == "TechCorp":
                    if "John Smith" in text and "TechCorp" in text and ("Co-Founder" in text or "CEO" in text):
                        john_context = text[max(0, text.find("John Smith")-50):text.find("John Smith")+200]
                        if "TechCorp" in john_context and ("Co-Founder" in john_context or ("CEO" in john_context and "Co-Founder" in text)):
                            found_in_chunk.append("John Smith")
                    if "Jane Doe" in text and "TechCorp" in text and "Co-Founder" in text:
                        jane_context = text[max(0, text.find("Jane Doe")-50):text.find("Jane Doe")+200]
                        if "TechCorp" in jane_context and "Co-Founder" in jane_context:
                            found_in_chunk.append("Jane Doe")
                    if "Mike Johnson" in text and "TechCorp" in text and "Co-Founder" in text:
                        mike_context = text[max(0, text.find("Mike Johnson")-50):text.find("Mike Johnson")+200]
                        if "TechCorp" in mike_context and "Co-Founder" in mike_context:
                            found_in_chunk.append("Mike Johnson")
                    if "Sarah Williams" in text and "TechCorp" in text and "Co-Founder" in text:
                        sarah_context = text[max(0, text.find("Sarah Williams")-50):text.find("Sarah Williams")+200]
                        if "TechCorp" in sarah_context and "Co-Founder" in sarah_context:
                            found_in_chunk.append("Sarah Williams")
                elif query_company == "DataSystems Inc.":
                    if "Alex Chen" in text and "DataSystems" in text and "Co-Founder" in text:
                        alex_context = text[max(0, text.find("Alex Chen")-50):text.find("Alex Chen")+200]
                        if "DataSystems" in alex_context and "Co-Founder" in alex_context:
                            found_in_chunk.append("Alex Chen")
                    if "Maria Rodriguez" in text and "DataSystems" in text and "Co-Founder" in text:
                        maria_context = text[max(0, text.find("Maria Rodriguez")-50):text.find("Maria Rodriguez")+200]
                        if "DataSystems" in maria_context and "Co-Founder" in maria_context:
                            found_in_chunk.append("Maria Rodriguez")
                    if "David Kim" in text and "DataSystems" in text and "Co-Founder" in text:
                        david_context = text[max(0, text.find("David Kim")-50):text.find("David Kim")+200]
                        if "DataSystems" in david_context and "Co-Founder" in david_context:
                            found_in_chunk.append("David Kim")
                    if "Robert Taylor" in text and "DataSystems" in text and "Co-Founder" in text:
                        robert_context = text[max(0, text.find("Robert Taylor")-50):text.find("Robert Taylor")+200]
                        if "DataSystems" in robert_context and "Co-Founder" in robert_context:
                            found_in_chunk.append("Robert Taylor")
                
                if found_in_chunk:
                    response_parts.append(f"  Chunk {i}: Found {', '.join(found_in_chunk)} from {query_company} (ignoring other companies' co-founders, continuing to read remaining chunks...)\n")
                else:
                    # Check if chunk has other companies' co-founders that we're ignoring
                    other_companies = []
                    if query_company == "TechCorp" and ("DataSystems" in text or "DataSystems Inc." in text):
                        other_companies.append("DataSystems Inc.")
                    elif query_company == "DataSystems Inc." and "TechCorp" in text:
                        other_companies.append("TechCorp")
                    
                    if other_companies:
                        response_parts.append(f"  Chunk {i}: No co-founders of {query_company} found (chunk mentions {', '.join(other_companies)} but we ignore those, continuing to read remaining chunks...)\n")
                    else:
                        response_parts.append(f"  Chunk {i}: No co-founders of {query_company} found (continuing to read remaining chunks...)\n")
            
            response_parts.append(f"\nComplete list of co-founders of {query_company} (from all {len(high_relevance_chunks)} HIGH relevance chunks):\n")
            if unique_names:
                # CRITICAL: List ALL co-founders found - do not stop after finding some
                # IMPORTANT: You must read ALL chunks completely and extract ALL co-founders
                # CRITICAL: Only list co-founders of {query_company}, NOT other companies
                for name in unique_names:
                    response_parts.append(f"- {name}\n")
                # Add explicit note about completeness and filtering
                response_parts.append(f"\n✅ COMPLETE: Found all {len(unique_names)} co-founder(s) of {query_company} by reading all {len(high_relevance_chunks)} HIGH relevance chunks completely.\n")
                response_parts.append(f"   ⚠️  IMPORTANT: Only extracted co-founders of {query_company}. Ignored co-founders of other companies (e.g., DataSystems Inc., TechCorp) that appeared in the same chunks.\n")
                response_parts.append(f"   Do not stop after finding some - you must read all chunks and extract all matches for the queried company only.\n")
            else:
                response_parts.append("(No co-founders found for the specified company in the HIGH relevance chunks)\n")
        
        elif "mission" in query.lower():
            # Extract mission statement, filtering out irrelevant content
            # CRITICAL: Must include "blockchain" and all relevant keywords
            # CRITICAL: Filter out irrelevant content like "annual report", "customer support", "quarterly targets"
            mission_parts = []
            irrelevant_keywords = ["annual report", "customer support", "quarterly targets", "board approved", "strategic plan", "fiscal year", "sales team", "marketing materials"]
            
            for chunk in high_relevance_chunks:
                text = chunk['text']
                # Find sentences containing mission-related keywords
                sentences = text.split('. ')
                # Filter out sentences with irrelevant keywords
                relevant_sentences = []
                for sentence in sentences:
                    # Check if sentence contains mission-related keywords
                    mission_keywords = ["mission", "enterprise intelligence", "governance", "AI-powered", "blockchain", "transform", "redefine"]
                    has_mission_content = any(keyword.lower() in sentence.lower() for keyword in mission_keywords)
                    # Check if sentence contains irrelevant keywords
                    has_irrelevant = any(keyword.lower() in sentence.lower() for keyword in irrelevant_keywords)
                    # Only include if it has mission content and doesn't have irrelevant content
                    if has_mission_content and not has_irrelevant:
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    mission_parts.append('. '.join(relevant_sentences) + '.')
                else:
                    # Fallback: extract mission sentence even if it's mixed with irrelevant content
                    for sentence in sentences:
                        if "mission" in sentence.lower():
                            # Extract just the mission part
                            mission_parts.append(sentence + '.')
                            break
                # Extract mission sentence and blockchain sentence
                mission_sentence = None
                blockchain_sentence = None
                for sent in sentences:
                    # Skip irrelevant sentences
                    if any(irrelevant in sent.lower() for irrelevant in ['annual report', 'customer support', 'quarterly', 'board approved']):
                        continue
                    # Find mission sentence
                    if "mission" in sent.lower() and ("enterprise" in sent.lower() or "governance" in sent.lower() or "AI-powered" in sent.lower()):
                        mission_sentence = sent.strip()
                    # Find blockchain sentence (might be separate)
                    if "blockchain" in sent.lower() and ("governance" in sent.lower() or "transform" in sent.lower() or "strategy" in sent.lower()):
                        blockchain_sentence = sent.strip()
                
                # Combine mission and blockchain sentences
                if mission_sentence:
                    mission_parts.append(mission_sentence)
                if blockchain_sentence and blockchain_sentence not in mission_parts:
                    mission_parts.append(blockchain_sentence)
                
                if mission_parts:
                    break
            
            if mission_parts:
                # Join mission parts, ensuring blockchain is included
                full_mission = '. '.join(mission_parts)
                # Verify blockchain is mentioned, if not add it
                if "blockchain" not in full_mission.lower():
                    # Try to find blockchain sentence from chunks
                    for chunk in high_relevance_chunks:
                        text = chunk['text']
                        if "blockchain" in text.lower():
                            sentences = text.split('. ')
                            for sent in sentences:
                                if "blockchain" in sent.lower() and not any(irrelevant in sent.lower() for irrelevant in ['annual report', 'customer support']):
                                    full_mission += ". " + sent.strip()
                                    break
                            break
                
                response_parts.append(f"Based on the HIGH relevance chunks, {full_mission}.\n")
            else:
                # Fallback: use complete mission statement
                response_parts.append("Based on the HIGH relevance chunks, TechCorp's mission is to redefine enterprise intelligence and governance at a global scale through AI-powered business intelligence solutions, integrating blockchain technology to transform governance, strategy, and financial operations.\n")
        
        else:
            response_parts.append("Based on the extracted information from HIGH relevance chunks:\n")
            for i, info in enumerate(extracted_info[:3], 1):
                response_parts.append(f"{i}. {info[:200]}...\n")
    
    # For analytical/personal queries, synthesize insights
    elif query_type in ["analytical", "personal"]:
        response_parts.append("ANALYSIS: Reading all chunks completely to identify patterns and themes.\n")
        response_parts.append(f"RELEVANCE EVALUATION:\n")
        response_parts.append(f"- HIGH relevance (score ≥0.70): {len(high_relevance_chunks)} chunks\n")
        response_parts.append(f"- MEDIUM relevance (0.50-0.69): {len(medium_relevance_chunks)} chunks\n")
        response_parts.append(f"- LOW relevance (score <0.50): {len(low_relevance_chunks)} chunks\n")
        # Add explicit note showing actual scores to emphasize they are HIGH
        if high_relevance_chunks:
            high_scores = [f"{c['score']:.2f}" for c in high_relevance_chunks]
            response_parts.append(f"\n✅ IMPORTANT: HIGH relevance chunks have scores: {', '.join(high_scores)}\n")
            response_parts.append(f"   All of these scores are ≥0.70, so they are ALL HIGH relevance, NOT MEDIUM.\n")
            response_parts.append(f"   For example: 0.85 ≥ 0.70 = HIGH, 0.82 ≥ 0.70 = HIGH, 0.80 ≥ 0.70 = HIGH, 0.78 ≥ 0.70 = HIGH\n")
            response_parts.append(f"   You must read ALL {len(high_relevance_chunks)} HIGH relevance chunks completely.\n")
        response_parts.append(f"\nEXTRACTING THEMES:\n")
        
        # Extract themes from high relevance chunks - read entire chunks but extract only relevant information
        # CRITICAL: Filter out irrelevant content (weather, office logistics, etc.)
        themes = []
        for chunk in high_relevance_chunks:
            full_text = chunk['text']
            # Extract only the relevant sentences that relate to the query
            sentences = full_text.split('. ')
            # For personal/analytical queries, extract sentences that contain emotional or analytical content
            query_lower = query.lower()
            if any(term in query_lower for term in ['turning point', 'pattern', 'theme', 'motivation', 'emotional', 'regret', 'responsibility', 'map', 'identity']):
                # Extract sentences with emotional/analytical content, EXCLUDING irrelevant content
                irrelevant_keywords = ['weather', 'hiking', 'apartment lease', 'funeral arrangements', 'quarterly earnings', 'board meeting', 'office renovation', 'cafeteria menu', 'holiday party', 'parking arrangements', 'customer satisfaction', 'HR department', 'IT department']
                relevant_sentences = []
                for s in sentences:
                    # Skip sentences with irrelevant keywords
                    if any(irrelevant in s.lower() for irrelevant in irrelevant_keywords):
                        continue
                    # Include sentences with relevant keywords
                    if any(word in s.lower() for word in ['decision', 'change', 'shift', 'growth', 'pattern', 'feeling', 'emotion', 'regret', 'responsibility', 'question', 'struggle', 'confront', 'reassess', 'reinvention', 'recognize', 'mortality', 'priorities', 'identity', 'turning point', '2015', '2018', '2020']):
                        relevant_sentences.append(s)
                
                if relevant_sentences:
                    themes.append('. '.join(relevant_sentences[:4]) + '.')  # Take top 4 relevant sentences
                else:
                    # Fallback: take first sentence that doesn't have irrelevant keywords
                    for s in sentences:
                        if not any(irrelevant in s.lower() for irrelevant in irrelevant_keywords):
                            themes.append(s + '.')
                            break
                    if not themes:
                        themes.append(full_text[:200] + '...')
            else:
                # For other analytical queries, extract sentences with analytical content
                irrelevant_keywords = ['quarterly earnings', 'board meeting', 'customer satisfaction', 'office renovation', 'cafeteria', 'holiday party']
                relevant_sentences = []
                for s in sentences:
                    if any(irrelevant in s.lower() for irrelevant in irrelevant_keywords):
                        continue
                    if any(word in s.lower() for word in ['mission', 'strategy', 'focus', 'goal', 'value', 'challenge', 'theme', 'pattern', 'direction', 'blockchain', 'AI-powered', 'enterprise intelligence']):
                        relevant_sentences.append(s)
                
                if relevant_sentences:
                    themes.append('. '.join(relevant_sentences[:3]) + '.')
                else:
                    themes.append(full_text[:200] + '...')
        
        response_parts.append(f"\nSYNTHESIS:\n")
        response_parts.append("Based on analyzing the HIGH relevance chunks, I've identified the following:\n\n")
        
        if "turning points" in query.lower() or "map" in query.lower():
            # Extract actual turning points from chunks, filtering irrelevant content
            turning_points = []
            for chunk in high_relevance_chunks:
                text = chunk['text']
                # Filter out irrelevant sentences
                sentences = text.split('. ')
                for s in sentences:
                    # Skip irrelevant content
                    if any(irrelevant in s.lower() for irrelevant in ['weather', 'hiking', 'apartment lease', 'quarterly', 'board meeting', 'office']):
                        continue
                    # Extract turning point information
                    if any(year in s for year in ['2015', '2018', '2020', '2019', '2021', '2022']):
                        if any(keyword in s.lower() for keyword in ['decision', 'leave', 'corporate', 'consulting', 'death', 'father', 'mortality', 'priorities', 'move', 'reinvention', 'recognize', 'pattern', 'fear']):
                            turning_points.append(s.strip())
            
            if turning_points:
                response_parts.append("Major turning points identified:\n")
                for i, tp in enumerate(turning_points[:4], 1):
                    # Extract year and key event
                    year = None
                    for y in ['2015', '2018', '2020', '2019', '2021']:
                        if y in tp:
                            year = y
                            break
                    if year:
                        if '2015' in tp and ('corporate' in tp.lower() or 'consulting' in tp.lower()):
                            response_parts.append(f"{i}. Career transition ({year}) - leaving corporate job for consulting\n")
                        elif '2018' in tp and ('father' in tp.lower() or 'death' in tp.lower() or 'mortality' in tp.lower()):
                            response_parts.append(f"{i}. Personal loss ({year}) - father's death leading to reassessment of priorities\n")
                        elif '2020' in tp and ('move' in tp.lower() or 'reinvention' in tp.lower()):
                            response_parts.append(f"{i}. Geographic and professional reinvention ({year}) - move to City A\n")
                        elif any(y in tp for y in ['2019', '2021']) and ('recognize' in tp.lower() or 'pattern' in tp.lower()):
                            response_parts.append(f"{i}. Personal growth period (2019-2021) - recognizing fear-based decision patterns\n")
                        else:
                            response_parts.append(f"{i}. {tp[:100]}...\n")
            else:
                # Fallback
                response_parts.append("Major turning points identified:\n")
                response_parts.append("1. Career transition (2015) - leaving corporate job for consulting\n")
                response_parts.append("2. Personal loss (2018) - father's death leading to reassessment of priorities\n")
                response_parts.append("3. Geographic and professional reinvention (2020) - move to City A\n")
                response_parts.append("4. Personal growth period (2019-2021) - recognizing fear-based decision patterns\n")
        
        elif "emotional themes" in query.lower() or "regret" in query.lower() or "responsibility" in query.lower():
            response_parts.append("Emotional themes identified:\n")
            response_parts.append("1. Regret: Described using metaphors of weight and burden ('carrying the weight of that decision')\n")
            response_parts.append("2. Responsibility: Framed as stewardship and care ('holding space for others', 'steward of trust')\n")
            response_parts.append("3. Self-reflection: Expressed through questions rather than statements ('Did I do enough?')\n")
        
        else:
            response_parts.append("Key insights extracted from the documents:\n")
            for i, theme in enumerate(themes[:3], 1):
                response_parts.append(f"{i}. {theme[:150]}...\n")
    
    response_parts.append("\nWhat would you like to explore further about this topic?")
    
    return "".join(response_parts)

def generate_conversation(query: str, chunks: List[Dict[str, Any]], query_type: str) -> Dict[str, Any]:
    """Generate a complete training conversation"""
    
    # Format chunks for context
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] Score: {chunk['score']:.3f}, File: {chunk['file']}, Preview: '{chunk['text'][:80]}...'")
        context_parts.append(f"[{i}] FULL CHUNK TEXT: '{chunk['text']}'")
    
    context = "\n".join(context_parts)
    
    # Create system prompt
    system_prompt = """You are an AI assistant trained to analyze RAG (Retrieval-Augmented Generation) chunks and extract relevant information.

Your task is to:
1. Read EVERY chunk completely from start to finish - DO NOT stop reading once you find relevant information
2. Evaluate relevance for each chunk:
   - HIGH (score ≥0.70): Information that directly and explicitly answers the query. IMPORTANT: Scores like 0.85, 0.82, 0.80, 0.78 are ALL HIGH relevance (≥0.70), NOT MEDIUM.
   - MEDIUM (0.50-0.69): Information that is related but requires inference
   - LOW (score <0.50): Information that mentions similar terms but doesn't actually answer the query
3. Extract only HIGH relevance information - be precise about what exactly matches the query
4. For list questions: Find EVERY matching item in EVERY chunk - read each chunk completely. DO NOT stop after finding some items - you must extract ALL matching items from ALL chunks.
5. For entity extraction (e.g., co-founders): Extract ALL entities that match the query criteria. If a chunk mentions multiple entities, extract ALL of them, not just the first one you find. Read through ALL chunks completely before listing entities. If there are 4 co-founders mentioned across chunks, you must list ALL 4, not just 2 or 3.
6. For company-specific queries: Only extract entities that belong to the specified company. Ignore entities from other companies even if they appear in the same chunks.
7. For analytical questions: Extract all relevant information from all chunks before synthesizing
8. Filter out irrelevant content: Do not include information about weather, hiking, apartment leases, funeral arrangements, annual reports, customer support, quarterly targets, or other unrelated details.
9. Format your analysis showing: RELEVANCE EVALUATION → EXTRACTING INFORMATION → SYNTHESIS → Final Answer

CRITICAL RULES:
- When extracting lists or entities, you must find and list EVERY matching item. Do not stop after finding some items - continue reading all chunks completely and extract all matches.
- Scores ≥0.70 are HIGH relevance. Scores 0.82, 0.80, 0.78 are HIGH, not MEDIUM.
- Read ALL chunks with HIGH relevance scores completely, not just the first one.
- Filter out irrelevant sentences that don't answer the query (weather, office logistics, etc.).

Always end with a brief, natural follow-up question."""

    # Create messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nRAG Chunks:\n{context}"
        },
        {
            "role": "assistant",
            "content": create_expected_response(query, chunks, query_type)
        }
    ]
    
    return {
        "messages": messages,
        "query_type": query_type,
        "num_chunks": len(chunks),
        "high_relevance_count": len([c for c in chunks if c["score"] >= 0.70])
    }

def generate_dataset(num_examples: int = 100) -> List[Dict[str, Any]]:
    """Generate the complete training dataset"""
    dataset = []
    
    # Generate factual queries (20 examples)
    for i in range(20):
        query = random.choice(FACTUAL_QUERIES).format(
            company="TechCorp",
            person=random.choice(["John Smith", "Jane Doe", "Mike Johnson", "Sarah Williams"])
        )
        if "co-founder" in query.lower() or "cofounder" in query.lower():
            chunks = generate_chunks_for_query("co_founders", query)
        else:
            chunks = generate_chunks_for_query("mission", query)
        conversation = generate_conversation(query, chunks, "factual")
        dataset.append(conversation)
    
    # Generate single-company co-founder queries (50 examples) - CRITICAL for learning complete extraction
    # More examples = better learning of the pattern
    for i in range(50):
        query = "who are the co-founders of TechCorp?"
        chunks = generate_chunks_for_query("co_founders", query)
        conversation = generate_conversation(query, chunks, "factual")
        dataset.append(conversation)
    
    # Generate multi-company co-founder queries (50 examples) - teaches entity differentiation
    # More examples = better learning of entity filtering
    for i in range(50):
        # Randomly choose which company to query about
        target_company = random.choice(["TechCorp", "DataSystems Inc."])
        query = f"who are the co-founders of {target_company}?"
        chunks = generate_chunks_for_query("co_founders_multi_company", query)
        conversation = generate_conversation(query, chunks, "factual")
        dataset.append(conversation)
    
    # Generate list queries (15 examples)
    for i in range(15):
        query = random.choice(LIST_QUERIES).format(company="TechCorp")
        chunks = generate_chunks_for_query("co_founders", query)
        conversation = generate_conversation(query, chunks, "list")
        dataset.append(conversation)
    
    # Generate analytical queries (20 examples)
    for i in range(20):
        query = random.choice(ANALYTICAL_QUERIES).format(company="TechCorp")
        chunks = generate_chunks_for_query("mission", query)
        conversation = generate_conversation(query, chunks, "analytical")
        dataset.append(conversation)
    
    # Generate personal reflection queries (45 examples)
    for i in range(45):
        query_template = random.choice(PERSONAL_REFLECTION_QUERIES)
        # Fill in placeholders
        try:
            query = query_template.format(
                topic=random.choice(["horses", "training", "discipline", "leadership"]),
                theme1=random.choice(["discipline", "compassion", "mastery", "growth"]),
                theme2=random.choice(["compassion", "mastery", "growth", "resilience"]),
                theme3=random.choice(["mastery", "growth", "resilience", "wisdom"])
            )
        except KeyError:
            # Template doesn't have placeholders
            query = query_template
        
        # Determine chunk type based on query content
        if "turning points" in query.lower() or ("patterns" in query.lower() and "thinking" in query.lower()):
            chunks = generate_chunks_for_query("personal_turning_points", query)
        elif "emotional" in query.lower() or "regret" in query.lower() or "responsibility" in query.lower():
            chunks = generate_chunks_for_query("emotional_themes", query)
        else:
            # Mix of both for variety
            if i % 2 == 0:
                chunks = generate_chunks_for_query("personal_turning_points", query)
            else:
                chunks = generate_chunks_for_query("emotional_themes", query)
        
        conversation = generate_conversation(query, chunks, "personal")
        dataset.append(conversation)
    
    # Shuffle for better training
    random.shuffle(dataset)
    
    return dataset

def main():
    """Main function to generate dataset"""
    print("=" * 80)
    print("Generating RAG Analysis Training Dataset")
    print("=" * 80)
    print()
    
    print("Generating 200 training examples (heavily weighted toward co-founder extraction for better learning)...")
    dataset = generate_dataset(200)
    
    print(f"✅ Generated {len(dataset)} training examples")
    print()
    
    # Count by type
    type_counts = {}
    for conv in dataset:
        qtype = conv["query_type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    print("Dataset breakdown:")
    for qtype, count in type_counts.items():
        print(f"  - {qtype}: {count} examples")
    print()
    
    # Save dataset
    output_file = "rag_analysis_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved to: {output_file}")
    print()
    print("=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the dataset: rag_analysis_dataset.json")
    print("2. Run training script: train_rag_analysis_colab.py")
    print()

if __name__ == "__main__":
    main()

