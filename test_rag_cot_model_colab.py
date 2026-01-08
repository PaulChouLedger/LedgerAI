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
    from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "outputs_rag_cot"  # Path to LoRA adapters or merged model
MAX_SEQ_LENGTH = 4096

# The exact Deep Scan system prompt used in training
SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items."""

# ============================================================================
# Test Scenarios
# ============================================================================

def get_test_scenarios():
    """Returns a list of test scenarios for evaluation"""
    return [
        {
            "name": "LedgerAI Co-Founders (4 co-founders)",
            "query": "Who are the co-founders of LedgerAI?",
            "context": """has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. As Co-Founder of Soler Salva LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures LedgerAI's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing LedgerAI's position as a trailblazer in AI-powered business intelligence. Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at Scarinci Hollenbeck, Attorneys at Law, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that LedgerAI continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-powered business intelligence. Liam Hugill is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as LedgerAI's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that LedgerAI's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying LedgerAI's brand, expanding its reach, and solidifying trust among investors and supporters.
---
Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering.
---
into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. A recognized thought leader, Paul has been featured on TED Talks and major global conferences for over a decade, solidifying his role as a pioneer at the forefront of AI, blockchain, and financial innovation. Bob Carella is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.""",
            "expected": ["Paul Chou", "Bob Carella", "David Lara", "Jorge Guinovart"],
            "not_expected": ["Albert Soler", "Will Specht", "Peter Moeller", "Liam Hugill"],
            "answer_type": "person"
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

def test_scenario(model, tokenizer, scenario):
    """Test a single scenario"""
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
                temperature=0.05,
                do_sample=True,
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
            
        # Extract and print full reasoning section
        reasoning_section = ""
        if "REASONING:" in assistant_response:
            # Find reasoning section
            reasoning_start = assistant_response.find("REASONING:")
            final_answer_start = assistant_response.find("FINAL ANSWER:")
            if final_answer_start > reasoning_start:
                reasoning_section = assistant_response[reasoning_start:final_answer_start].strip()
            else:
                reasoning_section = assistant_response[reasoning_start:].strip()
        
        print(f"\n📝 Model Response:")
        if reasoning_section:
            print(f"\n{'='*80}")
            print(f"🧠 FULL REASONING SECTION:")
            print(f"{'='*80}")
            print(reasoning_section)
            print(f"{'='*80}")
            
            # Extract DISCARD/KEEP decisions for debugging
            discard_items = []
            keep_items = []
            for line in reasoning_section.split('\n'):
                if '[DISCARD]' in line or '[ DISCARD]' in line:
                    # Extract item name before DISCARD
                    item_match = re.search(r'- Item:\s*([^\n-]+?)(?:\s*-|$)', line, re.IGNORECASE)
                    if item_match:
                        item_name = item_match.group(1).strip()
                        discard_items.append(item_name)
                elif '[KEEP]' in line or '[ KEEP]' in line:
                    # Extract item name before KEEP
                    item_match = re.search(r'- Item:\s*([^\n-]+?)(?:\s*-|$)', line, re.IGNORECASE)
                    if item_match:
                        item_name = item_match.group(1).strip()
                        keep_items.append(item_name)
            
            if keep_items:
                print(f"\n✅ Items marked [KEEP]: {keep_items}")
            if discard_items:
                print(f"❌ Items marked [DISCARD]: {discard_items}")
        else:
            print(f"{assistant_response[:500]}...")
        
        # Check for CoT
        has_cot, indicators = check_cot_reasoning(assistant_response)
        cot_status = "✅ CoT reasoning detected" if has_cot else "⚠️  Explicit CoT reasoning NOT detected"
        print(f"\n🧠 CoT Reasoning Check:\n   {cot_status} (found indicators: {indicators})")
        
        # Print final answer section
        if "FINAL ANSWER:" in assistant_response:
            final_answer_section = assistant_response.split("FINAL ANSWER:")[-1].strip()
            print(f"\n{'='*80}")
            print(f"📋 FINAL ANSWER SECTION:")
            print(f"{'='*80}")
            print(final_answer_section[:1000])  # Print first 1000 chars of final answer
            print(f"{'='*80}")
        
        
        # CLEAN RESPONSE: The only text that counts is after the FINAL ANSWER header
        temp_response = assistant_response.strip()
        if temp_response.startswith('t'): temp_response = temp_response[1:].strip()
        
        clean_response = ""
        if "FINAL ANSWER:" in temp_response:
            clean_response = temp_response.split("FINAL ANSWER:")[-1].strip()
        elif "Final Answer:" in temp_response:
            clean_response = temp_response.split("Final Answer:")[-1].strip()
        elif "- End of scan." in temp_response:
            # If FINAL ANSWER is missing but End of scan is there, the answer is usually after End of scan
            clean_response = temp_response.split("- End of scan.")[-1].strip()
        else:
            # Fallback: find the longest block of text that doesn't look like reasoning
            blocks = temp_response.split('\n\n')
            if len(blocks) > 1:
                clean_response = blocks[-1].strip()
            else:
                # If no double newlines, use the very last paragraph but STRIP any [KEEP/DISCARD] tags
                lines = temp_response.split('\n')
                clean_response = lines[-1].strip()
            
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
        
        # Calculate score
        if not expected:
            score = 100.0 if not incorrectly_included else 0.0
        else:
            # Weighted score: correct - 0.5 * incorrect
            points = len(correctly_found) / len(expected) * 100
            penalty = (len(incorrectly_included) / max(1, len(found_items))) * 50
            score = max(0, points - penalty)
        
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
        
        print(f"   📈 Score: {score:.2f}%")
        
        return {
            "name": scenario['name'],
            "score": score,
            "has_cot": has_cot,
            "correct": correctly_found,
            "missing": missing,
            "incorrect": incorrectly_included,
            "answer_type": answer_type
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
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model path '{MODEL_PATH}' not found!")
        return
    
    print(f"\n================================================================================")
    print(f"Loading Trained Model")
    print(f"================================================================================")
    print(f"✅ Found model at: {MODEL_PATH}")
    
    # Load model and tokenizer
    print(f"   Attempting to load model...")
    
    if HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    print(f"✅ Model loaded successfully")
    print(f"   Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Run tests
    scenarios = get_test_scenarios()
    results = []
    
    for scenario in scenarios:
        result = test_scenario(model, tokenizer, scenario)
        if result:
            results.append(result)
            
    # Summary
    print(f"\n{'='*80}")
    print(f"Test Summary")
    print(f"{'='*80}")
    
    total_score = 0
    cot_count = 0
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"   CoT Reasoning: {'✅' if r['has_cot'] else '❌'}")
        print(f"   Score: {r['score']:.2f}%")
        if r['correct']:
            print(f"   ✅ Found: {r['correct']}")
        if r['missing']:
            print(f"   ⚠️  Missing: {r['missing']}")
        if r['incorrect']:
            print(f"   ❌ Incorrect: {r['incorrect']}")
            
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
    if len(type_scores) > 1:
        print(f"\n   Breakdown by Query Type:")
        for qtype, scores in type_scores.items():
            avg_type_score = sum(scores) / len(scores)
            print(f"      {qtype.title()}: {avg_type_score:.2f}% ({len(scores)} tests)")
    print(f"{'='*80}")
    
    if avg_score > 80 and cot_pct > 75:
        print("\n✅ Model shows strong accuracy and CoT reasoning!")
    else:
        print("\n❌ Model needs more training. Consider:")
        print("   - Increasing training epochs (30-40)")
        print("   - Adding more training examples (15-20 total)")
        print("   - Adjusting learning rate")
        print("   - Fixing input truncation (LedgerAI test failed due to truncation)")

if __name__ == "__main__":
    run_tests()
