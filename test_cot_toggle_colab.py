#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoT Toggle Model Test Suite (Colab Version)
Tests the fine-tuned Qwen 2.5 model's conditional CoT behavior:
- WITH CoT when RAG context is provided (CoT system prompt)
- WITHOUT CoT for conversational queries (conversational system prompt)
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

MODEL_PATH = "outputs_cot_toggle_merged"  # Path to merged CoT toggle model
MAX_SEQ_LENGTH = 4096

# CoT System Prompt (for RAG queries)
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

# Conversational System Prompt (for non-RAG queries)
CONVERSATIONAL_SYSTEM_PROMPT = """You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation.
You act as a proactive AI agent guiding users to better outcomes through gentle guidance.

CRITICAL RULES:
- Only provide logical, factual responses. Avoid hallucination at all costs.
- IMPORTANT: Commands and instructions like 'Give me X', 'Tell me about Y', 'Show me Z' are VALID requests and should be answered normally using your general knowledge.
- For general knowledge questions (recipes, facts, etc.), use your general knowledge to provide helpful answers.
- Keep responses VERY SHORT - maximum 2-3 sentences total.
- Be conversational, friendly, and natural.
- Always end your response with a brief, natural question. Examples: 'Would you like more information about this?' or 'Is there anything else I can help you with?'"""

# ============================================================================
# Test Scenarios
# ============================================================================

def get_rag_test_scenarios():
    """RAG test scenarios (should use CoT)"""
    return [
        {
            "name": "LedgerAI Co-Founders (4 co-founders)",
            "query": "Who are the co-founders of LedgerAI?",
            "context": """has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. As Co-Founder of Soler Salva LLP, he has led high-profile cases in entertainment, media, and corporate law, specializing in federal and state litigation, licensing, sponsorships, and complex commercial transactions. His deep understanding of intellectual property protection, regulatory frameworks, and emerging technologies ensures LedgerAI's AI-driven innovations remain legally sound, compliant, and strategically positioned for growth. With extensive experience advising industry leaders, Albert provides critical oversight on AI governance, tokenized ecosystems, and enterprise partnerships, reinforcing LedgerAI's position as a trailblazer in AI-powered business intelligence. Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. With over a decade of experience in technology, legal services, and professional consulting, he has built a reputation for accelerating business expansion, optimizing market positioning, and forging high-value partnerships. As Chief Growth Officer at Scarinci Hollenbeck, Attorneys at Law, Peter has successfully led strategic business planning, market research, SEO management, content development, and enterprise relationship management—making him a key player in driving brand visibility and revenue growth. His expertise in business strategy, recruiting, and communications ensures that LedgerAI continues to expand its reach, attract top-tier clients, and solidify its position as a leader in AI-powered business intelligence. Liam Hugill is a master of influence, engagement, and community-building in the Web3 and cryptocurrency space, being a natural fit as LedgerAI's Ambassador of Influence and Engagement. With an unmatched ability to ignite passion, foster loyalty, and drive momentum, Liam ensures that LedgerAI's community remains informed, engaged, and excited about the project's vision and growth. His expertise in navigating the fast-paced, ever-evolving crypto landscape makes him a critical force in amplifying LedgerAI's brand, expanding its reach, and solidifying trust among investors and supporters.
---
Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering.
---
into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. Bob Carella is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.""",
            "expected": ["Paul Chou", "Bob Carella", "David Lara", "Jorge Guinovart"],
            "not_expected": ["Albert Soler", "Will Specht", "Peter Moeller", "Liam Hugill"],
            "answer_type": "person",
            "should_use_cot": True
        },
        {
            "name": "TechCorp Co-Founders (3 co-founders)",
            "query": "Who are the co-founders of TechCorp?",
            "context": """John Smith is the CEO and Co-Founder of TechCorp. Sarah Johnson is the Chief Technology Officer at TechCorp. Michael Brown is the Co-Founder and Chief Operating Officer of TechCorp. Emily Davis is the Head of Marketing. Robert Wilson is an External Advisor. David Martinez is the Co-Founder and Chief Product Officer. Lisa Anderson is the Chief Financial Officer.""",
            "expected": ["John Smith", "Michael Brown", "David Martinez"],
            "not_expected": ["Sarah Johnson", "Emily Davis", "Robert Wilson", "Lisa Anderson"],
            "answer_type": "person",
            "should_use_cot": True
        },
    ]

def get_conversational_test_scenarios():
    """Conversational test scenarios (should NOT use CoT)"""
    return [
        {
            "name": "Recipe Query",
            "query": "Give me a recipe for cooked chicken.",
            "expected_cot": False,  # Should NOT use CoT
            "expected_keywords": ["chicken", "cook", "recipe", "season"],
            "should_use_cot": False
        },
        {
            "name": "General Knowledge - Capital",
            "query": "What is the capital of France?",
            "expected_cot": False,  # Should NOT use CoT
            "expected_keywords": ["Paris", "France", "capital"],
            "should_use_cot": False
        },
        {
            "name": "Greeting",
            "query": "Hello!",
            "expected_cot": False,  # Should NOT use CoT
            "expected_keywords": ["hello", "hi", "help"],
            "should_use_cot": False
        },
        {
            "name": "General Knowledge - Planets",
            "query": "How many planets are in our solar system?",
            "expected_cot": False,  # Should NOT use CoT
            "expected_keywords": ["8", "eight", "planets", "solar system"],
            "should_use_cot": False
        },
    ]

# ============================================================================
# Helper Functions
# ============================================================================

def check_cot_reasoning(response: str) -> Tuple[bool, List[str]]:
    """Check if response contains CoT reasoning indicators"""
    indicators = []
    has_cot = False
    
    cot_patterns = [
        (r'REASONING:', 'REASONING:'),
        (r'Reasoning:', 'Reasoning:'),
        (r'FINAL ANSWER:', 'FINAL ANSWER:'),
        (r'Final Answer:', 'Final Answer:'),
        (r'- Item:', 'Item:'),
        (r'- Evidence:', 'Evidence:'),
        (r'- Action:', 'Action:'),
        (r'\[KEEP\]', '[KEEP]'),
        (r'\[DISCARD\]', '[DISCARD]'),
        (r'- End of scan', 'End of scan'),
    ]
    
    for pattern, label in cot_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            indicators.append(label)
            has_cot = True
    
    return has_cot, indicators

# ============================================================================
# Test Functions
# ============================================================================

def test_rag_scenario(model, tokenizer, scenario):
    """Test a RAG scenario (should use CoT)"""
    print(f"\n{'='*80}")
    print(f"Testing RAG Scenario: {scenario['name']}")
    print(f"{'='*80}")
    
    # Format with RAG context (CoT system prompt)
    user_prompt = f"Knowledge context: {scenario['context']}\n---\nQuestion: {scenario['query']}"
    
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
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
            prompt_text = f"{COT_SYSTEM_PROMPT}\n\n{user_prompt}"
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
        
        # Extract assistant response
        assistant_marker = "assistant\n"
        if assistant_marker in response:
            assistant_response = response.split(assistant_marker)[-1].strip()
        else:
            assistant_response = response.strip()
        
        print(f"\n📝 Model Response (first 1200 chars):\n{assistant_response[:1200]}...")
        
        # Check for CoT
        has_cot, indicators = check_cot_reasoning(assistant_response)
        if scenario.get('should_use_cot', True):
            if has_cot:
                print(f"✅ CoT reasoning detected (expected)")
            else:
                print(f"❌ CoT reasoning NOT detected (expected CoT for RAG queries)")
        else:
            if has_cot:
                print(f"❌ CoT reasoning detected (should NOT use CoT)")
            else:
                print(f"✅ No CoT reasoning (expected for non-RAG queries)")
        
        print(f"   Indicators found: {indicators}")
        
        # Extract final answer (if CoT)
        if has_cot:
            if "FINAL ANSWER:" in assistant_response:
                final_answer_section = assistant_response.split("FINAL ANSWER:")[-1].strip()
            elif "Final Answer:" in assistant_response:
                final_answer_section = assistant_response.split("Final Answer:")[-1].strip()
            else:
                final_answer_section = assistant_response
            
            # Print full FINAL ANSWER section for debugging
            print(f"\n🔍 Full FINAL ANSWER section (first 800 chars):\n{final_answer_section[:800]}")
            # Also check if there's more content after (might be truncated)
            if len(final_answer_section) > 800:
                print(f"   ... (truncated, total length: {len(final_answer_section)} chars)")
            clean_response = final_answer_section
        else:
            clean_response = assistant_response
        
        # Remove CoT markers
        clean_response = re.sub(r'\[(KEEP|DISCARD|Action|Result)\]', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'(?m)^- .*$', '', clean_response).strip()
        
        # Score (for RAG scenarios)
        if 'expected' in scenario:
            expected = scenario.get('expected', [])
            not_expected = scenario.get('not_expected', [])
            
            # Extract found items
            found_items = []
            for item in expected:
                if item.lower() in clean_response.lower():
                    found_items.append(item)
            
            # Check for incorrectly included items
            incorrect_items = []
            for item in not_expected:
                if item.lower() in clean_response.lower():
                    incorrect_items.append(item)
            
            # Calculate score
            if len(expected) == 0:
                score = 1.0 if len(incorrect_items) == 0 else 0.0
            else:
                correct = len(found_items)
                missing = len(expected) - correct
                score = correct / len(expected) if len(expected) > 0 else 0.0
            
            print(f"\n📊 Results:")
            print(f"   Expected: {expected}")
            print(f"   Found: {found_items}")
            print(f"   Missing: {[e for e in expected if e not in found_items]}")
            print(f"   Incorrectly included: {incorrect_items}")
            print(f"   Score: {score:.2%}")
            
            return {
                "name": scenario['name'],
                "has_cot": has_cot,
                "expected_cot": scenario.get('should_use_cot', True),
                "cot_correct": has_cot == scenario.get('should_use_cot', True),
                "score": score,
                "found_items": found_items,
                "missing_items": [e for e in expected if e not in found_items],
                "incorrect_items": incorrect_items
            }
        else:
            return {
                "name": scenario['name'],
                "has_cot": has_cot,
                "expected_cot": scenario.get('should_use_cot', False),
                "cot_correct": has_cot == scenario.get('should_use_cot', False),
            }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_conversational_scenario(model, tokenizer, scenario):
    """Test a conversational scenario (should NOT use CoT)"""
    print(f"\n{'='*80}")
    print(f"Testing Conversational Scenario: {scenario['name']}")
    print(f"{'='*80}")
    
    # Format without RAG context (conversational system prompt)
    messages = [
        {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
        {"role": "user", "content": scenario['query']}
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
            prompt_text = f"{CONVERSATIONAL_SYSTEM_PROMPT}\n\n{scenario['query']}"
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
                max_new_tokens=512,  # Shorter for conversational
                temperature=0.7,  # Higher for conversational
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        assistant_marker = "assistant\n"
        if assistant_marker in response:
            assistant_response = response.split(assistant_marker)[-1].strip()
        else:
            assistant_response = response.strip()
        
        print(f"\n📝 Model Response:\n{assistant_response}")
        
        # Check for CoT (should NOT have CoT)
        has_cot, indicators = check_cot_reasoning(assistant_response)
        if scenario.get('should_use_cot', False):
            if has_cot:
                print(f"✅ CoT reasoning detected (expected)")
            else:
                print(f"❌ CoT reasoning NOT detected (expected CoT)")
        else:
            if has_cot:
                print(f"❌ CoT reasoning detected (should NOT use CoT for conversational queries)")
                print(f"   Indicators found: {indicators}")
            else:
                print(f"✅ No CoT reasoning (expected for conversational queries)")
        
        # Check for expected keywords
        if 'expected_keywords' in scenario:
            found_keywords = []
            response_lower = assistant_response.lower()
            for keyword in scenario['expected_keywords']:
                if keyword.lower() in response_lower:
                    found_keywords.append(keyword)
            
            print(f"\n📊 Keyword Check:")
            print(f"   Expected keywords: {scenario['expected_keywords']}")
            print(f"   Found keywords: {found_keywords}")
            print(f"   Missing: {[k for k in scenario['expected_keywords'] if k.lower() not in response_lower]}")
        
        return {
            "name": scenario['name'],
            "has_cot": has_cot,
            "expected_cot": scenario.get('should_use_cot', False),
            "cot_correct": has_cot == scenario.get('should_use_cot', False),
            "response": assistant_response[:200]  # First 200 chars
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Main Test Function
# ============================================================================

def main():
    print("=" * 80)
    print("CoT Toggle Model Test Suite")
    print("=" * 80)
    print()
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model path '{MODEL_PATH}' not found!")
        print(f"   Please train the model first using train_cot_toggle_colab.py")
        return
    
    print(f"✅ Found model at: {MODEL_PATH}")
    
    # Load model and tokenizer
    print("\n📦 Loading model and tokenizer...")
    try:
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
        
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get test scenarios
    rag_scenarios = get_rag_test_scenarios()
    conversational_scenarios = get_conversational_test_scenarios()
    
    print(f"\n📋 Test Plan:")
    print(f"   RAG scenarios (should use CoT): {len(rag_scenarios)}")
    print(f"   Conversational scenarios (should NOT use CoT): {len(conversational_scenarios)}")
    print()
    
    # Test RAG scenarios
    print("=" * 80)
    print("Testing RAG Scenarios (Should Use CoT)")
    print("=" * 80)
    
    rag_results = []
    for scenario in rag_scenarios:
        result = test_rag_scenario(model, tokenizer, scenario)
        if result:
            rag_results.append(result)
    
    # Test conversational scenarios
    print("\n" + "=" * 80)
    print("Testing Conversational Scenarios (Should NOT Use CoT)")
    print("=" * 80)
    
    conv_results = []
    for scenario in conversational_scenarios:
        result = test_conversational_scenario(model, tokenizer, scenario)
        if result:
            conv_results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    # RAG results
    if rag_results:
        rag_cot_correct = sum(1 for r in rag_results if r.get('cot_correct', False))
        rag_scores = [r.get('score', 0) for r in rag_results if 'score' in r]
        avg_rag_score = sum(rag_scores) / len(rag_scores) if rag_scores else 0
        
        print(f"\n📊 RAG Scenarios:")
        print(f"   CoT behavior correct: {rag_cot_correct}/{len(rag_results)} ({rag_cot_correct/len(rag_results)*100:.1f}%)")
        if rag_scores:
            print(f"   Average accuracy: {avg_rag_score:.2%}")
    
    # Conversational results
    if conv_results:
        conv_cot_correct = sum(1 for r in conv_results if r.get('cot_correct', False))
        
        print(f"\n📊 Conversational Scenarios:")
        print(f"   CoT behavior correct: {conv_cot_correct}/{len(conv_results)} ({conv_cot_correct/len(conv_results)*100:.1f}%)")
    
    # Overall
    all_results = rag_results + conv_results
    if all_results:
        overall_cot_correct = sum(1 for r in all_results if r.get('cot_correct', False))
        print(f"\n✅ Overall CoT Toggle Accuracy: {overall_cot_correct}/{len(all_results)} ({overall_cot_correct/len(all_results)*100:.1f}%)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
