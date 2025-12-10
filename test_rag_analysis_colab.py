#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Chunk Analysis - Colab Test Script
========================================

This script tests the fine-tuned model's ability to:
1. Read RAG chunks completely from start to finish
2. Evaluate relevance (HIGH/MEDIUM/LOW) based on scores
3. Extract only HIGH relevance information
4. Synthesize answers from multiple chunks
5. Differentiate between entities (e.g., TechCorp vs DataSystems co-founders)
6. Handle mixed-content chunks (relevant + irrelevant information)

To use in Colab:
1. Upload this script and your fine-tuned model (trained on rag_analysis_dataset.json)
2. Run: !pip install unsloth transformers accelerate llama-cpp-python
3. Run this script
"""

import json
import re
import os
import glob
from typing import List, Dict, Optional, Tuple
import torch

# Try to import Unsloth (for HuggingFace format models)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available. Will try standard transformers.")

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available.")

# Try to import llama_cpp (for GGUF format models)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not available. Will try HuggingFace format.")

# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load fine-tuned model (tries multiple formats)."""
    print("\n" + "="*80)
    print("Model Load Debug")
    print("="*80)
    print(f"Working directory: {os.getcwd()}")
    print(f"Exists outputs_rag_analysis/: {os.path.exists('outputs_rag_analysis/')}")
    print(f"Exists gguf_model_rag_analysis/: {os.path.exists('gguf_model_rag_analysis/')}")
    print()
    # Try Unsloth format
    if UNSLOTH_AVAILABLE:
        try:
            if os.path.exists("outputs_rag_analysis/"):
                print("📦 Loading Unsloth model from outputs_rag_analysis/...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="outputs_rag_analysis/",
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=False,
                )
                model_type = "unsloth"
                print(f"✅ Loaded Unsloth model (type: {model_type}) from outputs_rag_analysis/")
                return model, tokenizer, model_type
        except Exception as e:
            print(f"⚠️  Could not load Unsloth model: {e}")
    
    # Try standard transformers
    if TRANSFORMERS_AVAILABLE:
        try:
            if os.path.exists("outputs_rag_analysis/"):
                print("📦 Loading HuggingFace model from outputs_rag_analysis/...")
                tokenizer = AutoTokenizer.from_pretrained("outputs_rag_analysis/")
                model = AutoModelForCausalLM.from_pretrained(
                    "outputs_rag_analysis/",
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                model_type = "transformers"
                print(f"✅ Loaded HuggingFace model (type: {model_type}) from outputs_rag_analysis/")
                return model, tokenizer, model_type
        except Exception as e:
            print(f"⚠️  Could not load HuggingFace model: {e}")
    
    # Try GGUF format
    if LLAMA_CPP_AVAILABLE:
        try:
            gguf_files = glob.glob("gguf_model_rag_analysis/*.gguf")
            if not gguf_files:
                gguf_files = glob.glob("gguf_model_rag_analysis/*-rag-analysis.gguf")
            if gguf_files:
                print(f"📦 Loading GGUF model from gguf_model_rag_analysis/: {gguf_files[0]}...")
                model = Llama(
                    model_path=gguf_files[0],
                    n_ctx=2048,
                    verbose=False,
                )
                model_type = "gguf"
                print(f"✅ Loaded GGUF model (type: {model_type})")
                return model, None, model_type
        except Exception as e:
            print(f"⚠️  Could not load GGUF model: {e}")
    
    raise RuntimeError("❌ Could not load any model format. Please ensure model files exist.")

# ============================================================================
# Inference
# ============================================================================

def generate_response(model, tokenizer, messages: List[Dict], model_type: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
    """Generate response from model."""
    if model_type == "unsloth":
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "transformers":
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    elif model_type == "gguf":
        # Format messages for GGUF
        formatted_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        formatted_text += "<|im_start|>assistant\n"
        
        response = model(
            formatted_text,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        return response["choices"][0]["text"].strip()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ============================================================================
# Test Cases
# ============================================================================

TEST_CASES = [
    {
        "name": "Co-Founders Query (Single Company)",
        "query": "who are the co-founders of TechCorp?",
        "chunks": [
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
        ],
        "expected_entities": ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Williams"],
        "should_not_contain": ["Alex Chen", "Maria Rodriguez", "DataSystems"]
    },
    {
        "name": "Co-Founders Query (Multi-Company - TechCorp)",
        "query": "who are the co-founders of TechCorp?",
        "chunks": [
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
        ],
        "expected_entities": ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Williams"],
        "should_not_contain": ["Alex Chen", "Maria Rodriguez", "David Kim", "Robert Taylor"]
    },
    {
        "name": "Co-Founders Query (Multi-Company - DataSystems)",
        "query": "who are the co-founders of DataSystems Inc.?",
        "chunks": [
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
        ],
        "expected_entities": ["Alex Chen", "Maria Rodriguez", "David Kim", "Robert Taylor"],
        "should_not_contain": ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Williams"]
    },
    {
        "name": "Mission Query (Mixed Content)",
        "query": "what is TechCorp's mission?",
        "chunks": [
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
        ],
        "expected_keywords": ["mission", "enterprise intelligence", "governance", "AI-powered", "blockchain"],
        "should_not_contain": ["annual report", "customer support", "quarterly targets"]
    },
    {
        "name": "Personal Reflection Query (Mixed Content)",
        "query": "help me map the major turning points of my life and how they shaped my identity",
        "chunks": [
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
        ],
        "expected_keywords": ["2015", "decision", "corporate job", "consulting", "2018", "father", "mortality", "priorities"],
        "should_not_contain": ["weather", "hiking", "apartment lease", "funeral arrangements"]
    },
]

# ============================================================================
# Test Execution
# ============================================================================

def format_rag_chunks(chunks: List[Dict]) -> str:
    """Format RAG chunks for the prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] Score: {chunk['score']:.3f}, File: {chunk['file']}, Preview: '{chunk['text'][:80]}...'")
        context_parts.append(f"[{i}] FULL CHUNK TEXT: '{chunk['text']}'")
    return "\n".join(context_parts)

def create_system_prompt() -> str:
    """Create system prompt for RAG analysis."""
    return """You are an AI assistant trained to analyze RAG (Retrieval-Augmented Generation) chunks and extract relevant information.

Your task is to:
1. Read every chunk completely from start to finish - DO NOT stop reading once you find relevant information
2. Evaluate relevance for each chunk:
   - HIGH (score ≥0.70): Information that directly and explicitly answers the query
   - MEDIUM (0.50-0.69): Information that is related but requires inference
   - LOW (score <0.50): Information that mentions similar terms but doesn't actually answer the query
3. Extract only HIGH relevance information - be precise about what exactly matches the query
4. For list questions: Find EVERY matching item in EVERY chunk - read each chunk completely
5. For analytical questions: Extract all relevant information from all chunks before synthesizing
6. Format your analysis showing: RELEVANCE EVALUATION → EXTRACTING INFORMATION → SYNTHESIS → Final Answer

Always end with a brief, natural follow-up question."""

def run_test_case(model, tokenizer, test_case: Dict, model_type: str) -> Dict:
    """Run a single test case and return results."""
    print(f"\n{'='*80}")
    print(f"Test: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Query: {test_case['query']}")
    print(f"\nChunks provided: {len(test_case['chunks'])}")
    
    # Format chunks
    chunks_text = format_rag_chunks(test_case['chunks'])
    
    # Create messages
    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": f"Query: {test_case['query']}\n\nRAG Chunks:\n{chunks_text}"}
    ]
    
    # Generate response
    print("\n🤖 Generating response...")
    response = generate_response(model, tokenizer, messages, model_type, max_tokens=800, temperature=0.7)
    
    print(f"\n📝 Response:\n{response}\n")
    
    # Evaluate response
    results = {
        "test_name": test_case['name'],
        "query": test_case['query'],
        "response": response,
        "passed": True,
        "issues": []
    }
    
    # Check for expected entities/keywords
    if "expected_entities" in test_case:
        for entity in test_case['expected_entities']:
            if entity not in response:
                results["passed"] = False
                results["issues"].append(f"Missing expected entity: {entity}")
            else:
                print(f"✅ Found expected entity: {entity}")
    
    if "expected_keywords" in test_case:
        for keyword in test_case['expected_keywords']:
            if keyword.lower() not in response.lower():
                results["passed"] = False
                results["issues"].append(f"Missing expected keyword: {keyword}")
            else:
                print(f"✅ Found expected keyword: {keyword}")
    
    # Check for entities/keywords that should NOT be present
    if "should_not_contain" in test_case:
        for item in test_case['should_not_contain']:
            if item in response:
                results["passed"] = False
                results["issues"].append(f"Should not contain: {item}")
            else:
                print(f"✅ Correctly excluded: {item}")
    
    # Check for proper analysis structure
    if "RELEVANCE EVALUATION" not in response:
        results["issues"].append("Missing RELEVANCE EVALUATION section")
    else:
        print("✅ Contains RELEVANCE EVALUATION section")
    
    if "EXTRACTING INFORMATION" not in response and "EXTRACTING THEMES" not in response:
        results["issues"].append("Missing EXTRACTING INFORMATION/THEMES section")
    else:
        print("✅ Contains EXTRACTING INFORMATION/THEMES section")
    
    if "SYNTHESIS" not in response:
        results["issues"].append("Missing SYNTHESIS section")
    else:
        print("✅ Contains SYNTHESIS section")
    
    return results

def run_all_tests(model, tokenizer, model_type: str):
    """Run all test cases."""
    print("\n" + "="*80)
    print("Running RAG Analysis Tests")
    print("="*80)
    
    all_results = []
    passed = 0
    failed = 0
    
    for test_case in TEST_CASES:
        try:
            result = run_test_case(model, tokenizer, test_case, model_type)
            all_results.append(result)
            
            if result["passed"]:
                passed += 1
                print(f"\n✅ Test PASSED: {test_case['name']}")
            else:
                failed += 1
                print(f"\n❌ Test FAILED: {test_case['name']}")
                if result["issues"]:
                    print("Issues:")
                    for issue in result["issues"]:
                        print(f"  - {issue}")
        except Exception as e:
            print(f"\n❌ Test ERROR: {test_case['name']}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(TEST_CASES)*100:.1f}%")
    print("="*80)
    
    return all_results

# ============================================================================
# Interactive Mode
# ============================================================================

def run_interactive_test(model, tokenizer, model_type: str):
    """Run interactive test mode."""
    print("\n" + "="*80)
    print("Interactive RAG Analysis Test")
    print("="*80)
    print("\nEnter queries to test RAG chunk analysis.")
    print("Type 'exit' to quit, 'test' to run all test cases.")
    print()
    
    messages = [
        {"role": "system", "content": create_system_prompt()}
    ]
    
    while True:
        query = input("\n💬 Query: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'test':
            run_all_tests(model, tokenizer, model_type)
            continue
        elif not query:
            continue
        
        # For interactive mode, create sample chunks
        print("\n📚 Using sample chunks...")
        sample_chunks = [
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
        ]
        
        chunks_text = format_rag_chunks(sample_chunks)
        user_message = f"Query: {query}\n\nRAG Chunks:\n{chunks_text}"
        
        messages.append({"role": "user", "content": user_message})
        
        print("\n🤖 Generating response...")
        response = generate_response(model, tokenizer, messages, model_type, max_tokens=800, temperature=0.7)
        
        print(f"\n📝 Response:\n{response}\n")
        
        messages.append({"role": "assistant", "content": response})

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RAG Chunk Analysis - Colab Test Script")
    print("=" * 80)
    print("\nLoading model...")
    
    try:
        model, tokenizer, model_type = load_model()
        print(f"\n✅ Model loaded successfully (type: {model_type})")
        
        print("\n" + "=" * 80)
        print("Choose mode:")
        print("1. Run all test cases")
        print("2. Interactive mode")
        print("=" * 80)
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            run_all_tests(model, tokenizer, model_type)
        elif choice == "2":
            run_interactive_test(model, tokenizer, model_type)
        else:
            print("Invalid choice. Running all tests...")
            run_all_tests(model, tokenizer, model_type)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

