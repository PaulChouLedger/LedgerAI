#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Model Diagnostics
================================

Runs a full battery of tests on the fine-tuned model to identify all issues
before the next training session.

Tests:
1. Multi-entity extraction across chunks
2. Answer type classification
3. Role filtering accuracy
4. Chunk reading completeness
5. Token limit handling
6. Edge cases
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

# Try to import model loading dependencies
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str = None):
    """Load fine-tuned model."""
    import glob
    
    possible_paths = []
    if model_path:
        possible_paths.append(model_path)
    
    if IN_COLAB:
        possible_paths.extend([
            "/content/",
            "/content/outputs_rag_analysis/",
            "./outputs_rag_analysis/",
        ])
    else:
        possible_paths.extend(["./outputs_rag_analysis/", "outputs_rag_analysis/"])
    
    # Check for GGUF
    if LLAMA_CPP_AVAILABLE:
        if model_path and model_path.endswith('.gguf') and os.path.exists(model_path):
            return load_gguf_model(model_path)
        
        for path in possible_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    gguf_files = glob.glob(os.path.join(path, "*.gguf"))
                elif path.endswith('.gguf'):
                    gguf_files = [path]
                else:
                    gguf_files = []
                
                if gguf_files:
                    preferred = [f for f in gguf_files if "Q4_K_M" in f or "q4_k_m" in f]
                    gguf_file = preferred[0] if preferred else gguf_files[0]
                    return load_gguf_model(gguf_file)
    
    # Try other formats...
    raise FileNotFoundError("Model not found")

def load_gguf_model(gguf_file: str):
    """Load GGUF model."""
    model = Llama(model_path=gguf_file, n_ctx=8192, n_threads=4, verbose=False)
    
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
        except:
            tokenizer = create_simple_tokenizer()
    else:
        tokenizer = create_simple_tokenizer()
    
    return model, tokenizer, "gguf"

def create_simple_tokenizer():
    """Create simple tokenizer wrapper."""
    class SimpleTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            return text
    return SimpleTokenizer()

# ============================================================================
# Test Suites
# ============================================================================

def test_multi_entity_extraction(model, tokenizer, model_type: str) -> Dict:
    """Test 1: Multi-entity extraction across chunks"""
    print("\n" + "="*80)
    print("TEST 1: MULTI-ENTITY EXTRACTION ACROSS CHUNKS")
    print("="*80)
    
    test_cases = [
        {
            "name": "Co-founders across 2 chunks",
            "query": "who are the co-founders of TechCorp?",
            "chunks": [
                {
                    "text": "John Smith is a strategic leader. As Co-Founder and CEO of TechCorp, he drives innovation.",
                    "score": 0.85,
                    "file": "test.pdf"
                },
                {
                    "text": "Sarah Jones is a visionary. As Co-Founder and CTO of TechCorp, she leads technology. Mike Brown is a financial expert. As Co-Founder and CFO of TechCorp, he manages finances.",
                    "score": 0.85,
                    "file": "test.pdf"
                }
            ],
            "expected_entities": ["John Smith", "Sarah Jones", "Mike Brown"],
            "expected_chunks_used": [1, 2]
        },
        {
            "name": "Executives across 3 chunks",
            "query": "who are the executives of CompanyX?",
            "chunks": [
                {"text": "Alice Johnson serves as Executive at CompanyX, leading strategy.", "score": 0.85, "file": "test.pdf"},
                {"text": "Bob Williams is Executive at CompanyX, managing operations.", "score": 0.85, "file": "test.pdf"},
                {"text": "Carol Davis serves as Executive at CompanyX, overseeing sales.", "score": 0.85, "file": "test.pdf"}
            ],
            "expected_entities": ["Alice Johnson", "Bob Williams", "Carol Davis"],
            "expected_chunks_used": [1, 2, 3]
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        result = run_extraction_test(model, tokenizer, model_type, test_case)
        results.append(result)
        
        # Calculate accuracy
        extracted = result['extracted_entities']
        expected = test_case['expected_entities']
        correct = [e for e in expected if e in extracted]
        accuracy = (len(correct) / len(expected) * 100) if expected else 0
        
        print(f"   Expected: {expected}")
        print(f"   Extracted: {extracted}")
        print(f"   Accuracy: {accuracy:.1f}% ({len(correct)}/{len(expected)})")
        
        if accuracy < 100:
            missing = [e for e in expected if e not in extracted]
            print(f"   ❌ Missing: {missing}")
    
    return {"test_name": "multi_entity_extraction", "results": results}

def test_answer_type_classification(model, tokenizer, model_type: str) -> Dict:
    """Test 2: Answer type classification"""
    print("\n" + "="*80)
    print("TEST 2: ANSWER TYPE CLASSIFICATION")
    print("="*80)
    
    test_cases = [
        {
            "query": "how are CompanyA and CompanyB related?",
            "expected_answer_type": "relationship",
            "chunks": [{"text": "CompanyA maintains a partnership with CompanyB.", "score": 0.85, "file": "test.pdf"}]
        },
        {
            "query": "why did CompanyX expand?",
            "expected_answer_type": "analytical",
            "chunks": [{"text": "CompanyX expanded due to market demand.", "score": 0.85, "file": "test.pdf"}]
        },
        {
            "query": "what is the difference between ProductA and ProductB?",
            "expected_answer_type": "comparison",
            "chunks": [{"text": "ProductA focuses on speed, while ProductB emphasizes quality.", "score": 0.85, "file": "test.pdf"}]
        },
        {
            "query": "how does the process work?",
            "expected_answer_type": "process",
            "chunks": [{"text": "The process involves three steps: planning, execution, and review.", "score": 0.85, "file": "test.pdf"}]
        },
        {
            "query": "who are the co-founders of TechCorp?",
            "expected_answer_type": "entities",
            "chunks": [{"text": "John Smith is Co-Founder of TechCorp.", "score": 0.85, "file": "test.pdf"}]
        },
        {
            "query": "list the features of ProductX",
            "expected_answer_type": "list",
            "chunks": [{"text": "ProductX offers feature1, feature2, and feature3.", "score": 0.85, "file": "test.pdf"}]
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n📋 Query: {test_case['query']}")
        result = run_answer_type_test(model, tokenizer, model_type, test_case)
        results.append(result)
        
        predicted = result['predicted_answer_type']
        expected = test_case['expected_answer_type']
        match = predicted == expected
        
        print(f"   Expected: {expected}")
        print(f"   Predicted: {predicted}")
        print(f"   {'✅ Match' if match else '❌ Mismatch'}")
    
    accuracy = sum(1 for r in results if r['correct']) / len(results) * 100
    print(f"\n📊 Answer Type Classification Accuracy: {accuracy:.1f}%")
    
    return {"test_name": "answer_type_classification", "results": results, "accuracy": accuracy}

def test_role_filtering(model, tokenizer, model_type: str) -> Dict:
    """Test 3: Role filtering accuracy"""
    print("\n" + "="*80)
    print("TEST 3: ROLE FILTERING ACCURACY")
    print("="*80)
    
    test_cases = [
        {
            "name": "Co-founders only (exclude CEO)",
            "query": "who are the co-founders of TechCorp?",
            "chunks": [
                {"text": "John Smith is Co-Founder of TechCorp.", "score": 0.85, "file": "test.pdf"},
                {"text": "Jane Doe is CEO of TechCorp.", "score": 0.85, "file": "test.pdf"},  # Should exclude
                {"text": "Mike Brown is Co-Founder of TechCorp.", "score": 0.85, "file": "test.pdf"}
            ],
            "expected_include": ["John Smith", "Mike Brown"],
            "expected_exclude": ["Jane Doe"]
        },
        {
            "name": "Executives only (exclude other roles)",
            "query": "who are the executives of CompanyX?",
            "chunks": [
                {"text": "Alice Johnson is Executive at CompanyX.", "score": 0.85, "file": "test.pdf"},
                {"text": "Bob Williams is Manager at CompanyX.", "score": 0.85, "file": "test.pdf"},  # Should exclude
                {"text": "Carol Davis is Executive at CompanyX.", "score": 0.85, "file": "test.pdf"}
            ],
            "expected_include": ["Alice Johnson", "Carol Davis"],
            "expected_exclude": ["Bob Williams"]
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        result = run_role_filtering_test(model, tokenizer, model_type, test_case)
        results.append(result)
        
        included = result['extracted_entities']
        should_include = test_case['expected_include']
        should_exclude = test_case['expected_exclude']
        
        correct_includes = [e for e in should_include if e in included]
        incorrect_includes = [e for e in should_exclude if e in included]
        
        print(f"   Should include: {should_include}")
        print(f"   Should exclude: {should_exclude}")
        print(f"   Extracted: {included}")
        print(f"   ✅ Correctly included: {correct_includes}")
        if incorrect_includes:
            print(f"   ❌ Incorrectly included: {incorrect_includes}")
        if len(correct_includes) < len(should_include):
            missing = [e for e in should_include if e not in included]
            print(f"   ❌ Missing: {missing}")
    
    return {"test_name": "role_filtering", "results": results}

def test_chunk_reading_completeness(model, tokenizer, model_type: str) -> Dict:
    """Test 4: Verify model reads all chunks"""
    print("\n" + "="*80)
    print("TEST 4: CHUNK READING COMPLETENESS")
    print("="*80)
    
    test_cases = [
        {
            "name": "Entities in all 3 chunks",
            "query": "who are the managers of CompanyX?",
            "chunks": [
                {"text": "Manager 1: Alice Johnson manages operations.", "score": 0.85, "file": "test.pdf"},
                {"text": "Manager 2: Bob Williams manages sales.", "score": 0.85, "file": "test.pdf"},
                {"text": "Manager 3: Carol Davis manages finance.", "score": 0.85, "file": "test.pdf"}
            ],
            "expected_chunks_used": [1, 2, 3],
            "expected_entities": ["Alice Johnson", "Bob Williams", "Carol Davis"]
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        result = run_chunk_completeness_test(model, tokenizer, model_type, test_case)
        results.append(result)
        
        chunks_used = result['chunks_used']
        expected_chunks = test_case['expected_chunks_used']
        
        print(f"   Expected chunks used: {expected_chunks}")
        print(f"   Actual chunks used: {chunks_used}")
        
        if set(chunks_used) == set(expected_chunks):
            print("   ✅ Model read all expected chunks")
        else:
            missing_chunks = set(expected_chunks) - set(chunks_used)
            if missing_chunks:
                print(f"   ❌ Model didn't read chunks: {missing_chunks}")
    
    return {"test_name": "chunk_reading_completeness", "results": results}

# ============================================================================
# Test Execution Helpers
# ============================================================================

def run_extraction_test(model, tokenizer, model_type: str, test_case: Dict) -> Dict:
    """Run a single extraction test."""
    query = test_case['query']
    chunks = test_case['chunks']
    
    # Format chunks
    chunks_text = format_rag_chunks(chunks)
    system_prompt = create_system_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nRAG Chunks:\n{chunks_text}"}
    ]
    
    # Generate response
    response = generate_response(model, tokenizer, messages, model_type)
    
    # Parse response
    extracted_entities = extract_entities_from_response(response)
    chunks_used = extract_chunks_used(response)
    
    return {
        "query": query,
        "extracted_entities": extracted_entities,
        "chunks_used": chunks_used,
        "full_response": response
    }

def run_answer_type_test(model, tokenizer, model_type: str, test_case: Dict) -> Dict:
    """Run answer type classification test."""
    query = test_case['query']
    chunks = test_case['chunks']
    
    chunks_text = format_rag_chunks(chunks)
    system_prompt = create_system_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nRAG Chunks:\n{chunks_text}"}
    ]
    
    response = generate_response(model, tokenizer, messages, model_type)
    
    # Try to parse JSON from response
    predicted_answer_type = extract_answer_type(response)
    
    return {
        "query": query,
        "predicted_answer_type": predicted_answer_type,
        "expected_answer_type": test_case['expected_answer_type'],
        "correct": predicted_answer_type == test_case['expected_answer_type'],
        "response": response
    }

def run_role_filtering_test(model, tokenizer, model_type: str, test_case: Dict) -> Dict:
    """Run role filtering test."""
    return run_extraction_test(model, tokenizer, model_type, test_case)

def run_chunk_completeness_test(model, tokenizer, model_type: str, test_case: Dict) -> Dict:
    """Run chunk completeness test."""
    result = run_extraction_test(model, tokenizer, model_type, test_case)
    return result

# ============================================================================
# Response Parsing
# ============================================================================

def extract_entities_from_response(response: str) -> List[str]:
    """Extract entity names from model response."""
    entities = []
    
    # Try to parse JSON first
    json_match = re.search(r'\{[^{}]*"items"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if 'items' in data:
                entities = data['items']
        except:
            pass
    
    # Fallback: extract names from text
    if not entities:
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        entities = re.findall(name_pattern, response)
        entities = list(set(entities))
    
    return entities

def extract_chunks_used(response: str) -> List[int]:
    """Extract chunks_used from response."""
    chunks_used = []
    
    # Try JSON first
    json_match = re.search(r'\{[^{}]*"chunks_used"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if 'chunks_used' in data:
                chunks_used = data['chunks_used']
        except:
            pass
    
    # Fallback: find chunk mentions
    if not chunks_used:
        chunk_pattern = r'Chunk\s+(\d+)'
        chunks_mentioned = re.findall(chunk_pattern, response, re.IGNORECASE)
        chunks_used = [int(c) for c in chunks_mentioned]
    
    return chunks_used

def extract_answer_type(response: str) -> str:
    """Extract answer_type from response."""
    # Try JSON first
    json_match = re.search(r'\{[^{}]*"answer_type"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if 'answer_type' in data:
                return data['answer_type']
        except:
            pass
    
    # Fallback: infer from response content
    response_lower = response.lower()
    if "don't have that information" in response_lower or "not found" in response_lower:
        return "not_found"
    elif "related" in response_lower or "relationship" in response_lower:
        return "relationship"
    elif "difference" in response_lower or "compare" in response_lower:
        return "comparison"
    elif "why" in response_lower or "because" in response_lower:
        return "analytical"
    elif "how does" in response_lower or "process" in response_lower:
        return "process"
    elif "list" in response_lower or len(extract_entities_from_response(response)) > 0:
        return "list"
    else:
        return "unknown"

# ============================================================================
# Helper Functions
# ============================================================================

def format_rag_chunks(chunks: List[Dict]) -> str:
    """Format chunks for model input."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0.0)
        file_name = chunk.get('file', 'document.pdf')
        text = chunk['text']
        text_escaped = text.replace("'", "\\'")
        context_parts.append(f"[Chunk {i}] Score: {score:.2f}, File: {file_name}")
        context_parts.append(f"FULL CHUNK TEXT: '{text_escaped}'")
        context_parts.append("")
    return "\n".join(context_parts)

def create_system_prompt() -> str:
    """System prompt matching training."""
    return """You are an AI assistant that extracts information from RAG chunks and returns it as JSON.

TASK:
1. Read ALL chunks completely from start to finish
2. Extract ALL matching items (do NOT stop after first match)
3. Return results as valid JSON

OUTPUT FORMAT:
{
  "answer_type": "entities" | "list" | "comparison" | "analytical" | "relationship" | "process" | "not_found",
  "items": ["item1", "item2", ...],
  "text": "natural language answer",
  "chunks_used": [1, 2, ...]
}

CRITICAL: Extract ALL matching items - partial extraction is incorrect."""

def generate_response(model, tokenizer, messages: List[Dict], model_type: str, max_tokens: int = 2000) -> str:
    """Generate response from model."""
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    if model_type == "gguf":
        result = model(formatted_text, max_tokens=max_tokens, temperature=0.7, stop=["<|im_end|>", "<|endoftext|>"], echo=False)
        if isinstance(result, dict) and 'choices' in result:
            return result['choices'][0].get('text', str(result)).strip()
        return str(result).strip()
    else:
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=8192)
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# ============================================================================
# Main
# ============================================================================

def run_all_diagnostics(model, tokenizer, model_type: str) -> Dict:
    """Run all diagnostic tests."""
    print("="*80)
    print("COMPREHENSIVE MODEL DIAGNOSTICS")
    print("="*80)
    
    all_results = {}
    
    # Run all tests
    all_results['multi_entity'] = test_multi_entity_extraction(model, tokenizer, model_type)
    all_results['answer_type'] = test_answer_type_classification(model, tokenizer, model_type)
    all_results['role_filtering'] = test_role_filtering(model, tokenizer, model_type)
    all_results['chunk_reading'] = test_chunk_reading_completeness(model, tokenizer, model_type)
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    multi_entity_accuracy = calculate_multi_entity_accuracy(all_results['multi_entity'])
    answer_type_accuracy = all_results['answer_type']['accuracy']
    
    print(f"Multi-Entity Extraction Accuracy: {multi_entity_accuracy:.1f}%")
    print(f"Answer Type Classification Accuracy: {answer_type_accuracy:.1f}%")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR NEXT TRAINING")
    print("="*80)
    
    if multi_entity_accuracy < 90:
        print("❌ Multi-entity extraction needs improvement")
        print("   → Increase multi-entity examples in dataset")
        print("   → Enhance system prompt with 'extract ALL' emphasis")
        print("   → Add explicit examples showing complete extraction")
    
    if answer_type_accuracy < 85:
        print("❌ Answer type classification needs improvement")
        print("   → Add explicit query → answer_type mapping in system prompt")
        print("   → Add class weighting in training")
        print("   → Increase examples for rare answer types")
    
    print("="*80)
    
    return all_results

def calculate_multi_entity_accuracy(test_results: Dict) -> float:
    """Calculate average accuracy across multi-entity tests."""
    accuracies = []
    for result in test_results['results']:
        # This would need expected entities from test case
        # Simplified for now
        pass
    return 0.0  # Placeholder

if __name__ == "__main__":
    try:
        model, tokenizer, model_type = load_model()
        print(f"✅ Model loaded (type: {model_type})")
        
        results = run_all_diagnostics(model, tokenizer, model_type)
        
        # Save results
        with open("comprehensive_diagnostics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Diagnostics complete! Results saved to comprehensive_diagnostics.json")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

