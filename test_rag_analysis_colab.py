#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple RAG Chunk Analysis Test Script
=====================================

Tests the fine-tuned model's ability to analyze RAG chunks and extract information.
The model should work with minimal guidance - just provide query and chunks.

Usage:
    - Interactive mode: Run script and provide query + chunks
    - Test mode: Define test cases with query and chunks (no hardcoded expectations)
"""

import json
import os
from typing import List, Dict, Optional

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import llama_cpp (for GGUF format models)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# ============================================================================
# Model Loading
# ============================================================================

def load_base_model():
    """Load base model (before fine-tuning) for comparison."""
    if TRANSFORMERS_AVAILABLE:
        print("📦 Loading BASE model (Qwen2.5-1.5B-Instruct)...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            import os
            
            # Disable Unsloth if it's interfering
            os.environ.setdefault('UNSLOTH_OFF', '1')
            
            # Try multiple possible model names (prefer non-Unsloth versions)
            model_names = [
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-1.5B",
            ]
            
            model = None
            tokenizer = None
            last_error = None
            
            for model_name in model_names:
                try:
                    print(f"  Trying: {model_name}...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    print(f"✅ Loaded BASE model: {model_name}")
                    return model, tokenizer, "transformers"
                except Exception as e:
                    last_error = e
                    print(f"    Failed: {str(e)[:100]}")
                    continue
            
            if model is None:
                print(f"⚠️  Could not load BASE model. Last error: {str(last_error)[:200]}")
                print("   Continuing without base model comparison...")
                return None, None, None
                
        except Exception as e:
            print(f"⚠️  Could not load BASE model: {e}")
            print("   Continuing without base model comparison...")
            return None, None, None
    
    return None, None, None

def load_model():
    """Load fine-tuned model (tries multiple formats)."""
    import glob
    
    # Try Unsloth format first
    if UNSLOTH_AVAILABLE and os.path.exists("outputs_rag_analysis/"):
        print("📦 Loading Unsloth model from outputs_rag_analysis/...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="outputs_rag_analysis/",
                max_seq_length=8192,
                dtype=None,
                load_in_4bit=False,
            )
            print("✅ Loaded Unsloth model")
            return model, tokenizer, "unsloth"
        except Exception as e:
            print(f"⚠️  Could not load Unsloth model: {e}")
    
    # Try HuggingFace format
    if TRANSFORMERS_AVAILABLE and os.path.exists("outputs_rag_analysis/"):
        print("📦 Loading HuggingFace model from outputs_rag_analysis/...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("outputs_rag_analysis/")
            model = AutoModelForCausalLM.from_pretrained(
                "outputs_rag_analysis/",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print("✅ Loaded HuggingFace model")
            return model, tokenizer, "transformers"
        except Exception as e:
            print(f"⚠️  Could not load HuggingFace model: {e}")
    
    # Try GGUF format
    if LLAMA_CPP_AVAILABLE:
        gguf_dir = "gguf_model_rag_analysis"
        if os.path.exists(gguf_dir):
            gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
            if gguf_files:
                preferred = [f for f in gguf_files if "Q4_K_M" in f or "q4_k_m" in f]
                gguf_file = preferred[0] if preferred else gguf_files[0]
                print(f"📦 Loading GGUF model from {gguf_file}...")
                try:
                    model = Llama(
                        model_path=gguf_file,
                        n_ctx=8192,
                        n_threads=4,
                        verbose=False
                    )
                    tokenizer = None
                    if TRANSFORMERS_AVAILABLE:
                        try:
                            if os.path.exists("outputs_rag_analysis/"):
                                tokenizer = AutoTokenizer.from_pretrained("outputs_rag_analysis/")
                            else:
                                tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
                        except:
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
                            tokenizer = SimpleTokenizer()
                    
                    if tokenizer is None:
                        raise ValueError("Could not load tokenizer for GGUF model")
                    
                    print("✅ Loaded GGUF model")
                    return model, tokenizer, "gguf"
                except Exception as e:
                    print(f"⚠️  Could not load GGUF model: {e}")
    
    raise FileNotFoundError(
        "Model not found. Please train the model first.\n"
        "Checked: outputs_rag_analysis/, gguf_model_rag_analysis/"
    )

# ============================================================================
# Response Generation
# ============================================================================

def generate_response(model, tokenizer, messages: List[Dict], model_type: str, 
                     max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """Generate response from model."""
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    if model_type == "gguf":
        response = model(
            formatted_text,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return response['choices'][0]['text'].strip()
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=8192)
    
    if model_type == "unsloth":
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

# ============================================================================
# Simple System Prompt (Minimal Guidance)
# ============================================================================

def create_system_prompt() -> str:
    """System prompt - matches the training dataset."""
    return """You are an AI assistant trained to analyze RAG chunks and extract relevant information.

CRITICAL: Always use the EXACT names, entities, or terms from the user's query. Never hallucinate or substitute different names.

Process:
1. Read each chunk COMPLETELY from start to finish (each chunk has 6-8 sentences)
2. Evaluate relevance using the provided score:
   - HIGH relevance (score ≥0.70): Extract information that directly answers the query
   - MEDIUM relevance (0.50-0.69): May contain related information, use with caution
   - LOW relevance (score <0.50): Likely irrelevant, ignore unless no HIGH relevance chunks available
3. Understand the MEANING in each chunk, not just keywords
4. Extract ONLY information that directly answers or addresses the query
5. IGNORE information that is similar but does NOT answer the query (even if in HIGH relevance chunks)
6. Use the score to determine if extracted information directly answers the query

Return ONLY the final answer. Do not include reasoning steps or process details."""

# ============================================================================
# Format Chunks
# ============================================================================

def format_rag_chunks(chunks: List[Dict]) -> str:
    """Format RAG chunks for the model (matches training dataset format)."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0.0)
        file_name = chunk.get('file', 'document.pdf')
        text = chunk['text']
        # Escape single quotes to match dataset format
        text_escaped = text.replace("'", "\\'")
        context_parts.append(f"[Chunk {i}] Score: {score:.3f}, File: {file_name}")
        context_parts.append(f"[{i}] FULL CHUNK TEXT: '{text_escaped}'")
        context_parts.append("")
    return "\n".join(context_parts)

# ============================================================================
# Test Execution
# ============================================================================

def analyze_rag_chunks(model, tokenizer, query: str, chunks: List[Dict], 
                       model_type: str, max_tokens: int = 2000) -> str:
    """
    Analyze RAG chunks and answer query.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        query: User query
        chunks: List of chunk dicts with 'text', 'score', and optionally 'file'
        model_type: Type of model ('unsloth', 'transformers', 'gguf')
        max_tokens: Maximum tokens to generate
    
    Returns:
        Model's response
    """
    chunks_text = format_rag_chunks(chunks)
    
    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": f"Query: {query}\n\nRAG Chunks:\n{chunks_text}"}
    ]
    
    response = generate_response(model, tokenizer, messages, model_type, max_tokens=max_tokens)
    return response

# ============================================================================
# Test Cases (Simple - Just Query + Chunks)
# ============================================================================

def get_test_cases() -> List[Dict]:
    """Get test cases - just query and chunks, no hardcoded expectations."""
    return [
        {
            "name": "CEO Query",
            "query": "who is the CEO of LedgerAI?",
            "chunks": [
                {
                    "text": "into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations.",
                    "score": 0.85,
                    "file": "ledgerai.pdf"
                },
                {
                    "text": "has spent two decades pioneering breakthrough tech...",
                    "score": 0.65,
                    "file": "ledgerai.pdf"
                }
            ]
        },
        {
            "name": "Co-Founders Query",
            "query": "who are the co-founders of TechCorp?",
            "chunks": [
                {
                    "text": "John Smith is a renowned leader in AI, blockchain, and institutional finance. As CEO and Co-Founder of TechCorp, he is driving the development of AI-powered business intelligence.",
                    "score": 0.85,
                    "file": "techcorp.pdf"
                },
                {
                    "text": "Jane Doe is a strategic powerhouse in AI-driven governance. As Co-Founder and Chief Operating Officer of TechCorp, she leads the execution of AI-powered intelligence solutions.",
                    "score": 0.82,
                    "file": "techcorp.pdf"
                },
                {
                    "text": "Mike Johnson is a visionary leader. As Co-Founder and Chief Marketing Officer of TechCorp, he is spearheading global adoption.",
                    "score": 0.80,
                    "file": "techcorp.pdf"
                },
                {
                    "text": "Sarah Williams is a driving force in finance, blockchain, and enterprise strategy. As Co-Founder and Chief Financial Officer of TechCorp, she architects the company's financial strategy, tokenomics, and investment framework.",
                    "score": 0.78,
                    "file": "techcorp.pdf"
                }
            ]
        }
    ]

# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_mode(model, tokenizer, model_type: str, base_model=None, base_tokenizer=None, base_model_type=None):
    """Interactive mode - user provides query and chunks."""
    print("\n" + "="*80)
    print("Interactive RAG Analysis Mode")
    print("="*80)
    print("\nEnter your query and RAG chunks. Type 'quit' to exit.\n")
    
    while True:
        query = input("Query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print("\nEnter chunks (one per line). Type 'done' when finished:")
        print("Note: Scores will default to 0.85 (HIGH relevance) if not specified.")
        chunks = []
        chunk_num = 1
        while True:
            print(f"\nChunk {chunk_num}:")
            text = input("  Text: ").strip()
            if text.lower() == 'done':
                break
            if not text:
                continue
            
            # Make score optional - default to HIGH relevance
            score_input = input("  Score (default 0.85, press Enter to skip): ").strip()
            if score_input:
                try:
                    score = float(score_input)
                except ValueError:
                    print("  ⚠️  Invalid score, using default 0.85")
                    score = 0.85
            else:
                score = 0.85  # Default to HIGH relevance
            
            # Make file optional
            file_input = input("  File (default document.pdf, press Enter to skip): ").strip()
            file_name = file_input if file_input else "document.pdf"
            
            chunks.append({
                "text": text,
                "score": score,
                "file": file_name
            })
            chunk_num += 1
        
        if not chunks:
            print("No chunks provided. Skipping...")
            continue
        
        print("\n🤖 Analyzing with FINE-TUNED model...")
        try:
            response_finetuned = analyze_rag_chunks(model, tokenizer, query, chunks, model_type)
            print(f"\n📝 FINE-TUNED Model Response:\n{response_finetuned}\n")
        except Exception as e:
            print(f"❌ Error with fine-tuned model: {e}")
            import traceback
            traceback.print_exc()
            response_finetuned = None
        
        # Test base model if available
        if base_model and base_tokenizer and base_model_type:
            print("\n🤖 Analyzing with BASE model (for comparison)...")
            try:
                response_base = analyze_rag_chunks(base_model, base_tokenizer, query, chunks, base_model_type)
                print(f"\n📝 BASE Model Response:\n{response_base}\n")
                
                print("\n" + "="*80)
                print("COMPARISON SUMMARY")
                print("="*80)
                print(f"\n✅ Fine-tuned model response length: {len(response_finetuned) if response_finetuned else 0} chars")
                print(f"✅ Base model response length: {len(response_base) if response_base else 0} chars")
                print("\n" + "="*80)
            except Exception as e:
                print(f"❌ Error with base model: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️  Base model not available for comparison.")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RAG Chunk Analysis Test Script")
    print("=" * 80)
    print("\nLoading model...")
    
    try:
        model, tokenizer, model_type = load_model()
        print(f"✅ Fine-tuned model loaded successfully (type: {model_type})\n")
        
        # Try to load base model for comparison
        base_model, base_tokenizer, base_model_type = load_base_model()
        if base_model:
            print("✅ Base model loaded for comparison\n")
        else:
            print("⚠️  Base model not available - will only test fine-tuned model\n")
        
        print("=" * 80)
        print("Choose mode:")
        print("1. Run test cases")
        print("2. Interactive mode")
        print("=" * 80)
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\n" + "=" * 80)
            print("Running Test Cases")
            print("=" * 80)
            
            # Try to load base model for comparison
            base_model, base_tokenizer, base_model_type = load_base_model()
            if base_model:
                print("✅ Base model loaded for comparison\n")
            else:
                print("⚠️  Base model not available - will only test fine-tuned model\n")
            
            test_cases = get_test_cases()
            for test_case in test_cases:
                print(f"\n{'='*80}")
                print(f"Test: {test_case['name']}")
                print(f"{'='*80}")
                print(f"Query: {test_case['query']}")
                print(f"Chunks: {len(test_case['chunks'])}")
                
                # Test fine-tuned model
                print("\n🤖 Generating response with FINE-TUNED model...")
                try:
                    response_finetuned = analyze_rag_chunks(
                        model, tokenizer, 
                        test_case['query'], 
                        test_case['chunks'], 
                        model_type
                    )
                    print(f"\n📝 FINE-TUNED Model Response:\n{response_finetuned}\n")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    response_finetuned = None
                
                # Test base model if available
                if base_model and base_tokenizer and base_model_type:
                    print("🤖 Generating response with BASE model...")
                    try:
                        response_base = analyze_rag_chunks(
                            base_model, base_tokenizer,
                            test_case['query'],
                            test_case['chunks'],
                            base_model_type
                        )
                        print(f"\n📝 BASE Model Response:\n{response_base}\n")
                        
                        print("\n" + "-"*80)
                        print("COMPARISON:")
                        print("-"*80)
                        print(f"Fine-tuned: {len(response_finetuned) if response_finetuned else 0} chars")
                        print(f"Base: {len(response_base) if response_base else 0} chars")
                        print("-"*80)
                    except Exception as e:
                        print(f"❌ Error with base model: {e}")
            
            print("\n" + "=" * 80)
            print("Tests Complete")
            print("=" * 80)
        
        elif choice == "2":
            interactive_mode(model, tokenizer, model_type, base_model, base_tokenizer, base_model_type)
        
        else:
            print("Invalid choice. Exiting.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
