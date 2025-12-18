#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Multi-Entity Extraction (Google Colab Version)
====================================================

Diagnostic tool to understand why the fine-tuned model only extracts
one entity instead of all matching entities.

Checks:
1. Is the model reading entire chunks?
2. Is it stopping after first match?
3. Are all chunks being processed?
4. Token limits / truncation issues?
5. Model's internal reasoning (if available)

Usage in Colab:
1. Upload your fine-tuned model folder to Colab (outputs_rag_analysis/)
2. Install dependencies: !pip install unsloth transformers accelerate bitsandbytes
3. Run this script
4. Provide query and chunks interactively or use test case
"""

import json
import os
import re
from typing import List, Dict, Optional

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✅ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("📋 Running locally")

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    if IN_COLAB:
        print("⚠️  Unsloth not installed. Run: !pip install unsloth")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    if IN_COLAB:
        print("⚠️  Transformers not installed. Run: !pip install transformers accelerate bitsandbytes")

# Try to import llama_cpp (for GGUF format)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    if IN_COLAB:
        print("⚠️  llama-cpp-python not installed. Run: !pip install llama-cpp-python (for GGUF models)")

# ============================================================================
# Model Loading (Colab-compatible)
# ============================================================================

def load_model(model_path: str = None):
    """
    Load fine-tuned model from Colab or local path.
    
    Args:
        model_path: Path to model folder. If None, tries common locations.
    """
    # Try to find model in common locations
    possible_paths = []
    
    if model_path:
        possible_paths.append(model_path)
    
    # Colab: Check common upload locations
    if IN_COLAB:
        possible_paths.extend([
            "/content/",  # Root directory - check for .gguf files here
            "/content/outputs_rag_analysis/",
            "/content/rag_analysis_model/",
            "./outputs_rag_analysis/",
            "./rag_analysis_model/",
            ".",  # Current directory
        ])
    else:
        # Local: Check current directory
        possible_paths.extend([
            "./outputs_rag_analysis/",
            "outputs_rag_analysis/",
        ])
    
    # Check for GGUF format first (quantized models often in GGUF)
    # Also check if model_path is directly a .gguf file
    import glob
    
    # Check if model_path is a direct .gguf file
    if model_path and model_path.endswith('.gguf') and os.path.exists(model_path):
        possible_paths.insert(0, os.path.dirname(model_path) if os.path.dirname(model_path) else ".")
        gguf_file = model_path
        print(f"📦 Found GGUF model file: {gguf_file}")
        try:
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python not installed. Run: !pip install llama-cpp-python")
            
            model = Llama(
                model_path=gguf_file,
                n_ctx=8192,
                n_threads=4,
                verbose=False
            )
            # Load tokenizer from base model (Qwen2.5-1.5B-Instruct)
            tokenizer = None
            if TRANSFORMERS_AVAILABLE:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        "Qwen/Qwen2.5-1.5B-Instruct", 
                        trust_remote_code=True
                    )
                    print("✅ Loaded tokenizer from base model")
                except:
                    # Create simple tokenizer wrapper
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
                        def encode(self, text, **kwargs):
                            # Simple tokenization - GGUF models handle this internally
                            return text
                        def decode(self, tokens, **kwargs):
                            return tokens if isinstance(tokens, str) else str(tokens)
                    tokenizer = SimpleTokenizer()
                    print("✅ Using simple tokenizer wrapper")
            
            if tokenizer is None:
                raise ValueError("Could not load tokenizer for GGUF model")
            
            print(f"✅ Loaded GGUF model: {os.path.basename(gguf_file)}")
            return model, tokenizer, "gguf"
        except Exception as e:
            print(f"⚠️  Could not load GGUF file {gguf_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Check for GGUF files in directories AND root directory
    if LLAMA_CPP_AVAILABLE:
        # First, check root directory (/content/) for .gguf files directly
        if IN_COLAB:
            root_gguf_files = glob.glob("/content/*.gguf")
            if root_gguf_files:
                # Prefer Q4_K_M quantization if available (matches user's file)
                preferred = [f for f in root_gguf_files if "Q4_K_M" in f or "q4_k_m" in f or "Q4" in f]
                gguf_file = preferred[0] if preferred else root_gguf_files[0]
                print(f"📦 Found GGUF model in /content/: {os.path.basename(gguf_file)}")
                try:
                    model = Llama(
                        model_path=gguf_file,
                        n_ctx=8192,
                        n_threads=4,
                        verbose=False
                    )
                    # Load tokenizer from base model
                    tokenizer = None
                    if TRANSFORMERS_AVAILABLE:
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(
                                "Qwen/Qwen2.5-1.5B-Instruct", 
                                trust_remote_code=True
                            )
                            print("✅ Loaded tokenizer from base model (Qwen2.5-1.5B-Instruct)")
                        except Exception as e:
                            print(f"⚠️  Could not load tokenizer: {e}, using simple wrapper")
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
                                def encode(self, text, **kwargs):
                                    return text
                                def decode(self, tokens, **kwargs):
                                    return tokens if isinstance(tokens, str) else str(tokens)
                            tokenizer = SimpleTokenizer()
                            print("✅ Using simple tokenizer wrapper")
                    
                    if tokenizer is None:
                        raise ValueError("Could not load tokenizer for GGUF model")
                    
                    print(f"✅ Loaded GGUF model: {os.path.basename(gguf_file)}")
                    return model, tokenizer, "gguf"
                except Exception as e:
                    print(f"⚠️  Could not load GGUF from /content/: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Then check other paths
        for path in possible_paths:
            if os.path.exists(path):
                # Check if it's a directory or file
                if os.path.isdir(path):
                    gguf_files = glob.glob(os.path.join(path, "*.gguf"))
                elif path.endswith('.gguf'):
                    gguf_files = [path]
                else:
                    gguf_files = []
                
                if gguf_files:
                    # Prefer Q4_K_M quantization if available (matches user's file)
                    preferred = [f for f in gguf_files if "Q4_K_M" in f or "q4_k_m" in f or "Q4" in f]
                    gguf_file = preferred[0] if preferred else gguf_files[0]
                    print(f"📦 Found GGUF model: {os.path.basename(gguf_file)}")
                    try:
                        model = Llama(
                            model_path=gguf_file,
                            n_ctx=8192,
                            n_threads=4,
                            verbose=False
                        )
                        # Try to load tokenizer from same directory or base model
                        tokenizer = None
                        if TRANSFORMERS_AVAILABLE:
                            try:
                                # Check if tokenizer files exist in same directory
                                tokenizer_path = os.path.dirname(gguf_file)
                                if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
                                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                                    print("✅ Loaded tokenizer from model directory")
                                else:
                                    # Fallback to base model tokenizer
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        "Qwen/Qwen2.5-1.5B-Instruct", 
                                        trust_remote_code=True
                                    )
                                    print("✅ Loaded tokenizer from base model (Qwen2.5-1.5B-Instruct)")
                            except Exception as e:
                                print(f"⚠️  Could not load tokenizer: {e}, using simple wrapper")
                                # Create simple tokenizer wrapper
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
                                    def encode(self, text, **kwargs):
                                        # Simple tokenization - GGUF models handle this internally
                                        return text
                                    def decode(self, tokens, **kwargs):
                                        return tokens if isinstance(tokens, str) else str(tokens)
                                tokenizer = SimpleTokenizer()
                                print("✅ Using simple tokenizer wrapper")
                        
                        if tokenizer is None:
                            raise ValueError("Could not load tokenizer for GGUF model")
                        
                        print(f"✅ Loaded GGUF model: {os.path.basename(gguf_file)}")
                        return model, tokenizer, "gguf"
                    except Exception as e:
                        print(f"⚠️  Could not load GGUF from {path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
    
    # Try Unsloth format (supports quantized models)
    if UNSLOTH_AVAILABLE:
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📦 Trying to load Unsloth model from: {path}")
                try:
                    # Unsloth handles quantized models automatically
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=path,
                        max_seq_length=8192,
                        dtype=None,
                        load_in_4bit=True,  # Use 4-bit for quantized models and memory efficiency
                    )
                    print(f"✅ Loaded Unsloth model from {path}")
                    return model, tokenizer, "unsloth"
                except Exception as e:
                    print(f"⚠️  Could not load from {path}: {e}")
                    continue
    
    # Try HuggingFace format (with quantization support)
    if TRANSFORMERS_AVAILABLE:
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📦 Trying to load HuggingFace model from: {path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    
                    # Check if model is already quantized (has quantization_config.json)
                    is_quantized = os.path.exists(os.path.join(path, "quantization_config.json"))
                    
                    if is_quantized:
                        # Model is already quantized, load with 4-bit config
                        print("  📊 Detected quantized model, loading with 4-bit config...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            path,
                            quantization_config=quantization_config,
                            device_map="auto",
                            trust_remote_code=True,
                        )
                    else:
                        # Regular model, load with float16
                        model = AutoModelForCausalLM.from_pretrained(
                            path,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                        )
                    
                    print(f"✅ Loaded HuggingFace model from {path}")
                    return model, tokenizer, "transformers"
                except Exception as e:
                    print(f"⚠️  Could not load from {path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # If nothing found, show helpful message
    print("\n❌ Model not found in any of these locations:")
    for path in possible_paths:
        print(f"   - {path}")
    
    # Check if any .gguf files exist in /content/
    if IN_COLAB:
        import glob
        gguf_files = glob.glob("/content/*.gguf")
        if gguf_files:
            print(f"\n💡 Found .gguf files in /content/: {[os.path.basename(f) for f in gguf_files]}")
            print("   But llama-cpp-python is not installed. Run: !pip install llama-cpp-python")
        else:
            print("\n💡 To use this script:")
            print("   1. Upload your .gguf file to /content/ in Colab")
            print("   2. Install llama-cpp-python: !pip install llama-cpp-python")
            print("   3. Or specify the path: load_model('/content/your_file.gguf')")
    else:
        print("\n💡 To use this script:")
        print("   1. Upload your fine-tuned model folder")
        print("   2. Or specify the path when calling load_model()")
        print("   3. Model folder should contain: config.json, tokenizer files, model files")
    
    raise FileNotFoundError(
        f"Model not found. Checked: {', '.join(possible_paths)}\n"
        f"Upload your model to Colab or specify the path."
    )

# ============================================================================
# Diagnostic Functions
# ============================================================================

def analyze_chunk_content(chunks: List[Dict], query: str) -> Dict:
    """Analyze chunks to find all expected entities."""
    print("\n" + "="*80)
    print("CHUNK ANALYSIS")
    print("="*80)
    
    # Extract query type
    query_lower = query.lower()
    is_cofounder_query = "co-founder" in query_lower or "cofounder" in query_lower
    is_executive_query = "executive" in query_lower
    is_founder_query = "founder" in query_lower and not is_cofounder_query
    
    # Find company name
    company_match = re.search(r'of\s+(\w+)', query, re.IGNORECASE)
    company = company_match.group(1) if company_match else None
    
    print(f"Query: {query}")
    print(f"Query Type: {'co-founder' if is_cofounder_query else 'founder' if is_founder_query else 'executive' if is_executive_query else 'other'}")
    print(f"Company: {company}")
    print()
    
    all_entities = []
    chunk_analysis = []
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        score = chunk.get('score', 0.0)
        
        # Find entities in this chunk
        entities_in_chunk = []
        
        if is_cofounder_query:
            # Look for "Co-Founder" patterns - extract actual person names
            # Pattern examples from actual text:
            # - "David Lara is... As Co-Founder and Chief Operating Officer of LedgerAI"
            # - "Paul Chou is... As CEO and Co-Founder of LedgerAI"
            # - "Bob Carella is... As Co-Founder and Chief Financial Officer of LedgerAI"
            # - "Jorge Guinovart is... As Co-Founder and Chief Marketing Officer of LedgerAI"
            
            patterns = [
                # Pattern 1: "Name is... As Co-Founder and Title"
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+is[^.]{0,300}As\s+Co-Founder',
                # Pattern 2: "Name is... As CEO and Co-Founder"
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+is[^.]{0,300}As\s+(?:CEO|CTO|CFO|COO|CMO)\s+and\s+Co-Founder',
                # Pattern 3: "As Co-Founder and Title, Name..." (less common)
                r'As\s+Co-Founder\s+and\s+[^,]+,\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            ]
            
            found_names = set()
            
            # Try the patterns
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    name = match.group(1).strip()
                    # Validate it's a real name (two capitalized words)
                    parts = name.split()
                    if len(parts) == 2 and parts[0][0].isupper() and parts[1][0].isupper():
                        # Exclude common false positives
                        excluded = ['he is', 'he leads', 'he architects', 'ceo and', 'he serves', 
                                   'he manages', 'he provides', 'he continues', 'he has', 'he was',
                                   'he holds', 'he brings', 'he brings', 'he brings']
                        if name.lower() not in excluded:
                            found_names.add(name)
            
            # Also try: find names that appear before "Co-Founder" within reasonable distance
            # This catches cases where the name appears earlier in the sentence
            cofounder_positions = [m.start() for m in re.finditer(r'Co-Founder', text, re.IGNORECASE)]
            for cofounder_pos in cofounder_positions:
                # Look backwards up to 200 chars for a name pattern
                search_start = max(0, cofounder_pos - 200)
                search_text = text[search_start:cofounder_pos]
                # Find capitalized name patterns
                name_matches = re.finditer(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', search_text)
                for name_match in name_matches:
                    name = name_match.group(1).strip()
                    parts = name.split()
                    if len(parts) == 2 and parts[0][0].isupper() and parts[1][0].isupper():
                        excluded = ['he is', 'he leads', 'he architects', 'ceo and', 'he serves']
                        if name.lower() not in excluded:
                            found_names.add(name)
            
            entities_in_chunk = list(found_names)
        elif is_founder_query:
            patterns = [
                r'(\w+\s+\w+)\s+is\s+(?:a\s+)?Founder',
                r'(\w+\s+\w+)\s+serves\s+as\s+(?:a\s+)?Founder',
                r'As\s+Founder[^,]*,\s+(\w+\s+\w+)',
            ]
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.group(1).strip()
                    if name and name not in entities_in_chunk:
                        entities_in_chunk.append(name)
        elif is_executive_query:
            patterns = [
                r'(\w+\s+\w+)\s+is\s+(?:an\s+)?Executive',
                r'(\w+\s+\w+)\s+serves\s+as\s+(?:an\s+)?Executive',
                r'As\s+Executive[^,]*,\s+(\w+\s+\w+)',
            ]
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.group(1).strip()
                    if name and name not in entities_in_chunk:
                        entities_in_chunk.append(name)
        
        chunk_analysis.append({
            'chunk_num': i,
            'score': score,
            'text_length': len(text),
            'entities_found': entities_in_chunk,
            'text_preview': text[:200] + "..." if len(text) > 200 else text
        })
        
        all_entities.extend(entities_in_chunk)
        
        print(f"Chunk {i} (Score: {score:.2f}, Length: {len(text)} chars):")
        print(f"  Entities found: {entities_in_chunk if entities_in_chunk else 'None'}")
        print(f"  Preview: {text[:150]}...")
        print()
    
    unique_entities = list(set(all_entities))
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total chunks: {len(chunks)}")
    print(f"Total entities found across all chunks: {len(all_entities)}")
    print(f"Unique entities: {len(unique_entities)}")
    print(f"Expected entities: {unique_entities}")
    print("="*80)
    
    return {
        'expected_entities': unique_entities,
        'chunk_analysis': chunk_analysis,
        'total_chunks': len(chunks)
    }

def check_token_limits(tokenizer, messages: List[Dict], max_length: int = 8192) -> Dict:
    """Check if input exceeds token limits."""
    if not hasattr(tokenizer, 'apply_chat_template'):
        return {'error': 'Tokenizer does not support chat template'}
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
    token_count = len(tokens)
    
    print("\n" + "="*80)
    print("TOKEN ANALYSIS")
    print("="*80)
    print(f"Input token count: {token_count}")
    print(f"Max sequence length: {max_length}")
    print(f"Token usage: {(token_count / max_length) * 100:.1f}%")
    
    if token_count > max_length:
        print(f"⚠️  WARNING: Input exceeds max_length! Will be truncated.")
        print(f"   Tokens will be cut at position {max_length}")
        truncated_tokens = tokens[:max_length]
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        print(f"   Last 200 chars of truncated input: ...{truncated_text[-200:]}")
    else:
        print("✅ Input fits within token limit")
    
    print("="*80)
    
    return {
        'token_count': token_count,
        'max_length': max_length,
        'exceeds_limit': token_count > max_length,
        'truncation_point': max_length if token_count > max_length else None
    }

def extract_model_reasoning(response: str) -> Dict:
    """Extract model's internal reasoning from response (if present)."""
    print("\n" + "="*80)
    print("MODEL REASONING ANALYSIS")
    print("="*80)
    
    reasoning = {
        'has_step1': 'STEP 1' in response or 'UNDERSTAND THE QUERY' in response,
        'has_step2': 'STEP 2' in response or 'READ EACH CHUNK' in response,
        'has_step3': 'STEP 3' in response or 'ANALYZE CHUNK MEANING' in response,
        'has_step4': 'STEP 4' in response or 'EXTRACT MATCHING' in response,
        'has_step5': 'STEP 5' in response or 'VERIFY COMPLETENESS' in response,
        'has_step6': 'STEP 6' in response or 'SYNTHESIZE RESPONSE' in response,
        'chunks_mentioned': [],
        'entities_mentioned': [],
    }
    
    # Find which chunks were mentioned
    chunk_pattern = r'Chunk\s+(\d+)'
    chunks_mentioned = re.findall(chunk_pattern, response, re.IGNORECASE)
    reasoning['chunks_mentioned'] = [int(c) for c in chunks_mentioned]
    
    # Find entity names mentioned
    name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    entities_mentioned = re.findall(name_pattern, response)
    reasoning['entities_mentioned'] = list(set(entities_mentioned))
    
    print(f"Step 1 (Understand Query): {'✅' if reasoning['has_step1'] else '❌'}")
    print(f"Step 2 (Read Chunks): {'✅' if reasoning['has_step2'] else '❌'}")
    print(f"Step 3 (Analyze Meaning): {'✅' if reasoning['has_step3'] else '❌'}")
    print(f"Step 4 (Extract Info): {'✅' if reasoning['has_step4'] else '❌'}")
    print(f"Step 5 (Verify Completeness): {'✅' if reasoning['has_step5'] else '❌'}")
    print(f"Step 6 (Synthesize): {'✅' if reasoning['has_step6'] else '❌'}")
    print()
    print(f"Chunks mentioned in response: {reasoning['chunks_mentioned']}")
    print(f"Entities mentioned in response: {reasoning['entities_mentioned']}")
    print()
    print("Full response:")
    print("-"*80)
    print(response)
    print("-"*80)
    
    return reasoning

# ============================================================================
# Main Diagnostic Function
# ============================================================================

def diagnose_extraction(model, tokenizer, query: str, chunks: List[Dict], 
                       model_type: str, max_tokens: int = 2000) -> Dict:
    """Run full diagnostic on entity extraction."""
    
    # 1. Analyze chunks to find expected entities
    chunk_analysis = analyze_chunk_content(chunks, query)
    expected_entities = chunk_analysis['expected_entities']
    
    # 2. Format prompt
    chunks_text = format_rag_chunks(chunks)
    system_prompt = create_system_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nRAG Chunks:\n{chunks_text}"}
    ]
    
    # 3. Check token limits
    token_analysis = check_token_limits(tokenizer, messages)
    
    # 4. Generate response
    print("\n" + "="*80)
    print("GENERATING MODEL RESPONSE")
    print("="*80)
    
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    # Handle different model types
    if model_type == "gguf":
        # GGUF models use different generation API
        print(f"🔍 Generating with GGUF model (max_tokens={max_tokens})...")
        print(f"   Input length: {len(formatted_text)} characters")
        try:
            # llama-cpp-python API
            result = model(
                formatted_text,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],  # Stop on multiple newlines too
                echo=False,  # Don't echo the input
            )
            
            # Handle different response formats
            if isinstance(result, dict):
                if 'choices' in result and len(result['choices']) > 0:
                    if 'text' in result['choices'][0]:
                        response = result['choices'][0]['text'].strip()
                    elif 'content' in result['choices'][0]:
                        response = result['choices'][0]['content'].strip()
                    else:
                        response = str(result['choices'][0]).strip()
                elif 'text' in result:
                    response = result['text'].strip()
                else:
                    response = str(result).strip()
            else:
                response = str(result).strip()
            
            if not response or len(response) < 5:
                print(f"⚠️  Warning: Response is very short ({len(response)} chars)")
                print(f"   Raw result: {result}")
            else:
                print(f"✅ Generated {len(response)} characters")
                print(f"   Preview: {response[:100]}...")
                
        except Exception as e:
            print(f"❌ Error generating with GGUF model: {e}")
            import traceback
            traceback.print_exc()
            response = f"Error generating response: {e}"
    else:
        # Unsloth and Transformers models
        if hasattr(tokenizer, 'encode') and callable(getattr(tokenizer, 'encode', None)):
            inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=8192)
            if hasattr(inputs, 'to'):
                inputs = inputs.to(model.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        else:
            # Simple tokenizer - use text directly
            inputs = {"input_ids": formatted_text}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
            eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
        )
        
        if model_type == "unsloth" or model_type == "transformers":
            if isinstance(inputs, dict) and 'input_ids' in inputs:
                input_length = inputs['input_ids'].shape[1] if hasattr(inputs['input_ids'], 'shape') else 0
            else:
                input_length = 0
            
            generated_tokens = outputs[0][input_length:] if input_length > 0 else outputs[0]
            
            if hasattr(tokenizer, 'decode'):
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                response = str(generated_tokens)
        else:
            response = str(outputs)
    
    # 5. Analyze response
    reasoning = extract_model_reasoning(response)
    
    # 6. Extract actual entities from response
    actual_entities = reasoning['entities_mentioned']
    
    # 7. Compare expected vs actual
    print("\n" + "="*80)
    print("EXTRACTION COMPARISON")
    print("="*80)
    print(f"Expected entities: {expected_entities}")
    print(f"Actual entities extracted: {actual_entities}")
    print()
    
    missing = [e for e in expected_entities if e not in actual_entities]
    extra = [e for e in actual_entities if e not in expected_entities]
    
    if missing:
        print(f"❌ Missing entities: {missing}")
    if extra:
        print(f"⚠️  Extra entities (not in chunks): {extra}")
    if not missing and not extra:
        print("✅ All entities extracted correctly!")
    
    print("="*80)
    
    # Root cause analysis
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    if not reasoning['chunks_mentioned']:
        print("❌ ISSUE #1: Model didn't mention any chunks in response")
        print("   → Model may not be reading chunks or following the 6-step process")
    
    if len(missing) > 0:
        print(f"\n❌ ISSUE #2: Model missed {len(missing)} co-founders")
        print(f"   Missing: {missing}")
        print("   → Model likely stopped after finding first few entities")
        print("   → Or didn't read Chunk 2 completely (where Paul Chou and Bob Carella are)")
    
    if len(extra) > 0:
        print(f"\n⚠️  ISSUE #3: Model extracted {len(extra)} incorrect entities")
        print(f"   Extra: {extra}")
        print("   → 'Will Specht' is Head of Engineering, NOT a co-founder")
        print("   → Model not filtering by role correctly")
    
    # Check which chunks have missing entities
    if missing:
        print(f"\n📋 Missing entities are in:")
        for chunk_info in chunk_analysis['chunk_analysis']:
            chunk_entities = chunk_info['entities_found']
            missing_in_chunk = [e for e in missing if e in chunk_entities]
            if missing_in_chunk:
                print(f"   Chunk {chunk_info['chunk_num']}: Contains {missing_in_chunk} but LLM didn't extract")
                print(f"      → LLM may not have read this chunk completely")
    
    print("="*80)
    
    return {
        'expected_entities': expected_entities,
        'actual_entities': actual_entities,
        'missing': missing,
        'extra': extra,
        'correct': correct,
        'accuracy': (len(correct) / len(expected_entities) * 100) if expected_entities else 0,
        'chunk_analysis': chunk_analysis,
        'token_analysis': token_analysis,
        'reasoning': reasoning,
        'full_response': response,
        'root_cause': {
            'no_chunks_mentioned': not reasoning['chunks_mentioned'],
            'incomplete_extraction': len(missing) > 0,
            'incorrect_extraction': len(extra) > 0,
            'missing_count': len(missing),
            'extra_count': len(extra),
        }
    }

# ============================================================================
# Helper Functions
# ============================================================================

def format_rag_chunks(chunks: List[Dict]) -> str:
    """Format RAG chunks for the model."""
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
    """System prompt - matches training dataset."""
    return """You are an AI assistant trained to analyze RAG chunks and extract relevant information.

CORE PRINCIPLES (SYSTEMATIC EVALUATION PROCESS):

STEP 1: UNDERSTAND THE QUERY
- Identify what information is being requested
- Note any specific filtering requirements (role, entity, attribute, relationship, etc.)
- Understand the scope and context of what needs to be extracted

STEP 2: READ EACH CHUNK COMPLETELY
- Read the entire chunk from start to finish
- Do not stop at keywords - read for full context and meaning
- Understand the complete context before making extraction decisions

STEP 3: ANALYZE CHUNK MEANING
- Understand the semantic meaning, not just surface-level keywords
- Identify entities, relationships, attributes, and concepts mentioned
- Recognize how information relates to the query

STEP 4: EXTRACT MATCHING INFORMATION
- Extract only information that passes the relevance evaluation
- Apply exact matching - use information exactly as it appears in chunks
- Track all matching items across all chunks

STEP 5: VERIFY COMPLETENESS
- Ensure you have read ALL chunks completely
- Verify you extracted ALL matching items (do not stop after first match)
- Confirm extraction is complete before finalizing response

STEP 6: SYNTHESIZE RESPONSE
- Combine information from all chunks into coherent answer
- Format naturally and directly address the query
- CRITICAL: Extract ALL matching entities, not just the first one

CRITICAL: Follow these steps in order for EVERY query. Chunk order does not change the answer - read all chunks before responding.

ESSENTIAL GUIDELINES:
- NEVER hallucinate - only use information that appears in the provided chunks
- Extract ALL matching items - do NOT stop after finding the first match
- Read ALL chunks completely before responding
- Use EXACT information from chunks - never substitute or modify names, terms, or entities"""

# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Interactive Mode (Colab-friendly)
# ============================================================================

def interactive_diagnostic(model, tokenizer, model_type: str):
    """Interactive mode for Colab - user provides query and chunks."""
    print("\n" + "="*80)
    print("Interactive Diagnostic Mode")
    print("="*80)
    print("\nEnter your query and chunks to diagnose extraction issues.\n")
    
    while True:
        query = input("\nQuery (or 'quit' to exit): ").strip()
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
            
            score_input = input("  Score (default 0.85, press Enter to skip): ").strip()
            if score_input:
                try:
                    score = float(score_input)
                except ValueError:
                    print("  ⚠️  Invalid score, using default 0.85")
                    score = 0.85
            else:
                score = 0.85
            
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
        
        # Run diagnostic
        try:
            results = diagnose_extraction(model, tokenizer, query, chunks, model_type)
            
            # Save results
            output_file = "extraction_diagnostic_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Diagnostic complete! Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error during diagnostic: {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# Test Case Mode
# ============================================================================

def run_test_case(model, tokenizer, model_type: str):
    """Run diagnostic on pre-defined test case."""
    print("\n" + "="*80)
    print("Running Test Case: Co-founders of LedgerAI")
    print("="*80)
    
    query = "who are the co-founders of LedgerAI?"
    chunks = [
        {
            "text": "Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management, bridging the gap between technology, operations, and policymaking. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions, driving efficiency and transforming enterprise decision-making. He is also the CEO of Petra Capital & Advisory, focusing on AI technology and fintech investments, and Co-Founder of SuperCity AI, a next-generation super app revolutionizing government services, digital payments, and civic engagement. His extensive experience spans both public and private sectors, having served as a Partner at Ichor Strategies (2020–2023) and held senior leadership roles in New York's city and state governments, including Chief Administrative Officer and Deputy Director of Budget, where he managed multi-billion-dollar budgets, strategic initiatives, and fiscal oversight. David holds an MS in Material Science and Engineering from the University of Washington and a Master's in Public Affairs from the University of Texas, equipping him with a unique blend of technical expertise and policy leadership. With a proven track record of optimizing complex systems and integrating AI into high-stakes environments, David is driving LedgerAI's mission to redefine enterprise intelligence and governance at a global scale. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance, driving the future of intelligent digital ecosystems. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion, ensuring LedgerAI becomes the premier AI-driven business intelligence platform. In addition, as Founder and CEO of AlphaCityAI, he is pioneering AI integration within the metaverse, transforming how businesses and consumers interact in virtual economies. Through Bank, a next-generation Web3 financial platform, he is reshaping the future of decentralized banking and digital asset solutions. With an unparalleled ability to bridge AI, blockchain, and next-gen financial products, Jorge is driving innovation, growth, and disruption across multiple industries. Will Specht is a technological architect with over 20 years of experience in engineering, AI infrastructure, and enterprise software development, leading LedgerAI's cutting-edge engineering efforts as Head of Engineering. With an impressive track record at Remesh, Medallion, Plusgrade, Ladders, and Siemens, he has built and scaled complex systems that power AI-driven analytics, high-frequency data processing, and secure enterprise platforms. A University of Delaware engineering graduate, Will has spent two decades pioneering breakthrough technologies in AI, automation, and decentralized systems, ensuring that LedgerAI's infrastructure is built for speed, security, and scalability. His leadership is the driving force behind AuraVision's seamless integration, real-time intelligence capabilities, and next-generation AI deployment, positioning LedgerAI at the forefront of enterprise AI solutions.",
            "score": 0.85,
            "file": "ledgerai.pdf"
        },
        {
            "text": "into enterprises worldwide. Paul Chou is a renowned leader in AI, blockchain, and institutional finance, shaping the future of intelligent enterprise solutions and digital assets. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence, integrating blockchain technology to transform governance, strategy, and financial operations. A graduate of MIT with degrees in Mathematics and Electrical Engineering & Computer Science, Paul's expertise spans high-frequency trading, decentralized finance, and AI-driven analytics. Previously, he co-founded LedgerX (2014–2020), the first U.S. federally regulated crypto derivatives exchange, revolutionizing institutional Bitcoin options trading. Before that, he was a high-level trader at Goldman Sachs (2010–2014), mastering complex markets. As the Founder of Foundation Coin, he continues to push the boundaries of next-generation cryptocurrency architectures. A recognized thought leader, Paul has been featured on TED Talks and major global conferences for over a decade, solidifying his role as a pioneer at the forefront of AI, blockchain, and financial innovation. Bob Carella is a driving force in finance, blockchain, and enterprise strategy, bringing deep expertise in financial operations, tokenized ecosystems, and corporate finance. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy, tokenomics, and investment framework, ensuring long-term sustainability and growth. In addition, as Founder and CEO of BobFi, he provides advisory services in payroll, human capital, and financial structuring. Previously, as Global Head of Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs, navigating global compliance, financial operations, and digital asset compensation models. A passionate educator, he serves as an Adjunct Professor at Drew University, teaching Innovative Cryptocurrency Solutions and helping shape the next generation of fintech leaders.",
            "score": 0.85,
            "file": "ledgerai.pdf"
        }
    ]
    
    # Run diagnostic
    results = diagnose_extraction(model, tokenizer, query, chunks, model_type)
    
    # Save results
    output_file = "extraction_diagnostic_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Diagnostic complete! Results saved to {output_file}")
    return results

# ============================================================================
# Main (Colab-compatible)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Entity Extraction Diagnostic Tool (Google Colab)")
    print("=" * 80)
    
    # Load model
    try:
        # For GGUF files, you can specify the full path to the .gguf file
        # Example: model_path = "/content/Qwen2.5-1.5B-Instruct.Q4_K_M-2.gguf"
        # Or just upload to /content/ and it will auto-detect
        model_path = None  # Set to custom path if needed, e.g., "/content/Qwen2.5-1.5B-Instruct.Q4_K_M-2.gguf"
        
        # If you uploaded the .gguf file directly, uncomment and set the path:
        # model_path = "/content/Qwen2.5-1.5B-Instruct.Q4_K_M-2.gguf"
        
        model, tokenizer, model_type = load_model(model_path)
        print(f"✅ Model loaded (type: {model_type})\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. For GGUF files: Upload your .gguf file to Colab")
        print("   2. Install llama-cpp-python: !pip install llama-cpp-python")
        print("   3. Specify the path: model_path = '/content/Qwen2.5-1.5B-Instruct.Q4_K_M-2.gguf'")
        print("   4. Or upload to /content/ and it will auto-detect .gguf files")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Choose mode
    if IN_COLAB:
        print("=" * 80)
        print("Choose mode:")
        print("1. Run test case (Co-founders of LedgerAI)")
        print("2. Interactive mode (enter your own query and chunks)")
        print("=" * 80)
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            run_test_case(model, tokenizer, model_type)
        elif choice == "2":
            interactive_diagnostic(model, tokenizer, model_type)
        else:
            print("Invalid choice. Running test case by default...")
            run_test_case(model, tokenizer, model_type)
    else:
        # Local mode - run test case
        run_test_case(model, tokenizer, model_type)
