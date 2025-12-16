#!/usr/bin/env python3
"""
Post-Training Model Evaluation Script for Colab
Evaluates trained RAG analysis model and provides comprehensive analysis
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    SEMANTIC_SIMILARITY_AVAILABLE = False
    print("⚠️  sentence-transformers not available - using string similarity only")
    print("   Install with: pip install sentence-transformers")

# Try to import training dependencies (may not be available in all environments)
try:
    from unsloth import FastLanguageModel
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available - will use basic evaluation only")

def similarity_score(str1: str, str2: str) -> float:
    """
    Calculate similarity score between two strings.
    Uses semantic similarity (embeddings) if available, falls back to string similarity.
    """
    # Try semantic similarity first (more accurate for paraphrased content)
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Lazy load model (cache it)
        if not hasattr(similarity_score, '_model'):
            similarity_score._model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embeddings
        embeddings = similarity_score._model.encode([str1, str2])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # Convert to percentage (0-1 -> 0-100)
        return float(similarity * 100)
    except (ImportError, Exception) as e:
        # Fallback to string similarity if embeddings not available
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() * 100

def detect_cot_leakage(text: str) -> Tuple[bool, List[str]]:
    """
    Detect if model output contains CoT leakage (intermediate steps)
    Returns: (has_leakage, detected_patterns)
    """
    patterns = [
        r'STEP\s*[1-6]',
        r'Step\s*[1-6]',
        r'Extract information from Chunk',
        r'Chunk\s*\d+',
        r'Step \d+:',
        r'STEP \d+:',
        r'Final Answer:',
        r'\[Final Answer\]',
        r'Extract information from',
    ]
    
    detected = []
    has_leakage = False
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            has_leakage = True
            detected.extend(matches)
    
    return has_leakage, list(set(detected))

def clean_output(text: str) -> str:
    """Remove CoT leakage patterns from output - AGGRESSIVE cleaning"""
    if not text:
        return text
    
    original_text = text
    
    # Pattern 1: Remove lines that are ONLY extraction instructions
    # Match lines like "Extract information from Chunk X" or "Extract information from Chunk X and Chunk Y"
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are only extraction instructions
        if re.match(r'^Extract information from Chunk\s*\d+', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^Extract information from\s*$', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^Chunk\s*\d+[:\-]?\s*$', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^STEP\s*[1-6][:\-]?\s*$', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^Step\s*[1-6][:\-]?\s*$', line_stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Pattern 2: Remove extraction instructions at the start
    text = re.sub(r'^Extract information from Chunk\s*\d+.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Extract information from Chunk\s*\d+\s*\[and Chunk\s*\d+\].*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Extract information from Chunk\s*\d+\s*and Chunk\s*\d+.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 3: Remove standalone extraction phrases
    text = re.sub(r'Extract information from Chunk\s*\d+\s*\[and Chunk\s*\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Extract information from Chunk\s*\d+\s*and Chunk\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Extract information from Chunk\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Extract information from\s*', '', text, flags=re.IGNORECASE)
    
    # Pattern 4: Remove STEP markers
    text = re.sub(r'STEP\s*[1-6][:\-]?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Step\s*[1-6][:\-]?\s*', '', text, flags=re.IGNORECASE)
    
    # Pattern 5: Remove "Final Answer:" markers but keep content after
    text = re.sub(r'Final Answer:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Final Answer\]\s*', '', text, flags=re.IGNORECASE)
    
    # Pattern 6: Remove standalone "Chunk X:" lines
    text = re.sub(r'^Chunk\s*\d+[:\-]?\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 7: Remove analysis sections that are just instructions
    text = re.sub(r'Step \d+:\s*.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'STEP \d+:\s*.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 8: Remove lines that are just "Extract information from Chunk X" repeated
    # Split by newlines and filter out repetitive extraction lines
    lines = text.split('\n')
    filtered_lines = []
    prev_was_extraction = False
    for line in lines:
        is_extraction = bool(re.search(r'Extract information|Chunk\s*\d+', line, re.IGNORECASE))
        if is_extraction and prev_was_extraction:
            # Skip consecutive extraction lines
            continue
        prev_was_extraction = is_extraction
        if not is_extraction or len(line.strip()) > 50:  # Keep if it has substantial content
            filtered_lines.append(line)
    
    text = '\n'.join(filtered_lines)
    
    # Pattern 9: If text is mostly extraction instructions, try to find actual content
    # Look for content after "Final Answer", "Answer:", or after extraction instructions
    if len(text.strip()) < 20:  # Very short output might be just instructions
        # Try to extract from original if it had more content
        final_answer_match = re.search(r'Final Answer[:\-]?\s*(.+?)(?:\n\n|\Z)', original_text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            text = final_answer_match.group(1).strip()
        
        # Look for content after extraction instructions
        after_extraction = re.search(r'Extract information from Chunk\s*\d+.*?\n\n(.+?)(?:\n\n|\Z)', original_text, re.IGNORECASE | re.DOTALL)
        if after_extraction:
            text = after_extraction.group(1).strip()
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
    text = text.strip()
    
    # If result is empty or too short, return a message
    if len(text.strip()) < 10:
        return "[Output contained only extraction instructions - no actual answer extracted]"
    
    return text

def load_model(model_path: str = None):
    """
    Load fine-tuned model (tries multiple formats).
    Uses same logic as test_rag_analysis_colab.py
    """
    import glob
    
    # If path is explicitly provided, use it
    if model_path:
        model_path = os.path.expanduser(model_path)
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
    else:
        # Try to get from environment variable
        model_path = os.getenv("MODEL_PATH", None)
        if model_path:
            model_path = os.path.expanduser(model_path)
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
    
    # Try Unsloth format first (from outputs_rag_analysis/)
    if UNSLOTH_AVAILABLE:
        # Check explicit path first
        if model_path and os.path.exists(model_path):
            check_path = model_path
        else:
            # Default to outputs_rag_analysis (same as test_rag_analysis_colab.py)
            check_path = "outputs_rag_analysis"
        
        if os.path.exists(check_path):
            print(f"📦 Loading Unsloth model from {check_path}/...")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=check_path,
                    max_seq_length=8192,
                    dtype=None,
                    load_in_4bit=False,
                )
                FastLanguageModel.for_inference(model)
                print("✅ Loaded Unsloth model")
                return model, tokenizer
            except Exception as e:
                print(f"⚠️  Could not load Unsloth model: {e}")
    
    # Try HuggingFace format (fallback)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        if model_path and os.path.exists(model_path):
            check_path = model_path
        else:
            check_path = "outputs_rag_analysis"
        
        if os.path.exists(check_path):
            print(f"📦 Loading HuggingFace model from {check_path}/...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(check_path)
                model = AutoModelForCausalLM.from_pretrained(
                    check_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                print("✅ Loaded HuggingFace model")
                return model, tokenizer
            except Exception as e:
                print(f"⚠️  Could not load HuggingFace model: {e}")
    except ImportError:
        pass
    
    # If we get here, model not found
    raise FileNotFoundError(
        "Model not found. Please train the model first.\n"
        f"Checked: {check_path if 'check_path' in locals() else 'outputs_rag_analysis/'}\n"
        "To specify a different path, set: os.environ['MODEL_PATH'] = './your_path'"
    )

def generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int = 2000):
    """Generate prediction from model (matches test_rag_analysis_colab.py logic)"""
    if model is None or tokenizer is None:
        return None
    
    # Validate prompt is not empty
    if not prompt or not prompt.strip():
        return "[Empty prompt - cannot generate]"
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        
        # CRITICAL: Check if tokenization produced empty sequence
        if inputs['input_ids'].shape[1] == 0:
            return "[Tokenization produced empty sequence]"
        
        # Move inputs to model device (critical for Unsloth models)
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif hasattr(model, 'module') and hasattr(model.module, 'device'):
            inputs = {k: v.to(model.module.device) for k, v in inputs.items()}
        else:
            # Try to detect device from model parameters
            try:
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except:
                pass  # Keep inputs on default device
        
        # Get EOS token ID
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.pad_token_id
        
        # Validate input length one more time after device move
        if inputs['input_ids'].shape[1] == 0:
            return "[Input sequence is empty after device move]"
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=eos_token_id,
                eos_token_id=eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        # Check if we got any generated tokens
        if len(generated_tokens) == 0:
            return "[Model generated empty response]"
        
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    except Exception as e:
        print(f"❌ Error generating prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_dataset(dataset_path: str = "rag_analysis_dataset_v2.json"):
    """Load evaluation dataset"""
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    print(f"📂 Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} examples")
    return data

def check_list_completeness(query: str, answer: str, expected: str) -> Dict:
    """
    Check if list query extracted all items (for queries like "who are the co-founders")
    Returns completeness metrics
    """
    # Detect if this is a list query
    list_keywords = ["who are", "list", "what are", "all", "co-founders", "founders", "members", 
                     "executives", "directors", "managers", "leaders", "features", "services", 
                     "capabilities", "components", "benefits", "advantages"]
    is_list_query = any(kw in query.lower() for kw in list_keywords)
    
    if not is_list_query:
        return {
            'is_list_query': False,
            'complete': True,
            'completeness_score': 1.0,
            'missing_items': [],
            'extra_items': []
        }
    
    # Extract items from answer and expected (split by comma, semicolon, or "and")
    def extract_items(text: str) -> set:
        if not text or text.strip().lower() == "i don't have that information in the provided documents":
            return set()
        # Split by common separators
        items = re.split(r'[,;]| and | & ', text.lower())
        # Clean and filter
        items = {item.strip() for item in items if len(item.strip()) > 2}
        # Remove common prefixes/suffixes
        cleaned = set()
        for item in items:
            # Remove leading "the", "a", "an"
            item = re.sub(r'^(the|a|an)\s+', '', item)
            # Remove trailing punctuation
            item = item.rstrip('.,;:')
            if len(item) > 2:
                cleaned.add(item)
        return cleaned
    
    answer_items = extract_items(answer)
    expected_items = extract_items(expected)
    
    if not expected_items:
        return {
            'is_list_query': True,
            'complete': True,
            'completeness_score': 1.0,
            'missing_items': [],
            'extra_items': list(answer_items)
        }
    
    missing = expected_items - answer_items
    extra = answer_items - expected_items
    completeness_score = len(answer_items & expected_items) / len(expected_items) if expected_items else 0.0
    
    return {
        'is_list_query': True,
        'complete': len(missing) == 0,
        'completeness_score': completeness_score,
        'missing_items': list(missing),
        'extra_items': list(extra),
        'answer_count': len(answer_items),
        'expected_count': len(expected_items)
    }

def format_example_for_model(example: Dict, tokenizer) -> str:
    """
    Format example messages using tokenizer's chat template.
    Handles both 'messages' format and 'formatted_text' format.
    """
    # If already formatted, use it
    if 'formatted_text' in example and example.get('formatted_text'):
        return example['formatted_text']
    
    # Otherwise, format from messages
    messages = example.get('messages', [])
    if not messages:
        return None
    
    # Use tokenizer's chat template (same as training script)
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Add assistant prompt for generation
            )
            return formatted_text
        else:
            # Fallback: simple formatting
            formatted = ""
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                formatted += f"{role}: {content}\n"
            formatted += "assistant: "
            return formatted
    except Exception as e:
        print(f"⚠️  Error formatting example: {e}")
        return None

def evaluate_example(model, tokenizer, example: Dict, verbose: bool = False) -> Dict:
    """Evaluate a single example"""
    query = example.get('query', '')
    expected = example.get('expected_output', '')
    
    # Extract query and expected from messages if not directly available
    messages = example.get('messages', [])
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user' and not query:
            # Try to extract query from user message (look for "Query: ..." pattern)
            query_match = re.search(r'Query:\s*(.+?)(?:\n|$)', content, re.IGNORECASE | re.DOTALL)
            if query_match:
                query = query_match.group(1).strip()
            elif content:
                # Use first line or first 200 chars as query
                query = content.split('\n')[0][:200].strip()
        elif role == 'assistant' and not expected:
            expected = content.strip()
    
    # Format example for model (handles both formats)
    formatted_text = format_example_for_model(example, tokenizer)
    
    # Validate formatted_text is not empty
    if not formatted_text or not formatted_text.strip():
        return {
            'query': query,
            'expected': expected,
            'prediction': None,
            'match_score': 0.0,
            'has_cot_leakage': False,
            'cot_patterns': [],
            'error': 'Could not format example for model (empty messages or formatting error)',
            'completeness': {'is_list_query': False, 'complete': True, 'completeness_score': 0.0}
        }
    
    # Generate prediction
    prediction = generate_prediction(model, tokenizer, formatted_text)
    
    if prediction is None:
        return {
            'query': query,
            'expected': expected,
            'prediction': None,
            'match_score': 0.0,
            'has_cot_leakage': False,
            'cot_patterns': [],
            'error': 'Failed to generate prediction'
        }
    
    # Clean prediction (remove CoT leakage) - use aggressive cleaning
    cleaned_prediction = clean_output(prediction)
    
    # Calculate match score on cleaned prediction
    match_score = similarity_score(expected, cleaned_prediction)
    
    # Also calculate score on raw prediction for comparison
    raw_match_score = similarity_score(expected, prediction)
    
    # Detect CoT leakage
    has_leakage, patterns = detect_cot_leakage(prediction)
    
    # Check if cleaning improved the score
    cleaning_improved = match_score > raw_match_score + 5  # 5% threshold
    
    # Check list completeness (for queries like "who are the co-founders")
    completeness = check_list_completeness(query, cleaned_prediction, expected)
    
    result = {
        'query': query,
        'expected': expected[:200] + '...' if len(expected) > 200 else expected,
        'prediction': prediction[:200] + '...' if len(prediction) > 200 else prediction,
        'cleaned_prediction': cleaned_prediction[:200] + '...' if len(cleaned_prediction) > 200 else cleaned_prediction,
        'match_score': match_score,
        'raw_match_score': raw_match_score,
        'has_cot_leakage': has_leakage,
        'cot_patterns': patterns,
        'cleaning_improved': cleaning_improved,
        'completeness': completeness,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"Expected: {expected[:150]}...")
        print(f"Raw Prediction: {prediction[:150]}...")
        print(f"Cleaned Prediction: {cleaned_prediction[:150]}...")
        print(f"Match Score (cleaned): {match_score:.2f}%")
        print(f"Match Score (raw): {raw_match_score:.2f}%")
        print(f"CoT Leakage: {'Yes' if has_leakage else 'No'}")
        if has_leakage:
            print(f"Patterns: {patterns}")
        if cleaning_improved:
            print(f"✅ Cleaning improved score by {match_score - raw_match_score:.2f}%")
    
    return result

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze evaluation results"""
    if not results:
        return {}
    
    # Basic statistics
    match_scores = [r['match_score'] for r in results if r.get('match_score') is not None]
    raw_match_scores = [r.get('raw_match_score', r['match_score']) for r in results if r.get('match_score') is not None]
    cot_leakage_count = sum(1 for r in results if r.get('has_cot_leakage', False))
    cleaning_improved_count = sum(1 for r in results if r.get('cleaning_improved', False))
    
    # List completeness statistics
    list_queries = [r for r in results if r.get('completeness', {}).get('is_list_query', False)]
    incomplete_list_queries = [r for r in list_queries if not r.get('completeness', {}).get('complete', True)]
    completeness_scores = [r.get('completeness', {}).get('completeness_score', 1.0) for r in list_queries]
    
    # Score distribution
    excellent = sum(1 for s in match_scores if s >= 90)
    good = sum(1 for s in match_scores if 70 <= s < 90)
    fair = sum(1 for s in match_scores if 50 <= s < 70)
    poor = sum(1 for s in match_scores if s < 50)
    
    # CoT leakage patterns
    all_patterns = []
    for r in results:
        all_patterns.extend(r.get('cot_patterns', []))
    pattern_counts = defaultdict(int)
    for pattern in all_patterns:
        pattern_counts[pattern] += 1
    
    # Calculate improvement from cleaning
    avg_improvement = 0
    if raw_match_scores and match_scores:
        improvements = [cleaned - raw for cleaned, raw in zip(match_scores, raw_match_scores)]
        avg_improvement = np.mean(improvements) if improvements else 0
    
    analysis = {
        'total_examples': len(results),
        'match_scores': {
            'mean': np.mean(match_scores) if match_scores else 0,
            'median': np.median(match_scores) if match_scores else 0,
            'std': np.std(match_scores) if match_scores else 0,
            'min': np.min(match_scores) if match_scores else 0,
            'max': np.max(match_scores) if match_scores else 0,
        },
        'raw_match_scores': {
            'mean': np.mean(raw_match_scores) if raw_match_scores else 0,
        },
        'cleaning_impact': {
            'avg_improvement': avg_improvement,
            'examples_improved': cleaning_improved_count,
            'improvement_percentage': (cleaning_improved_count / len(results)) * 100 if results else 0,
        },
        'score_distribution': {
            'excellent (≥90%)': excellent,
            'good (70-89%)': good,
            'fair (50-69%)': fair,
            'poor (<50%)': poor,
        },
        'cot_leakage': {
            'total_with_leakage': cot_leakage_count,
            'percentage': (cot_leakage_count / len(results)) * 100 if results else 0,
            'common_patterns': dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        },
        'list_completeness': {
            'total_list_queries': len(list_queries),
            'incomplete_list_queries': len(incomplete_list_queries),
            'incomplete_percentage': (len(incomplete_list_queries) / len(list_queries) * 100) if list_queries else 0,
            'avg_completeness_score': np.mean(completeness_scores) if completeness_scores else 1.0,
            'min_completeness_score': np.min(completeness_scores) if completeness_scores else 1.0,
        },
    }
    
    return analysis

def generate_recommendations(analysis: Dict) -> List[str]:
    """Generate recommendations based on analysis"""
    recommendations = []
    
    mean_score = analysis['match_scores']['mean']
    cot_percentage = analysis['cot_leakage']['percentage']
    
    # Match score recommendations
    if mean_score < 50:
        recommendations.append("❌ CRITICAL: Mean match score < 50% - Model is underperforming")
        recommendations.append("   → Consider: Increasing LoRA rank (8-16), more training epochs, or better dataset")
    elif mean_score < 70:
        recommendations.append("⚠️  Mean match score 50-70% - Model needs improvement")
        recommendations.append("   → Consider: Increasing LoRA rank to 8, adjusting learning rate, or more training")
    elif mean_score < 85:
        recommendations.append("✅ Mean match score 70-85% - Good performance")
        recommendations.append("   → Minor improvements possible: Fine-tune learning rate or add more diverse examples")
    else:
        recommendations.append("✅ EXCELLENT: Mean match score ≥ 85% - Model performing well")
    
    # CoT leakage recommendations
    if cot_percentage > 50:
        recommendations.append("❌ CRITICAL: >50% of outputs have CoT leakage")
        recommendations.append("   → Consider: Post-processing filter, adjust training data format, or increase regularization")
    elif cot_percentage > 25:
        recommendations.append("⚠️  CoT leakage in 25-50% of outputs")
        recommendations.append("   → Consider: Post-processing to remove intermediate steps, or adjust prompt format")
    elif cot_percentage > 10:
        recommendations.append("⚠️  Minor CoT leakage (10-25%)")
        recommendations.append("   → Consider: Simple post-processing filter to clean outputs")
    else:
        recommendations.append("✅ Good: CoT leakage < 10%")
    
    # Score distribution recommendations
    poor_count = analysis['score_distribution']['poor (<50%)']
    poor_percentage = (poor_count / analysis['total_examples']) * 100 if analysis['total_examples'] > 0 else 0
    
    if poor_percentage > 30:
        recommendations.append("⚠️  >30% of examples have poor match scores")
        recommendations.append("   → Review: Query types with low scores may need more training examples")
    
    # Specific pattern recommendations
    if analysis['cot_leakage']['common_patterns']:
        top_pattern = list(analysis['cot_leakage']['common_patterns'].keys())[0]
        recommendations.append(f"💡 Most common CoT pattern: '{top_pattern}'")
        recommendations.append("   → Consider: Add regex filter for this pattern in post-processing")
    
    return recommendations

def print_analysis_report(analysis: Dict, recommendations: List[str]):
    """Print comprehensive analysis report"""
    print("\n" + "="*70)
    print("  POST-TRAINING MODEL EVALUATION REPORT")
    print("="*70)
    
    print("\n📊 MATCH SCORE STATISTICS")
    print("-"*70)
    scores = analysis['match_scores']
    raw_scores = analysis.get('raw_match_scores', {})
    print(f"  Mean (cleaned):   {scores['mean']:.2f}%")
    if raw_scores.get('mean'):
        print(f"  Mean (raw):       {raw_scores['mean']:.2f}%")
        improvement = scores['mean'] - raw_scores['mean']
        print(f"  Improvement:      {improvement:+.2f}%")
    print(f"  Median: {scores['median']:.2f}%")
    print(f"  Std:    {scores['std']:.2f}%")
    print(f"  Min:    {scores['min']:.2f}%")
    print(f"  Max:    {scores['max']:.2f}%")
    
    cleaning = analysis.get('cleaning_impact', {})
    if cleaning.get('avg_improvement', 0) > 0:
        print(f"\n  🧹 Post-processing Impact:")
        print(f"     Average improvement: {cleaning['avg_improvement']:.2f}%")
        print(f"     Examples improved: {cleaning['examples_improved']} ({cleaning['improvement_percentage']:.1f}%)")
    
    print("\n📈 SCORE DISTRIBUTION")
    print("-"*70)
    dist = analysis['score_distribution']
    total = analysis['total_examples']
    for category, count in dist.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {category:20s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\n🔍 CoT LEAKAGE ANALYSIS")
    print("-"*70)
    cot = analysis['cot_leakage']
    print(f"  Examples with leakage: {cot['total_with_leakage']} ({cot['percentage']:.1f}%)")
    if cot['common_patterns']:
        print(f"  Common patterns:")
        for pattern, count in list(cot['common_patterns'].items())[:5]:
            print(f"    - '{pattern}': {count} occurrences")
    
    print("\n📋 LIST QUERY COMPLETENESS")
    print("-"*70)
    completeness = analysis.get('list_completeness', {})
    if completeness.get('total_list_queries', 0) > 0:
        print(f"  Total list queries: {completeness['total_list_queries']}")
        print(f"  Incomplete extractions: {completeness['incomplete_list_queries']} ({completeness['incomplete_percentage']:.1f}%)")
        print(f"  Avg completeness score: {completeness['avg_completeness_score']:.2%}")
        print(f"  Min completeness score: {completeness['min_completeness_score']:.2%}")
        if completeness['incomplete_list_queries'] > 0:
            print(f"  ⚠️  Model is missing items in list queries - may need more training on STEP 5 (verify completeness)")
    else:
        print("  No list queries found in evaluation set")
    
    print("\n💡 RECOMMENDATIONS")
    print("-"*70)
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "="*70)

def main():
    """Main evaluation function"""
    print("="*70)
    print("  POST-TRAINING MODEL EVALUATION")
    print("="*70)
    print("\nThis script evaluates your trained RAG analysis model")
    print("and provides comprehensive analysis and recommendations.\n")
    
    # Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", None)  # Set in Colab: os.environ["MODEL_PATH"] = "./outputs_rag_analysis"
    DATASET_PATH = os.getenv("DATASET_PATH", "rag_analysis_dataset_v2.json")
    NUM_EVAL_EXAMPLES = int(os.getenv("NUM_EVAL_EXAMPLES", "100"))  # Evaluate on subset
    VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
    
    # Normalize MODEL_PATH if provided
    if MODEL_PATH:
        MODEL_PATH = os.path.expanduser(MODEL_PATH)  # Handle ~ in paths
        MODEL_PATH = os.path.abspath(MODEL_PATH) if not os.path.isabs(MODEL_PATH) else MODEL_PATH
    
    # Load model
    try:
        model_result = load_model(MODEL_PATH)
        if model_result is None:
            print("\n❌ Cannot proceed without model. Please check model path.")
            return
        model, tokenizer = model_result
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load dataset
    dataset = load_dataset(DATASET_PATH)
    if dataset is None:
        return
    
    # Select evaluation subset
    eval_examples = dataset[:NUM_EVAL_EXAMPLES]
    
    # Filter out invalid examples (must have messages or formatted_text)
    valid_examples = []
    skipped_count = 0
    for example in eval_examples:
        # Check if example has messages or formatted_text
        has_messages = example.get('messages') and len(example.get('messages', [])) > 0
        has_formatted = example.get('formatted_text') and example.get('formatted_text').strip()
        
        if has_messages or has_formatted:
            # Also check that we can get expected output
            expected = example.get('expected_output', '')
            if not expected:
                # Try to get from assistant message
                messages = example.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        expected = msg.get('content', '')
                        break
            
            if expected:  # Must have expected output
                valid_examples.append(example)
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"\n⚠️  Skipped {skipped_count} examples (missing messages/formatted_text or expected_output)")
    
    print(f"\n🔍 Evaluating on {len(valid_examples)} valid examples...")
    
    if len(valid_examples) == 0:
        print("❌ No valid examples to evaluate!")
        return
    
    # Evaluate examples
    results = []
    for i, example in enumerate(valid_examples):
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(valid_examples)} examples...")
        
        result = evaluate_example(model, tokenizer, example, verbose=VERBOSE)
        results.append(result)
    
    print(f"\n✅ Evaluation complete!")
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis)
    
    # Print report
    print_analysis_report(analysis, recommendations)
    
    # Save detailed results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'analysis': analysis,
            'recommendations': recommendations,
            'sample_results': results[:20],  # Save first 20 for inspection
            'all_results': results,  # Save all results for detailed analysis
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Print sample failures for inspection
    print("\n" + "="*70)
    print("  SAMPLE FAILURES (for inspection)")
    print("="*70)
    
    # Show worst performing examples
    worst = sorted(results, key=lambda x: x.get('match_score', 0))[:5]
    print("\n🔴 Worst 5 examples (lowest match scores):")
    for i, result in enumerate(worst, 1):
        print(f"\n{i}. Match Score: {result.get('match_score', 0):.2f}%")
        print(f"   Query: {result.get('query', 'N/A')[:100]}")
        print(f"   Expected: {result.get('expected', 'N/A')[:100]}")
        print(f"   Prediction: {result.get('prediction', 'N/A')[:100]}")
        if result.get('completeness', {}).get('is_list_query'):
            print(f"   ⚠️  List query - Missing: {result.get('completeness', {}).get('missing_items', [])}")
    
    # Show best performing examples
    best = sorted(results, key=lambda x: x.get('match_score', 0), reverse=True)[:5]
    print("\n✅ Best 5 examples (highest match scores):")
    for i, result in enumerate(best, 1):
        print(f"\n{i}. Match Score: {result.get('match_score', 0):.2f}%")
        print(f"   Query: {result.get('query', 'N/A')[:100]}")
        print(f"   Expected: {result.get('expected', 'N/A')[:100]}")
        print(f"   Prediction: {result.get('prediction', 'N/A')[:100]}")
    
    print("\n" + "="*70)
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
