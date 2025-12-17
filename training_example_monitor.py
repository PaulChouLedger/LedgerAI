#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Example Monitor - Real-Time Example Visualization
==========================================================

Shows how training examples are being processed in real-time to help determine
if training will be successful before wasting compute power.

Features:
- Shows sample examples being processed
- Displays model predictions vs expected outputs
- Shows loss progression with example context
- Identifies if model is learning correctly
"""

import json
import re
from typing import Dict, List, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

class ExampleMonitorCallback(TrainerCallback):
    """Callback to monitor training examples in real-time."""
    
    def __init__(
        self,
        dataset: Any,
        tokenizer: Any = None,
        model: Any = None,
        sample_every_n_steps: int = 50,
        num_samples: int = 3,
        show_predictions: bool = False,  # Default False - predictions are slow
        show_input: bool = False,
        show_chunks: bool = True  # NEW: Show actual chunk text for verification
    ):
        """
        Args:
            dataset: The training dataset
            tokenizer: Tokenizer for decoding
            model: Model for generating predictions
            sample_every_n_steps: Show examples every N training steps
            num_samples: Number of examples to show each time
            show_predictions: Whether to show model predictions
            show_input: Whether to show full input (can be verbose)
            show_chunks: Whether to show actual chunk text (helps verify what model sees)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.show_predictions = show_predictions
        self.show_input = show_input
        self.show_chunks = show_chunks
        self.last_step = -1
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when training logs are written."""
        if state.global_step % self.sample_every_n_steps == 0 and state.global_step != self.last_step:
            self.last_step = state.global_step
            
            print("\n" + "=" * 80)
            print(f"📊 TRAINING EXAMPLE MONITOR - Step {state.global_step}")
            print("=" * 80)
            
            # Get current loss
            current_loss = logs.get('loss', 'N/A')
            learning_rate = logs.get('learning_rate', 'N/A')
            epoch = logs.get('epoch', 'N/A')
            
            print(f"\n📈 Training Metrics:")
            print(f"   Loss: {current_loss:.4f}" if isinstance(current_loss, float) else f"   Loss: {current_loss}")
            print(f"   Learning Rate: {learning_rate:.2e}" if isinstance(learning_rate, float) else f"   Learning Rate: {learning_rate}")
            print(f"   Epoch: {epoch:.2f}" if isinstance(epoch, float) else f"   Epoch: {epoch}")
            
            # Sample random examples from dataset
            import random
            sample_indices = random.sample(range(len(self.dataset)), min(self.num_samples, len(self.dataset)))
            
            print(f"\n📝 Sample Examples Being Processed ({len(sample_indices)} examples):")
            print("-" * 80)
            
            # Early training warning
            early_training = state.global_step < 200
            if early_training and self.show_predictions:
                print("\n   ⚠️  NOTE: Early training stage - predictions will be poor until model learns CoT structure")
                print("   💡 Focus on loss decreasing rather than prediction quality at this stage\n")
            
            for idx, example_idx in enumerate(sample_indices, 1):
                example = self.dataset[example_idx]
                text = example.get('text', '')
                
                # Parse the example to extract query and expected answer
                query, expected_answer = self._parse_example(text)
                
                print(f"\n   Example {idx} (Dataset Index {example_idx}):")
                print(f"   {'─' * 76}")
                
                if query:
                    print(f"   📋 Query: {query[:150]}..." if len(query) > 150 else f"   📋 Query: {query}")
                
                if expected_answer:
                    print(f"   ✅ Expected: {expected_answer[:150]}..." if len(expected_answer) > 150 else f"   ✅ Expected: {expected_answer}")
                
                # Show chunk understanding diagnostics (NEW)
                chunks_analysis = self._analyze_chunks(text, query, expected_answer, show_chunk_text=self.show_chunks)
                if chunks_analysis:
                    print(f"\n   🔍 Chunk Understanding Analysis:")
                    for chunk_info in chunks_analysis:
                        # Check if this is chunk text (starts with "Chunk X Text:")
                        if chunk_info.startswith("Chunk") and "Text:" in chunk_info:
                            print(f"      {chunk_info}")
                        elif chunk_info.startswith("   "):  # Indented chunk text
                            print(f"      {chunk_info}")
                        else:
                            print(f"      {chunk_info}")
                
                # Show model prediction if enabled (skip during very early training for better performance)
                if self.show_predictions and self.model is not None and not early_training:
                    try:
                        # Get raw prediction first (before filtering)
                        raw_prediction = self._generate_prediction(text, apply_filter=False)
                        
                        if raw_prediction:
                            # Check if CoT leakage was present before filtering
                            has_leakage = self._has_cot_leakage(raw_prediction)
                            
                            # Apply filter to get cleaned prediction
                            if has_leakage:
                                try:
                                    from cot_leakage_filter import clean_cot_leakage
                                    prediction = clean_cot_leakage(raw_prediction, aggressive=True)
                                except ImportError:
                                    prediction = raw_prediction
                            else:
                                prediction = raw_prediction
                            
                            if prediction:
                                # Display prediction with indication if leakage was filtered
                                if has_leakage:
                                    print(f"   🤖 Model Output (cleaned): {prediction[:150]}..." if len(prediction) > 150 else f"   🤖 Model Output (cleaned): {prediction}")
                                    print(f"   🧹 CoT leakage filtered from raw output")
                                else:
                                    print(f"   🤖 Model Output: {prediction[:150]}..." if len(prediction) > 150 else f"   🤖 Model Output: {prediction}")
                                
                                # Check if prediction matches expected (simple check)
                                if expected_answer:
                                    match_score = self._calculate_match_score(expected_answer, prediction)
                                    if match_score > 0.5:
                                        print(f"   ✅ Match Score: {match_score:.2%} (Good)")
                                    else:
                                        print(f"   ⚠️  Match Score: {match_score:.2%} (Needs improvement)")
                                
                                # Show model's understanding vs chunks (NEW - JSON mode)
                                if self.show_predictions and prediction:
                                    model_understanding = self._analyze_model_output(prediction, chunks_analysis, query)
                                    if model_understanding:
                                        print(f"\n   🤖 Model's Understanding:")
                                        for info in model_understanding:
                                            print(f"      {info}")
                    except Exception as e:
                        print(f"   ⚠️  Could not generate prediction: {e}")
                elif self.show_predictions and early_training:
                    print(f"   ⏭️  Predictions skipped (too early - model hasn't learned CoT structure yet)")
                
                # Show input if enabled (can be very verbose)
                if self.show_input:
                    print(f"\n   📄 Full Input (first 300 chars):")
                    print(f"   {text[:300]}...")
            
            print("\n" + "=" * 80)
            print("💡 Interpretation:")
            if early_training:
                print("   ⚠️  EARLY TRAINING STAGE:")
                print("   - Loss should decrease gradually (currently very early)")
                print("   - Model hasn't learned CoT structure yet - poor predictions are normal")
                print("   - Focus on loss trend, not prediction quality until step 200+")
                print("   - After step 200, predictions should start improving")
            else:
                print("   - Loss should decrease over time")
                print("   - Model outputs should gradually match expected answers")
                print("   - If loss plateaus or predictions don't improve, consider:")
                print("     • Adjusting learning rate")
                print("     • Adding more training examples")
                print("     • Increasing LoRA rank")
                print("     • Checking if examples are too complex")
            print("=" * 80 + "\n")
    
    def _parse_example(self, text: str) -> tuple:
        """Extract query and expected answer from example text."""
        query = None
        expected_answer = None
        
        # The formatted text from Qwen chat template contains:
        # <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_response}<|im_end|>
        
        # Extract query from user message section
        # Look for user section and extract Query: line
        user_match = re.search(r'<\|im_start\|>user\s*\n(.*?)<\|im_end\|>', text, re.DOTALL)
        if user_match:
            user_content = user_match.group(1)
            query_match = re.search(r'Query:\s*(.+?)(?:\n\nRAG|$)', user_content, re.DOTALL)
            if query_match:
                query = query_match.group(1).strip()
        
        # Extract expected answer from assistant message section
        # NEW FORMAT: Assistant message contains ONLY final answer (no STEP markers)
        # OLD FORMAT: Assistant message contains STEP 6/7 markers
        assistant_match = re.search(r'<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>', text, re.DOTALL)
        if assistant_match:
            assistant_content = assistant_match.group(1)
            
            # Check if this is the new format (no STEP markers - just final answer)
            has_step_markers = re.search(r'STEP\s+\d+:', assistant_content)
            
            if not has_step_markers:
                # NEW FORMAT: Assistant message IS the final answer (no CoT steps)
                expected_answer = assistant_content.strip()
            else:
                # OLD FORMAT: Extract from STEP 6 or STEP 7
                # Look for STEP 6: SYNTHESIZE RESPONSE (new 6-step structure)
                step6_match = re.search(r'STEP 6:\s*SYNTHESIZE RESPONSE\s*\n(.+?)(?=\n\nSTEP |\nSTEP |$)', assistant_content, re.DOTALL)
                if step6_match:
                    expected_answer = step6_match.group(1).strip()
                else:
                    # Fallback: look for STEP 7: SYNTHESIZE RESPONSE (old 7-step structure)
                    step7_match = re.search(r'STEP 7:\s*SYNTHESIZE RESPONSE\s*\n(.+?)(?=\n\nSTEP |\nSTEP |$)', assistant_content, re.DOTALL)
                    if step7_match:
                        expected_answer = step7_match.group(1).strip()
                    else:
                        # Last resort: look for any text after last STEP marker
                        # Find the last STEP marker and get everything after it
                        last_step_match = re.search(r'STEP \d+:\s*[^\n]+\s*\n(.+?)$', assistant_content, re.DOTALL)
                        if last_step_match:
                            expected_answer = last_step_match.group(1).strip()
        
        # Clean up expected answer - remove any remaining STEP markers or formatting
        if expected_answer:
            # Remove any STEP markers that might be in the answer (for old format)
            expected_answer = re.sub(r'^STEP\s+\d+:.*?\n', '', expected_answer, flags=re.MULTILINE)
            # Remove any special tokens
            expected_answer = re.sub(r'<\|[^|]+\|>', '', expected_answer)
            expected_answer = expected_answer.strip()
        
        return query, expected_answer
    
    def _get_prediction(self, text: str, max_new_tokens: int = 300) -> Optional[str]:
        """Get model prediction for the input text (with CoT leakage filtering).
        
        Note: The text should already be in chat template format (as used during training).
        The model will generate from where the assistant response should start.
        """
        return self._generate_prediction(text, max_new_tokens, apply_filter=True)
    
    def _generate_prediction(self, text: str, max_new_tokens: int = 300, apply_filter: bool = True) -> Optional[str]:
        """Generate prediction from model, optionally applying CoT leakage filter."""
        try:
            # The text is already in chat template format from training
            # We need to find where the assistant response should start
            # In Qwen format: <|im_start|>assistant\n[generation starts here]
            
            # Find the assistant start token
            assistant_start = text.find('<|im_start|>assistant')
            if assistant_start == -1:
                # Fallback: just use the full text
                prompt_text = text
            else:
                # Get everything up to and including the assistant start
                prompt_text = text[:assistant_start] + '<|im_start|>assistant\n'
            
            # Tokenize input
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the generated part
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Remove any special tokens that might be at the start
            prediction = prediction.replace('<|im_end|>', '').strip()
            
            # Extract STEP 6 if present
            if "STEP 6: SYNTHESIZE RESPONSE" in prediction:
                step6_start = prediction.find("STEP 6: SYNTHESIZE RESPONSE")
                prediction = prediction[step6_start + len("STEP 6: SYNTHESIZE RESPONSE"):].strip()
            elif "STEP 7: SYNTHESIZE RESPONSE" in prediction:
                step7_start = prediction.find("STEP 7: SYNTHESIZE RESPONSE")
                prediction = prediction[step7_start + len("STEP 7: SYNTHESIZE RESPONSE"):].strip()
            
            # Apply CoT leakage filter to clean the prediction (if requested)
            # This shows what the model would output after post-processing
            if apply_filter:
                try:
                    from cot_leakage_filter import clean_cot_leakage
                    prediction = clean_cot_leakage(prediction, aggressive=True)
                except ImportError:
                    # If filter not available, just continue with raw prediction
                    pass
            
            return prediction.strip()
        except Exception as e:
            return None
    
    def _has_cot_leakage(self, text: str) -> bool:
        """Check if text contains CoT leakage patterns."""
        if not text:
            return False
        
        cot_patterns = [
            r'STEP\s*[1-6]',
            r'Step\s*[1-6]',
            r'Extract information from Chunk',
            r'Chunk\s*\d+[:\-]?\s*$',  # Standalone chunk references
        ]
        
        for pattern in cot_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_match_score(self, expected: str, predicted: str) -> float:
        """Calculate a match score between expected and predicted.
        
        Uses a combination of:
        1. Key phrase matching (for multi-item answers)
        2. Word overlap (for general similarity)
        """
        if not expected or not predicted:
            return 0.0
        
        expected_lower = expected.lower()
        predicted_lower = predicted.lower()
        
        # For list-style answers (comma/and separated), check if key items are present
        # Split by common separators
        import re
        expected_items = [item.strip() for item in re.split(r'[,;]| and | or ', expected_lower) if item.strip()]
        predicted_items = [item.strip() for item in re.split(r'[,;]| and | or ', predicted_lower) if item.strip()]
        
        # If we have multiple expected items, check how many are found
        if len(expected_items) > 1:
            found_items = 0
            for exp_item in expected_items:
                # Check if this item (or its key words) appear in predicted
                exp_words = set(exp_item.split())
                for pred_item in predicted_items:
                    pred_words = set(pred_item.split())
                    # If significant overlap (>=50% of expected words), count as found
                    if len(exp_words) > 0 and len(exp_words.intersection(pred_words)) >= len(exp_words) * 0.5:
                        found_items += 1
                        break
            
            if len(expected_items) > 0:
                item_score = found_items / len(expected_items)
            else:
                item_score = 0.0
        else:
            item_score = 0.0
        
        # Also calculate word overlap for general similarity
        expected_words = set(re.findall(r'\b\w+\b', expected_lower))
        predicted_words = set(re.findall(r'\b\w+\b', predicted_lower))
        
        if not expected_words:
            return 0.0
        
        word_overlap = len(expected_words.intersection(predicted_words))
        word_score = word_overlap / len(expected_words)
        
        # Combine scores (weight item matching more for list answers)
        if len(expected_items) > 1:
            # For list answers, weight item matching 70%, word overlap 30%
            return item_score * 0.7 + word_score * 0.3
        else:
            # For single answers, use word overlap
            return word_score
    
    def _analyze_chunks(self, text: str, query: str, expected_answer: str = None, show_chunk_text: bool = True) -> List[str]:
        """
        Analyze chunks to show what entities/items are actually in each chunk.
        Helps diagnose if model is reading chunks completely.
        
        Args:
            text: Full training example text
            query: The query being asked
            expected_answer: Expected answer (for comparison)
            show_chunk_text: Whether to include actual chunk text in output
        
        Returns:
            List of strings describing what's in each chunk
        """
        analysis = []
        
        try:
            # Extract chunks from user message
            user_match = re.search(r'<\|im_start\|>user\s*\n(.*?)<\|im_end\|>', text, re.DOTALL)
            if not user_match:
                return analysis
            
            user_content = user_match.group(1)
            
            # Find all chunks with their scores
            # Handle both single and double quotes, and escaped quotes
            chunk_pattern = r'\[Chunk (\d+)\] Score: ([\d.]+).*?FULL CHUNK TEXT: [\'"](.+?)[\'"]'
            chunks = re.findall(chunk_pattern, user_content, re.DOTALL)
            
            # Also handle escaped quotes in chunk text
            chunks_cleaned = []
            for chunk_num, chunk_score, chunk_text in chunks:
                # Remove escape sequences
                chunk_text_clean = chunk_text.replace("\\'", "'").replace('\\"', '"')
                chunks_cleaned.append((chunk_num, chunk_score, chunk_text_clean))
            chunks = chunks_cleaned
            
            if not chunks:
                return analysis
            
            # Determine query type
            is_entity_query = any(word in query.lower() for word in ['who are', 'who is', 'executives', 'managers', 'directors', 'founders', 'co-founders', 'leaders', 'members'])
            is_list_query = any(word in query.lower() for word in ['list', 'what are', 'features', 'benefits', 'components', 'capabilities', 'services'])
            
            # Parse expected answer if JSON
            expected_items = []
            if expected_answer:
                try:
                    expected_json = json.loads(expected_answer)
                    expected_items = expected_json.get('items', [])
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, try to parse as natural language
                    if is_entity_query or is_list_query:
                        # Split by comma/and
                        expected_items = [item.strip() for item in re.split(r'[,;]| and ', expected_answer) if item.strip()]
            
            # Analyze each chunk
            for chunk_num, chunk_score, chunk_text in chunks:
                chunk_num = int(chunk_num)
                chunk_score = float(chunk_score)
                chunk_info = []
                
                # Show actual chunk text if requested (helps verify what model sees)
                if show_chunk_text:
                    # Show full chunk text (truncated if very long)
                    if len(chunk_text) > 400:
                        chunk_preview = chunk_text[:400] + "..."
                        chunk_info.append(f"Chunk {chunk_num} (Score: {chunk_score:.2f}) - Full Text ({len(chunk_text)} chars, showing first 400):")
                    else:
                        chunk_preview = chunk_text
                        chunk_info.append(f"Chunk {chunk_num} (Score: {chunk_score:.2f}) - Full Text ({len(chunk_text)} chars):")
                    # Indent the text for readability
                    chunk_info.append(f"   \"{chunk_preview}\"")
                    chunk_info.append("")  # Empty line for readability
                
                # Find entities/items in this chunk
                if is_entity_query:
                    # Look for person names (common patterns)
                    # Pattern: "Name serves as ROLE at Company" or "As ROLE, Name..." or "Name is ROLE"
                    name_patterns = [
                        r'([A-Z][a-z]+ [A-Z][a-z]+) serves as',
                        r'As [^,]+, ([A-Z][a-z]+ [A-Z][a-z]+)',
                        r'([A-Z][a-z]+ [A-Z][a-z]+) holds the position',
                        r'([A-Z][a-z]+ [A-Z][a-z]+) is (?:executive|manager|director|founder|co-founder|leader|member)',
                        r'([A-Z][a-z]+ [A-Z][a-z]+) is [^,]+,',
                        r'([A-Z][a-z]+ [A-Z][a-z]+), (?:executive|manager|director|founder|co-founder|leader|member)',
                    ]
                    
                    found_names = set()
                    for pattern in name_patterns:
                        matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                        found_names.update(matches)
                    
                    # Also try simple pattern: Capitalized First Last name
                    # This catches names that might not match the above patterns
                    simple_names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', chunk_text)
                    # Filter out common false positives (company names, etc.)
                    false_positives = {'Smart Systems', 'Data Systems', 'Cloud Systems', 'AI Systems', 'Tech Systems'}
                    for name in simple_names:
                        if name not in false_positives and len(name.split()) == 2:
                            # Check if it appears in context that suggests it's a person
                            name_lower = name.lower()
                            chunk_lower = chunk_text.lower()
                            name_idx = chunk_lower.find(name_lower)
                            if name_idx >= 0:
                                # Check surrounding context (50 chars before/after)
                                context_start = max(0, name_idx - 50)
                                context_end = min(len(chunk_lower), name_idx + len(name) + 50)
                                context = chunk_lower[context_start:context_end]
                                # If context suggests person (role words nearby), include it
                                role_words = ['executive', 'manager', 'director', 'founder', 'co-founder', 'leader', 'member', 'serves', 'holds', 'position']
                                if any(role in context for role in role_words):
                                    found_names.add(name)
                    
                    if found_names:
                        chunk_info.append(f"Chunk {chunk_num}: Found {len(found_names)} entities: {', '.join(sorted(found_names)[:5])}")
                        if len(found_names) > 5:
                            chunk_info[-1] += f" (+{len(found_names)-5} more)"
                    else:
                        chunk_info.append(f"Chunk {chunk_num}: No entities found")
                
                elif is_list_query:
                    # Look for list items (features, benefits, etc.)
                    # Common patterns: "offers X", "provides X", "X is available"
                    list_patterns = [
                        r'offers ([^,\.]+)',
                        r'provides ([^,\.]+)',
                        r'([^,\.]+) is available',
                        r'key (?:features|benefits|components): ([^,\.]+)',
                    ]
                    
                    found_items = set()
                    for pattern in list_patterns:
                        matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                        found_items.update([m.strip() for m in matches if len(m.strip()) > 3])
                    
                    if found_items:
                        chunk_info.append(f"Chunk {chunk_num}: Found {len(found_items)} items: {', '.join(list(found_items)[:3])}")
                        if len(found_items) > 3:
                            chunk_info[-1] += f" (+{len(found_items)-3} more)"
                    else:
                        chunk_info.append(f"Chunk {chunk_num}: No list items found")
                
                # Check if expected items are in this chunk
                if expected_items:
                    found_expected = []
                    for item in expected_items:
                        # Simple check: is this item mentioned in chunk?
                        item_words = set(item.lower().split())
                        chunk_words = set(chunk_text.lower().split())
                        # If significant overlap, consider it found
                        if len(item_words) > 0 and len(item_words.intersection(chunk_words)) >= len(item_words) * 0.6:
                            found_expected.append(item)
                    
                    if found_expected:
                        chunk_info.append(f"         ✅ Contains {len(found_expected)}/{len(expected_items)} expected items: {', '.join(found_expected[:3])}")
                        if len(found_expected) > 3:
                            chunk_info[-1] += f" (+{len(found_expected)-3} more)"
                
                if chunk_info:
                    analysis.extend(chunk_info)
            
            # Summary
            if expected_items and len(chunks) > 0:
                total_chunks = len(chunks)
                analysis.append(f"📊 Summary: {total_chunks} chunks, {len(expected_items)} expected items")
        
        except Exception as e:
            analysis.append(f"⚠️  Error analyzing chunks: {e}")
        
        return analysis
    
    def _analyze_model_output(self, prediction: str, chunks_analysis: List[str], query: str) -> List[str]:
        """
        Analyze model's output to show what it extracted vs what's in chunks.
        Helps diagnose extraction completeness issues.
        
        Returns:
            List of strings describing model's understanding
        """
        analysis = []
        
        try:
            # Try to parse as JSON
            try:
                # Remove markdown code blocks if present
                prediction_clean = prediction.strip()
                if prediction_clean.startswith('```json'):
                    prediction_clean = prediction_clean[7:]
                if prediction_clean.startswith('```'):
                    prediction_clean = prediction_clean[3:]
                if prediction_clean.endswith('```'):
                    prediction_clean = prediction_clean[:-3]
                prediction_clean = prediction_clean.strip()
                
                model_json = json.loads(prediction_clean)
                model_items = model_json.get('items', [])
                model_answer_type = model_json.get('answer_type', 'unknown')
                model_chunks_used = model_json.get('chunks_used', [])
                
                analysis.append(f"Answer type: {model_answer_type}")
                analysis.append(f"Extracted {len(model_items)} items: {', '.join(model_items[:5])}")
                if len(model_items) > 5:
                    analysis[-1] += f" (+{len(model_items)-5} more)"
                
                # Check for duplicates
                unique_items = set(model_items)
                if len(unique_items) < len(model_items):
                    duplicates = len(model_items) - len(unique_items)
                    analysis.append(f"⚠️  Found {duplicates} duplicate(s) - model may be repeating same entity")
                
                # Check which chunks were used
                if model_chunks_used:
                    analysis.append(f"Chunks used: {model_chunks_used}")
                else:
                    analysis.append(f"⚠️  No chunks_used specified - model may not be tracking source")
                
                # Compare with expected (if available from chunks_analysis)
                # This is a simple heuristic - could be improved
                if chunks_analysis:
                    # Count expected items from chunks_analysis
                    expected_count = 0
                    for line in chunks_analysis:
                        if 'expected items' in line.lower():
                            match = re.search(r'(\d+)/(\d+) expected items', line)
                            if match:
                                expected_count = int(match.group(2))
                                break
                    
                    if expected_count > 0:
                        if len(model_items) < expected_count:
                            missing = expected_count - len(model_items)
                            analysis.append(f"❌ Missing {missing} item(s) - incomplete extraction")
                        elif len(model_items) == expected_count:
                            analysis.append(f"✅ All {expected_count} items extracted")
                        else:
                            analysis.append(f"⚠️  Extracted {len(model_items)} items (expected {expected_count}) - may have duplicates")
            
            except json.JSONDecodeError:
                # Not JSON, try natural language parsing
                is_entity_query = any(word in query.lower() for word in ['who are', 'who is', 'executives', 'managers'])
                if is_entity_query:
                    # Try to extract names
                    names = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', prediction)
                    if names:
                        unique_names = list(set(names))
                        analysis.append(f"Extracted {len(unique_names)} unique entities: {', '.join(unique_names[:5])}")
                        if len(names) > len(unique_names):
                            analysis.append(f"⚠️  Found {len(names) - len(unique_names)} duplicate(s)")
        
        except Exception as e:
            analysis.append(f"⚠️  Error analyzing model output: {e}")
        
        return analysis


def create_example_monitor(
    dataset: Any,
    tokenizer: Any,
    model: Any,
    sample_every_n_steps: int = 50,
    num_samples: int = 3,
    show_predictions: bool = True,
    show_chunks: bool = True  # NEW: Show actual chunk text
) -> ExampleMonitorCallback:
    """
    Create an example monitor callback for training.
    
    Args:
        dataset: Training dataset
        tokenizer: Tokenizer
        model: Model being trained
        sample_every_n_steps: Show examples every N steps
        num_samples: Number of examples to show
        show_predictions: Whether to show model predictions (slower but more informative)
        show_chunks: Whether to show actual chunk text (helps verify what model sees)
    
    Returns:
        ExampleMonitorCallback instance
    """
    return ExampleMonitorCallback(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        sample_every_n_steps=sample_every_n_steps,
        num_samples=num_samples,
        show_predictions=show_predictions,
        show_input=False,  # Set to True for debugging, but it's very verbose
        show_chunks=show_chunks  # Show chunk text for verification
    )


# ============================================================================
# Usage Example
# ============================================================================

"""
To use in train_rag_analysis_colab.py:

1. Import the callback:
   from training_example_monitor import create_example_monitor

2. Create the monitor before training:
   example_monitor = create_example_monitor(
       dataset=train_dataset,
       tokenizer=tokenizer,
       model=model,
       sample_every_n_steps=50,  # Show examples every 50 steps
       num_samples=3,  # Show 3 examples each time
       show_predictions=True  # Show model predictions (slower)
   )

3. Add to trainer:
   trainer = SFTTrainer(
       model=model,
       tokenizer=tokenizer,
       train_dataset=train_dataset,
       dataset_text_field="text",
       max_seq_length=MAX_SEQ_LENGTH,
       args=training_args,
       callbacks=[example_monitor]  # Add callback here
   )

This will show:
- Sample examples being processed
- Expected vs predicted outputs
- Match scores to see if model is learning
- Loss progression with context
"""
