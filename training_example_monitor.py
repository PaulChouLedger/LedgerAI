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
        show_input: bool = False
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
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.show_predictions = show_predictions
        self.show_input = show_input
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
                
                # Show model prediction if enabled (skip during very early training for better performance)
                if self.show_predictions and self.model is not None and not early_training:
                    try:
                        prediction = self._get_prediction(text)
                        if prediction:
                            print(f"   🤖 Model Output: {prediction[:150]}..." if len(prediction) > 150 else f"   🤖 Model Output: {prediction}")
                            
                            # Check if prediction matches expected (simple check)
                            if expected_answer:
                                match_score = self._calculate_match_score(expected_answer, prediction)
                                if match_score > 0.5:
                                    print(f"   ✅ Match Score: {match_score:.2%} (Good)")
                                else:
                                    print(f"   ⚠️  Match Score: {match_score:.2%} (Needs improvement)")
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
        """Get model prediction for the input text.
        
        Note: The text should already be in chat template format (as used during training).
        The model will generate from where the assistant response should start.
        """
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
            
            return prediction.strip()
        except Exception as e:
            return None
    
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


def create_example_monitor(
    dataset: Any,
    tokenizer: Any,
    model: Any,
    sample_every_n_steps: int = 50,
    num_samples: int = 3,
    show_predictions: bool = True
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
        show_input=False  # Set to True for debugging, but it's very verbose
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
