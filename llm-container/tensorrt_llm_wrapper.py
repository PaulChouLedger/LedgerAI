"""
TensorRT-LLM Wrapper
Provides unified interface for TensorRT-LLM models (Qwen, Llama, etc.)
"""

import os
from typing import List, Dict, Optional, Union, Iterator
import numpy as np

try:
    from tensorrt_llm import ModelRunner, SamplingConfig
    from tensorrt_llm.runtime import PYTHON_BINDINGS_AVAILABLE
    TENSORRT_LLM_AVAILABLE = True
except ImportError:
    TENSORRT_LLM_AVAILABLE = False
    print("[TensorRT-LLM] ⚠️ TensorRT-LLM not available - ensure TensorRT-LLM is installed")


class TensorRTLLMWrapper:
    """
    Unified wrapper for TensorRT-LLM models
    Supports both Qwen and Llama model families
    """
    
    def __init__(self, engine_dir: str, tokenizer_dir: Optional[str] = None):
        """
        Initialize TensorRT-LLM model
        
        Args:
            engine_dir: Path to TensorRT-LLM engine directory
            tokenizer_dir: Path to tokenizer directory (defaults to engine_dir)
        """
        if not TENSORRT_LLM_AVAILABLE:
            raise ImportError("TensorRT-LLM is not available. Install TensorRT-LLM.")
        
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir or engine_dir
        
        print(f"[TensorRT-LLM] 🚀 Loading engine from: {engine_dir}")
        
        # Load model runner
        try:
            self.runner = ModelRunner.from_dir(
                engine_dir,
                tokenizer_dir=self.tokenizer_dir,
                debug_mode=False
            )
            print(f"[TensorRT-LLM] ✅ Model loaded successfully")
        except Exception as e:
            print(f"[TensorRT-LLM] ❌ Failed to load model: {e}")
            raise
        
        # Get tokenizer
        self.tokenizer = self.runner.tokenizer
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.15,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Generate response from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            repeat_penalty: Repeat penalty
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            stop: List of stop sequences
            stream: Enable streaming
            **kwargs: Additional parameters
        
        Returns:
            Generated text (str) or iterator of text chunks (if stream=True)
        """
        # Convert messages to prompt using tokenizer's chat template
        prompt = self._messages_to_prompt(messages)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = np.array([input_ids], dtype=np.int32)
        
        # Create sampling config
        sampling_config = SamplingConfig(
            num_beams=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_new_tokens=max_tokens,
            stop_words_list=stop or [],
        )
        
        # Generate
        if stream:
            return self._generate_streaming(input_ids, sampling_config)
        else:
            output_ids = self.runner.generate(
                input_ids,
                sampling_config=sampling_config
            )
            
            # Decode output (remove input tokens)
            output_text = self.tokenizer.decode(
                output_ids[0][0][len(input_ids[0]):],
                skip_special_tokens=True
            )
            
            return output_text
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt using tokenizer's chat template"""
        # Use tokenizer's apply_chat_template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Fallback: manual formatting
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"
    
    def _generate_streaming(
        self,
        input_ids: np.ndarray,
        sampling_config: SamplingConfig
    ) -> Iterator[str]:
        """Generate tokens in streaming mode"""
        # TODO: Implement streaming generation
        # For now, generate full response and yield in chunks
        output_ids = self.runner.generate(
            input_ids,
            sampling_config=sampling_config
        )
        
        # Decode and yield token by token
        output_tokens = output_ids[0][0][len(input_ids[0]):]
        for token_id in output_tokens:
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if token_text:
                yield token_text
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.15,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict, Iterator[Dict]]:
        """
        OpenAI-style chat completion API
        
        Returns OpenAI-compatible response format
        """
        response_text = self.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
            stream=stream,
            **kwargs
        )
        
        if stream:
            # Return iterator of OpenAI-format chunks
            def stream_generator():
                for chunk_text in response_text:
                    yield {
                        "choices": [{
                            "delta": {
                                "content": chunk_text
                            }
                        }]
                    }
            
            return stream_generator()
        else:
            # Return OpenAI-format response
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }]
            }

