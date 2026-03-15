"""
Shared Base Class for LLM Containers
====================================
Common functionality for both generic and medical LLM containers:
- Model loading and management
- LLM chat wrapper
- Health checks
- Sentence tagging for TTS
- Common utilities
"""

import os
import json
import threading
from typing import Optional, Iterator, Dict, Any
from llama_cpp import Llama
from flask import Flask, jsonify


class BaseLLMContainer:
    """Base class for LLM containers with shared functionality"""
    
    # Thread safety
    llm_lock = threading.Lock()
    
    # Default LLM parameters (can be overridden by subclasses)
    LLM_TEMPERATURE_SIMPLE = 0.7
    LLM_TOP_P = 0.95
    LLM_TOP_K = 40
    LLM_REPEAT_PENALTY = 1.2  # Increased to match test script (was 1.1)
    LLM_NUM_PREDICT_DEFAULT = 200
    SIMPLE_N_CTX = 4096
    SIMPLE_CHAT_FORMAT = "qwen"
    N_THREADS = 8
    N_BATCH = 256
    CACHE_PROMPT = True
    
    def __init__(self, service_name: str, default_model_path: str = "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"):
        """
        Initialize base LLM container
        
        Args:
            service_name: Name of the service (e.g., "aura-llm-generic", "aura-llm-medical")
            default_model_path: Default model path if not found in settings/env
        """
        self.service_name = service_name
        self.default_model_path = default_model_path
        self.model_path = None
        self.llm_simple = None
        self._model_loaded = False
    
    def resolve_model_path(self) -> str:
        """
        Resolve model path from app_settings.json or environment variable.
        Priority: app_settings.json > SIMPLE_MODEL_PATH env var > default
        """
        # 1) Try app_settings.json first
        try:
            settings_path = "/app/data/app_settings.json"
            if os.path.isfile(settings_path):
                with open(settings_path, "r") as f:
                    data = json.load(f)
                    name = (data.get("llm_model") or "").strip()
                    if name:
                        candidate = f"/models/{name}" if not name.startswith("/") else name
                        if os.path.isfile(candidate):
                            print(f"[{self.service_name}] 🎯 Using model from settings: {candidate}")
                            return candidate
                        else:
                            print(f"[{self.service_name}] ⚠️ Model from settings not found: {candidate}")
        except Exception as e:
            print(f"[{self.service_name}] ⚠️ Failed reading app settings: {e}")
        
        # 2) Use environment variable (set by Dockerfile) as fallback
        env_path = os.getenv("SIMPLE_MODEL_PATH", "")
        if env_path and os.path.isfile(env_path):
            print(f"[{self.service_name}] 🛟 Using model from environment: {env_path}")
            return env_path
        
        # 3) Final fallback
        print(f"[{self.service_name}] 🛟 Using default model: {self.default_model_path}")
        return self.default_model_path
    
    def load_model(self, n_gpu_layers: int = 0, use_mlock: bool = True, use_mmap: bool = True) -> bool:
        """
        Load the LLM model. Must be called before using the container.
        
        Args:
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)
            use_mlock: Lock memory to prevent swapping
            use_mmap: Use memory mapping for model loading
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_loaded and self.llm_simple is not None:
            return True
        
        try:
            self.model_path = self.resolve_model_path()
            print(f"[{self.service_name}] 📦 Loading model: {self.model_path}")
            
            if n_gpu_layers != 0:
                print(f"[{self.service_name}] 🚀 GPU acceleration: {n_gpu_layers} layers offloaded to GPU")
            
            self.llm_simple = Llama(
                model_path=self.model_path,
                n_ctx=self.SIMPLE_N_CTX,
                n_threads=1,  # Use 1 thread for deterministic output (temperature=0 alone isn't enough)
                n_batch=self.N_BATCH,
                n_gpu_layers=n_gpu_layers,
                chat_format=self.SIMPLE_CHAT_FORMAT,
                use_mlock=use_mlock,
                use_mmap=use_mmap,
                verbose=False
            )
            
            self._model_loaded = True
            print(f"[{self.service_name}] ✅ Model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"[{self.service_name}] ❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_llm_response_content(self, response) -> str:
        """Extract text content from LLM response"""
        if isinstance(response, dict):
            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content']
            elif 'content' in response:
                return response['content']
        return str(response)
    
    def llm_chat_simple(self, messages, max_tokens=None, temperature=None, stream=False, **kwargs):
        """
        Wrapper for LLM chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate (defaults to LLM_NUM_PREDICT_DEFAULT)
            temperature: Sampling temperature (defaults to LLM_TEMPERATURE_SIMPLE)
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
        
        Returns:
            If stream=True: Iterator of response chunks
            If stream=False: String response content
        """
        if not self._model_loaded or self.llm_simple is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if temperature is None:
            temperature = float(self.LLM_TEMPERATURE_SIMPLE)
        
        if max_tokens is None:
            max_tokens = int(self.LLM_NUM_PREDICT_DEFAULT)
        
        generation_params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.pop("top_p", float(self.LLM_TOP_P)),
            "top_k": kwargs.pop("top_k", int(self.LLM_TOP_K)),
            "repeat_penalty": kwargs.pop("repeat_penalty", float(self.LLM_REPEAT_PENALTY)),
            "stream": stream,
            **kwargs  # This allows seed and other params to be passed through
        }
        
        # Default stop tokens to empty list if not provided (can be overridden via kwargs)
        if "stop" not in generation_params:
            generation_params["stop"] = []
        
        with self.llm_lock:
            try:
                response = self.llm_simple.create_chat_completion(**generation_params)
                
                if stream:
                    # Check if response is actually an iterator
                    if hasattr(response, '__iter__'):
                        # Return iterator directly without debug wrapper
                        return response
                    else:
                        print(f"[{self.service_name}] ⚠️ WARNING: LLM did not return iterator for stream=True, got: {type(response)}")
                        return iter([])
                
                return self.extract_llm_response_content(response)
            except Exception as e:
                print(f"[{self.service_name}] ❌ Error in llm_chat_simple: {e}")
                import traceback
                traceback.print_exc()
                if stream:
                    return iter([])
                return ""
    
    def health_check_response(self, additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate health check response
        
        Args:
            additional_info: Additional fields to include in health check
        
        Returns:
            Health check dict
        """
        try:
            simple_loaded = self.llm_simple is not None
            
            response = {
                "status": "ok",
                "service": self.service_name,
                "models": {
                    "simple_loaded": simple_loaded,
                    "simple_path": self.model_path or "not_loaded"
                }
            }
            
            if additional_info:
                response.update(additional_info)
            
            return response
        except Exception as e:
            return {
                "status": "error",
                "service": self.service_name,
                "error": str(e)
            }
    
    def sentence_tag_stream(self, word_stream: Iterator[str]) -> Iterator[str]:
        """
        Wrap word stream with <sentence_start>/<sentence_end> markers, splitting on sentence boundaries.
        Each complete sentence/phrase gets its own tags for natural TTS playback.
        Expands abbreviations like "e.g." → "for example", "i.e." → "that is", "etc." → "etcetera".
        
        IMPORTANT: This processes words incrementally with minimal buffering (1 token lookahead),
        allowing tokens to be sent to TTS as they're generated.
        
        Args:
            word_stream: Iterator of word tokens from LLM
        
        Yields:
            Tagged tokens with <sentence_start> and <sentence_end> markers
        """
        print(f"[{self.service_name}] 🔍 DEBUG: sentence_tag_stream CALLED - creating generator")
        sentence_buffer = ""
        sentence_open = False
        prev_word = None
        buffered_word = None  # One-token lookahead buffer for multi-token abbreviations
        
        # Abbreviation expansions (abbrev -> full text)
        abbrev_expansions = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'dr.': 'doctor',
            'mr.': 'mister',
            'mrs.': 'missus',
            'ms.': 'miss',
            'prof.': 'professor',
            'sr.': 'senior',
            'jr.': 'junior',
        }
        
        # Multi-token abbreviation patterns (first part -> (second part, expansion))
        multi_token_abbrevs = {
            'e.': ('g.', 'for example'),  # e.g.
            'i.': ('e.', 'that is'),  # i.e.
        }
        
        def yield_word(word_to_yield):
            """Helper to yield a word, expanding abbreviations if needed"""
            nonlocal sentence_buffer, sentence_open
            
            word_stripped = word_to_yield.strip()
            
            # Special handling for standalone dashes: they start new sentences for list items
            if word_stripped == '-':
                # Close previous sentence if open
                if sentence_open:
                    yield "<sentence_end>"
                    sentence_buffer = ""
                # Start new sentence for list item (dash is first word)
                sentence_open = True
                yield "<sentence_start>"
                yield word_to_yield
                return
            
            # Check for abbreviations (full match)
            if word_stripped.lower() in abbrev_expansions:
                expansion = abbrev_expansions[word_stripped.lower()]
                yield word_to_yield.replace(word_stripped, expansion)
                return
            
            # Check for multi-token abbreviations (e.g., "e." followed by "g.")
            if prev_word and prev_word.strip().lower() in multi_token_abbrevs:
                abbrev_info = multi_token_abbrevs[prev_word.strip().lower()]
                if word_stripped.lower() == abbrev_info[0]:
                    # Replace previous word with expansion
                    sentence_buffer = sentence_buffer.rsplit(prev_word, 1)[0] + abbrev_info[1]
                    yield abbrev_info[1] + " "
                    return
            
            # Normal word processing
            yield word_to_yield
        
        # Track if we've yielded anything
        has_yielded_anything = False
        
        # Process word stream
        word_count = 0
        
        # Force first iteration to trigger the chain
        try:
            first_word = next(word_stream)
            word_count += 1
            
            # Process first word
            if not first_word:
                pass  # Skip empty words
            else:
                has_yielded_anything = True
                
                # Handle buffered word from lookahead
                if buffered_word is not None:
                    for token in yield_word(buffered_word):
                        yield token
                    buffered_word = None
                
                # Check if current word could be first part of multi-token abbreviation
                word_stripped = first_word.strip()
                word_clean = word_stripped.lstrip('(').lstrip('[').lstrip('{').lower()
                if len(word_clean) == 2 and word_clean[0].isalpha() and word_clean[-1] == '.':
                    if word_clean in multi_token_abbrevs:
                        # Buffer this word to check next token
                        buffered_word = first_word
                        prev_word = first_word
                    else:
                        # Normal processing - not part of multi-token abbreviation
                        if not sentence_open:
                            yield "<sentence_start>"
                            sentence_open = True
                            sentence_buffer = ""
                        for item in yield_word(first_word):
                            yield item
                        sentence_buffer += first_word
                        prev_word = first_word
                else:
                    # Normal processing - not part of multi-token abbreviation
                    if not sentence_open:
                        yield "<sentence_start>"
                        sentence_open = True
                        sentence_buffer = ""
                    for item in yield_word(first_word):
                        yield item
                    sentence_buffer += first_word
                    prev_word = first_word
                    
                    # Check for sentence endings
                    word_clean = first_word.strip()
                    if word_clean:
                        # Check if word ends with sentence punctuation
                        if word_clean[-1] in ('.', '!', '?'):
                            # Check if it's an abbreviation (common patterns)
                            is_abbrev = (
                                len(word_clean) <= 4 and word_clean[-1] == '.' and
                                word_clean[0].isupper() and
                                word_clean[:-1].isalpha()
                            )
                            
                            if not is_abbrev:
                                # Real sentence ending
                                yield "<sentence_end>"
                                sentence_open = False
                                sentence_buffer = ""
            
            # Continue with rest of iterator
            for word in word_stream:
                word_count += 1
                if not word:
                    continue
                
                has_yielded_anything = True
                
                # Handle buffered word from lookahead
                if buffered_word is not None:
                    for token in yield_word(buffered_word):
                        yield token
                    buffered_word = None
                
                # Check if this might be part of a multi-token abbreviation
                word_stripped = word.strip().lower()
                if word_stripped in ['e.', 'i.']:
                    # Buffer this word and look ahead
                    buffered_word = word
                    prev_word = word
                    continue
                
                # Start sentence if not already open
                if not sentence_open:
                    yield "<sentence_start>"
                    sentence_open = True
                    sentence_buffer = ""
                
                # Process word
                for token in yield_word(word):
                    yield token
                
                sentence_buffer += word
                prev_word = word
                
                # Check for sentence endings
                word_clean = word.strip()
                if word_clean:
                    # Check if word ends with sentence punctuation
                    if word_clean[-1] in ('.', '!', '?'):
                        # Check if it's an abbreviation (common patterns)
                        is_abbrev = (
                            len(word_clean) <= 4 and word_clean[-1] == '.' and
                            word_clean[0].isupper() and
                            word_clean[:-1].isalpha()
                        )
                        
                        if not is_abbrev:
                            # Real sentence ending
                            yield "<sentence_end>"
                            sentence_open = False
                            sentence_buffer = ""
        except StopIteration:
            print(f"[{self.service_name}] ⚠️ WARNING: sentence_tag_stream: word_stream is EMPTY (StopIteration on first next())")
            # Yield empty sentence tags so speaker knows stream ended
            yield "<sentence_start>"
            yield "<sentence_end>"
            return
        except Exception as e:
            print(f"[{self.service_name}] ⚠️ ERROR in sentence_tag_stream: {e}")
            import traceback
            traceback.print_exc()
            # Yield empty sentence tags so speaker knows stream ended
            yield "<sentence_start>"
            yield "<sentence_end>"
            return
        
        # Handle any remaining buffered word
        if buffered_word is not None:
            has_yielded_anything = True
            if not sentence_open:
                yield "<sentence_start>"
                sentence_open = True
            for token in yield_word(buffered_word):
                yield token
        
        # Close any remaining open sentence
        if sentence_open and sentence_buffer.strip():
            yield "<sentence_end>"
        elif not has_yielded_anything:
            # If no tokens were yielded at all, send an empty sentence to indicate completion
            # This ensures the speaker module knows the stream ended
            yield "<sentence_start>"
            yield "<sentence_end>"
    
    def register_health_check(self, app: Flask):
        """
        Register health check endpoint on Flask app
        
        Args:
            app: Flask application instance
        """
        @app.route("/health", methods=["GET"])
        def health_check():
            """Health check endpoint to verify models are loaded"""
            response = self.health_check_response()
            status_code = 200 if response.get("status") == "ok" else 500
            return jsonify(response), status_code

