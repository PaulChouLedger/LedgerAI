# === container_rest.py — Aura Generic LLM Container with TensorRT-LLM
# Dual-model system:
# - Qwen2.5-7B-Instruct (TensorRT-LLM) for RAG and complex questions
# - Llama-3.2-1B (TensorRT-LLM) for simple tasks and greetings
# Supports streaming and non-streaming chat endpoints

from flask import Flask, request, jsonify, stream_with_context, Response
from dotenv import load_dotenv
import os, re, json, string, threading, time
from datetime import datetime, timedelta
from glob import glob
import requests

# Import modular RAG client (supports both GPU and CPU modes)
from rag import get_rag_client

# Import fuzzy matcher for handling typos and transcription errors
from fuzzy_matcher import get_fuzzy_matcher

# TensorRT-LLM imports - REQUIRED, no fallback
try:
    from tensorrt_llm_wrapper import load_tensorrt_model
    TENSORRT_AVAILABLE = True
except ImportError as e:
    print(f"[TensorRT-LLM] ❌ TensorRT-LLM wrapper not available: {e}")
    print("[TensorRT-LLM] ❌ TensorRT-LLM is required - no fallback available")
    TENSORRT_AVAILABLE = False

app = Flask(__name__)
load_dotenv()

# === Thread Safety ===
llm_lock = threading.Lock()

# === Model Config ===
MODEL_PATH_COMPLEX = os.getenv("MODEL_PATH_COMPLEX", "/models/qwen2.5-7b-instruct-trt/engine")
MODEL_PATH_SIMPLE = os.getenv("MODEL_PATH_SIMPLE", "/models/llama-3.2-1b-instruct-trt/engine")
N_CTX_COMPLEX = int(os.getenv("N_CTX_COMPLEX", "4096"))
N_CTX_SIMPLE = int(os.getenv("N_CTX_SIMPLE", "2048"))
CHAT_FORMAT_COMPLEX = os.getenv("CHAT_FORMAT_COMPLEX", "mistral-instruct")
CHAT_FORMAT_SIMPLE = os.getenv("CHAT_FORMAT_SIMPLE", "llama-3")

# Models will be loaded in __main__ block
llm_complex = None  # Qwen2.5-7B for RAG and complex questions
llm_simple = None   # Llama-3.2-1B for simple tasks

# === Helper Functions ===
def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def is_complex_query(prompt: str, has_rag_context: bool = False) -> bool:
    """
    Determine if query should use complex model (Qwen2.5-7B)
    
    Uses complex model for:
    - RAG queries (has_rag_context)
    - Long prompts (>50 words)
    - Questions starting with "what", "why", "how", "explain"
    - Technical/academic language
    """
    if has_rag_context:
        return True
    
    prompt_lower = prompt.lower().strip()
    
    # Long prompts
    if len(prompt.split()) > 50:
        return True
    
    # Complex question words
    complex_triggers = ['what is', 'what are', 'why', 'how', 'explain', 'describe', 
                       'analyze', 'compare', 'discuss', 'evaluate']
    if any(prompt_lower.startswith(trigger) for trigger in complex_triggers):
        return True
    
    # Technical/academic terms
    technical_terms = ['theory', 'concept', 'principle', 'mechanism', 'analysis', 
                       'research', 'study', 'methodology', 'framework']
    if any(term in prompt_lower for term in technical_terms):
        return True
    
    # Default to simple model
    return False

def extract_llm_response_content(response) -> str:
    """Extract text content from LLM response"""
    if isinstance(response, dict):
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        elif 'content' in response:
            return response['content']
    return str(response)

def llm_chat(messages, max_tokens=512, temperature=None, stream=False, 
             use_complex=False, **kwargs):
    """
    Wrapper for LLM chat completion
    
    Args:
        messages: Chat messages
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        stream: Enable streaming
        use_complex: Use complex model (Qwen2.5-7B) instead of simple (Llama-3.2-1B)
    """
    if temperature is None:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Select model
    if use_complex and llm_complex is not None:
        llm = llm_complex
        model_name = "Qwen2.5-7B"
    elif llm_simple is not None:
        llm = llm_simple
        model_name = "Llama-3.2-1B"
    else:
        raise RuntimeError("No LLM model available")
    
    generation_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(os.getenv("LLM_TOP_P", "0.85"))),
        "top_k": kwargs.pop("top_k", int(os.getenv("LLM_TOP_K", "30"))),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(os.getenv("LLM_REPEAT_PENALTY", "1.15"))),
        "stream": stream,
        **kwargs
    }
    
    with llm_lock:
        try:
            if not TENSORRT_AVAILABLE:
                raise RuntimeError("TensorRT-LLM not available - models must be in TensorRT format")
            
            # TensorRT-LLM API
            response = llm.create_chat_completion(messages=messages, **generation_params)
            
            if stream:
                return response
            return extract_llm_response_content(response)
        except Exception as e:
            print(f"[LLM] ❌ Error in llm_chat ({model_name}): {e}")
            import traceback
            traceback.print_exc()
            if stream:
                return iter([])
            return ""

def build_rag_prompt(user_query: str, rag_context: str = None) -> list:
    """Build prompt with RAG context if available"""
    if rag_context:
        system_prompt = f"""You are a helpful assistant. Use the following information to answer the user's question accurately:

{rag_context}

Please provide a helpful and accurate response based on the information above."""
    else:
        system_prompt = "You are a helpful assistant. Provide accurate and helpful responses to user questions."
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

# === Health Check Endpoint ===
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify models are loaded"""
    try:
        complex_loaded = llm_complex is not None
        simple_loaded = llm_simple is not None
        
        return jsonify({
            "status": "ok",
            "service": "aura-llm-generic-tensorrt",
            "tensorrt_available": TENSORRT_AVAILABLE,
            "models": {
                "complex_loaded": complex_loaded,
                "complex_path": MODEL_PATH_COMPLEX if complex_loaded else None,
                "simple_loaded": simple_loaded,
                "simple_path": MODEL_PATH_SIMPLE if simple_loaded else None,
            },
            "rag": {
                "available": get_rag_client() is not None
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "service": "aura-llm-generic-tensorrt",
            "error": str(e)
        }), 500

# === CPU FAISS Auto-Ingestion Endpoints ===
@app.route('/cpu-faiss/ingest', methods=['POST'])
def cpu_faiss_ingest():
    """Trigger CPU FAISS auto-ingestion manually"""
    try:
        rag_client = get_rag_client()
        
        if not rag_client or not hasattr(rag_client, '_auto_ingest') or rag_client._auto_ingest is None:
            return jsonify({'error': 'CPU FAISS auto-ingestion not available'}), 500
        
        result = rag_client._auto_ingest.scan_and_process()
        
        return jsonify({
            'status': 'success',
            'processed': result['processed'],
            'skipped': result['skipped'],
            'errors': result['errors'],
            'total_chunks': result['total_chunks'],
            'message': 'CPU FAISS auto-ingestion completed'
        })
        
    except Exception as e:
        print(f"[RAG] ❌ Error in CPU FAISS auto-ingestion: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/cpu-faiss/status', methods=['GET'])
def cpu_faiss_status():
    """Get CPU FAISS status"""
    try:
        rag_client = get_rag_client()
        
        if not rag_client or not hasattr(rag_client, '_auto_ingest') or rag_client._auto_ingest is None:
            return jsonify({'error': 'CPU FAISS auto-ingestion not available'}), 500
        
        auto_ingest = rag_client._auto_ingest
        
        return jsonify({
            'status': 'active',
            'watching': auto_ingest.watching,
            'total_chunks': len(auto_ingest.chunks),
            'processed_files': len(auto_ingest.state.get('processed_files', {})),
            'input_directory': str(auto_ingest.input_dir),
            'cpu_embeddings_directory': str(auto_ingest.cpu_embeddings_dir),
            'model_name': auto_ingest.model_name
        })
        
    except Exception as e:
        print(f"[RAG] ❌ Error getting CPU FAISS status: {e}")
        return jsonify({'error': str(e)}), 500

# === Main Chat Endpoint ===
@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint with RAG support and dual-model routing
    Supports both streaming and non-streaming modes
    """
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "default").strip()
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 512)
    use_rag = data.get("use_rag", True)  # Enable RAG by default
    force_complex = data.get("force_complex", False)  # Force complex model
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    try:
        # Apply fuzzy correction to query for better matching
        fuzzy_matcher = get_fuzzy_matcher()
        corrected_query = fuzzy_matcher.fuzzy_correct(prompt)
        if corrected_query != prompt.lower():
            print(f"[Fuzzy] 🔧 Corrected query: '{prompt}' → '{corrected_query}'")
        
        # Get RAG context if available and enabled
        rag_context = None
        if use_rag:
            try:
                rag_client = get_rag_client()
                if rag_client:
                    # Use corrected query for RAG search
                    results = rag_client.search(query=corrected_query, k=3)
                    if results:
                        rag_context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(results)])
                        print(f"[RAG] ✅ Retrieved {len(results)} relevant chunks")
            except Exception as e:
                print(f"[RAG] ⚠️ RAG search failed: {e}")
        
        # Determine which model to use
        use_complex = force_complex or is_complex_query(prompt, rag_context is not None)
        model_name = "Qwen2.5-7B" if use_complex else "Llama-3.2-1B"
        print(f"[Chat] 🎯 Using {model_name} ({'complex' if use_complex else 'simple'})")
        
        # Build messages with RAG context
        messages = build_rag_prompt(prompt, rag_context)
        
        if stream:
            # Streaming response
            def generate():
                try:
                    stream_gen = llm_chat(messages, max_tokens=max_tokens, 
                                         stream=True, use_complex=use_complex)
                    for chunk in stream_gen:
                        if isinstance(chunk, dict):
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        elif isinstance(chunk, str):
                            yield chunk
                except Exception as e:
                    print(f"[Chat] ❌ Streaming error: {e}")
                    yield ""
            
            return Response(stream_with_context(generate()), mimetype='text/plain')
        else:
            # Non-streaming response
            response = llm_chat(messages, max_tokens=max_tokens, 
                              stream=False, use_complex=use_complex)
            return jsonify({
                "response": response,
                "session_id": session_id,
                "used_rag": rag_context is not None,
                "model_used": model_name
            })
            
    except Exception as e:
        print(f"[Chat] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Telegram Chat Endpoint ===
@app.route("/chat-tg", methods=["POST"])
def chat_tg():
    """Non-streaming chat endpoint for Telegram bot"""
    return chat()  # Reuse main chat endpoint

# === Server Startup ===
if __name__ == "__main__":
    # Validate TensorRT-LLM is available (REQUIRED, no fallback)
    if not TENSORRT_AVAILABLE:
        print("[TensorRT-LLM] ❌ TensorRT-LLM is REQUIRED but not available!")
        print("[TensorRT-LLM] ❌ Container cannot start without TensorRT-LLM")
        exit(1)
    
    print("[TensorRT-LLM] ✅ TensorRT-LLM available")
    print("[TensorRT-LLM] 🚀 Loading models...")
    
    # Load complex model (Qwen2.5-7B-Instruct)
    print(f"[TensorRT-LLM] 📦 Loading complex model: {MODEL_PATH_COMPLEX}")
    if os.path.exists(MODEL_PATH_COMPLEX):
        try:
            llm_complex = load_tensorrt_model(MODEL_PATH_COMPLEX, model_type="qwen")
            if llm_complex:
                print(f"[TensorRT-LLM] ✅ Complex model loaded: Qwen2.5-7B-Instruct")
            else:
                print(f"[TensorRT-LLM] ⚠️ Failed to load complex model")
        except Exception as e:
            print(f"[TensorRT-LLM] ⚠️ Failed to load complex model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[TensorRT-LLM] ⚠️ Complex model not found: {MODEL_PATH_COMPLEX}")
    
    # Load simple model (Llama-3.2-1B-Instruct)
    print(f"[TensorRT-LLM] 📦 Loading simple model: {MODEL_PATH_SIMPLE}")
    if os.path.exists(MODEL_PATH_SIMPLE):
        try:
            llm_simple = load_tensorrt_model(MODEL_PATH_SIMPLE, model_type="llama")
            if llm_simple:
                print(f"[TensorRT-LLM] ✅ Simple model loaded: Llama-3.2-1B-Instruct")
            else:
                print(f"[TensorRT-LLM] ⚠️ Failed to load simple model")
        except Exception as e:
            print(f"[TensorRT-LLM] ⚠️ Failed to load simple model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[TensorRT-LLM] ⚠️ Simple model not found: {MODEL_PATH_SIMPLE}")
    
    # Validate at least one model is loaded
    if llm_complex is None and llm_simple is None:
        print("[TensorRT-LLM] ❌ No models loaded!")
        print("[TensorRT-LLM] ❌ At least one TensorRT-LLM model must be available")
        exit(1)
    
    print("[Aura-LLM] 🚀 Starting Aura Generic LLM Container (TensorRT-LLM)")
    print("[Aura-LLM] 📋 Configuration:")
    print("  - TensorRT-LLM: Enabled (Required)")
    print("  - Complex Model: Qwen2.5-7B-Instruct (RAG, complex questions) - " + 
          ("✅ Loaded" if llm_complex else "❌ Not available"))
    print("  - Simple Model: Llama-3.2-1B-Instruct (greetings, simple tasks) - " + 
          ("✅ Loaded" if llm_simple else "❌ Not available"))
    print("  - RAG Integration: Document-based Q&A")
    print("  - Fuzzy Matching: Typo and transcription error correction")
    
    app.run(host='0.0.0.0', port=11435, debug=False)
