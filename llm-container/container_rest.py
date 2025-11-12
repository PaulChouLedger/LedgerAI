# === container_rest.py — Aura Generic Conversational Container ===
# Provides general conversation with RAG-powered knowledge

from flask import Flask, request, jsonify, stream_with_context, Response
from llama_cpp import Llama
from dotenv import load_dotenv
import os, threading, atexit
import requests
from typing import List, Optional

# Conversation management for passive listening and keyword activation
from conversation_manager import ConversationMemoryIndex, ConversationOrchestrator

# Import modular RAG client (supports both GPU and CPU modes)
from rag import get_rag_client

app = Flask(__name__)
load_dotenv()

# === Thread Safety ===
llm_lock = threading.Lock()

# === Model Config ===
SIMPLE_MODEL_PATH = os.getenv("SIMPLE_MODEL_PATH", "/models/Llama-3.2-1B.Q4_K_M.gguf")
SIMPLE_N_CTX = int(os.getenv("SIMPLE_N_CTX", "2048"))
SIMPLE_CHAT_FORMAT = os.getenv("SIMPLE_CHAT_FORMAT", "llama-3")

llm_simple = None

# === Conversation Memory / Activation Config ===
def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


ACTIVATION_KEYWORDS = [
    kw.strip().lower()
    for kw in os.getenv("ACTIVATION_KEYWORDS", "hey aura").split(",")
    if kw.strip()
]
ACTIVATION_WINDOW_SECONDS = _get_env_float("ACTIVATION_WINDOW_SECONDS", 15.0)
ACTIVATION_COOLDOWN_SECONDS = _get_env_float("ACTIVATION_COOLDOWN_SECONDS", 3.0)
CONVERSATION_MEMORY_DIR = os.getenv(
    "CONVERSATION_MEMORY_DIR", "data/learning/conversation_memory"
)
CONVERSATION_MEMORY_PERSIST_EVERY = _get_env_int(
    "CONVERSATION_MEMORY_PERSIST_EVERY", 10
)
CONVERSATION_MEMORY_MAX_ENTRIES = _get_env_int(
    "CONVERSATION_MEMORY_MAX_ENTRIES", 5000
)
CONVERSATION_MEMORY_TOP_K = _get_env_int("CONVERSATION_MEMORY_TOP_K", 3)
CONVERSATION_MEMORY_MIN_SCORE = _get_env_float(
    "CONVERSATION_MEMORY_MIN_SCORE", 0.35
)

# === Health Check Endpoint ===
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify models are loaded"""
    try:
        simple_loaded = llm_simple is not None
        
        return jsonify({
            "status": "ok",
            "service": "aura-llm-generic",
            "models": {
                "simple_loaded": simple_loaded,
                "simple_path": SIMPLE_MODEL_PATH
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "service": "aura-llm-generic",
            "error": str(e)
        }), 500

def extract_llm_response_content(response) -> str:
    """Extract text content from LLM response"""
    if isinstance(response, dict):
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        elif 'content' in response:
            return response['content']
    return str(response)

def llm_chat_simple(messages, max_tokens=None, temperature=None, stream=False, **kwargs):
    """Wrapper for LLM chat completion"""
    if temperature is None:
        temperature = float(os.getenv("LLM_TEMPERATURE_SIMPLE"))
    
    # Handle max_tokens: use LLM_NUM_PREDICT as default if not provided
    if max_tokens is None:
        num_predict_env = os.getenv("LLM_NUM_PREDICT")
        if num_predict_env and num_predict_env.isdigit():
            max_tokens = int(num_predict_env)
        else:
            raise ValueError("LLM_NUM_PREDICT must be set in environment")
    
    generation_params = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": kwargs.pop("top_p", float(os.getenv("LLM_TOP_P"))),
        "top_k": kwargs.pop("top_k", int(os.getenv("LLM_TOP_K"))),
        "repeat_penalty": kwargs.pop("repeat_penalty", float(os.getenv("LLM_REPEAT_PENALTY"))),
        "stream": stream,
        **kwargs
    }
    
    stop_env = os.getenv("LLM_STOP", "").strip()
    if stop_env:
        generation_params["stop"] = [s for s in stop_env.split(",") if s]
    
    with llm_lock:
        try:
            response = llm_simple.create_chat_completion(**generation_params)
            if stream:
                return response
            return extract_llm_response_content(response)
        except Exception as e:
            print(f"[LLM] ❌ Error in llm_chat_simple: {e}")
            if stream:
                return iter([])
            return ""

# === Conversational Logic ===
def handle_conversation(
    prompt: str, session_id: str, memory_context: Optional[str] = None
):
    """Handle general conversation with optional RAG"""
    
    # Try RAG first for knowledge queries
    RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() == "true"
    rag_context = ""
    if RAG_ENABLED:
        try:
            rag_client = get_rag_client()
            results = rag_client.search(query=prompt, k=3)
            
            if results and len(results) > 0:
                rag_context = "\n".join(
                    [r.get("text", "") for r in results[:3] if r.get("text")]
                )
        except Exception as e:
            print(f"[Generic] ⚠️ RAG failed, using direct LLM: {e}")
    
    contextual_sections: List[str] = []
    if rag_context:
        contextual_sections.append(f"Knowledge context:\n{rag_context}")
    if memory_context:
        contextual_sections.append(f"Conversation memory:\n{memory_context}")
    combined_context = "\n\n".join(contextual_sections).strip()

    if combined_context:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the provided knowledge context "
                    "and conversation memory to answer the user's question.\n\n"
                    f"{combined_context}\n\n"
                    f"User question: {prompt}\n\n"
                    "If the provided context does not contain the answer, acknowledge "
                    "that and answer based on your general knowledge."
                ),
            }
        ]
        return llm_chat_simple(messages, max_tokens=300)

    # Fallback to direct LLM conversation without external context
    system_prompt = (
        "You are a helpful, friendly assistant. Keep responses concise and conversational."
    )
    if memory_context:
        system_prompt += f"\n\nConversation memory you can reference:\n{memory_context}"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return llm_chat_simple(messages, max_tokens=300)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        rag_client = get_rag_client()
        return rag_client.embed(texts)
    except Exception as exc:
        print(f"[Memory] ⚠️ Failed to generate embeddings: {exc}")
        return []


conversation_memory = ConversationMemoryIndex(
    storage_dir=CONVERSATION_MEMORY_DIR,
    persist_every=CONVERSATION_MEMORY_PERSIST_EVERY,
    max_entries=CONVERSATION_MEMORY_MAX_ENTRIES,
)

conversation_orchestrator = ConversationOrchestrator(
    memory_index=conversation_memory,
    embed_fn=_embed_texts,
    conversation_handler=handle_conversation,
    activation_keywords=ACTIVATION_KEYWORDS,
    activation_window=ACTIVATION_WINDOW_SECONDS,
    activation_cooldown=ACTIVATION_COOLDOWN_SECONDS,
    memory_top_k=CONVERSATION_MEMORY_TOP_K,
    memory_min_score=CONVERSATION_MEMORY_MIN_SCORE,
)

atexit.register(conversation_orchestrator.flush_memory)

# === Chat Endpoints ===
@app.route("/chat-tg", methods=["POST"])
def chat_tg():
    """Non-streaming chat endpoint for Telegram"""
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or "default").strip()
    
    if not prompt:
        return jsonify({"response": "Please provide a message."})
    
    print(f"[Generic] 💬 Session: {session_id}, Prompt: '{prompt[:50]}...'")
    
    try:
        response = handle_conversation(prompt, session_id)
        return jsonify({"response": response})
    except Exception as e:
        print(f"[Generic] ❌ Error: {e}")
        return jsonify({"response": "I apologize, I encountered an error processing your request."})

@app.route("/chat-tts", methods=["POST"])
def chat_tts():
    """Streaming chat endpoint for TTS/Voice"""
    data = request.get_json()
    prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("chat_id") or data.get("session_id") or None)
    
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400
    
    print(f"[Generic] 💬 Streaming Session: {session_id}, Prompt: '{prompt[:50]}...'")
    
    def generate_response():
        try:
            response = handle_conversation(prompt, session_id or "default")
            yield response
        except Exception as e:
            print(f"[Generic] ❌ Error: {e}")
            yield "I apologize, I encountered an error."
    
    return Response(stream_with_context(generate_response()), mimetype="text/plain")


@app.route("/voice/transcript", methods=["POST"])
def voice_transcript():
    """
    Passive transcript ingestion endpoint.
    
    Accepts continuous text from the SST pipeline, indexes it for long-term memory,
    and returns an LLM response only when an activation keyword window is open.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    session_id = str(
        data.get("session_id") or data.get("chat_id") or data.get("conversation_id") or "default"
    )
    is_final = bool(data.get("is_final", True))
    timestamp = data.get("timestamp")
    metadata = data.get("metadata") or {}

    if not text:
        return jsonify({"error": "Missing text"}), 400

    result = conversation_orchestrator.process_chunk(
        session_id=session_id,
        text=text,
        is_final=is_final,
        timestamp=timestamp,
        metadata=metadata,
    )

    return jsonify(
        {
            "status": "ok",
            "session_id": session_id,
            **result,
        }
    )

# === CPU FAISS Auto-Ingestion Endpoints ===
@app.route('/cpu-faiss/ingest', methods=['POST'])
def cpu_faiss_ingest():
    """Trigger CPU FAISS auto-ingestion manually"""
    try:
        # Get RAG client instance
        from rag import get_rag_client
        rag_client = get_rag_client()
        
        if not rag_client or not hasattr(rag_client, '_auto_ingest') or rag_client._auto_ingest is None:
            return jsonify({'error': 'CPU FAISS auto-ingestion not available'}), 500
        
        # Trigger manual scan
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
        print(f"[Generic] ❌ Error in CPU FAISS auto-ingestion: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/cpu-faiss/status', methods=['GET'])
def cpu_faiss_status():
    """Get CPU FAISS status"""
    try:
        # Get RAG client instance
        from rag import get_rag_client
        rag_client = get_rag_client()
        
        if not rag_client or not hasattr(rag_client, '_auto_ingest') or rag_client._auto_ingest is None:
            return jsonify({'error': 'CPU FAISS auto-ingestion not available'}), 500
        
        auto_ingest = rag_client._auto_ingest
        
        return jsonify({
            'status': 'active',
            'watching': auto_ingest.watching,
            'total_chunks': len(auto_ingest.chunks),
            'processed_files': len(auto_ingest.state.get('processed_files', {}))
        })
        
    except Exception as e:
        print(f"[Generic] ❌ Error getting CPU FAISS status: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("[Generic] 🚀 Starting Aura Generic LLM Container...")
    
    # Load model
    print(f"[Generic] 📦 Loading model: {SIMPLE_MODEL_PATH}")
    llm_simple = Llama(
        model_path=SIMPLE_MODEL_PATH,
        n_ctx=SIMPLE_N_CTX,
        chat_format=SIMPLE_CHAT_FORMAT,
        verbose=False
    )
    print(f"[Generic] ✅ Model loaded: {SIMPLE_MODEL_PATH}")
    
    print("[Generic] ✅ LLM Container ready!")
    print("[Generic] 🌐 Starting Flask server on 0.0.0.0:11436...")
    
    app.run(host="0.0.0.0", port=11436, threaded=True, debug=False)
