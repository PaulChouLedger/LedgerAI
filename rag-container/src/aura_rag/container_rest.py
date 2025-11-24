"""
Modern RAG API using cuDF, cuVS, and LlamaIndex.
Simple, clean, GPU-accelerated.
"""
from flask import Flask, request, jsonify
import os
import traceback
from typing import Dict, Any

from aura_rag.rag_engine import RAGEngine
from aura_rag.config import AuraRAGConfig

# === Flask App ===
app = Flask(__name__)

# Global RAG engine instance
rag_engine = None

def initialize_rag():
    """Initialize the RAG engine."""
    global rag_engine
    try:
        print("Initializing modern RAG engine...")
        
        # Load configuration
        config = AuraRAGConfig.from_env()
        print(f"Loaded configuration: {config.to_dict()}")
        
        # Initialize RAG engine with config
        rag_engine = RAGEngine(config)
        print("RAG engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG engine: {e}")
        traceback.print_exc()

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    if rag_engine is None:
        return jsonify({"status": "unhealthy", "error": "RAG engine not initialized"}), 500
    
    try:
        stats = rag_engine.get_stats()
        return jsonify({
            "status": "healthy", 
            "service": "aura-rag",
            "stats": stats
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/rag", methods=["POST"])
def rag_query():
    """Main RAG endpoint - takes user query, returns LLM response with context."""
    if rag_engine is None:
        return jsonify({"error": "RAG engine not initialized"}), 500
    
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    query = data["query"]
    top_k = data.get("top_k", 3)
    
    if not query.strip():
        return jsonify({"error": "Query cannot be empty"}), 400
    
    try:
        print(f"Processing RAG query: {query}")
        response = rag_engine.query(query, top_k=top_k)
        
        return jsonify({
            "query": query,
            "response": response,
            "top_k": top_k
        }), 200
        
    except Exception as e:
        print(f"Error processing RAG query: {e}")
        traceback.print_exc()
        return jsonify({"error": f"RAG processing failed: {str(e)}"}), 500

@app.route("/rebuild", methods=["POST"])
def rebuild_index():
    """Rebuild the vector index from documents."""
    if rag_engine is None:
        return jsonify({"error": "RAG engine not initialized"}), 500
    
    try:
        print("Rebuilding vector index...")
        rag_engine._build_index()
        rag_engine._setup_index()
        
        stats = rag_engine.get_stats()
        return jsonify({
            "message": "Index rebuilt successfully",
            "stats": stats
        }), 200
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Index rebuild failed: {str(e)}"}), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get RAG system statistics."""
    if rag_engine is None:
        return jsonify({"error": "RAG engine not initialized"}), 500
    
    try:
        stats = rag_engine.get_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/config", methods=["GET"])
def get_config():
    """Get current configuration."""
    if rag_engine is None:
        return jsonify({"error": "RAG engine not initialized"}), 500
    
    try:
        config_dict = rag_engine.config.to_dict()
        return jsonify(config_dict), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Initialize RAG engine on startup
    initialize_rag()
    
    # Start Flask app
    print("Starting Aura RAG service on port 5003...")
    app.run(host="0.0.0.0", port=5003, debug=False)
