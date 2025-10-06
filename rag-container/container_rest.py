#!/usr/bin/env python3
"""
Aura RAG Container REST API
Dedicated container for RAG functionality with FAISS and sentence transformers
"""

import os
import sys
import time
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Add current directory to path
sys.path.append('/app')

from rag import AuraRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global RAG instance
rag_system = None

def initialize_rag():
    """Initialize RAG system"""
    global rag_system
    try:
        print("[RAG Container] 🚀 Starting Aura RAG Container...")
        
        # Initialize RAG system
        rag_system = AuraRAG(
            index_path="data/embeddings/index.faiss",
            chunks_path="data/embeddings/doc_chunks.npy",
            model_name="all-MiniLM-L6-v2",
            relevance_threshold=0.3
        )
        
        print("[RAG Container] ✅ RAG system initialized successfully")
        return True
        
    except Exception as e:
        print(f"[RAG Container] ❌ Failed to initialize RAG: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'aura-rag',
        'timestamp': time.time()
    })

@app.route('/rag/stats', methods=['GET'])
def get_rag_stats():
    """Get RAG system statistics"""
    try:
        if rag_system is None:
            return jsonify({
                'status': 'not_ready',
                'health_score': 0,
                'chunks_loaded': 0,
                'index_loaded': False,
                'encoder_loaded': False
            })
        
        stats = rag_system.get_health_status()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'health_score': 0,
            'chunks_loaded': 0
        }), 500

@app.route('/rag/search', methods=['POST'])
def search_documents():
    """Search documents using RAG"""
    try:
        if rag_system is None:
            return jsonify({
                'error': 'RAG system not initialized',
                'results': []
            }), 503
        
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({
                'error': 'Query is required',
                'results': []
            }), 400
        
        # Perform search
        results = rag_system.search(query, top_k=top_k)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return jsonify({
            'error': str(e),
            'results': []
        }), 500

@app.route('/rag/encode', methods=['POST'])
def encode_text():
    """Encode text using sentence transformer"""
    try:
        if rag_system is None or rag_system.encoder is None:
            return jsonify({
                'error': 'Sentence transformer not available',
                'embedding': []
            }), 503
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'error': 'Text is required',
                'embedding': []
            }), 400
        
        # Encode text
        embedding = rag_system.encoder.encode([text])
        
        return jsonify({
            'text': text,
            'embedding': embedding[0].tolist(),
            'dimension': len(embedding[0])
        })
        
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        return jsonify({
            'error': str(e),
            'embedding': []
        }), 500

@app.route('/rag/reload', methods=['POST'])
def reload_rag():
    """Reload RAG system"""
    try:
        global rag_system
        rag_system = None
        
        # Reinitialize
        success = initialize_rag()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'RAG system reloaded'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to reload RAG system'
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading RAG: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("[RAG Container] 🚀 Starting Aura RAG Container...")
    
    # Initialize RAG system
    if initialize_rag():
        print("[RAG Container] ✅ RAG system ready")
        print("[RAG Container] 🌐 Starting REST API on port 11435...")
        
        # Start Flask app
        app.run(
            host='0.0.0.0',
            port=11435,
            debug=False,
            threaded=True
        )
    else:
        print("[RAG Container] ❌ Failed to initialize RAG system")
        sys.exit(1)
