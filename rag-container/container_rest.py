#!/usr/bin/env python3
"""
Aura Communication Container REST API
Minimal container for communication between services
"""

import os
import sys
import time
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Import RAG functionality
from rag import get_rag, search_medical_info, smart_search_medical_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Service configuration
SERVICE_NAME = "aura-communication"
LLM_SERVICE_URL = "http://localhost:11434"  # LLM container
WHISPER_SERVICE_URL = "http://localhost:11436"  # Whisper container

# Configuration parameters
RAG_THRESHOLD = float(os.environ.get('RAG_THRESHOLD', '0.3'))
TOP_K = int(os.environ.get('TOP_K', '3'))

def initialize_service():
    """Initialize communication service and pre-load RAG model"""
    try:
        print(f"[{SERVICE_NAME}] 🚀 Starting Aura Communication Service...")
        
        # Pre-load RAG model during startup to avoid first-request delay
        print(f"[{SERVICE_NAME}] 🔧 Pre-loading RAG model...")
        try:
            rag = get_rag()
            print(f"[{SERVICE_NAME}] ✅ RAG model pre-loaded successfully")
            print(f"[{SERVICE_NAME}] 📊 Loaded {rag.get_stats().get('chunks_loaded', 0)} document chunks")
        except Exception as e:
            print(f"[{SERVICE_NAME}] ⚠️ RAG pre-load failed (will retry on first request): {e}")
            # Don't fail startup if RAG fails - it can initialize on first request
        
        print(f"[{SERVICE_NAME}] ✅ Communication service ready")
        return True
        
    except Exception as e:
        print(f"[{SERVICE_NAME}] ❌ Failed to initialize service: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with RAG status"""
    try:
        rag = get_rag()
        stats = rag.get_stats()
        return jsonify({
            'status': 'healthy' if stats.get('status') == 'ready' else 'degraded',
            'service': SERVICE_NAME,
            'rag_status': stats.get('status', 'unknown'),
            'rag_components': {
                'index_size': stats.get('index_size', 0),
                'chunks_loaded': stats.get('chunks_loaded', 0),
                'model_name': stats.get('model_name', 'unknown')
            },
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': SERVICE_NAME,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/services/status', methods=['GET'])
def get_services_status():
    """Check status of all services"""
    try:
        services_status = {}
        
        # Check LLM service
        try:
            response = requests.get(f"{LLM_SERVICE_URL}/health", timeout=5)
            services_status['llm'] = response.json() if response.status_code == 200 else {'status': 'error'}
        except:
            services_status['llm'] = {'status': 'unavailable'}
        
        # Check Whisper service
        try:
            response = requests.get(f"{WHISPER_SERVICE_URL}/health", timeout=5)
            services_status['whisper'] = response.json() if response.status_code == 200 else {'status': 'error'}
        except:
            services_status['whisper'] = {'status': 'unavailable'}
        
        return jsonify({
            'status': 'success',
            'services': services_status
        })
        
    except Exception as e:
        logger.error(f"Error checking services: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/proxy/llm', methods=['POST'])
def proxy_llm():
    """Proxy requests to LLM service"""
    try:
        data = request.get_json()
        response = requests.post(f"{LLM_SERVICE_URL}/generate", json=data, timeout=30)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Error proxying to LLM: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/proxy/whisper', methods=['POST'])
def proxy_whisper():
    """Proxy requests to Whisper service"""
    try:
        data = request.get_json()
        response = requests.post(f"{WHISPER_SERVICE_URL}/transcribe", json=data, timeout=30)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Error proxying to Whisper: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config/threshold', methods=['GET'])
def get_threshold():
    """Get current RAG threshold"""
    return jsonify({
        'threshold': RAG_THRESHOLD,
        'top_k': TOP_K
    })

@app.route('/config/threshold', methods=['POST'])
def set_threshold():
    """Set RAG threshold"""
    try:
        data = request.get_json()
        new_threshold = data.get('threshold')
        new_top_k = data.get('top_k')
        
        if new_threshold is not None:
            global RAG_THRESHOLD
            RAG_THRESHOLD = float(new_threshold)
            
        if new_top_k is not None:
            global TOP_K
            TOP_K = int(new_top_k)
            
        return jsonify({
            'threshold': RAG_THRESHOLD,
            'top_k': TOP_K,
            'message': 'Configuration updated'
        })
        
    except Exception as e:
        logger.error(f"Error setting threshold: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/init', methods=['POST'])
def rag_init():
    """Initialize RAG system with robust error handling"""
    try:
        print(f"[{SERVICE_NAME}] 🔍 RAG init requested")
        
        # Initialize RAG system with detailed error reporting
        try:
            rag = get_rag()
            stats = rag.get_stats()
            
            # Verify all components are loaded
            if stats.get('status') != 'ready':
                error_msg = f"RAG components not ready: {stats}"
                logger.error(error_msg)
                return jsonify({
                    'status': 'error',
                    'message': error_msg,
                    'stats': stats
                }), 500
            
            print(f"[{SERVICE_NAME}] ✅ RAG system initialized successfully")
            return jsonify({
                'status': 'success',
                'message': 'RAG system initialized',
                'stats': stats
            })
            
        except RuntimeError as e:
            error_msg = f"RAG initialization failed: {e}"
            logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'retry_recommended': True
            }), 500
            
    except Exception as e:
        error_msg = f"Unexpected error during RAG initialization: {e}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'retry_recommended': True
        }), 500

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    """Get RAG system statistics"""
    try:
        rag = get_rag()
        stats = rag.get_stats()
        stats.update({
            'service': SERVICE_NAME,
            'threshold': RAG_THRESHOLD,
            'top_k': TOP_K
        })
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/diagnose', methods=['GET'])
def rag_diagnose():
    """Diagnose RAG system issues"""
    try:
        rag = get_rag()
        diagnosis = rag.diagnose_index_issues()
        return jsonify(diagnosis)
    except Exception as e:
        logger.error(f"Error diagnosing RAG: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/search', methods=['POST'])
def rag_search():
    """Search using RAG system"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', TOP_K)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Use actual RAG functionality
        rag = get_rag()
        results = rag.search(query, k)
        
        # Format results for LLM container compatibility
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result.get('chunk', ''),
                'score': result.get('score', 0.0),
                'distance': result.get('distance', 0.0),
                'rank': result.get('rank', 0)
            })
        
        # Also provide a properly formatted prompt for LLM compatibility
        if formatted_results:
            context_parts = []
            for result in formatted_results:
                context_parts.append(result['text'])
            context = "\n\n".join(context_parts)
            formatted_prompt = f"Based on the following information:\n\n{context}\n\nPlease answer: {query}"
        else:
            formatted_prompt = query
        
        return jsonify({
            'query': query,
            'results': formatted_results,
            'count': len(formatted_results),
            'formatted_prompt': formatted_prompt,
            'used_rag': len(formatted_results) > 0
        })
        
    except Exception as e:
        logger.error(f"Error in RAG search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/augment', methods=['POST'])
def rag_augment():
    """Augment prompt with RAG context"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', TOP_K)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Use search_medical_info to get properly formatted prompt
        augmented_prompt = search_medical_info(query, k)
        
        return jsonify({
            'query': query,
            'augmented_prompt': augmented_prompt,
            'used_rag': augmented_prompt != query
        })
        
    except Exception as e:
        logger.error(f"Error in RAG augment: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize service
    if initialize_service():
        print(f"[{SERVICE_NAME}] 🚀 Starting server on port 11435...")
        app.run(host='0.0.0.0', port=11435, debug=False)
    else:
        print(f"[{SERVICE_NAME}] ❌ Failed to initialize service")
        sys.exit(1)