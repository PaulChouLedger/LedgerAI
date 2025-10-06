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
    """Initialize communication service"""
    try:
        print(f"[{SERVICE_NAME}] 🚀 Starting Aura Communication Service...")
        print(f"[{SERVICE_NAME}] ✅ Communication service ready")
        return True
        
    except Exception as e:
        print(f"[{SERVICE_NAME}] ❌ Failed to initialize service: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': SERVICE_NAME,
        'timestamp': time.time()
    })

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
    """Initialize RAG system"""
    try:
        print(f"[{SERVICE_NAME}] 🔍 RAG init requested")
        # Since this is a communication container, we just return success
        # The actual RAG functionality would be handled by the LLM container
        return jsonify({
            'status': 'success',
            'message': 'RAG communication service ready'
        })
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    """Get RAG system statistics"""
    try:
        # Since this is a communication container, return basic stats
        return jsonify({
            'status': 'ready',
            'service': SERVICE_NAME,
            'threshold': RAG_THRESHOLD,
            'top_k': TOP_K,
            'chunks_loaded': 0,  # Would be populated by actual RAG service
            'index_loaded': False
        })
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/search', methods=['POST'])
def rag_search():
    """Search using RAG system - proxy to LLM container for efficiency"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', TOP_K)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Proxy to LLM container for actual RAG processing (more efficient)
        try:
            response = requests.post(
                f"{LLM_SERVICE_URL}/rag/search", 
                json=data, 
                timeout=10
            )
            return jsonify(response.json()), response.status_code
        except requests.exceptions.RequestException:
            # Fallback: return empty results if LLM container not available
            return jsonify({
                'query': query,
                'results': [],
                'count': 0,
                'message': 'LLM container not available'
            })
        
    except Exception as e:
        logger.error(f"Error in RAG search: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize service
    if initialize_service():
        print(f"[{SERVICE_NAME}] 🚀 Starting server on port 11435...")
        app.run(host='0.0.0.0', port=11435, debug=False)
    else:
        print(f"[{SERVICE_NAME}] ❌ Failed to initialize service")
        sys.exit(1)