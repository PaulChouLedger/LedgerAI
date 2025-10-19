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

@app.route('/rag/names', methods=['GET'])
def get_person_names():
    """
    Extract all person names from RAG database for name correction
    Returns list of names found in document chunks
    """
    try:
        rag = get_rag()
        
        # Check if chunks is empty (handle numpy arrays properly)
        if rag.chunks is None or len(rag.chunks) == 0:
            return jsonify({'names': []}), 200
        
        import re
        names = set()
        
        # Extract capitalized names (2+ words) from all chunks
        for chunk in rag.chunks:
            # Find patterns like "Rafael Cabello", "Bob Carella", etc.
            chunk_names = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)', chunk)
            names.update(chunk_names)
        
        # Convert to sorted list
        names_list = sorted(list(names))
        
        logger.info(f"[RAG] 📋 Extracted {len(names_list)} person names from database")
        
        return jsonify({
            'names': names_list,
            'count': len(names_list)
        })
        
    except Exception as e:
        logger.error(f"Error extracting names: {e}")
        return jsonify({'error': str(e), 'names': []}), 500

@app.route('/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check - blocks until RAG is fully initialized
    Use this before starting services that depend on RAG
    """
    try:
        rag = get_rag()
        stats = rag.get_stats()
        
        # Check if RAG is actually ready
        is_ready = (
            stats.get('status') == 'ready' and
            stats.get('index_size', 0) > 0 and
            stats.get('chunks_loaded', 0) > 0
        )
        
        if is_ready:
            return jsonify({
                'ready': True,
                'service': SERVICE_NAME,
                'rag_components': {
                    'index_size': stats.get('index_size', 0),
                    'chunks_loaded': stats.get('chunks_loaded', 0),
                    'model_name': stats.get('model_name', 'unknown')
                },
                'message': 'RAG system fully initialized and ready',
                'timestamp': time.time()
            })
        else:
            return jsonify({
                'ready': False,
                'service': SERVICE_NAME,
                'rag_status': stats.get('status', 'unknown'),
                'message': 'RAG system not yet ready',
                'timestamp': time.time()
            }), 503  # Service Unavailable
            
    except Exception as e:
        return jsonify({
            'ready': False,
            'service': SERVICE_NAME,
            'error': str(e),
            'message': 'RAG system initialization failed',
            'timestamp': time.time()
        }), 503

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

def extract_names_from_text(text: str) -> list:
    """Extract person names from text (simple heuristic: capitalized consecutive words)"""
    import re
    # Find sequences of 2-3 capitalized words (likely names)
    # This pattern looks for full names like "Rafael Cabello", "Bob Carella", etc.
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
    names = re.findall(name_pattern, text)
    
    # Filter out common false positives
    stopwords = {'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Who', 'How', 'Why'}
    filtered_names = [name for name in names if not any(word in stopwords for word in name.split())]
    
    # Return unique names
    return list(set(filtered_names))

@app.route('/rag/guideline/<guideline_name>', methods=['GET'])
def get_guideline_chunks(guideline_name):
    """Get ALL chunks from a specific medical guideline"""
    try:
        rag = get_rag()
        
        # Get all chunks from this guideline
        results = rag.get_all_chunks_from_guideline(guideline_name)
        
        if not results:
            return jsonify({
                'error': f'No chunks found for guideline: {guideline_name}',
                'results': []
            }), 404
        
        # Format for consistency with /rag/search
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result.get('text', ''),
                'score': result.get('score', 1.0),
                'metadata': result.get('metadata', {})
            })
        
        return jsonify({
            'results': formatted_results,
            'guideline_name': guideline_name,
            'chunk_count': len(formatted_results)
        })
        
    except Exception as e:
        print(f"[API] ❌ Error getting guideline chunks: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/rag/search', methods=['POST'])
def rag_search():
    """Search using RAG system"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', TOP_K)
        disable_keyword_filter = data.get('disable_keyword_filter', False)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Use actual RAG functionality
        rag = get_rag()
        results = rag.search(query, k, disable_keyword_filter=disable_keyword_filter)
        
        # Format results for LLM container compatibility
        formatted_results = []
        all_extracted_names = []
        
        for result in results:
            chunk_text = result.get('chunk', '')
            formatted_results.append({
                'text': chunk_text,
                'score': result.get('score', 0.0),
                'distance': result.get('distance', 0.0),
                'rank': result.get('rank', 0)
            })
            
            # Extract names from this chunk
            names_in_chunk = extract_names_from_text(chunk_text)
            all_extracted_names.extend(names_in_chunk)
        
        # Get unique names
        unique_names = list(set(all_extracted_names))
        
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
            'used_rag': len(formatted_results) > 0,
            'extracted_names': unique_names  # NEW: Names found in the chunks
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

@app.route('/rag/ingest', methods=['POST'])
def rag_ingest():
    """Trigger auto-ingest to process files from data/input (text extraction only)"""
    try:
        from ingest import AutoIngest
        
        # Get RAG instance
        rag = get_rag()
        if not rag:
            return jsonify({'error': 'RAG not initialized'}), 500
        
        # Create ingest instance (reuses RAG encoder)
        ingest = AutoIngest(rag)
        
        # Scan and process files (text extraction only, no embedding generation)
        result = ingest.scan_and_process()
        
        return jsonify({
            'status': 'success',
            'processed': result['processed'],
            'skipped': result['skipped'],
            'errors': result['errors'],
            'message': 'Text extracted - host will generate embeddings'
        })
        
    except Exception as e:
        logger.error(f"Error in auto-ingest: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/rag/reload', methods=['POST'])
def rag_reload():
    """Reload RAG index after host generates new embeddings"""
    try:
        import faiss
        import numpy as np
        from pathlib import Path
        
        # Get RAG instance
        rag = get_rag()
        if not rag:
            return jsonify({'error': 'RAG not initialized'}), 500
        
        # Reload index and chunks
        embeddings_dir = Path("data/embeddings")
        
        print("[RAG] 🔄 Reloading index after host embedding generation...")
        rag.index = faiss.read_index(str(embeddings_dir / "index.faiss"))
        rag.chunks = np.load(str(embeddings_dir / "doc_chunks.npy"), allow_pickle=True)
        
        # Reload CUDA vectors for faiss_lite
        print("[RAG] 🔧 Reloading CUDA vectors...")
        rag._prepare_cuda_data()
        
        print(f"[RAG] ✅ Reloaded: {rag.index.ntotal} vectors")
        
        return jsonify({
            'status': 'success',
            'total_chunks': rag.index.ntotal
        })
        
    except Exception as e:
        logger.error(f"Error reloading RAG: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/embed', methods=['POST'])
def embed_texts():
    """
    Generate embeddings for texts using RAG's embedding model
    Used by LLM container for semantic similarity scoring
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        # Get RAG instance (has embedding model)
        rag = get_rag()
        if not rag or not rag.encoder:
            return jsonify({'error': 'RAG encoder not initialized'}), 500
        
        # Generate embeddings
        import numpy as np
        embeddings = rag.encoder.encode(texts, convert_to_numpy=True)
        
        # Convert to list for JSON serialization
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        return jsonify({
            'embeddings': embeddings_list,
            'count': len(embeddings_list),
            'model': 'all-distilroberta-v1'
        })
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize service
    if initialize_service():
        print(f"[{SERVICE_NAME}] 🚀 Starting server on port 11435...")
        app.run(host='0.0.0.0', port=11435, debug=False)
    else:
        print(f"[{SERVICE_NAME}] ❌ Failed to initialize service")
        sys.exit(1)