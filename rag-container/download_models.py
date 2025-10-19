#!/usr/bin/env python3
"""
Download all embedding models for offline use
"""

from sentence_transformers import SentenceTransformer
import time

models = [
    'all-MiniLM-L6-v2',
    'all-MiniLM-L12-v2', 
    'paraphrase-MiniLM-L6-v2',
    'multi-qa-MiniLM-L6-cos-v1',
    'all-mpnet-base-v2',
    'all-distilroberta-v1'
]

print('📥 Pre-downloading all embedding models...')
for i, model_name in enumerate(models, 1):
    print(f'[{i}/{len(models)}] Downloading {model_name}...')
    start_time = time.time()
    try:
        SentenceTransformer(model_name)
        load_time = time.time() - start_time
        print(f'✅ {model_name} downloaded in {load_time:.1f}s')
    except Exception as e:
        print(f'❌ Failed to download {model_name}: {e}')

print('🎉 All models pre-downloaded successfully!')
