#!/usr/bin/env python3
"""
Embedding Model Benchmarking Script

Compares different sentence transformer models for:
- Lateralization performance (left vs right)
- General similarity discrimination
- Processing speed
- Embedding quality metrics

Usage:
    python3 benchmark_embedding_models.py
    python3 benchmark_embedding_models.py --model all-mpnet-base-v2
    python3 benchmark_embedding_models.py --compare-all
"""

import argparse
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import sys

# Model configurations
MODELS = {
    'all-MiniLM-L6-v2': {
        'size': '22MB',
        'dimensions': 384,
        'description': 'Small, fast, general purpose'
    },
    'all-MiniLM-L12-v2': {
        'size': '33MB', 
        'dimensions': 384,
        'description': 'Larger MiniLM, better quality'
    },
    'paraphrase-MiniLM-L6-v2': {
        'size': '22MB',
        'dimensions': 384,
        'description': 'Optimized for paraphrase/similarity'
    },
    'multi-qa-MiniLM-L6-cos-v1': {
        'size': '22MB',
        'dimensions': 384,
        'description': 'QA-optimized with cosine similarity'
    },
    'all-mpnet-base-v2': {
        'size': '420MB',
        'dimensions': 768,
        'description': 'Large, high-quality, general purpose'
    },
    'all-distilroberta-v1': {
        'size': '290MB',
        'dimensions': 768,
        'description': 'Distilled RoBERTa, balanced performance'
    }
}

# Test cases for benchmarking
TEST_CASES = {
    'lateralization': [
        ('left', 'right'),
        ('left side', 'right side'),
        ('left arm', 'right arm'),
        ('left chest', 'right chest'),
        ('left lower quadrant', 'right lower quadrant'),
        ('left upper quadrant', 'right upper quadrant')
    ],
    'medical_anatomy': [
        ('chest', 'abdomen'),
        ('chest pain', 'abdominal pain'),
        ('retrosternal', 'epigastric'),
        ('flank', 'groin'),
        ('thoracic', 'lumbar'),
        ('cardiac', 'gastrointestinal')
    ],
    'general_similarity': [
        ('cat', 'dog'),
        ('pain', 'discomfort'),
        ('sharp', 'dull'),
        ('acute', 'chronic'),
        ('severe', 'mild'),
        ('constant', 'intermittent')
    ],
    'identical': [
        ('left', 'left'),
        ('chest', 'chest'),
        ('pain', 'pain'),
        ('acute', 'acute')
    ]
}

def benchmark_model(model_name: str) -> Dict:
    """Benchmark a single embedding model"""
    print(f"\n{'='*80}")
    print(f"🔬 BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    
    # Model info
    model_info = MODELS.get(model_name, {})
    print(f"📊 Model Info:")
    print(f"   Size: {model_info.get('size', 'Unknown')}")
    print(f"   Dimensions: {model_info.get('dimensions', 'Unknown')}")
    print(f"   Description: {model_info.get('description', 'Unknown')}")
    
    # Load model and measure time
    print(f"\n⏱️  Loading model...")
    start_time = time.time()
    try:
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    results = {
        'model_name': model_name,
        'load_time': load_time,
        'dimensions': model_info.get('dimensions', 0),
        'size': model_info.get('size', 'Unknown'),
        'test_results': {}
    }
    
    # Run all test categories
    for category, test_pairs in TEST_CASES.items():
        print(f"\n🧪 Testing {category.upper()}:")
        print(f"{'─'*60}")
        
        category_results = []
        total_inference_time = 0
        
        for pair1, pair2 in test_pairs:
            # Measure inference time
            start_time = time.time()
            emb1 = model.encode([pair1])[0]
            emb2 = model.encode([pair2])[0]
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Calculate similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Calculate embedding quality metrics
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            max_diff = np.max(np.abs(emb1 - emb2))
            is_identical = np.array_equal(emb1, emb2)
            
            result = {
                'pair1': pair1,
                'pair2': pair2,
                'similarity': similarity,
                'inference_time': inference_time,
                'norm1': norm1,
                'norm2': norm2,
                'max_diff': max_diff,
                'is_identical': is_identical
            }
            
            category_results.append(result)
            
            # Print result
            status = "🟢" if similarity > 0.8 else "🟡" if similarity > 0.5 else "🔴"
            print(f"{status} {pair1:20} vs {pair2:20}: {similarity:.3f} ({inference_time*1000:.1f}ms)")
        
        # Category summary
        similarities = [r['similarity'] for r in category_results]
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        avg_inference = total_inference_time / len(test_pairs)
        
        print(f"\n📈 {category.upper()} Summary:")
        print(f"   Average similarity: {avg_similarity:.3f} ± {std_similarity:.3f}")
        print(f"   Average inference time: {avg_inference*1000:.1f}ms")
        print(f"   Total inference time: {total_inference_time*1000:.1f}ms")
        
        results['test_results'][category] = {
            'results': category_results,
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'avg_inference_time': avg_inference,
            'total_inference_time': total_inference_time
        }
    
    # Overall model summary
    print(f"\n🎯 OVERALL MODEL SUMMARY:")
    print(f"{'─'*60}")
    
    lateralization_avg = results['test_results']['lateralization']['avg_similarity']
    medical_avg = results['test_results']['medical_anatomy']['avg_similarity']
    general_avg = results['test_results']['general_similarity']['avg_similarity']
    identical_avg = results['test_results']['identical']['avg_similarity']
    
    total_inference = sum(cat['total_inference_time'] for cat in results['test_results'].values())
    
    print(f"📊 Performance Metrics:")
    print(f"   Lateralization (left vs right): {lateralization_avg:.3f}")
    print(f"   Medical anatomy discrimination: {medical_avg:.3f}")
    print(f"   General similarity: {general_avg:.3f}")
    print(f"   Identical word matching: {identical_avg:.3f}")
    print(f"   Total inference time: {total_inference*1000:.1f}ms")
    print(f"   Average per comparison: {total_inference/len(TEST_CASES)/sum(len(pairs) for pairs in TEST_CASES.values())*1000:.1f}ms")
    
    # Quality assessment
    print(f"\n🏆 Quality Assessment:")
    if identical_avg > 0.99:
        print(f"   ✅ Identical matching: EXCELLENT ({identical_avg:.3f})")
    elif identical_avg > 0.95:
        print(f"   ✅ Identical matching: GOOD ({identical_avg:.3f})")
    else:
        print(f"   ❌ Identical matching: POOR ({identical_avg:.3f})")
    
    if lateralization_avg < 0.4:
        print(f"   ✅ Lateralization: EXCELLENT ({lateralization_avg:.3f})")
    elif lateralization_avg < 0.6:
        print(f"   🟡 Lateralization: GOOD ({lateralization_avg:.3f})")
    else:
        print(f"   ❌ Lateralization: POOR ({lateralization_avg:.3f})")
    
    if medical_avg < 0.5:
        print(f"   ✅ Medical discrimination: EXCELLENT ({medical_avg:.3f})")
    elif medical_avg < 0.7:
        print(f"   🟡 Medical discrimination: GOOD ({medical_avg:.3f})")
    else:
        print(f"   ❌ Medical discrimination: POOR ({medical_avg:.3f})")
    
    return results

def compare_all_models():
    """Compare all available models"""
    print(f"\n🚀 COMPREHENSIVE MODEL COMPARISON")
    print(f"{'='*80}")
    
    all_results = []
    
    for model_name in MODELS.keys():
        result = benchmark_model(model_name)
        if result:
            all_results.append(result)
    
    # Generate comparison table
    print(f"\n📊 COMPARISON TABLE:")
    print(f"{'='*120}")
    print(f"{'Model':<25} {'Size':<8} {'Dim':<4} {'Load(s)':<8} {'Lateral':<8} {'Medical':<8} {'General':<8} {'Identical':<9} {'Speed(ms)':<10}")
    print(f"{'─'*120}")
    
    for result in all_results:
        lateral = result['test_results']['lateralization']['avg_similarity']
        medical = result['test_results']['medical_anatomy']['avg_similarity']
        general = result['test_results']['general_similarity']['avg_similarity']
        identical = result['test_results']['identical']['avg_similarity']
        total_time = sum(cat['total_inference_time'] for cat in result['test_results'].values())
        avg_speed = total_time / sum(len(pairs) for pairs in TEST_CASES.values()) * 1000
        
        print(f"{result['model_name']:<25} {result['size']:<8} {result['dimensions']:<4} "
              f"{result['load_time']:<8.2f} {lateral:<8.3f} {medical:<8.3f} {general:<8.3f} "
              f"{identical:<9.3f} {avg_speed:<10.1f}")
    
    # Find best models
    print(f"\n🏆 BEST MODELS BY CATEGORY:")
    print(f"{'─'*60}")
    
    # Best lateralization (lowest similarity for left vs right)
    best_lateral = min(all_results, key=lambda x: x['test_results']['lateralization']['avg_similarity'])
    print(f"🥇 Best Lateralization: {best_lateral['model_name']} ({best_lateral['test_results']['lateralization']['avg_similarity']:.3f})")
    
    # Best medical discrimination (lowest similarity for medical terms)
    best_medical = min(all_results, key=lambda x: x['test_results']['medical_anatomy']['avg_similarity'])
    print(f"🥇 Best Medical Discrimination: {best_medical['model_name']} ({best_medical['test_results']['medical_anatomy']['avg_similarity']:.3f})")
    
    # Fastest model
    fastest = min(all_results, key=lambda x: sum(cat['total_inference_time'] for cat in x['test_results'].values()))
    total_time = sum(cat['total_inference_time'] for cat in fastest['test_results'].values())
    print(f"🥇 Fastest: {fastest['model_name']} ({total_time*1000:.1f}ms total)")
    
    # Best overall (balanced score)
    def overall_score(result):
        lateral = result['test_results']['lateralization']['avg_similarity']
        medical = result['test_results']['medical_anatomy']['avg_similarity']
        identical = result['test_results']['identical']['avg_similarity']
        # Lower lateral and medical is better, higher identical is better
        return (1 - lateral) + (1 - medical) + identical
    
    best_overall = max(all_results, key=overall_score)
    print(f"🥇 Best Overall: {best_overall['model_name']} (score: {overall_score(best_overall):.3f})")

def main():
    parser = argparse.ArgumentParser(description='Benchmark embedding models for medical diagnosis')
    parser.add_argument('--model', type=str, help='Specific model to benchmark')
    parser.add_argument('--compare-all', action='store_true', help='Compare all available models')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("📋 Available Models:")
        for model_name, info in MODELS.items():
            print(f"  {model_name:<25} - {info['size']:<8} - {info['description']}")
        return
    
    if args.compare_all:
        compare_all_models()
    elif args.model:
        if args.model not in MODELS:
            print(f"❌ Model '{args.model}' not found. Available models:")
            for model_name in MODELS.keys():
                print(f"  - {model_name}")
            sys.exit(1)
        benchmark_model(args.model)
    else:
        # Default: benchmark the current best model
        print("🔬 Running default benchmark on all-mpnet-base-v2...")
        benchmark_model('all-mpnet-base-v2')

if __name__ == "__main__":
    main()
