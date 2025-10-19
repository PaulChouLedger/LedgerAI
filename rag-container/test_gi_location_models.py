#!/usr/bin/env python3
"""
Test script to benchmark embedding models against GI guideline LOCATION sections.
Tests semantic similarity for patient language vs medical guideline locations.
"""

import sys
import os
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# Add the current directory to Python path for imports
sys.path.append('/app')

def load_gi_guidelines() -> List[Dict]:
    """Load GI guidelines and extract LOCATION sections from OLDCARTS."""
    try:
        # Try to load from the medical guidelines directory
        guidelines_path = "/app/medical/guidelines"
        if not os.path.exists(guidelines_path):
            # Fallback to hardcoded GI guidelines for testing
            return get_hardcoded_gi_guidelines()
        
        gi_guidelines = []
        for filename in os.listdir(guidelines_path):
            if filename.startswith("GI_") and filename.endswith(".json"):
                filepath = os.path.join(guidelines_path, filename)
                with open(filepath, 'r') as f:
                    guideline = json.load(f)
                    if 'oldcarts' in guideline and 'location' in guideline['oldcarts']:
                        gi_guidelines.append({
                            'name': guideline.get('name', filename),
                            'location': guideline['oldcarts']['location']
                        })
        
        return gi_guidelines
    except Exception as e:
        print(f"⚠️  Could not load guidelines from file: {e}")
        return get_hardcoded_gi_guidelines()

def get_hardcoded_gi_guidelines() -> List[Dict]:
    """Fallback hardcoded GI guidelines for testing."""
    return [
        {
            'name': 'Acute Appendicitis',
            'location': 'Pain MIGRATES from periumbilical to right lower quadrant (RLQ) over 12-24 hours - highly specific migration pattern. Localizes to McBurney\'s point in RLQ.'
        },
        {
            'name': 'Acute Cholecystitis', 
            'location': 'Right upper quadrant (RUQ), precisely localized just below right rib cage. RADIATES TO RIGHT SHOULDER OR SCAPULA (phrenic nerve referred pain).'
        },
        {
            'name': 'Acute Pancreatitis',
            'location': 'Epigastric (upper mid-abdomen) and periumbilical. RADIATES STRAIGHT THROUGH TO THE BACK in \'boring\' pattern.'
        },
        {
            'name': 'Acute Gastroenteritis',
            'location': 'PERIUMBILICAL or DIFFUSE throughout abdomen. NOT localized to one quadrant. Cramping moves around.'
        },
        {
            'name': 'Biliary Colic',
            'location': 'Right upper quadrant (RUQ) or epigastric. May RADIATE TO RIGHT SHOULDER OR BACK.'
        },
        {
            'name': 'Small Bowel Obstruction',
            'location': 'Periumbilical and diffuse (not localized to one quadrant). Cramping throughout mid-abdomen.'
        },
        {
            'name': 'Acute Diverticulitis',
            'location': 'LEFT LOWER QUADRANT (LLQ) - key differentiator from appendicitis (RLQ). LOCALIZED and CONSTANT. Sometimes palpable tender mass.'
        },
        {
            'name': 'GERD',
            'location': 'RETROSTERNAL (behind breastbone) and EPIGASTRIC. BURNING rises from stomach toward throat. No radiation.'
        },
        {
            'name': 'Gastric Outlet Obstruction',
            'location': 'Epigastric (upper mid-abdomen), may radiate to back.'
        },
        {
            'name': 'Acute Gastritis',
            'location': 'EPIGASTRIC (upper mid-abdomen), diffuse. NO radiation.'
        },
        {
            'name': 'Acute Hepatitis',
            'location': 'RIGHT UPPER QUADRANT (liver area) discomfort. RUQ tenderness with hepatomegaly (enlarged liver).'
        },
        {
            'name': 'Inflammatory Bowel Disease Flare',
            'location': 'Diffuse cramping throughout abdomen, or RLQ if Crohn\'s (terminal ileum involved).'
        },
        {
            'name': 'Irritable Bowel Syndrome (IBS)',
            'location': 'LOWER ABDOMEN, diffuse. Cramping migrates, not fixed to one spot.'
        },
        {
            'name': 'Incarcerated Inguinal/Femoral Hernia',
            'location': 'Groin (inguinal or femoral region), may have lower abdominal pain if bowel involved.'
        },
        {
            'name': 'Mallory-Weiss Tear',
            'location': 'Epigastric (upper mid-abdomen) or lower chest discomfort.'
        },
        {
            'name': 'Acute Mesenteric Ischemia',
            'location': 'Diffuse, PERIUMBILICAL. Not localized to one quadrant.'
        },
        {
            'name': 'Peptic Ulcer Disease',
            'location': 'EPIGASTRIC, midline upper abdomen. No radiation typically.'
        },
        {
            'name': 'Perforated Viscus',
            'location': 'Initially EPIGASTRIC (perforated ulcer) or localized, then becomes DIFFUSE as peritonitis develops.'
        },
        {
            'name': 'Sigmoid Volvulus',
            'location': 'Left lower quadrant (LLQ) or diffuse lower abdomen.'
        },
        {
            'name': 'Kidney Stone',
            'location': 'Unilateral FLANK PAIN. RADIATES from flank→groin→testicle (males) or labia (females). Follows ureter path.'
        }
    ]

def get_patient_gi_prompts() -> List[str]:
    """Get 20 patient GI-related prompts for testing."""
    return [
        "left lower part of my abdomen towards my pelvis",
        "right upper side under my ribs",
        "middle of my stomach area",
        "all over my belly, it moves around",
        "right lower side near my hip bone",
        "upper middle part of my stomach",
        "left side of my belly",
        "behind my breastbone and upper stomach",
        "right shoulder and upper right belly",
        "lower part of my stomach, moves around",
        "right side under my ribs, goes to my back",
        "middle belly area around my belly button",
        "left lower belly, stays in one spot",
        "upper stomach behind my chest bone",
        "right upper belly, goes to my shoulder",
        "all over my abdomen, not in one place",
        "right lower belly near my hip",
        "upper middle stomach, goes through to my back",
        "left side of my lower belly",
        "right upper part under my ribs, goes to my back"
    ]

def compute_similarity(model: SentenceTransformer, text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using the model."""
    # Generate embeddings
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    
    # Ensure embeddings are normalized (unit vectors)
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Compute cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    cosine_similarity = dot_product / norm_product
    
    return float(cosine_similarity)

def test_model_performance(model_name: str, guidelines: List[Dict], patient_prompts: List[str]) -> Dict:
    """Test a single model's performance on all patient prompts vs guidelines."""
    print(f"\n🔄 Testing model: {model_name}")
    
    # Load model and measure loading time
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
        'total_inference_time': 0,
        'similarities': [],
        'best_matches': []
    }
    
    # Test each patient prompt against all guidelines
    total_start = time.time()
    
    for i, prompt in enumerate(patient_prompts):
        prompt_similarities = []
        
        for guideline in guidelines:
            start_inference = time.time()
            similarity = compute_similarity(model, prompt, guideline['location'])
            inference_time = time.time() - start_inference
            
            results['total_inference_time'] += inference_time
            
            prompt_similarities.append({
                'guideline': guideline['name'],
                'location': guideline['location'],
                'similarity': similarity,
                'inference_time': inference_time
            })
        
        # Sort by similarity and get best match
        prompt_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        best_match = prompt_similarities[0]
        
        results['similarities'].append({
            'prompt': prompt,
            'best_match': best_match,
            'all_similarities': prompt_similarities
        })
        
        results['best_matches'].append({
            'prompt': prompt,
            'best_guideline': best_match['guideline'],
            'similarity': best_match['similarity']
        })
        
        print(f"  📝 Prompt {i+1:2d}: '{prompt[:50]}...' → {best_match['guideline']} ({best_match['similarity']:.3f})")
    
    total_time = time.time() - total_start
    results['total_time'] = total_time
    
    print(f"⏱️  Total inference time: {results['total_inference_time']:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    return results

def analyze_results(all_results: List[Dict]) -> None:
    """Analyze and compare results across all models."""
    print("\n" + "="*100)
    print("📊 MODEL PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Summary table
    print(f"\n{'Model':<25} {'Load Time':<10} {'Inference Time':<15} {'Avg Similarity':<15}")
    print("-" * 70)
    
    for result in all_results:
        if result is None:
            continue
            
        avg_similarity = np.mean([match['similarity'] for match in result['best_matches']])
        print(f"{result['model_name']:<25} {result['load_time']:<10.2f} {result['total_inference_time']:<15.2f} {avg_similarity:<15.3f}")
    
    # Detailed analysis for each model
    for result in all_results:
        if result is None:
            continue
            
        print(f"\n🔍 DETAILED ANALYSIS: {result['model_name']}")
        print("="*100)
        
        # Show all matches in formatted table
        print(f"\n{'#':<3} {'User Prompt':<45} {'Guideline Location':<45} {'Score':<8}")
        print("-" * 100)
        
        for i, similarity_data in enumerate(result['similarities']):
            prompt = similarity_data['prompt']
            best_match = similarity_data['best_match']
            score = best_match['similarity']
            
            # Truncate long strings
            prompt_display = prompt[:42] + "..." if len(prompt) > 45 else prompt
            location_display = best_match['location'][:42] + "..." if len(best_match['location']) > 45 else best_match['location']
            
            print(f"{i+1:<3} {prompt_display:<45} {location_display:<45} {score:<8.3f}")
        
        # Show specific test case with top 3 matches
        left_lower_prompt = "left lower part of my abdomen towards my pelvis"
        left_lower_result = next((r for r in result['similarities'] if r['prompt'] == left_lower_prompt), None)
        if left_lower_result:
            print(f"\n🎯 SPECIFIC TEST CASE: '{left_lower_prompt}'")
            print("-" * 100)
            print(f"{'Rank':<5} {'Guideline':<30} {'Location':<50} {'Score':<8}")
            print("-" * 100)
            for i, match in enumerate(left_lower_result['all_similarities'][:5]):
                location_display = match['location'][:47] + "..." if len(match['location']) > 50 else match['location']
                print(f"{i+1:<5} {match['guideline']:<30} {location_display:<50} {match['similarity']:<8.3f}")
        
        print("\n" + "="*100)

def main():
    """Main test function."""
    print("🧪 GI LOCATION MODEL BENCHMARKING")
    print("="*50)
    
    # Load guidelines and patient prompts
    guidelines = load_gi_guidelines()
    patient_prompts = get_patient_gi_prompts()
    
    print(f"📋 Loaded {len(guidelines)} GI guidelines")
    print(f"📝 Testing with {len(patient_prompts)} patient prompts")
    
    # Models to test
    models = [
        'all-MiniLM-L6-v2',
        'all-MiniLM-L12-v2', 
        'paraphrase-MiniLM-L6-v2',
        'multi-qa-MiniLM-L6-cos-v1',
        'all-mpnet-base-v2',
        'all-distilroberta-v1'
    ]
    
    all_results = []
    
    # Test each model
    for model_name in models:
        result = test_model_performance(model_name, guidelines, patient_prompts)
        all_results.append(result)
    
    # Analyze results
    analyze_results(all_results)
    
    # Save detailed results to file
    output_file = "/app/gi_location_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n✅ Benchmarking complete!")

if __name__ == "__main__":
    main()
