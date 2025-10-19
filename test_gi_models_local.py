#!/usr/bin/env python3
"""
Local test script to benchmark embedding models against GI guideline LOCATION sections.
Run this locally before testing in Docker container.
"""

import sys
import os
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

def get_hardcoded_gi_guidelines() -> List[Dict]:
    """Hardcoded GI guidelines for testing."""
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

def get_patient_gi_prompts() -> List[Dict]:
    """Get 40 patient GI-related prompts for testing with expected matches."""
    return [
        {
            'prompt': "left lower part of my abdomen towards my pelvis",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis', 'Acute Pancreatitis']
        },
        {
            'prompt': "right upper side under my ribs",
            'expected': ['Acute Cholecystitis', 'Biliary Colic', 'Acute Hepatitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "middle of my stomach area",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "all over my belly, it moves around",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "right lower side near my hip bone",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "upper middle part of my stomach",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "left side of my belly",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "behind my breastbone and upper stomach",
            'expected': ['GERD', 'Acute Pancreatitis', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "right shoulder and upper right belly",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "lower part of my stomach, moves around",
            'expected': ['Irritable Bowel Syndrome (IBS)', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Pancreatitis']
        },
        {
            'prompt': "right side under my ribs, goes to my back",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "middle belly area around my belly button",
            'expected': ['Acute Pancreatitis', 'Small Bowel Obstruction', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "left lower belly, stays in one spot",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "upper stomach behind my chest bone",
            'expected': ['GERD', 'Acute Pancreatitis', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "right upper belly, goes to my shoulder",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "all over my abdomen, not in one place",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "right lower belly near my hip",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "upper middle stomach, goes through to my back",
            'expected': ['Acute Pancreatitis', 'Gastric Outlet Obstruction'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "left side of my lower belly",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "right upper part under my ribs, goes to my back",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        # Additional 20 prompts for comprehensive testing
        {
            'prompt': "pain in my right side that goes to my shoulder",
            'expected': ['Acute Cholecystitis', 'Biliary Colic'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "left side pain that stays in one place",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "burning pain behind my chest bone",
            'expected': ['GERD', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "cramping pain all over my stomach",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "sharp pain in my right lower belly",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "dull ache in my upper middle abdomen",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain that moves around my belly button area",
            'expected': ['Acute Pancreatitis', 'Small Bowel Obstruction', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "constant pain in my left lower side",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "pain under my right ribs that radiates",
            'expected': ['Acute Cholecystitis', 'Biliary Colic', 'Acute Hepatitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "severe pain in my upper stomach",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain that goes from my belly to my back",
            'expected': ['Acute Pancreatitis', 'Gastric Outlet Obstruction'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "intermittent pain in my right upper abdomen",
            'expected': ['Biliary Colic', 'Acute Cholecystitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain in my groin area on the right",
            'expected': ['Incarcerated Inguinal/Femoral Hernia', 'Acute Appendicitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "burning sensation in my upper belly",
            'expected': ['GERD', 'Acute Gastritis', 'Peptic Ulcer Disease'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "pain that started around my belly button",
            'expected': ['Acute Pancreatitis', 'Small Bowel Obstruction', 'Acute Gastroenteritis'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "left side abdominal pain that's constant",
            'expected': ['Acute Diverticulitis', 'Sigmoid Volvulus'],
            'should_reject': ['Acute Appendicitis', 'Acute Cholecystitis']
        },
        {
            'prompt': "pain in my right flank area",
            'expected': ['Kidney Stone', 'Acute Hepatitis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "diffuse abdominal pain that moves",
            'expected': ['Acute Gastroenteritis', 'Irritable Bowel Syndrome (IBS)', 'Small Bowel Obstruction'],
            'should_reject': ['Acute Cholecystitis', 'Acute Diverticulitis']
        },
        {
            'prompt': "pain in my epigastric region",
            'expected': ['Acute Pancreatitis', 'Peptic Ulcer Disease', 'Acute Gastritis'],
            'should_reject': ['Acute Diverticulitis', 'Acute Appendicitis']
        },
        {
            'prompt': "right lower quadrant pain",
            'expected': ['Acute Appendicitis', 'Incarcerated Inguinal/Femoral Hernia'],
            'should_reject': ['Acute Diverticulitis', 'Acute Cholecystitis']
        }
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

def test_model_performance(model_name: str, guidelines: List[Dict], patient_prompts: List[Dict]) -> Dict:
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
        'best_matches': [],
        'accuracy_analysis': []
    }
    
    # Test each patient prompt against all guidelines
    total_start = time.time()
    
    for i, prompt_data in enumerate(patient_prompts):
        prompt = prompt_data['prompt']
        expected = prompt_data['expected']
        should_reject = prompt_data['should_reject']
        
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
        
        # Analyze accuracy
        is_correct = best_match['guideline'] in expected
        is_wrong = best_match['guideline'] in should_reject
        
        accuracy_status = "✅ CORRECT" if is_correct else ("❌ WRONG" if is_wrong else "⚠️  NEUTRAL")
        
        results['similarities'].append({
            'prompt': prompt,
            'best_match': best_match,
            'all_similarities': prompt_similarities,
            'expected': expected,
            'should_reject': should_reject,
            'is_correct': is_correct,
            'is_wrong': is_wrong
        })
        
        results['best_matches'].append({
            'prompt': prompt,
            'best_guideline': best_match['guideline'],
            'similarity': best_match['similarity'],
            'is_correct': is_correct,
            'is_wrong': is_wrong
        })
        
        results['accuracy_analysis'].append({
            'prompt': prompt,
            'best_match': best_match['guideline'],
            'similarity': best_match['similarity'],
            'expected': expected,
            'should_reject': should_reject,
            'is_correct': is_correct,
            'is_wrong': is_wrong,
            'status': accuracy_status
        })
        
        print(f"  📝 Prompt {i+1:2d}: '{prompt[:50]}...' → {best_match['guideline']} ({best_match['similarity']:.3f}) {accuracy_status}")
    
    total_time = time.time() - total_start
    results['total_time'] = total_time
    
    # Calculate accuracy metrics
    correct_count = sum(1 for match in results['best_matches'] if match['is_correct'])
    wrong_count = sum(1 for match in results['best_matches'] if match['is_wrong'])
    total_count = len(results['best_matches'])
    
    results['accuracy_metrics'] = {
        'correct': correct_count,
        'wrong': wrong_count,
        'neutral': total_count - correct_count - wrong_count,
        'total': total_count,
        'accuracy_percentage': (correct_count / total_count) * 100,
        'error_percentage': (wrong_count / total_count) * 100
    }
    
    print(f"⏱️  Total inference time: {results['total_inference_time']:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📊 Accuracy: {correct_count}/{total_count} ({results['accuracy_metrics']['accuracy_percentage']:.1f}%) correct, {wrong_count} wrong")
    
    return results

def analyze_results(all_results: List[Dict]) -> None:
    """Analyze and compare results across all models."""
    print("\n" + "="*100)
    print("📊 MODEL PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Summary table
    print(f"\n{'Model':<25} {'Load Time':<10} {'Inference Time':<15} {'Avg Similarity':<15} {'Accuracy':<10} {'Errors':<8}")
    print("-" * 90)
    
    for result in all_results:
        if result is None:
            continue
            
        avg_similarity = np.mean([match['similarity'] for match in result['best_matches']])
        accuracy = result['accuracy_metrics']['accuracy_percentage']
        errors = result['accuracy_metrics']['wrong']
        print(f"{result['model_name']:<25} {result['load_time']:<10.2f} {result['total_inference_time']:<15.2f} {avg_similarity:<15.3f} {accuracy:<10.1f}% {errors:<8}")
    
    # Detailed analysis for each model
    for result in all_results:
        if result is None:
            continue
            
        print(f"\n🔍 DETAILED ANALYSIS: {result['model_name']}")
        print("="*100)
        
        # Show accuracy summary
        metrics = result['accuracy_metrics']
        print(f"\n📊 ACCURACY SUMMARY:")
        print(f"   ✅ Correct: {metrics['correct']}/{metrics['total']} ({metrics['accuracy_percentage']:.1f}%)")
        print(f"   ❌ Wrong: {metrics['wrong']}/{metrics['total']} ({metrics['error_percentage']:.1f}%)")
        print(f"   ⚠️  Neutral: {metrics['neutral']}/{metrics['total']} ({100-metrics['accuracy_percentage']-metrics['error_percentage']:.1f}%)")
        
        # Show all matches in formatted table with accuracy status
        print(f"\n{'#':<3} {'User Prompt':<40} {'Best Match':<25} {'Score':<8} {'Status':<10}")
        print("-" * 100)
        
        for i, accuracy_data in enumerate(result['accuracy_analysis']):
            prompt = accuracy_data['prompt']
            best_match = accuracy_data['best_match']
            score = accuracy_data['similarity']
            status = accuracy_data['status']
            
            # Truncate long strings
            prompt_display = prompt[:37] + "..." if len(prompt) > 40 else prompt
            match_display = best_match[:22] + "..." if len(best_match) > 25 else best_match
            
            print(f"{i+1:<3} {prompt_display:<40} {match_display:<25} {score:<8.3f} {status:<10}")
        
        # Show wrong matches in detail
        wrong_matches = [data for data in result['accuracy_analysis'] if data['is_wrong']]
        if wrong_matches:
            print(f"\n❌ WRONG MATCHES ({len(wrong_matches)}):")
            print("-" * 100)
            for match in wrong_matches:
                print(f"   '{match['prompt'][:50]}...'")
                print(f"   Expected: {', '.join(match['expected'])}")
                print(f"   Got: {match['best_match']} (score: {match['similarity']:.3f})")
                print(f"   Should reject: {', '.join(match['should_reject'])}")
                print()
        
        # Show specific test case with top 5 matches
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
    print("🧪 GI LOCATION MODEL BENCHMARKING (LOCAL)")
    print("="*50)
    
    # Load guidelines and patient prompts
    guidelines = get_hardcoded_gi_guidelines()
    patient_prompts = get_patient_gi_prompts()
    
    print(f"📋 Loaded {len(guidelines)} GI guidelines")
    print(f"📝 Testing with {len(patient_prompts)} patient prompts")
    print(f"🎯 Each prompt has expected matches and should-reject conditions for accuracy analysis")
    print(f"🔬 Comprehensive testing with 40 diverse patient location descriptions")
    
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
    output_file = "gi_location_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n✅ Benchmarking complete!")

if __name__ == "__main__":
    main()
