#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Dataset with CoT and Expected Output Formats
=======================================================

This script:
1. Generates base dataset (6000 examples)
2. Enhances with targeted examples (250 additional)
3. Converts to CoT format (100%)
4. Updates system prompts with explicit expected output formats
"""

import json
import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, args=None):
    """Run a Python script and return success status"""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def check_file_exists(filepath):
    """Check if file exists"""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

def main():
    print("=" * 80)
    print("REGENERATING DATASET WITH CoT AND EXPECTED OUTPUT FORMATS")
    print("=" * 80)
    
    # Step 1: Generate base dataset
    base_dataset = "rag_analysis_dataset_v2_base.json"
    generated_file = "rag_analysis_dataset_v2.json"  # generate_rag_dataset_v2.py hardcodes this
    
    if not check_file_exists(base_dataset):
        print("\n📝 Step 1: Generating base dataset (6000 examples)...")
        if not run_script("generate_rag_dataset_v2.py"):
            print("❌ Failed to generate base dataset")
            return 1
        
        # Rename generated file to base_dataset
        if check_file_exists(generated_file):
            import shutil
            shutil.move(generated_file, base_dataset)
            print(f"✅ Renamed {generated_file} to {base_dataset}")
    else:
        print(f"\n✅ Base dataset already exists: {base_dataset}")
    
    # Step 2: Enhance dataset with targeted examples
    enhanced_dataset = "rag_analysis_dataset_v2_enhanced.json"
    if not check_file_exists(enhanced_dataset):
        print("\n📝 Step 2: Enhancing dataset with targeted examples (250 additional)...")
        if not run_script("enhance_rag_dataset.py", [base_dataset, enhanced_dataset]):
            print("❌ Failed to enhance dataset")
            return 1
    else:
        print(f"\n✅ Enhanced dataset already exists: {enhanced_dataset}")
    
    # Step 3: Convert to CoT format
    cot_dataset = "rag_analysis_dataset_v2_cot.json"
    if not check_file_exists(cot_dataset):
        print("\n📝 Step 3: Converting to CoT format (100%)...")
        if not run_script("add_cot_to_dataset.py", [enhanced_dataset, cot_dataset, "true"]):
            print("❌ Failed to convert to CoT")
            return 1
    else:
        print(f"\n✅ CoT dataset already exists: {cot_dataset}")
    
    # Step 4: Update system prompts with expected output formats
    final_dataset = "rag_analysis_dataset_v2.json"
    print("\n📝 Step 4: Updating system prompts with explicit expected output formats...")
    if not run_script("update_cot_system_prompt_with_expected_outputs.py", [cot_dataset, final_dataset]):
        print("❌ Failed to update system prompts")
        return 1
    
    # Verify final dataset
    if check_file_exists(final_dataset):
        with open(final_dataset, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print("\n" + "=" * 80)
        print("✅ DATASET REGENERATION COMPLETE!")
        print("=" * 80)
        print(f"Final dataset: {final_dataset}")
        print(f"Total examples: {len(dataset)}")
        print(f"File size: {os.path.getsize(final_dataset) / (1024*1024):.1f} MB")
        
        # Check if system prompt has expected output formats
        if len(dataset) > 0:
            system_msg = None
            for msg in dataset[0].get('messages', []):
                if msg.get('role') == 'system':
                    system_msg = msg.get('content', '')
                    break
            
            if system_msg and 'EXPECTED OUTPUT FORMAT FOR STEP' in system_msg:
                print("\n✅ System prompt includes explicit expected output formats!")
            else:
                print("\n⚠️  Warning: System prompt may not include expected output formats")
        
        return 0
    else:
        print("\n❌ Final dataset not found!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
