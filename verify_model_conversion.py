#!/usr/bin/env python3
"""
Verify which model was actually converted to GGUF
Checks if the source model is fine-tuned and if the GGUF matches
"""

import os
import json
import glob

def check_model_status(model_path):
    """Check if a model path contains fine-tuned weights"""
    print(f"\n{'='*80}")
    print(f"Checking: {model_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(model_path):
        print(f"❌ Path does not exist: {model_path}")
        return False
    
    # Check for config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        print(f"✅ config.json found")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"   Model type: {config.get('model_type', 'unknown')}")
            print(f"   Architecture: {config.get('architectures', ['unknown'])[0]}")
        except Exception as e:
            print(f"   ⚠️  Could not read config: {e}")
    else:
        print(f"❌ config.json NOT found - may be LoRA adapters only")
    
    # Check for training artifacts
    training_state = os.path.join(model_path, "training_state.json")
    training_args = os.path.join(model_path, "training_args.json")
    
    if os.path.exists(training_state):
        print(f"✅ training_state.json found - model was fine-tuned!")
        try:
            with open(training_state, 'r') as f:
                state = json.load(f)
            print(f"   Training step: {state.get('global_step', 'unknown')}")
            print(f"   Epoch: {state.get('epoch', 'unknown')}")
        except:
            pass
    elif os.path.exists(training_args):
        print(f"✅ training_args.json found - model was fine-tuned!")
    else:
        print(f"⚠️  No training artifacts found - may be base model")
    
    # Check for model files
    safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
    bin_files = glob.glob(os.path.join(model_path, "*.bin"))
    
    if safetensors:
        total_size = sum(os.path.getsize(f) for f in safetensors) / (1024**3)
        print(f"✅ Found {len(safetensors)} safetensors file(s) ({total_size:.2f} GB)")
    elif bin_files:
        total_size = sum(os.path.getsize(f) for f in bin_files) / (1024**3)
        print(f"✅ Found {len(bin_files)} .bin file(s) ({total_size:.2f} GB)")
    else:
        print(f"⚠️  No model weight files found")
    
    # Check for adapter files (LoRA)
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"⚠️  adapter_config.json found - this is LoRA adapters, not merged model")
        return False
    
    return True

def main():
    print("=" * 80)
    print("Model Conversion Verification")
    print("=" * 80)
    
    # Check source model
    source_model = "outputs_rag_cot"
    is_fine_tuned = check_model_status(source_model)
    
    # Check merged model location (if exists)
    merged_in_gguf = "gguf_model_rag_cot"
    if os.path.exists(os.path.join(merged_in_gguf, "config.json")):
        print(f"\n⚠️  Found merged model in {merged_in_gguf} directory")
        print(f"   This might be what was converted instead of {source_model}")
        check_model_status(merged_in_gguf)
    
    # Check GGUF output
    print(f"\n{'='*80}")
    print("GGUF Model Check")
    print(f"{'='*80}")
    gguf_dir = "gguf_model_rag_cot"
    if os.path.exists(gguf_dir):
        gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
        if gguf_files:
            for gguf_file in gguf_files:
                size_mb = os.path.getsize(gguf_file) / (1024**2)
                print(f"✅ Found: {os.path.basename(gguf_file)} ({size_mb:.2f} MB)")
        else:
            print(f"❌ No GGUF files found in {gguf_dir}")
    else:
        print(f"❌ GGUF directory does not exist: {gguf_dir}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("Recommendations")
    print(f"{'='*80}")
    
    if not is_fine_tuned:
        print("❌ Source model does not appear to be fine-tuned!")
        print("   Action: Re-run training to ensure fine-tuning completes")
    else:
        print("✅ Source model appears to be fine-tuned")
        if os.path.exists(os.path.join(merged_in_gguf, "config.json")):
            print("⚠️  WARNING: A merged model exists in gguf_model_rag_cot")
            print("   This might be from a previous conversion attempt")
            print("   Action: Delete the merged model and re-run conversion:")
            print(f"      rm -rf {merged_in_gguf}/*.safetensors {merged_in_gguf}/config.json")
            print(f"      Then re-run convert_to_gguf_colab.py")
        else:
            print("✅ No conflicting merged model found")
            print("   Action: Re-run conversion script to convert fine-tuned model")

if __name__ == "__main__":
    main()
