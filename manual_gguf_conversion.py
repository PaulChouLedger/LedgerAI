#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual GGUF Conversion Script
Captures actual error output from llama.cpp conversion
"""

import os
import subprocess
import sys

def convert_to_gguf_manual():
    """Manually convert HuggingFace model to GGUF with error capture"""
    
    print("=" * 80)
    print("Manual GGUF Conversion for RAG Analysis Model")
    print("=" * 80)
    
    # Use outputs_rag_analysis (has complete model + tokenizer)
    model_dir = "outputs_rag_analysis"
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        print("   Make sure training completed.")
        return False
    
    print(f"✅ Found model directory: {model_dir}")
    
    # Check if config.json exists and is valid
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        import json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config.get("model_type") == "qwen2":
                print(f"✅ Valid Qwen2 config found")
            else:
                print(f"⚠️  Config model_type is: {config.get('model_type')}, expected 'qwen2'")
                print("   This might cause conversion issues.")
        except Exception as e:
            print(f"⚠️  Error reading config.json: {e}")
    else:
        print(f"⚠️  config.json not found in {model_dir}/")
    
    # Check if tokenizer files exist
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    missing_files = [f for f in tokenizer_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing_files:
        print(f"⚠️  Missing tokenizer files: {missing_files}")
        print("   Conversion may fail at tokenizer step.")
    
    # Check for llama.cpp
    llama_cpp_dir = "llama.cpp"
    if not os.path.exists(llama_cpp_dir):
        print(f"📦 Cloning llama.cpp...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                check=True,
                capture_output=True
            )
            print("✅ Cloned llama.cpp")
        except Exception as e:
            print(f"❌ Failed to clone llama.cpp: {e}")
            return False
    else:
        print(f"✅ Found llama.cpp directory")
    
    # Check for converter script
    converter_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
    if not os.path.exists(converter_script):
        # Try unsloth version
        converter_script = os.path.join(llama_cpp_dir, "unsloth_convert_hf_to_gguf.py")
        if not os.path.exists(converter_script):
            print(f"❌ Converter script not found")
            print(f"   Tried: llama.cpp/convert-hf-to-gguf.py")
            print(f"   Tried: llama.cpp/unsloth_convert_hf_to_gguf.py")
            return False
    
    print(f"✅ Found converter script: {converter_script}")
    
    # Output file
    output_file = "Qwen2.5-1.5B-Instruct-rag-analysis.gguf"
    
    print(f"\n🔄 Converting to GGUF format...")
    print(f"   Input: {model_dir}/")
    print(f"   Output: {output_file}")
    print(f"   This may take 10-15 minutes...")
    print()
    
    # Run conversion with full output capture
    try:
        # Use f16 format (more compatible than bf16)
        # Specify model type explicitly to avoid RAG config detection
        cmd = [
            sys.executable,
            converter_script,
            "--outfile", output_file,
            "--outtype", "f16",  # Use f16 instead of bf16 for better compatibility
            model_dir
        ]
        
        print("   Note: If you see 'RAG config' error, the config.json might be corrupted.")
        print("   Try: !cat outputs_rag_analysis/config.json | head -20")
        
        print(f"Running: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on error, we want to see the output
        )
        
        # Print all output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Conversion successful!")
            print(f"   Output file: {output_file}")
            
            # Now quantize to q4_k_m
            print(f"\n🔄 Quantizing to q4_k_m...")
            quantized_file = output_file.replace(".gguf", "-q4_k_m.gguf")
            
            quantize_cmd = [
                os.path.join(llama_cpp_dir, "llama-quantize"),
                output_file,
                quantized_file,
                "Q4_K_M"
            ]
            
            print(f"Running: {' '.join(quantize_cmd)}")
            print()
            
            quantize_result = subprocess.run(
                quantize_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if quantize_result.stdout:
                print("STDOUT:")
                print(quantize_result.stdout)
            if quantize_result.stderr:
                print("STDERR:")
                print(quantize_result.stderr)
            
            if quantize_result.returncode == 0:
                print(f"\n✅ Quantization successful!")
                print(f"   Quantized file: {quantized_file}")
                return True
            else:
                print(f"\n⚠️  Quantization failed, but bf16 file is available: {output_file}")
                return True
        else:
            print(f"\n❌ Conversion failed with exit code: {result.returncode}")
            print("\nTroubleshooting:")
            print("1. Check disk space: !df -h")
            print("2. Check if model files are complete in:", model_dir)
            print("3. Try converting from outputs_rag_analysis/ instead")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def try_alternative_conversion():
    """Try converting using base model + adapter merge"""
    print("\n" + "=" * 80)
    print("Trying Alternative: Load and merge model, then convert")
    print("=" * 80)
    
    try:
        from unsloth import FastLanguageModel
        import shutil
        
        print("📦 Loading model with Unsloth...")
        # Load from outputs_rag_analysis (has LoRA adapters)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="outputs_rag_analysis/",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        
        print("✅ Model loaded")
        print("🔄 Merging LoRA adapters...")
        # Check if model has PEFT adapters and merge them
        try:
            from peft import PeftModel
            # Check if model is wrapped in PEFT
            if hasattr(model, 'peft_config') or isinstance(model, PeftModel):
                print("   Model has LoRA adapters, merging...")
                model = model.merge_and_unload()
                print("✅ LoRA adapters merged")
            else:
                # Check if base_model has PEFT
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
                    print("   Model has LoRA adapters in base_model, merging...")
                    model = model.base_model.merge_and_unload()
                    print("✅ LoRA adapters merged")
                else:
                    print("⚠️  Model doesn't appear to have LoRA adapters, may already be merged")
        except Exception as e:
            print(f"⚠️  Could not merge adapters: {e}")
            print("   Proceeding with model as-is (may already be merged)")
        
        print("💾 Saving merged model with correct config...")
        merged_dir = "merged_model_for_gguf"
        
        # Remove directory if it exists
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        
        # Save model and tokenizer
        model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        
        # Verify and fix config.json
        import json
        config_path = os.path.join(merged_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config.get("model_type") != "qwen2":
                print(f"⚠️  Fixing config.json: model_type was {config.get('model_type')}, setting to 'qwen2'")
                config["model_type"] = "qwen2"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print("✅ Config fixed")
        else:
            print("⚠️  config.json not found after save, creating one...")
            # Get config from model
            if hasattr(model, 'config'):
                config = model.config.to_dict()
                config["model_type"] = "qwen2"  # Ensure correct type
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print("✅ Created config.json")
        
        print(f"✅ Merged model saved to {merged_dir}/")
        
        # Now convert
        llama_cpp_dir = "llama.cpp"
        if not os.path.exists(llama_cpp_dir):
            print("❌ llama.cpp not found")
            return False
        
        converter_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
        if not os.path.exists(converter_script):
            converter_script = os.path.join(llama_cpp_dir, "unsloth_convert_hf_to_gguf.py")
        
        if not os.path.exists(converter_script):
            print("❌ Converter script not found")
            return False
        
        output_file = "Qwen2.5-1.5B-Instruct-rag-analysis.gguf"
        
        print(f"\n🔄 Converting from {merged_dir}/ to {output_file}...")
        
        cmd = [
            sys.executable,
            converter_script,
            "--outfile", output_file,
            "--outtype", "f16",
            merged_dir
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Conversion successful!")
            print(f"   Output file: {output_file}")
            return True
        else:
            print(f"\n❌ Conversion failed with exit code: {result.returncode}")
            return False
            
    except ImportError:
        print("❌ Unsloth not available. Cannot merge adapters.")
        return False
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("This script will help diagnose and fix GGUF conversion issues.")
    print("Run this in Colab after training completes.\n")
    
    # Try main conversion first
    success = convert_to_gguf_manual()
    
    # If that fails, try alternative
    if not success:
        print("\n" + "=" * 80)
        print("Main conversion failed. Trying alternative method...")
        print("=" * 80)
        try_alternative_conversion()

