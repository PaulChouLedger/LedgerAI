#!/usr/bin/env python3
"""
Debug script to understand what TensorRT-LLM expects from a checkpoint directory.
This script will attempt to load the checkpoint and show what paths it's looking for.
"""

import os
import sys

checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "/models/tensorrt-llm/llama-3.2-1b-instruct/checkpoint"

print(f"Debugging TensorRT-LLM checkpoint expectations...")
print(f"Checkpoint directory: {checkpoint_dir}")
print()

# Check what files exist
print("Files in checkpoint root:")
for item in sorted(os.listdir(checkpoint_dir)):
    item_path = os.path.join(checkpoint_dir, item)
    if os.path.isfile(item_path):
        size = os.path.getsize(item_path)
        print(f"  📄 {item} ({size:,} bytes)")
    elif os.path.isdir(item_path):
        print(f"  📁 {item}/")
        for subitem in sorted(os.listdir(item_path)):
            subitem_path = os.path.join(item_path, subitem)
            if os.path.isfile(subitem_path):
                size = os.path.getsize(subitem_path)
                print(f"    📄 {subitem} ({size:,} bytes)")

print()
print("Attempting to import TensorRT-LLM and check from_checkpoint behavior...")
try:
    import tensorrt_llm
    from tensorrt_llm.models import LLaMAForCausalLM
    import json
    
    # Read config to understand model structure
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Config loaded: {len(config)} fields")
        print(f"   Model type: {config.get('model_type', 'unknown')}")
        print(f"   Hidden size: {config.get('hidden_size', 'unknown')}")
    else:
        print("❌ config.json not found")
    
    # Try to understand what from_checkpoint expects
    print()
    print("Attempting to call from_checkpoint (this may fail but will show what path it's looking for)...")
    try:
        model = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)
        print("✅ from_checkpoint succeeded!")
    except AssertionError as e:
        print(f"❌ AssertionError: {e}")
        print("This shows the assertion that failed - TensorRT-LLM couldn't find weights at the expected path")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ Could not import TensorRT-LLM: {e}")
    print("This script needs to run inside the TensorRT-LLM container")

