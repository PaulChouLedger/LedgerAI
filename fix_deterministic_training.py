#!/usr/bin/env python3
"""
Fix training script to ensure deterministic/repeatable results by setting all random seeds.
"""

import re

# Read training script
with open('train_rag_cot_colab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if seeds are already set
has_torch_seed = 'torch.manual_seed' in content
has_numpy_seed = 'np.random.seed' in content or 'numpy.random.seed' in content
has_random_seed = 'random.seed' in content
has_cuda_deterministic = 'torch.backends.cudnn.deterministic' in content

print("=" * 80)
print("ANALYZING TRAINING SCRIPT FOR DETERMINISM")
print("=" * 80)
print()

print("Current seed setup:")
print(f"  torch.manual_seed: {'✅' if has_torch_seed else '❌'}")
print(f"  numpy.random.seed: {'✅' if has_numpy_seed else '❌'}")
print(f"  random.seed: {'✅' if has_random_seed else '❌'}")
print(f"  CUDA deterministic: {'✅' if has_cuda_deterministic else '❌'}")
print()

if has_torch_seed and has_numpy_seed and has_random_seed and has_cuda_deterministic:
    print("✅ All seeds already set - script is deterministic!")
    exit(0)

# Find the import section
import_match = re.search(r'(^import .+\n)+', content, re.MULTILINE)
if not import_match:
    print("❌ Could not find import section")
    exit(1)

import_end = import_match.end()

# Find where to insert seed setup (after imports, before GPU check)
gpu_check_match = re.search(r'# =+.*\n# GPU Check', content)
if not gpu_check_match:
    print("❌ Could not find GPU Check section")
    exit(1)

insert_pos = gpu_check_match.start()

# Create seed setup code
seed_code = """
# ============================================================================
# Set Random Seeds for Deterministic Training
# ============================================================================

import random
import numpy as np

SEED = 3407  # Match seed used in TrainingArguments

# Set Python random seed
random.seed(SEED)

# Set NumPy random seed
np.random.seed(SEED)

# Set PyTorch random seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enable deterministic CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for hash randomization
import os
os.environ['PYTHONHASHSEED'] = str(SEED)

print("=" * 80)
print("Random Seeds Set for Deterministic Training")
print("=" * 80)
print(f"✅ Python random seed: {SEED}")
print(f"✅ NumPy random seed: {SEED}")
print(f"✅ PyTorch random seed: {SEED}")
print(f"✅ CUDA deterministic: True")
print(f"✅ CUDA benchmark: False")
print(f"✅ PYTHONHASHSEED: {SEED}")
print()

"""

# Insert seed code
new_content = content[:insert_pos] + seed_code + content[insert_pos:]

# Write updated script
with open('train_rag_cot_colab.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Training script updated with deterministic seed setup!")
print()
print("Added seed initialization:")
print("  - random.seed(3407)")
print("  - np.random.seed(3407)")
print("  - torch.manual_seed(3407)")
print("  - torch.cuda.manual_seed_all(3407)")
print("  - torch.backends.cudnn.deterministic = True")
print("  - torch.backends.cudnn.benchmark = False")
print("  - os.environ['PYTHONHASHSEED'] = '3407'")
print()
print("This ensures:")
print("  ✅ Training results are reproducible")
print("  ✅ Same dataset + same settings = same model weights")
print("  ✅ CUDA operations are deterministic")
print()
print("Note: Training will be slightly slower with CUDA deterministic=True")
print("      but results will be fully repeatable.")
