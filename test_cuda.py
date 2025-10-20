#!/usr/bin/env python3
"""
Test CUDA availability in RAG container
"""
import sys

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available - checking why...")
        print(f"PyTorch built with CUDA: {torch.cuda.is_available()}")
        print(f"CUDA runtime version: {torch.version.cuda}")
except ImportError as e:
    print(f"PyTorch not available: {e}")

try:
    import faiss
    print(f"FAISS version: {faiss.__version__}")
    print(f"FAISS GPU support: {faiss.get_num_gpus()}")
except ImportError as e:
    print(f"FAISS not available: {e}")

try:
    import sys
    sys.path.append('/opt/faiss_lite')
    from faiss_lite import cudaKNN, cudaL2Norm, cudaAllocMapped
    print("faiss_lite CUDA functions available")
except ImportError as e:
    print(f"faiss_lite not available: {e}")

print("CUDA test complete")
