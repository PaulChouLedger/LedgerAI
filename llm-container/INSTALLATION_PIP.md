# TensorRT-LLM Installation via pip

This approach installs TensorRT-LLM using pip in a base container, following the [official TensorRT-LLM installation guide](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).

## Advantages

1. **Python Compatibility**: Uses Python 3.12 (tested), avoiding Python 3.10 compatibility issues
2. **Official Tools**: Full access to TensorRT-LLM's conversion scripts without import errors
3. **More Control**: Complete control over the environment and dependencies
4. **Latest Features**: Direct access to the latest TensorRT-LLM features

## Base Image

The Dockerfile uses `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.1-py3` which provides:
- Jetson-compatible CUDA environment
- PyTorch pre-installed
- Python 3.x (compatible with TensorRT-LLM)

## Installation Process

1. **Prerequisites** (handled by base image):
   - CUDA Toolkit (via base image)
   - Python 3.x (via base image)

2. **TensorRT-LLM Installation**:
   ```bash
   pip3 install --upgrade pip setuptools
   pip3 install tensorrt_llm
   ```

3. **Additional Dependencies**:
   - transformers, torch, safetensors (for model conversion)
   - Application-specific packages (flask, etc.)

## Building Engines

With this setup, you can use TensorRT-LLM's official conversion scripts:

```bash
# Inside container
python3 /path/to/tensorrt_llm/models/llama/convert_checkpoint.py \
    --model_dir /models/Llama/Llama-3.2-1B-Instruct \
    --output_dir /models/tensorrt-llm/llama-3.2-1b-instruct/checkpoint \
    --dtype float16
```

The `build_tensorrt_engine.sh` script will now work properly with the official conversion tools.

## Verification

After building, verify TensorRT-LLM is installed correctly:

```python
from tensorrt_llm import LLM, SamplingParams

# Test with a small model
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, my name is"], sampling_params)
print(outputs[0].outputs[0].text)
```

## Known Limitations

- **Build Time**: Installing TensorRT-LLM via pip takes longer than using pre-built containers
- **Disk Space**: pip installation may require more disk space
- **Jetson Compatibility**: Ensure the base image matches your Jetson JetPack version (r36.4.0)

## Troubleshooting

If you encounter issues:

1. **CUDA Version**: Ensure `CUDA_HOME` is set correctly
2. **Python Version**: TensorRT-LLM tested on Python 3.12, but Python 3.10+ should work
3. **Memory**: Building engines requires significant RAM/VRAM

