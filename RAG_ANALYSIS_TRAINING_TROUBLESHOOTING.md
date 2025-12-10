# RAG Analysis Training - Troubleshooting Guide

## GGUF Conversion Failed

If you see an error like:
```
RuntimeError: Unsloth: GGUF conversion failed: ...
```

**Don't worry!** The model training was successful. The GGUF conversion is optional.

### ✅ What You Have

The model is saved in **HuggingFace format** at `outputs_rag_analysis/` which is fully usable:

- ✅ Model weights: `model.safetensors` or `pytorch_model.bin`
- ✅ Tokenizer: `tokenizer.json`, `tokenizer_config.json`
- ✅ Config: `config.json`
- ✅ LoRA adapters: `adapter_model.safetensors`

### Using the HuggingFace Model

The HuggingFace format model works perfectly with:

1. **Unsloth** (recommended for Colab):
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="outputs_rag_analysis/",
       max_seq_length=2048,
       dtype=None,
       load_in_4bit=False,
   )
   ```

2. **Transformers**:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   tokenizer = AutoTokenizer.from_pretrained("outputs_rag_analysis/")
   model = AutoModelForCausalLM.from_pretrained("outputs_rag_analysis/")
   ```

3. **Test Script**: The `test_rag_analysis_colab.py` script automatically detects and uses the HuggingFace format.

### Manual GGUF Conversion (Optional)

If you need GGUF format for deployment, you can convert manually:

#### Option 1: Using llama.cpp (Recommended)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# Convert HuggingFace to GGUF
python convert-hf-to-gguf.py outputs_rag_analysis/ --outfile rag-analysis-model.gguf --outtype f16

# Quantize (optional, for smaller file size)
./llama-quantize rag-analysis-model.gguf rag-analysis-model-q4_k_m.gguf q4_k_m
```

#### Option 2: Using Unsloth (Retry)

Sometimes the conversion fails due to temporary issues. You can retry:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs_rag_analysis/",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

# Retry GGUF conversion
model.save_pretrained_gguf(
    "gguf_model_rag_analysis",
    tokenizer,
    quantization_method="q4_k_m"
)
```

### Common GGUF Conversion Issues

1. **Out of Memory**: GGUF conversion requires significant RAM. Try:
   - Using a smaller quantization (q4_k_m instead of q8_0)
   - Converting on a machine with more RAM
   - Using CPU mode instead of GPU

2. **llama.cpp Build Issues**: If llama.cpp fails to build:
   - Ensure you have required build tools: `sudo apt-get install build-essential cmake`
   - Check llama.cpp GitHub issues for your specific error

3. **Model Too Large**: If the model is too large:
   - Use lower precision quantization
   - Split the model into multiple files

### Testing Without GGUF

You can test the model immediately using the HuggingFace format:

```python
python test_rag_analysis_colab.py
```

The test script will automatically:
- Detect the HuggingFace model in `outputs_rag_analysis/`
- Load it using Unsloth or Transformers
- Run all test cases

### Downloading from Colab

If running in Colab, download the model:

```python
from google.colab import files
import shutil

# Create zip of HuggingFace model
shutil.make_archive('rag_analysis_model', 'zip', 'outputs_rag_analysis')
files.download('rag_analysis_model.zip')
```

### Summary

- ✅ **Training successful**: Model is ready to use
- ✅ **HuggingFace format**: Fully functional, works with all frameworks
- ⚠️ **GGUF conversion**: Optional, can be done later if needed
- ✅ **Testing**: Can test immediately with HuggingFace format

The HuggingFace format is actually preferred for most use cases and is easier to work with for testing and further fine-tuning.

