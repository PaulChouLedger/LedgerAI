# Fine-Tuning Medical Bot with Unsloth on Jetson

This guide walks you through fine-tuning your medical chatbot using Unsloth on a Jetson device.

## Prerequisites

- Jetson device (AGX Orin, Xavier, etc.) with JetPack installed
- CUDA support (cu126 packages available, but compatible with other CUDA versions)
- Python 3.10 or 3.12
- At least 16GB RAM (32GB recommended)
- Medical SFT dataset (`medical_sft_dataset.json` in `LLM_tuning/` directory)

## Installation

### 1. Install Unsloth from Jetson AI Lab PyPI

The Jetson AI Lab provides optimized packages for Jetson devices. **The wheel file should handle all dependencies automatically:**

**Important:** If you have existing unsloth installations, uninstall them first for a clean install:

```bash
# Clean up existing installations (optional but recommended)
pip3 uninstall -y unsloth unsloth_zoo
```

Then install from the cu126 index:

```bash
# Install Unsloth from cu126 index
# Note: Use package name with version, not wheel filename
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth==2025.7.9

# Install unsloth_zoo explicitly (required dependency, may not auto-install)
pip3 install unsloth_zoo
```

**Alternative:** If you want to download and install the wheel file locally:

```bash
# Download the wheel file first
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/edc/0ac127024b8f9/unsloth-2025.7.9-py3-none-any.whl

# Install the local wheel file - dependencies will be installed automatically
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    ./unsloth-2025.7.9-py3-none-any.whl
```

**Note:** The wheel is available in the `cu126` index. If you need a different CUDA version, check the corresponding index (cu128, cu129, etc.) at [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/jp6/).

**Note:** You may need to install `unsloth_zoo` separately as it's not always installed automatically:
```bash
pip3 install unsloth_zoo
```

If you encounter issues, see the troubleshooting section.

### 2. Install Additional Dependencies

**CRITICAL: Install dependencies in this exact order to avoid conflicts:**

```bash
# 1. Install PyTorch first
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install transformers with compatible version FIRST (before unsloth)
# This is critical - transformers 4.46+ removed top_k_top_p_filtering which unsloth needs
pip3 install --force-reinstall --no-cache-dir "transformers>=4.40.0,<4.46.0"

# 3. Install trl with compatible version (before unsloth)
# CRITICAL: Use constraint to prevent transformers from being upgraded
echo "transformers<4.46.0" > /tmp/transformers_constraint.txt
pip3 install --force-reinstall --no-cache-dir --constraint /tmp/transformers_constraint.txt "trl>=0.7.0,<0.8.0"
rm -f /tmp/transformers_constraint.txt
# Verify transformers wasn't upgraded
python3 -c "import transformers; v=transformers.__version__; assert v < '4.46.0', f'Transformers upgraded to {v}!'; print(f'✅ Transformers: {v}')"

# 4. Install unsloth (with constraint to prevent transformers upgrade)
echo "transformers<4.46.0" > /tmp/transformers_constraint.txt
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --constraint /tmp/transformers_constraint.txt unsloth==2025.7.9
rm -f /tmp/transformers_constraint.txt
# Verify transformers wasn't upgraded
python3 -c "import transformers; v=transformers.__version__; assert v < '4.46.0', f'Transformers upgraded to {v}!'; print(f'✅ Transformers: {v}')"

# 5. Install unsloth_zoo if needed (with --no-deps to avoid pulling incompatible transformers)
pip3 install --no-deps unsloth_zoo || echo "unsloth_zoo may already be included in unsloth"

# 6. Install other dependencies
pip3 install datasets peft accelerate bitsandbytes scipy sentencepiece "fsspec>=2023.1.0,<=2025.9.0"
```

**Important Version Constraints:**
- `transformers>=4.40.0,<4.46.0`: Versions 4.46+ removed `top_k_top_p_filtering` which unsloth requires. **Must install BEFORE unsloth to prevent dependency conflicts.**
- `trl>=0.7.0,<0.8.0`: Newer trl (0.24+) requires transformers>=4.56.1, which is incompatible. Install compatible version BEFORE unsloth.
- `unsloth_zoo`: Newer versions (2025.11.3+) require transformers>=4.51.3. Use `--no-deps` when installing to avoid pulling incompatible transformers.

**Note:** The installation order is critical. Installing transformers and trl first prevents unsloth from pulling incompatible versions. The setup script handles this automatically.

### 3. Clean Virtual Environment (RECOMMENDED)

**This is the easiest and most reliable approach** - it avoids all dependency conflicts by starting with a fresh environment:

```bash
cd LLM_tuning

# Run the clean environment setup script
bash setup_clean_env.sh
```

This will:
1. Create a new virtual environment (`unsloth-env`)
2. Install PyTorch
3. Install Unsloth from the wheel (which handles its own dependencies)
4. Install additional required packages
5. Verify everything works

**To use the environment:**
```bash
cd LLM_tuning
source unsloth-env/bin/activate

# Now you can run fine-tuning
python3 finetune_unsloth.py --dataset_path ./medical_sft_dataset.json
```

**To deactivate when done:**
```bash
deactivate
```

**Advantages:**
- ✅ No dependency conflicts
- ✅ Isolated from system Python packages
- ✅ Easy to recreate if something goes wrong
- ✅ Unsloth wheel handles dependencies automatically

### 4. Quick Setup Script (System-wide Installation)

Or use the provided setup script for system-wide installation:

```bash
cd LLM_tuning
chmod +x setup_unsloth.sh
./setup_unsloth.sh
```

### 4. Verify Installation

```bash
python3 -c "from unsloth import FastLanguageModel; print('✅ Unsloth installed successfully')"
```

## Dataset Format

Your dataset (`medical_sft_dataset.json` in `LLM_tuning/` directory) should be in the format:

```json
[
  {
    "messages": [
      { "role": "user", "content": "I have chest pain" },
      { "role": "assistant", "content": "I understand..." }
    ]
  }
]
```

The script automatically converts this to the format Unsloth expects.

## Model Download

**No manual download required!** The model (`unsloth/Llama-3.2-1B-Instruct-bnb-4bit`) will be automatically downloaded from Hugging Face on first use. 

The model will be cached in `~/.cache/huggingface/hub/` (typically ~1-2GB for the 4-bit quantized version).

**Optional: Pre-download the model** (useful if you have slow internet or want to test connectivity):

```bash
python3 -c "
from unsloth import FastLanguageModel
print('Downloading model...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-1B-Instruct-bnb-4bit',
    max_seq_length=2048,
    load_in_4bit=True,
)
print('✅ Model downloaded and cached')
"
```

## Fine-Tuning

### Basic Usage

From the `LLM_tuning` directory:

```bash
cd LLM_tuning
python3 finetune_unsloth.py \
    --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --dataset_path ./medical_sft_dataset.json \
    --output_dir ../models/finetuned_medical \
    --num_epochs 3 \
    --batch_size 2
```

### Advanced Options

```bash
cd LLM_tuning
python3 finetune_unsloth.py \
    --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --dataset_path ./medical_sft_dataset.json \
    --output_dir ../models/finetuned_medical \
    --num_epochs 5 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_seq_length 2048 \
    --warmup_steps 10 \
    --save_steps 50 \
    --logging_steps 5
```

### Memory-Constrained Jetson Devices

For devices with limited memory (e.g., Jetson Xavier):

```bash
cd LLM_tuning
python3 finetune_unsloth.py \
    --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --batch_size 1 \
    --max_seq_length 1024 \
    --num_epochs 2
```

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | Pre-quantized model from Unsloth |
| `--dataset_path` | `./medical_sft_dataset.json` | Path to your SFT dataset (in LLM_tuning/ directory) |
| `--output_dir` | `../models/finetuned_medical` | Where to save the fine-tuned model (relative to LLM_tuning/) |
| `--num_epochs` | `3` | Number of training epochs |
| `--batch_size` | `2` | Batch size (reduce if OOM) |
| `--learning_rate` | `2e-4` | Learning rate |
| `--max_seq_length` | `2048` | Maximum sequence length |
| `--warmup_steps` | `5` | Warmup steps for learning rate |
| `--max_steps` | `-1` | Max steps (-1 = use epochs) |
| `--save_steps` | `100` | Save checkpoint every N steps |
| `--logging_steps` | `10` | Log every N steps |

## Available Models

Unsloth provides pre-quantized models optimized for Jetson. **All models are automatically downloaded on first use:**

- `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` (Recommended for Jetson, ~1-2GB)
- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (Requires more memory, ~2-3GB)
- `unsloth/Mistral-7B-Instruct-bnb-4bit` (Larger, more capable, ~4-5GB)

**Note:** Models are cached in `~/.cache/huggingface/hub/` after first download, so subsequent runs won't need to download again.

## Memory Optimization

Unsloth automatically uses:
- **4-bit quantization** (BNB) for memory efficiency
- **LoRA** (Low-Rank Adaptation) instead of full fine-tuning
- **Gradient checkpointing** to reduce memory usage
- **8-bit AdamW optimizer** for efficient training

### If You Run Out of Memory

1. **Reduce batch size:**
   ```bash
   --batch_size 1
   ```

2. **Reduce sequence length:**
   ```bash
   --max_seq_length 1024
   ```

3. **Reduce gradient accumulation:**
   Edit `LLM_tuning/finetune_unsloth.py` and change `gradient_accumulation_steps=4` to `2` or `1`

4. **Use a smaller model:**
   ```bash
   --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit
   ```

## Using the Fine-Tuned Model

After training, load and use your fine-tuned model:

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../models/finetuned_medical",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

# Create a conversation
messages = [
    {"role": "user", "content": "I have chest pain"}
]

# Format for inference
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Monitoring Training

The script logs:
- Training loss
- Learning rate
- GPU memory usage
- Training steps

Watch for:
- **Loss decreasing**: Good sign
- **Loss plateauing**: May need more epochs or learning rate adjustment
- **OOM errors**: Reduce batch size or sequence length

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
--batch_size 1

# Reduce sequence length
--max_seq_length 1024
```

### Issue: "ModuleNotFoundError: unsloth"

**Solution:**
```bash
# Install from cu126 index using package name
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth==2025.7.9
# Or install latest version:
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth
# If unsloth_zoo is still missing:
pip3 install unsloth_zoo
```

### Issue: "Unsloth: Please install unsloth_zoo" or "No module named 'unsloth_zoo.utils'"

This usually means the wheel didn't install dependencies properly. **Solutions:**

**Option 1: Reinstall from index (Recommended)**
```bash
# Force reinstall to ensure dependencies are installed
pip3 install --force-reinstall --no-cache-dir \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    unsloth==2025.7.9
```

**Option 2: Install unsloth_zoo separately**
```bash
pip3 install unsloth_zoo
```

**Option 3: Upgrade both together (if versions are incompatible)**
```bash
pip3 install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
```

**Option 4: Download and install wheel locally**
```bash
# Download the wheel first
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/edc/0ac127024b8f9/unsloth-2025.7.9-py3-none-any.whl

# Install the local wheel file
pip3 install --force-reinstall --no-cache-dir \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    ./unsloth-2025.7.9-py3-none-any.whl
```

### Issue: "cannot import name 'top_k_top_p_filtering' from 'transformers'"

This error occurs when using transformers 4.46.0 or newer, which removed this function. **Even if you have transformers 4.45.2 installed, cached bytecode files can cause this error.**

**First, run the diagnostic script:**
```bash
cd LLM_tuning
python3 diagnose_imports.py
```

This will show you:
- Current transformers version and location
- Whether `top_k_top_p_filtering` is available
- If there are multiple transformers installations
- Cache file locations

**Solution:**

```bash
# 1. First, uninstall incompatible packages
pip3 uninstall -y transformers trl unsloth_zoo

# 2. Install compatible versions in order
pip3 install --force-reinstall --no-cache-dir "transformers>=4.40.0,<4.46.0"
pip3 install --force-reinstall --no-cache-dir "trl>=0.7.0,<0.8.0"

# 3. Reinstall unsloth
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 unsloth==2025.7.9

# 4. Install unsloth_zoo without dependencies to avoid conflicts
pip3 install --no-deps unsloth_zoo

# 5. Clear Python cache (CRITICAL - stale cache can cause import errors)
echo "Clearing Python cache..."
# Clear user site-packages cache
USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
find "$USER_SITE" -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find "$USER_SITE" -name "*.pyc" -delete 2>/dev/null || true
find "$USER_SITE" -name "*.pyo" -delete 2>/dev/null || true
# Specifically clear transformers cache
find "$USER_SITE/transformers" -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find "$USER_SITE/transformers" -name "*.pyc" -delete 2>/dev/null || true

# Clear system site-packages cache
for SITE_DIR in $(python3 -c "import site; print(' '.join(site.getsitepackages()))"); do
    if [ -d "$SITE_DIR" ]; then
        find "$SITE_DIR" -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
        find "$SITE_DIR" -name "*.pyc" -delete 2>/dev/null || true
    fi
done

echo "Cache cleared"
```

**Note:** If you already have transformers 4.46+ installed, you must use `--force-reinstall` to downgrade it. Simply running `pip3 install` without `--force-reinstall` may not downgrade an existing newer version.

Then retry the import or run the fine-tuning script.

### Issue: Dependency conflicts (unsloth-zoo/trl requiring newer transformers)

If you see errors like:
- `unsloth-zoo 2025.11.3 requires transformers>=4.51.3`
- `trl 0.24.0 requires transformers>=4.56.1`

This means newer versions of these packages were installed, which require incompatible transformers versions. **Solution:**

```bash
# 1. Uninstall incompatible versions
pip3 uninstall -y unsloth_zoo trl transformers

# 2. Install compatible versions in the correct order with constraints
pip3 install --force-reinstall --no-cache-dir transformers==4.45.2

# Use constraint to prevent transformers from being upgraded when installing trl
echo "transformers<4.46.0" > /tmp/transformers_constraint.txt
pip3 install --force-reinstall --no-cache-dir --constraint /tmp/transformers_constraint.txt trl==0.7.11
rm -f /tmp/transformers_constraint.txt

# Verify transformers wasn't upgraded
python3 -c "import transformers; v=transformers.__version__; assert v < '4.46.0', f'Transformers upgraded to {v}! Downgrade with: pip3 install --force-reinstall transformers==4.45.2'"

# 3. Reinstall unsloth with constraint
echo "transformers<4.46.0" > /tmp/transformers_constraint.txt
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --constraint /tmp/transformers_constraint.txt unsloth==2025.7.9
rm -f /tmp/transformers_constraint.txt

# Verify transformers again
python3 -c "import transformers; v=transformers.__version__; assert v < '4.46.0', f'Transformers upgraded to {v}! Downgrade with: pip3 install --force-reinstall transformers==4.45.2'"

# 4. Install unsloth_zoo with --no-deps to prevent pulling incompatible transformers
pip3 install --no-deps unsloth_zoo

# 5. Verify installation
python3 -c "import transformers; print('transformers:', transformers.__version__)"
python3 -c "from unsloth import FastLanguageModel; print('✅ Unsloth works!')"
```

**Key points:**
- Install `transformers` FIRST with compatible version (4.45.2)
- Use `--constraint` when installing `trl` and `unsloth` to prevent transformers from being upgraded
- Use `--no-deps` when installing `unsloth_zoo` to prevent it from pulling incompatible transformers
- Always verify transformers version after each installation step
- The setup script handles this automatically if you run it fresh

**If transformers gets upgraded during installation:**
```bash
# Immediately downgrade it
pip3 install --force-reinstall --no-cache-dir transformers==4.45.2
```

### Issue: "IndexError: list index out of range" during Unsloth import

This is a known compatibility issue with unsloth's patching mechanism. **Solutions:**

**Option 1: Use compatible trl version (Recommended)**
```bash
pip3 install "trl>=0.7.0,<0.8.0"
pip3 install --upgrade unsloth
```

**Option 2: Try continuing anyway**
The error may be non-critical for SFT (Supervised Fine-Tuning). Try running the fine-tuning script - it may still work:
```bash
cd LLM_tuning
python3 finetune_unsloth.py
```

**Option 3: Downgrade trl**
```bash
pip3 install trl==0.7.11
```

**Option 4: Skip RL trainer patching (if not needed)**
If you only need SFT and not RLHF, you can try:
```bash
export UNSLOTH_SKIP_RL_PATCH=1
python3 finetune_unsloth.py
```

### Issue: "Slow training"

**Solution:**
- Ensure you're using CUDA (check with `nvidia-smi`)
- Use smaller batch size for faster iterations
- Reduce `max_seq_length` if possible

### Issue: "Model not loading"

**Solution:**
- Check that the model path is correct
- Ensure you have enough disk space
- Verify CUDA is properly installed

## Integration with Your Medical Bot

After fine-tuning, integrate the model into `llm-medical-container/container_rest.py`:

```python
from unsloth import FastLanguageModel

# Load fine-tuned model (from container perspective)
model, tokenizer = FastLanguageModel.from_pretrained(
    "/path/to/models/finetuned_medical",  # Adjust path as needed
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Use in your medical navigator
def generate_medical_response(messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Performance Tips

1. **Use 4-bit quantization**: Already enabled by default
2. **Enable gradient checkpointing**: Already enabled
3. **Use LoRA**: Already configured (r=16, alpha=16)
4. **Batch size**: Start with 2, reduce if needed
5. **Sequence length**: 2048 is good, reduce to 1024 if memory constrained

## Next Steps

1. **Evaluate the model**: Test with sample medical conversations
2. **Iterate**: Add more examples to your dataset if needed
3. **Deploy**: Integrate into your medical bot container
4. **Monitor**: Track performance in production

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Jetson AI Lab PyPI - cu126](https://pypi.jetson-ai-lab.io/jp6/cu126) (where unsloth wheel is available)
- [Jetson AI Lab PyPI - cu129](https://pypi.jetson-ai-lab.io/jp6/cu129) (alternative index)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

