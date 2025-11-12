# Fine-Tuning Medical Bot with Unsloth on Jetson

This guide walks you through fine-tuning your medical chatbot using Unsloth on a Jetson device.

## Prerequisites

- Jetson device (AGX Orin, Xavier, etc.) with JetPack installed
- CUDA 12.9 support (for cu129 packages)
- Python 3.10 or 3.12
- At least 16GB RAM (32GB recommended)
- Medical SFT dataset (`medical_sft_dataset.json` in `LLM_tuning/` directory)

## Installation

### 1. Install Unsloth from Jetson AI Lab PyPI

The Jetson AI Lab provides optimized packages for Jetson devices. Install Unsloth from their custom PyPI index:

```bash
# Install Unsloth and dependencies
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 \
    unsloth-2025.7.9-py3-none-any.whl \
    unsloth_zoo \
    transformers \
    datasets \
    trl \
    peft \
    accelerate \
    bitsandbytes
```

**Alternative:** If you have the wheel file locally:

```bash
# Download the wheel file first
wget https://pypi.jetson-ai-lab.io/jp6/cu129/+f/edc/0ac127024b8f9/unsloth-2025.7.9-py3-none-any.whl

# Install it
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 \
    unsloth-2025.7.9-py3-none-any.whl
```

### 2. Install Additional Dependencies

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install "transformers>=4.40.0,<4.46.0" datasets "trl>=0.7.0,<0.8.0" peft accelerate bitsandbytes unsloth_zoo
```

**Important Version Constraints:**
- `transformers>=4.40.0,<4.46.0`: Versions 4.46+ removed `top_k_top_p_filtering` which unsloth requires
- `trl>=0.7.0,<0.8.0`: Avoids compatibility issues with unsloth's patching mechanism

**Important:** `unsloth_zoo` is a required dependency for Unsloth. Make sure it's installed:
```bash
pip3 install unsloth_zoo
```

### 3. Quick Setup Script

Or use the provided setup script from the `LLM_tuning` directory:

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
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 unsloth
pip3 install unsloth_zoo
```

### Issue: "Unsloth: Please install unsloth_zoo"

**Solution:**
```bash
pip3 install unsloth_zoo
```

### Issue: "cannot import name 'top_k_top_p_filtering' from 'transformers'"

This error occurs when using transformers 4.46.0 or newer, which removed this function. **Solution:**

```bash
pip3 install "transformers>=4.40.0,<4.46.0"
```

Then retry the import or run the fine-tuning script.

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
- [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/jp6/cu129)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

