# Medical Bot Fine-Tuning Guide

This guide explains how to properly fine-tune a Llama-3.2 model to behave as a medical bot using the `medical_sft_dataset.json` dataset.

## 🎯 Key Improvements

The updated training script (`train_medical_bot.py`) includes several critical improvements over the original Colab notebook:

### 1. **Proper Chat Template Usage**
- ✅ Uses Llama-3.2's native chat template format
- ✅ Properly formats system, user, and assistant messages
- ✅ Ensures the model understands conversation structure

### 2. **Medical-Specific System Prompt**
- ✅ Adds a professional medical assistant system prompt
- ✅ Guides the model to follow structured medical frameworks (SOCRATES, OLD CARTS)
- ✅ Emphasizes empathy, professionalism, and thoroughness

### 3. **Correct Data Formatting**
- ✅ Preserves the original message structure from the dataset
- ✅ Applies chat template during tokenization (not as plain text)
- ✅ Maintains conversation context and flow

### 4. **Optimized Training Parameters**
- ✅ Increased warmup steps for better convergence
- ✅ Proper batch size and gradient accumulation
- ✅ Medical conversation-specific optimizations

### 5. **GGUF Export**
- ✅ Exports in Q4_K_M quantization (compatible with your system)
- ✅ Ready for deployment in llama.cpp-based containers

## 📋 Prerequisites

### For Local Training:
```bash
pip install unsloth trl peft accelerate bitsandbytes datasets transformers
```

### For Google Colab:
The script includes installation instructions. Just run:
```python
!pip install unsloth trl peft accelerate bitsandbytes datasets
```

## 🚀 Usage

### Option 1: Local Training

```bash
python train_medical_bot.py
```

### Option 2: Google Colab

1. Upload `medical_sft_dataset.json` to Colab
2. Run `train_medical_bot_colab.py` in a Colab notebook cell
3. The script will automatically download the GGUF file when complete

## 📊 Training Configuration

The script uses the following optimized settings:

- **Model**: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- **LoRA Rank**: 64 (good balance of capacity and memory)
- **LoRA Alpha**: 128 (2x rank for optimal scaling)
- **Max Sequence Length**: 2048 tokens
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 4 steps (effective batch size = 8)
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Warmup Steps**: 50

## 🔧 Customization

### Adjust Training Parameters

Edit these variables in the script:

```python
MAX_SEQ_LENGTH = 2048  # Increase if you have longer conversations
num_train_epochs=3,    # Adjust based on dataset size
learning_rate=2e-4,    # Lower for more stable training
r=64,                  # LoRA rank (higher = more capacity)
```

### Modify System Prompt

Edit the `MEDICAL_SYSTEM_PROMPT` variable to customize the bot's behavior:

```python
MEDICAL_SYSTEM_PROMPT = """Your custom medical assistant instructions here..."""
```

## 📁 Output Files

After training, you'll get:

1. **HuggingFace Format** (`outputs/`)
   - Full model with LoRA adapters
   - Can be used for further fine-tuning
   - Compatible with HuggingFace ecosystem

2. **GGUF Format** (`gguf_model/`)
   - Quantized model ready for deployment
   - Compatible with llama.cpp
   - Use this in your LedgerAI containers

## 🔄 Deploying the Trained Model

1. **Copy GGUF file to models directory:**
   ```bash
   cp gguf_model/*.gguf /path/to/LedgerAI/models/
   ```

2. **Update Dockerfile** (if needed):
   ```dockerfile
   COPY models/Your-Model-Name.gguf /models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
   ```

3. **Update environment variable:**
   ```bash
   SIMPLE_MODEL_PATH=/models/Your-Model-Name.gguf
   ```

## 🐛 Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size` to 1
- Reduce `MAX_SEQ_LENGTH` to 1024
- Reduce LoRA rank `r` to 32

### Slow Training
- Ensure you're using a GPU (CUDA)
- Check that `load_in_4bit=True` is set
- Reduce `dataset_num_proc` if CPU-bound

### Poor Model Performance
- Increase training epochs
- Increase LoRA rank (r=128)
- Add more data to `medical_sft_dataset.json`
- Adjust learning rate (try 1e-4 or 3e-4)

## 📝 Dataset Format

Your `medical_sft_dataset.json` should follow this format:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "I have chest pain"
      },
      {
        "role": "assistant",
        "content": "I understand you're experiencing chest pain..."
      }
    ]
  }
]
```

The script automatically:
- Adds a medical system prompt if not present
- Formats messages using Llama-3.2 chat template
- Validates conversation structure

## 🎓 Understanding the Changes

### Why Chat Template Matters

**Before (incorrect):**
```python
conversation_text = ""
for msg in messages:
    if role == "user":
        conversation_text += f"User: {content}\n\n"
    elif role == "assistant":
        conversation_text += f"Assistant: {content}\n\n"
```

**After (correct):**
```python
# Uses proper chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

The chat template ensures:
- Proper tokenization of special tokens (`<|start_header_id|>`, `<|eot_id|>`, etc.)
- Model understands conversation structure
- Better generation quality

### Why System Prompt Matters

The medical system prompt:
- Sets the model's role and behavior
- Guides structured medical questioning
- Ensures professional, empathetic responses
- Maintains consistency across conversations

## 📚 Additional Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Llama-3.2 Chat Template](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [LoRA Fine-tuning Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)

## ✅ Verification

After training, test your model:

```python
from llama_cpp import Llama

llm = Llama(model_path="gguf_model/your_model.gguf")

messages = [
    {"role": "system", "content": "You are a medical assistant..."},
    {"role": "user", "content": "I have chest pain"}
]

response = llm.create_chat_completion(messages=messages)
print(response['choices'][0]['message']['content'])
```

The model should respond with professional, structured medical questions following the patterns in your training data.

