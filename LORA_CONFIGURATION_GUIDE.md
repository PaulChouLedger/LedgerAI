# LoRA Configuration Guide

## Current Configuration
- **LoRA Rank (r)**: 128
- **LoRA Alpha**: 256
- **Trainable Parameters**: 90M / 1.3B (6.80%)
- **Target Modules**: 7 (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

## What is LoRA Rank?

LoRA (Low-Rank Adaptation) trains a small adapter matrix instead of all parameters. The rank (`r`) determines the size of this adapter:
- **Lower rank (r=64)**: Fewer parameters, faster training, less memory, but may underfit complex patterns
- **Higher rank (r=256-512)**: More parameters, better capacity, but slower training and more memory

## Options to Train More Parameters

### Option 1: Increase LoRA Rank (Recommended)
**Current**: r=128 → 90M parameters (6.80%)
**Higher rank**: r=256 → ~180M parameters (~13.6%)
**Even higher**: r=512 → ~360M parameters (~27.2%)

**Trade-offs**:
- ✅ Better capacity to learn complex patterns (OLD CARTS sequence, clinical reasoning)
- ✅ Better performance on nuanced medical questions
- ❌ More VRAM required (~2x for r=256, ~4x for r=512)
- ❌ Slightly slower training

**When to use**:
- If you have enough VRAM (16GB+ for r=256, 24GB+ for r=512)
- If the model is underfitting (not learning OLD CARTS sequence well)
- If you want better clinical reasoning quality

### Option 2: Add More Target Modules
**Current**: 7 modules (attention + MLP)
**Extended**: Add `embed_tokens`, `lm_head`, `layer_norm`

**Trade-offs**:
- ✅ Can improve token generation quality
- ✅ Better vocabulary understanding
- ❌ More parameters to train
- ❌ May not help much if rank is already high

**When to use**:
- If model struggles with specific medical terms
- If vocabulary/terminology is an issue

### Option 3: Full Fine-Tuning
**Train all 1.3B parameters**

**Trade-offs**:
- ✅ Maximum capacity and best performance
- ❌ Requires 40GB+ VRAM (not feasible on most GPUs)
- ❌ Much slower training
- ❌ Risk of overfitting with small datasets

**When to use**:
- Only if you have enterprise-grade GPUs (A100, H100)
- Very large datasets (10,000+ examples)
- Maximum performance is critical

## Recommended Configurations

### For Colab Free Tier (T4, 15GB VRAM)
```python
r=128,  # Current - good balance
lora_alpha=256,
```

### For Colab Pro (V100, 16GB VRAM)
```python
r=256,  # 2x more parameters
lora_alpha=512,  # Keep alpha = 2x rank
```

### For A100 (40GB+ VRAM)
```python
r=512,  # 4x more parameters
lora_alpha=1024,
# Optionally add: target_modules += ["embed_tokens", "lm_head"]
```

## Does Training More Parameters Make a Difference?

**Yes, but with diminishing returns:**

1. **r=64 → r=128**: Significant improvement (better sequence following)
2. **r=128 → r=256**: Moderate improvement (better reasoning, fewer errors)
3. **r=256 → r=512**: Small improvement (marginal gains)
4. **r=512 → Full fine-tuning**: Minimal improvement for most tasks

**For your use case (medical bot with OLD CARTS):**
- **r=128** (current): Good for basic OLD CARTS sequence
- **r=256**: Better for complex clinical reasoning and differential diagnosis
- **r=512**: Best if you have VRAM, but may be overkill

## How to Change Configuration

Edit `train_medical_bot_colab.py` line 274:

```python
# Current (6.80% trainable)
r=128,
lora_alpha=256,

# Higher capacity (13.6% trainable)
r=256,
lora_alpha=512,

# Maximum capacity (27.2% trainable)
r=512,
lora_alpha=1024,
```

## Memory Requirements

| Rank | Trainable Params | VRAM (4-bit) | VRAM (8-bit) |
|------|------------------|--------------|--------------|
| r=64 | ~45M (3.4%) | ~8GB | ~12GB |
| r=128 | ~90M (6.8%) | ~10GB | ~15GB |
| r=256 | ~180M (13.6%) | ~14GB | ~20GB |
| r=512 | ~360M (27.2%) | ~22GB | ~30GB |

## Recommendation

**For your current dataset (214 conversations):**
- **r=128** is sufficient if training is going well
- **r=256** if you have VRAM and want better clinical reasoning
- **r=512** only if you have A100 or similar high-end GPU

**Test first**: Try r=256 and see if loss decreases faster or final performance improves. If not, r=128 is fine.


