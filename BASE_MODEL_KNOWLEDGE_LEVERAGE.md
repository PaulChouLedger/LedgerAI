# Leveraging Base Model Medical Knowledge

## The Question

**Does the fine-tuning leverage the base model's existing medical knowledge?**

Yes! The base model (Qwen 2.5-1.5B-Instruct) already understands:
- Anatomical relationships (e.g., "right upper quadrant pain" → liver/gallbladder)
- Medical terminology (e.g., "epigastric" → stomach/pancreas)
- Clinical patterns (e.g., "pleuritic pain" → pulmonary/pleural)

## How LoRA Preserves Base Knowledge

**LoRA (Low-Rank Adaptation) fine-tuning:**
- ✅ **Base model weights are FROZEN** - all medical knowledge is preserved
- ✅ **Only adapter layers are trained** - ~13.6% of parameters (256 rank)
- ✅ **Base model's medical knowledge remains intact** - anatomy, terminology, patterns
- ✅ **Training adapts the model to the conversation format** - not replacing knowledge

## What We're Training

The fine-tuning teaches the model:
1. **Conversation format** - How to structure medical history taking
2. **Question sequence** - Empathy → Chronicity → Demographics → OLD CARTS
3. **Response format** - JSON for scoring, clinical reasoning format
4. **Element identification** - Which OLD CARTS element an answer addresses

**NOT training:**
- ❌ Medical facts (already in base model)
- ❌ Anatomical relationships (already in base model)
- ❌ Clinical patterns (already in base model)

## Enhanced Dataset Features

### 1. Base Knowledge Examples (4 new examples)
Added examples that **explicitly show using base medical knowledge**:
- **RUQ pain + fatty meal** → Demonstrates understanding gallbladder/biliary anatomy
- **Epigastric pain** → Shows understanding of stomach/pancreas anatomy
- **LLQ pain in female** → Shows understanding of sigmoid/ovary anatomy
- **Pleuritic chest pain** → Shows understanding of pulmonary vs cardiac patterns

### 2. System Prompt Updates
Updated system prompt to explicitly tell the model:
- "Use your extensive medical knowledge to understand patient descriptions"
- "Understand anatomical relationships"
- "Recognize medical terminology"
- "Apply clinical patterns"

## Example: Right Upper Quadrant Pain

**Base Model Knowledge:**
- Qwen 2.5-1.5B already knows: RUQ → liver, gallbladder, biliary system

**What Training Adds:**
- How to ask about RUQ pain in the conversation format
- How to use RUQ knowledge in clinical reasoning
- How to structure the response with OLD CARTS framework

**Result:**
- Model uses its base knowledge (RUQ = gallbladder) ✅
- Model follows conversation format ✅
- Model provides structured clinical reasoning ✅

## Training Configuration

```python
# LoRA Configuration - Only trains adapter layers
LORA_RANK = 256  # ~13.6% of parameters
# Base model weights: FROZEN (medical knowledge preserved)
# Adapter layers: TRAINED (conversation format learned)
```

## Verification

To verify the model is using base knowledge:

1. **Test with anatomical terms:**
   - "I have right upper quadrant pain" → Should recognize liver/gallbladder
   - "I have epigastric pain" → Should recognize stomach/pancreas
   - "I have left lower quadrant pain" → Should recognize sigmoid/ovary

2. **Test with medical terminology:**
   - "I have pleuritic chest pain" → Should recognize pulmonary/pleural
   - "I have substernal chest pain" → Should recognize cardiac/esophageal

3. **Test with clinical patterns:**
   - "Pain after fatty meal" → Should connect to gallbladder
   - "Pain worse with breathing" → Should connect to pulmonary

## Expected Behavior

**Before Training:**
- Base model knows RUQ = gallbladder ✅
- But doesn't know conversation format ❌
- Doesn't know OLD CARTS structure ❌

**After Training:**
- Base model knows RUQ = gallbladder ✅ (preserved)
- Knows conversation format ✅ (learned)
- Knows OLD CARTS structure ✅ (learned)
- Uses medical knowledge in reasoning ✅ (learned to apply)

## Summary

✅ **Base model knowledge is PRESERVED** (LoRA freezes base weights)
✅ **Training teaches FORMAT and STRUCTURE** (not medical facts)
✅ **Enhanced dataset shows HOW to use base knowledge** (anatomical examples)
✅ **System prompt explicitly tells model to use medical knowledge**

The fine-tuning **leverages** the base model's medical knowledge by:
1. Preserving it (frozen weights)
2. Teaching how to use it (conversation format)
3. Showing examples of using it (anatomical examples)
4. Explicitly instructing it (system prompt)

