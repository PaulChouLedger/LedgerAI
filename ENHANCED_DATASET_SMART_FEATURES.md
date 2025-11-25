# Enhanced Dataset with Smart Features

## Overview

The `medical_sft_dataset_enhanced_smart.json` dataset builds upon `medical_sft_dataset_enhanced.json` with two key enhancements:

1. **Smart OLD CARTS Question Selection**: Automatically identifies and marks irrelevant OLD CARTS elements
2. **British Slang Variations**: Includes British English variants for UK market deployment

## Dataset Statistics

- **Total Conversations**: 726
- **American Variants**: 363 (original)
- **British Variants**: 363 (new)
- **All conversations include smart features**: Yes

## Smart Features

### 1. Relevance Detection

The dataset automatically determines which OLD CARTS elements are relevant for each chief complaint:

**Example: "coffee ground vomit"**
- ✅ Relevant: Onset, Duration, Aggravating, Alleviating, Timing, Severity
- ❌ Not Relevant: Location, Character, Radiation

**Example: "chest pain"**
- ✅ Relevant: All 9 OLD CARTS elements

**Example: "hypertension"**
- ✅ Relevant: Onset, Duration, Aggravating, Alleviating, Timing, Severity
- ❌ Not Relevant: Location, Character, Radiation

### 2. Skip Tags

When an OLD CARTS element is not relevant, the dataset includes a skip message:

```json
{
  "role": "assistant",
  "content": "[SKIP:L] This OLD CARTS element is not relevant for this chief complaint and should be skipped.",
  "metadata": {
    "skip": true,
    "element": "L",
    "reason": "Not relevant for chief complaint: I have coffee ground vomit"
  }
}
```

This teaches the fine-tuned model to:
- Recognize when questions don't make sense
- Skip irrelevant OLD CARTS elements
- Ask only appropriate questions

### 3. Relevance Metadata

Each conversation includes `relevant_oldcarts` metadata:

```json
{
  "relevant_oldcarts": {
    "O": true,
    "L": false,
    "D": true,
    "C": false,
    "A_aggravating": true,
    "A_alleviating": true,
    "R": false,
    "T": true,
    "S": true
  }
}
```

## British Slang Variations

### Conversion Examples

**American → British:**
- "I have" → "I've got"
- "hurting" → "proper painful"
- "really bad" → "quite bad"
- "awful" → "rubbish"
- "right here" → "over here" / "round here"
- "out of nowhere" → "out of the blue"
- "really messes me up" → "proper sets it off"

### British-Specific Phrases

- "bloody", "blooming", "ruddy" (intensifiers)
- "quite", "rather", "proper" (modifiers)
- "alright" instead of "okay"
- "of course" instead of "sure"

### What Gets Converted

✅ **Converted:**
- User responses
- Conversational assistant messages
- Empathetic statements
- Initial questions

❌ **Not Converted:**
- System prompts
- Clinical reasoning sections
- Final diagnostic reasoning
- Medical terminology

## Usage with Training Script

The enhanced dataset is fully compatible with `train_medical_bot_colab.py`:

```python
# The training script will automatically:
# 1. Load conversations with skip tags
# 2. Learn to skip irrelevant questions
# 3. Handle both American and British variants
# 4. Use relevance metadata for context
```

## Benefits

### For Training
1. **Better Question Selection**: Model learns to ask only relevant questions
2. **Language Diversity**: Handles both American and British English
3. **Clinical Accuracy**: Avoids nonsensical questions like "Where is your hypertension located?"
4. **Efficiency**: Fewer irrelevant questions = faster, better patient experience

### For Deployment
1. **UK Market Ready**: British slang variants for UK deployment
2. **Smart Questioning**: Automatically skips irrelevant OLD CARTS elements
3. **Natural Conversations**: More natural, context-aware questioning
4. **Reduced Errors**: Less likely to ask confusing or inappropriate questions

## File Structure

```
medical_sft_dataset_enhanced_smart.json
├── Conversation 1 (American)
│   ├── messages: [...]
│   ├── smart_features: true
│   ├── relevant_oldcarts: {...}
│   └── variant: "american"
├── Conversation 1 (British)
│   ├── messages: [...]
│   ├── smart_features: true
│   ├── relevant_oldcarts: {...}
│   └── variant: "british"
└── ...
```

## Integration with Advanced Medical Navigator

The `advanced_medical_navigator.py` can use the relevance metadata:

```python
# Check if element should be skipped
relevant = conversation.get("relevant_oldcarts", {})
if not relevant.get("L", True):
    # Skip location question
    continue
```

## Regeneration

To regenerate the enhanced dataset:

```bash
python3 enhance_dataset_with_smart_features.py
```

This will:
1. Load `medical_sft_dataset_enhanced.json`
2. Add smart features to all conversations
3. Create British variants
4. Save to `medical_sft_dataset_enhanced_smart.json`

## Next Steps

1. **Train Model**: Use the enhanced dataset with `train_medical_bot_colab.py`
2. **Test Skip Logic**: Verify model learns to skip irrelevant questions
3. **Deploy**: Use British variants for UK market
4. **Monitor**: Track question relevance in production

