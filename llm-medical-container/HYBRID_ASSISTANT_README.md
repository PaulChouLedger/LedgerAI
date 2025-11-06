# Hybrid Medical Assistant - Natural, Context-Aware Conversations

## Overview

The `hybrid_medical_assistant.py` is a complete redesign from scratch, focusing on natural, human-like medical conversations using a hybrid LLM/Rules/ML approach.

## Key Features

### 1. **Natural Conversations**
- Human-like, empathetic responses
- Context-aware question generation
- Fluid conversation flow (not rigid)

### 2. **Smart Anatomical Understanding**
- "pain on my right side" → automatically infers "right upper quadrant"
- Uses LLM + Rules for intelligent inference
- Context-aware location extraction

### 3. **Context-Aware Question Generation**
- "bloody diarrhea" → naturally asks "Can you tell me about the color of the stool? How many episodes have you had?"
- Uses guidelines + FAISS to determine what to ask next
- Questions flow naturally from conversation context

### 4. **Hybrid Approach for ALL Interactions**
- **LLM**: Natural language understanding and generation
- **Rules**: Fast pattern matching (medical_rules.json)
- **ML**: Learned patterns (future enhancement)
- **FAISS**: Semantic matching to guidelines

## Architecture

### Components

1. **ConversationContext**
   - Manages conversation state
   - Tracks extracted information
   - Maintains active guidelines

2. **HybridExtractor**
   - Extracts anatomical locations (smart inference)
   - Extracts clinical information (OLDCARTS)
   - Uses LLM + Rules hybrid approach

3. **FAISSGuidelineAssistant**
   - Semantic matching to guidelines
   - Determines missing information
   - Provides hints for next questions

4. **QuestionGenerator**
   - Generates natural, context-aware questions
   - Uses guidelines + FAISS for context
   - LLM-powered for natural language

5. **HybridMedicalAssistant**
   - Main orchestrator
   - Manages conversation flow
   - Integrates all components

## Usage

### Basic Example

```python
from hybrid_medical_assistant import HybridMedicalAssistant
from ml.medical_rule_engine import MedicalRuleEngine
from sentence_transformers import SentenceTransformer

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
medical_rule_engine = MedicalRuleEngine(embedding_model=embedding_model)

# Initialize assistant
assistant = HybridMedicalAssistant(
    llm_chat_fn=your_llm_function,
    embedding_model=embedding_model,
    medical_rule_engine=medical_rule_engine
)

# Process messages
session_id = "user_123"
response = assistant.process_message(session_id, "I have bloody diarrhea")
print(response['response'])
# Output: "I understand you're experiencing bloody diarrhea. Can you tell me about the color of the stool? How many episodes have you had?"

response = assistant.process_message(session_id, "bright red, about 3 times today")
print(response['response'])
# Output: Natural follow-up question based on context
```

### Example: Smart Anatomical Inference

```python
# User: "pain on my right side"
response = assistant.process_message(session_id, "pain on my right side")

# System automatically:
# 1. Extracts: horizontal="right"
# 2. Does NOT assume quadrant (could be upper or lower)
# 3. If more context provided (e.g., "upper right", "near ribs"), infers quadrant
# 4. Generates context-aware question to clarify location
```

**Note**: The system does NOT assume a quadrant from "right side" alone - it could be upper or lower quadrant. The system will ask clarifying questions to determine the exact location.

### Example: Context-Aware Questions

```python
# Chief complaint: "bloody diarrhea"
# System:
# 1. Finds relevant guidelines (Acute Lower GI Bleed)
# 2. Extracts character terms: "bright red blood", "dark red blood"
# 3. Extracts timing terms: "with bowel movement", "episodic"
# 4. Generates: "Can you tell me about the color of the stool? How many episodes have you had?"
```

## Conversation Flow

### Phase 1: Greeting
- Natural greeting response
- Invites patient to share concern

### Phase 2: Chief Complaint
- Extracts chief complaint
- Finds relevant guidelines
- Generates context-aware first question

### Phase 3: Assessment
- Extracts clinical information
- Tracks OLDCARTS elements
- Generates natural follow-up questions
- Uses FAISS to match to guidelines

### Phase 4: Follow-up
- Continues assessment
- Refines understanding
- Completes information gathering

## Smart Features

### 1. Anatomical Location Inference

**Input**: "pain on my right side"

**Processing**:
1. Rule-based: Extracts `horizontal: "right"`
2. Smart inference: Defaults to `quadrant: "right_upper"` (most common)
3. LLM refinement: Can override if context suggests otherwise

**Output**: `{'horizontal': 'right', 'quadrant': 'right_upper', 'vertical': 'upper'}`

### 2. Context-Aware Question Generation

**Chief Complaint**: "bloody diarrhea"

**Processing**:
1. Finds relevant guidelines (Acute Lower GI Bleed)
2. Extracts character terms: "bright red blood", "dark red blood"
3. Extracts timing terms: "with bowel movement", "episodic"
4. LLM generates natural question incorporating these terms

**Output**: "Can you tell me about the color of the stool? How many episodes have you had?"

### 3. Dynamic Question Selection

The system determines what to ask next based on:
- What information is missing (OLDCARTS elements)
- What's most relevant to the chief complaint
- What terms are available in guidelines
- Conversation context

## Integration

### With Existing System

The hybrid assistant can work alongside the existing `adaptive_diagnostic_engine.py`:

```python
# Option 1: Use hybrid assistant for all interactions
assistant = HybridMedicalAssistant(...)

# Option 2: Use hybrid assistant for specific features
# (e.g., anatomical extraction, question generation)
```

### Environment Variables

No special environment variables needed. The assistant uses:
- `LLM_TEMPERATURE_SIMPLE` (if using LLM)
- Standard medical_rule_engine configuration

## Advantages Over Rigid System

### 1. **Natural Language**
- Questions feel human, not robotic
- Responses are empathetic
- Conversation flows naturally

### 2. **Context Awareness**
- Understands conversation history
- Asks relevant questions based on context
- Adapts to patient responses

### 3. **Smart Inference**
- "right side" → "right upper quadrant" (intelligent default)
- Uses context to refine understanding
- LLM + Rules hybrid for accuracy

### 4. **Dynamic Flow**
- Not rigid question sequences
- Adapts based on what's needed
- Uses guidelines to guide questions

## Future Enhancements

1. **ML Model Training**
   - Learn from conversation patterns
   - Improve question generation
   - Better anatomical inference

2. **Active Learning**
   - Learn from user corrections
   - Adapt to conversation style
   - Improve over time

3. **Multi-turn Context**
   - Better conversation memory
   - Refine understanding over time
   - Handle clarifications naturally

## Testing

### Test Cases

1. **Anatomical Inference**
   ```python
   # Test: "pain on my right side"
   # Expected: Infers right upper quadrant
   ```

2. **Context-Aware Questions**
   ```python
   # Test: "bloody diarrhea"
   # Expected: Asks about color and episodes naturally
   ```

3. **Natural Flow**
   ```python
   # Test: Full conversation
   # Expected: Questions flow naturally, not rigid
   ```

## Performance

- **Latency**: Similar to current system (LLM calls are primary bottleneck)
- **Accuracy**: Improved through context awareness
- **Naturalness**: Significantly improved (human-like)

## Dependencies

- `sentence-transformers` (for embeddings)
- `medical_rule_engine` (for FAISS matching)
- LLM function (for natural language)
- `medical_rules.json` (for anatomical rules)
- Guidelines (JSON files in `medical/guidelines/`)

## Notes

- This is a complete redesign, not a modification of existing code
- Can run in parallel with existing system
- Designed to be more natural and human-like
- Uses hybrid approach for all interactions (not just extraction)

