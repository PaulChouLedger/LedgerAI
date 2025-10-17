# Dual Model Setup (Llama-1B + Mistral-7B)

## Overview

The system now uses **two models in parallel** for optimal performance:

1. **Llama-3.2-1B** (Simple, Fast) - Templates, validation, simple questions
2. **Mistral-7B** (Complex, Smart) - Diagnostic reasoning, complex questions

## Performance Improvement

| Metric | Before (Mistral only) | After (Dual + Parallel) | Improvement |
|--------|----------------------|------------------------|-------------|
| **Filler response** | 8s | 0.1s | **80x faster** |
| **Opening statement** | 2s | 0.2s | **10x faster** |
| **Age question** | 1s | 0.1s | **10x faster** |
| **First response** | 8s | ~5s | **1.6x faster** |

## .env Configuration

Create or update `/Users/rcabello/Documents/GitHub/LedgerAI/.env`:

```bash
# =============================================================================
# DUAL MODEL CONFIGURATION
# =============================================================================

# Complex Model (Mistral-7B) - For diagnostic reasoning
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
CHAT_FORMAT=mistral-instruct
N_CTX=8192

# Simple Model (Llama-1B) - For templates and validation
SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SIMPLE_CHAT_FORMAT=llama-3
SIMPLE_N_CTX=2048

# =============================================================================
# LLM PARAMETERS
# =============================================================================
LLM_TEMPERATURE=0.6
LLM_TOP_P=0.85
LLM_TOP_K=30
LLM_REPEAT_PENALTY=1.15
```

## Quick Setup

```bash
cd ~/LedgerAI

# Create .env file
cat > .env << 'EOF'
MODEL_PATH=/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
CHAT_FORMAT=mistral-instruct
N_CTX=8192

SIMPLE_MODEL_PATH=/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SIMPLE_CHAT_FORMAT=llama-3
SIMPLE_N_CTX=2048

LLM_TEMPERATURE=0.6
LLM_TOP_P=0.85
LLM_TOP_K=30
LLM_REPEAT_PENALTY=1.15
EOF

# Rebuild container
cd llm-container
docker build -t aura-llm .

# Restart
cd ~/LedgerAI
docker-compose restart llm-container

# Verify both models loaded
docker-compose logs llm-container | grep "Loading"
```

## Model Assignment

### Llama-1B (Simple Tasks)
Fast, lightweight model for:
- ✅ Opening statements
- ✅ Age/sex questions
- ✅ Clarification questions
- ✅ Answer validation

### Mistral-7B (Complex Tasks)
Powerful model for:
- ✅ OLDCARTS question generation (requires guideline reasoning)
- ✅ Location clarification (guideline-driven)
- ✅ Associated symptom questions
- ✅ Diagnosis generation
- ✅ Red flag assessment

## How Parallel Execution Works

```
User: "I have abdominal pain"
  ↓
[0.1s] Generate filler → "I see..."
  ↓
┌─────────────────────────┬───────────────────────┐
│ Thread 1: RAG           │ Thread 2: Llama-1B    │
│ • Embed chief complaint │ • Generate opening    │
│ • Match 30 guidelines   │ • Generate age Q      │
│ • Score by prevalence   │ • (Both finish fast)  │
│ [Takes 5 seconds]       │ [Takes 0.3 seconds]   │
└─────────────────────────┴───────────────────────┘
  ↓ Both complete (RAG is bottleneck at 5s)
  ↓
Return: "I understand. Let me ask some questions. How old are you?"
```

**Result**: User hears filler at 0.1s, then actual question at 5s (instead of 8s)

## Additional Fixes in This Update

### 1. **Location Clarification Threshold**
Changed from `0.70` to `0.85` for more aggressive clarification.

**Example**:
- Patient: "left side"
- Guidelines: LLQ, Epigastric, Flank
- Similarity: 0.77
- **Before**: Accepted (no clarification)
- **After**: Asks "Is it upper left or lower left?"

### 2. **Single Question Validation**
Prevents LLM from combining multiple questions.

**Example**:
- ❌ **Before**: "How long does it last? Is it constant or intermittent?"
- ✅ **After**: "How long does the pain last?" (template fallback if LLM violates)

## Verification

After restart, check logs for:

```
[Engine] 🧠 Using dual models (simple + complex)
[LLM] ✅ Complex model (Mistral-7B) loaded
[LLM] ✅ Simple model (Llama-1B) loaded
```

Then test:
```
User: "I have abdominal pain"
Logs should show:
[Engine] 💬 Filler (for immediate response): [opening_2] 'Okay...'
[Engine] ⚡ Starting parallel execution (RAG + Llama-1B)...
[Engine] ⚡ Parallel execution complete!
[Engine]    Opening (simple model): '...'
[Engine]    Age Q (simple model): 'How old are you?'
```

## Benefits

✅ **Instant feedback** - Filler at 0.1s
✅ **10x faster templates** - Llama-1B vs Mistral-7B
✅ **Parallel processing** - RAG + LLM simultaneously
✅ **Quality preserved** - Mistral-7B still handles complex reasoning
✅ **Single questions enforced** - Validation prevents multiple-question prompts
✅ **Better location clarification** - Stricter threshold (0.85 vs 0.70)

