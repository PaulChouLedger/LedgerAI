# HYBRID_ON Toggle Guide

## Overview

The `HYBRID_ON` environment variable allows you to switch between two medical assistant systems:

1. **Hybrid Medical Assistant** (`HYBRID_ON=true`) - New natural, context-aware conversations
2. **Adaptive Diagnostic Engine** (`HYBRID_ON=false` or unset) - Default guideline-based system

## Usage

### Enable Hybrid Assistant

```bash
export HYBRID_ON=true
```

### Use Default Adaptive Engine

```bash
export HYBRID_ON=false
# or simply don't set it
```

### In Docker Compose

Add to your `.env` file or `docker-compose.yml`:

```yaml
environment:
  - HYBRID_ON=true  # or false
```

## What Changes

### When `HYBRID_ON=true`:

- Uses `HybridMedicalAssistant` for all symptom assessments
- Natural, human-like conversations
- Context-aware question generation
- Smart anatomical inference ("right side" → "right upper quadrant")
- Dynamic, fluid conversation flow

### When `HYBRID_ON=false` (default):

- Uses `AdaptiveDiagnosticEngine` for symptom assessments
- Guideline-based questioning
- Structured OLDCARTS flow
- Rule-based anatomical extraction

## Features Comparison

| Feature | Hybrid Assistant | Adaptive Engine |
|---------|------------------|-----------------|
| **Conversation Style** | Natural, human-like | Structured, guideline-based |
| **Question Generation** | Context-aware, dynamic | Rule-based, sequential |
| **Anatomical Inference** | Smart (LLM + Rules) | Rule-based only |
| **Context Awareness** | High (full conversation) | Medium (OLDCARTS tracking) |
| **Flexibility** | Very high | Medium |

## Examples

### Example 1: Chief Complaint Handling

**Hybrid Assistant** (`HYBRID_ON=true`):
```
User: "I have bloody diarrhea"
Assistant: "I understand you're experiencing bloody diarrhea. Can you tell me about the color of the stool? How many episodes have you had?"
```

**Adaptive Engine** (`HYBRID_ON=false`):
```
User: "I have bloody diarrhea"
Assistant: "I understand you're experiencing bloody diarrhea. Is this a new problem or an ongoing issue?"
```

### Example 2: Anatomical Location

**Hybrid Assistant** (`HYBRID_ON=true`):
```
User: "pain on my right side"
System: Automatically infers "right upper quadrant"
```

**Adaptive Engine** (`HYBRID_ON=false`):
```
User: "pain on my right side"
System: Extracts "right" (horizontal), asks for more specificity
```

## Switching Between Systems

You can switch at any time by changing the environment variable and restarting the service:

```bash
# Switch to hybrid
export HYBRID_ON=true
# Restart service

# Switch back to adaptive
export HYBRID_ON=false
# Restart service
```

## Fallback Behavior

If the hybrid assistant fails to initialize, the system automatically falls back to the adaptive engine:

```
[Clinician] ❌ Failed to initialize hybrid assistant: ...
[Clinician] ⚠️ Falling back to adaptive engine
```

## Debugging

Check which system is active:

```
[Clinician] 🔀 HYBRID_ON=true - Using new Hybrid Medical Assistant
# or
[Clinician] 🔀 HYBRID_ON=false - Using Adaptive Diagnostic Engine (default)
```

## Notes

- Both systems use the same underlying guidelines and medical rules
- Both systems support the same medical categories (GI, CARDIO, PULMONARY, etc.)
- The hybrid assistant is designed to be more natural and conversational
- The adaptive engine is more structured and follows a strict OLDCARTS flow
- You can test both systems in parallel by running different instances with different settings

