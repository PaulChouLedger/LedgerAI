# TTS Chunking Analysis - Sentence Boundary Detection Issues

## Problem

Sentences are being split incorrectly, causing audio artifacts. Example from logs:

```
"...jacket emblazoned with "NXIVM."
```

The sentence is split at the period inside the quote, then the closing quote `"` becomes a separate chunk, causing audio artifacts.

## Root Cause

**Location**: `llm-container/container_rest.py`, lines 92, 1761-1777

The sentence boundary detection logic:

1. **Uses simple punctuation check**: `SENTENCE_ENDINGS = ('.', '!', '?')`
2. **Immediately ends sentence** when punctuation is detected (line 1766):
   ```python
   if word_to_yield.rstrip().endswith(punct):
   ```
3. **No context awareness**: Doesn't check what comes after the punctuation

## Specific Issues

### Issue 1: Quoted Text
- When LLM outputs: `"NXIVM."` followed by `"`
- Period triggers `<sentence_end>` immediately
- Closing quote starts new sentence
- Result: Sentence split mid-quote → audio artifact

### Issue 2: No Lookahead
- Code doesn't peek at next token before ending sentence
- Can't distinguish between:
  - Period at end of sentence: `"Word." Next sentence`
  - Period in quote: `"Word."` (quote continues)
  - Period in abbreviation: `"Dr. Smith"` (not sentence end)

### Issue 3: Immediate Splitting
- Sentences split as soon as punctuation is seen
- No buffering to check context
- Multi-token punctuation (quotes) not handled

## Current Flow (Problematic)

```
Token: "NXIVM."
  → Period detected → <sentence_end> emitted immediately
Token: "  
  → New sentence starts → <sentence_start> emitted
  → Audio artifact: sentence split mid-quote
```

## Suggested Improvements

### 1. Add Quote Context Awareness
- Track if inside quotes (opening quote seen, closing quote not yet seen)
- Don't end sentence on period if inside quotes AND next token is closing quote
- Buffer until closing quote is seen

### 2. Add Lookahead Buffer
- When period detected, peek at next 1-2 tokens
- If next token is:
  - Closing quote (`"`) → don't end sentence
  - Lowercase letter → don't end sentence (abbreviation)
  - Uppercase letter → end sentence
  - End of stream → end sentence

### 3. Smart Period Detection
- Period + quote → keep in same sentence
- Period + lowercase → keep in same sentence (abbreviation)
- Period + uppercase → end sentence
- Period + end of stream → end sentence

### 4. Quote Pairing
- Track quote depth (opening vs closing)
- Don't split sentences mid-quote
- Keep quoted text together as one chunk

### 5. Context-Aware Sentence Endings
- Consider punctuation + following context, not just punctuation alone
- Period + quote = likely not sentence end
- Period + space + uppercase = likely sentence end

## Desired Flow

```
Token: "NXIVM."
  → Period detected → check next token
Token: "
  → Next token is quote → don't end sentence yet
  → Buffer together: "NXIVM."
  → After quote, check if sentence should end
  → Emit complete chunk: "...with "NXIVM.""
```

## Code Locations

- **Sentence endings constant**: Line 92
- **Sentence detection logic**: Lines 1761-1777
- **Word stream processing**: Lines 1782-1826
- **Abbreviation handling**: Lines 1713-1815

## Impact

- **Audio artifacts**: Sentences split mid-quote cause unnatural pauses
- **TTS quality**: Incomplete chunks sent to TTS reduce naturalness
- **User experience**: Robotic-sounding speech due to incorrect chunking

## Testing Scenarios

1. **Quoted text with period**: `"NXIVM."` → should stay together
2. **Abbreviations**: `Dr. Smith` → should stay together
3. **End of sentence**: `Word. Next` → should split
4. **Quoted sentence end**: `"Word." Next` → should split after quote
5. **Nested quotes**: `"He said 'Hello.'"` → should stay together

