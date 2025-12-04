# Co-Founders Analysis - LLM Scoring Issue

## Query
"Who are the co-founders of Ledger AI?"

## Chunks Retrieved by RAG

### Chunk 1 (Score: 0.582 → LLM Verification: 0.900) ✅ PASSED
**Content:**
- Mentions Will Specht (Head of Engineering - **NOT a co-founder**)
- Mentions Albert Soler (Co-Founder of **Soler Salva LLP** - **NOT LedgerAI**)
- States Albert Soler is "External Counsel & Advisor" to LedgerAI

**Problem:** This chunk got a HIGH score (0.900) but contains incorrect information. Albert Soler is NOT a co-founder of LedgerAI.

---

### Chunk 2 (Score: 0.552 → LLM Verification: 0.800) ✅ PASSED
**Content:**
- **Bob Carella** - "Co-Founder and Chief Financial Officer of LedgerAI" ✅
- **David Lara** - "Co-Founder and Chief Operating Officer of LedgerAI" ✅
- **Jorge Guinovart** - "Co-Founder and Chief Marketing Officer of LedgerAI" ✅
- Will Specht (Head of Engineering - **NOT a co-founder**)

**Correct Information:** This chunk correctly identifies 3 co-founders.

---

### Chunk 3 (Score: 0.523 → LLM Verification: 0.200) ❌ REJECTED
**Content:**
- **Paul Chou** - "CEO and Co-Founder of LedgerAI" ✅
- **Bob Carella** - "Co-Founder and Chief Financial Officer of LedgerAI" ✅

**Problem:** This chunk was REJECTED despite explicitly stating co-founder relationships. It has the clearest information about Paul Chou being a co-founder.

---

## System's Incorrect Answer
The system said: **"Albert Soler and David Lara"**

**Why this is wrong:**
- ❌ Albert Soler is NOT a co-founder of LedgerAI (he's co-founder of Soler Salva LLP)
- ✅ David Lara IS a co-founder (correct)

---

## Actual Co-Founders (from chunks)
Based on the chunks retrieved:

1. **Paul Chou** - CEO and Co-Founder (from Chunk 3, but rejected)
2. **Bob Carella** - Co-Founder and CFO (from Chunks 2 and 3)
3. **David Lara** - Co-Founder and COO (from Chunk 2)
4. **Jorge Guinovart** - Co-Founder and CMO (from Chunk 2)

---

## Issues with LLM Verification

1. **False Negative:** Chunk 3 was incorrectly rejected (score: 0.200) despite explicitly stating co-founder relationships
2. **False Positive:** Chunk 1 was incorrectly accepted (score: 0.900) despite mentioning someone who is NOT a co-founder of LedgerAI
3. **Incomplete Answer:** System only mentioned 2 people when there are actually 4 co-founders mentioned across the chunks

---

## Root Cause

The LLM verification prompt needs to be more explicit about:
1. Distinguishing between "Co-Founder of Company X" vs "Co-Founder of LedgerAI"
2. Recognizing that chunks explicitly stating "Co-Founder of LedgerAI" should get high scores
3. Rejecting chunks that mention people who are co-founders of OTHER companies

## Fix Applied

Made the LLM verification prompt generic and applicable to all types of queries:
1. Removed specific focus on relationship questions - now works for any query type
2. Generic scoring guidelines that apply to factual, definitional, procedural, and relationship questions
3. Focus on whether chunks actually answer the question, not just mention keywords
4. Handles name variations generically (useful for any context with person names)
5. Emphasizes accuracy - chunks with incorrect or misleading information get low scores

---

## Summary: Co-Founders of LedgerAI (from retrieved chunks)

Based on the chunks retrieved and analyzed:

1. **Paul Chou** - CEO and Co-Founder ✅ (mentioned in Chunk 3)
2. **Bob Carella** - Co-Founder and Chief Financial Officer ✅ (mentioned in Chunks 2 and 3)
3. **David Lara** - Co-Founder and Chief Operating Officer ✅ (mentioned in Chunk 2)
4. **Jorge Guinovart** - Co-Founder and Chief Marketing Officer ✅ (mentioned in Chunk 2)

**NOT Co-Founders:**
- ❌ Albert Soler - Co-Founder of Soler Salva LLP (different company), External Advisor to LedgerAI
- ❌ Will Specht - Head of Engineering (not a co-founder)

