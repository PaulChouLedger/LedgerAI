# Quick Training Checklist
## Step-by-Step Guide for Next Training Session

## ✅ Pre-Training Checklist

- [ ] **Run baseline diagnostics on current model**
  ```bash
  python3 comprehensive_model_diagnostics.py
  ```
  - Document: Multi-entity accuracy = __%
  - Document: Answer type accuracy = __%
  - Document: Role filtering accuracy = __%

- [ ] **Regenerate dataset with fixes**
  ```bash
  python3 generate_rag_dataset_v3_json.py
  ```
  - Verify: 2500+ multi_chunk examples
  - Verify: System prompt has "extract ALL" examples
  - Verify: System prompt has answer_type mapping

- [ ] **Verify training script config**
  - LoRA rank = 8 ✅
  - System prompt includes answer_type mapping ✅

---

## 🚀 Training Execution

- [ ] **Start training**
  ```bash
  python3 train_rag_analysis_colab.py
  ```

- [ ] **Monitor during training**
  - Watch for extraction completeness in logs
  - Check if loss decreases
  - Note any persistent issues

---

## 🧪 Post-Training Testing

- [ ] **Run comprehensive diagnostics**
  ```bash
  python3 comprehensive_model_diagnostics.py
  ```

- [ ] **Test real-world case**
  ```bash
  python3 debug_multi_entity_extraction.py
  ```
  - Query: "who are the co-founders of LedgerAI?"
  - Expected: 4 co-founders
  - Target: Extract all 4

- [ ] **Compare with baseline**
  - Multi-entity: __% (target: >90%)
  - Answer type: __% (target: >85%)
  - Role filtering: __% (target: >95%)

---

## 📊 Success Criteria

✅ **PASS if:**
- Multi-entity extraction >90%
- Answer type classification >85%
- Role filtering >95%

❌ **FAIL if:**
- Any metric below targets
- Model still stops after first few entities
- Model still defaults to "comparison"

---

## 🔄 If Targets Not Met

1. **Analyze failures:**
   - Which test cases fail?
   - What's the pattern?

2. **Apply additional fixes:**
   - Increase LoRA rank to 10
   - Add class weighting
   - Increase epochs to 10
   - Add more multi-entity examples (3000+)

3. **Re-train and test again**

---

## 📝 Quick Reference

**Key Files:**
- `generate_rag_dataset_v3_json.py` - Dataset generator (updated)
- `train_rag_analysis_colab.py` - Training script (already optimized)
- `comprehensive_model_diagnostics.py` - Full test suite
- `debug_multi_entity_extraction.py` - Single query diagnostic

**Key Changes:**
- Multi-chunk examples: 1500 → 2500
- System prompt: Enhanced with "extract ALL" examples
- System prompt: Added answer_type mapping
- Chunk distribution: Forces multi-chunk reading

**Target Metrics:**
- Multi-entity extraction: >90%
- Answer type classification: >85%
- Role filtering: >95%

