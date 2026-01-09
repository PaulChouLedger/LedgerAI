#!/usr/bin/env python3
"""
Strengthen CoT RAG logic in training dataset.

Focus areas:
1. Explicit co-founder identification rules
2. FINAL ANSWER completeness (must include ALL [KEEP] items)
3. DISCARD enforcement (items marked [DISCARD] must NEVER appear in FINAL ANSWER)
4. Evidence extraction accuracy (extract exact quote that shows role)
"""

import json
import re

# Enhanced system prompt with explicit co-founder rules
ENHANCED_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER - this is ABSOLUTE and MANDATORY.
- FINAL ANSWER must ONLY include items marked [KEEP] - no exceptions.
- If you mark an item [DISCARD] in reasoning, it is FORBIDDEN to mention it in FINAL ANSWER.
- Before writing FINAL ANSWER, verify: ONLY include items that were marked [KEEP] in REASONING.
- Read entire descriptions/chunks completely - titles may appear later in the text.
- SCAN ALL CHUNKS COMPLETELY - read each chunk from start to finish before moving to next.
- IMPORTANT: Identify ALL relevant items in REASONING before making any decisions. Do NOT stop after finding some matches - scan completely and list EVERY item that could be relevant.
- Extract Evidence from the EXACT ITEM MENTION that matches the query - find where the item is explicitly mentioned in relation to what you're looking for, not from descriptions or unrelated text.
- IMPORTANT: Do NOT include headers (like 'AURA VISION...'), page numbers, or metadata in Evidence. Extract ONLY the exact quote that shows the role or information.

CO-FOUNDER IDENTIFICATION RULES (CRITICAL):
- If evidence contains "Co-Founder" OR "co-founder" OR "Co-Founder and [Role]", the person IS a co-founder.
- If evidence says "As Co-Founder and Chief Operating Officer", this person IS a co-founder - mark [KEEP].
- If evidence says "As Co-Founder and [Any Role]", this person IS a co-founder - mark [KEEP].
- If evidence says "CEO and Co-Founder", this person IS a co-founder - mark [KEEP].
- If evidence does NOT mention "Co-Founder" or "co-founder", the person is NOT a co-founder (unless explicitly stated as "founder").
- Example: "As Co-Founder and Chief Operating Officer" → [KEEP] (co-founder confirmed).
- Example: "serves as Chief Operating Officer" → [DISCARD] (no co-founder mention).

FINAL ANSWER COMPLETENESS (CRITICAL):
- FINAL ANSWER must include EVERY item marked [KEEP] in REASONING.
- Count how many items you marked [KEEP] - ALL of them must appear in FINAL ANSWER.
- If you marked 4 items [KEEP], FINAL ANSWER must include all 4 items.
- Before writing FINAL ANSWER, list all [KEEP] items and ensure each one is included.
- If an item is marked [KEEP] in REASONING but missing from FINAL ANSWER, your answer is incomplete.
"""

def update_system_prompts(dataset):
    """Update all system prompts in the dataset with enhanced rules."""
    updated = 0
    for example in dataset:
        if "messages" in example and len(example["messages"]) > 0:
            if example["messages"][0].get("role") == "system":
                example["messages"][0]["content"] = ENHANCED_SYSTEM_PROMPT
                updated += 1
    return updated

def add_explicit_cofounder_examples(dataset):
    """Add explicit examples showing co-founder identification rules."""
    
    new_examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": ENHANCED_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": "Knowledge context: David Lara is a strategic leader in AI and fintech. As Co-Founder and Chief Operating Officer of LedgerAI, he leads operations. He is also the CEO of Petra Capital. John Smith serves as the Chief Technology Officer at LedgerAI, bringing deep technical expertise. Michael Brown is an operations manager at LedgerAI, responsible for day-to-day operations.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: David Lara\n  - Evidence: \"As Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: John Smith\n  - Evidence: \"serves as the Chief Technology Officer at LedgerAI\"\n  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only CTO role).\n- Item: Michael Brown\n  - Evidence: \"is an operations manager at LedgerAI\"\n  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only operations manager role).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founder of LedgerAI is David Lara."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": ENHANCED_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Paul Chou is a renowned leader. As CEO and Co-Founder of LedgerAI, he drives the company vision. Bob Carella is the CFO and Co-Founder of LedgerAI, managing financial strategy. David Lara is the COO and Co-Founder of LedgerAI, overseeing operations. Jorge Guinovart is the CMO and Co-Founder of LedgerAI, leading marketing. Will Specht serves as Head of Engineering at LedgerAI, leading technical development.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Paul Chou\n  - Evidence: \"As CEO and Co-Founder of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: Bob Carella\n  - Evidence: \"CFO and Co-Founder of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: David Lara\n  - Evidence: \"COO and Co-Founder of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: Jorge Guinovart\n  - Evidence: \"CMO and Co-Founder of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: Will Specht\n  - Evidence: \"serves as Head of Engineering at LedgerAI\"\n  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only Head of Engineering role).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of LedgerAI are Paul Chou, Bob Carella, David Lara, and Jorge Guinovart."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": ENHANCED_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": "Knowledge context: John Smith is an experienced executive. As CEO and Co-Founder of TechCorp, he leads the company. Sarah Johnson serves as the Chief Technology Officer at TechCorp, bringing technical expertise. Michael Brown is an operations leader. As Co-Founder and Chief Operating Officer of TechCorp, he manages operations. Emily Davis serves as the Head of Marketing at TechCorp, driving brand awareness. David Martinez is a product leader. As Co-Founder and Chief Product Officer of TechCorp, he leads product development. Lisa Anderson serves as the Chief Financial Officer at TechCorp, managing finances.\n---\nQuestion: Who are the co-founders of TechCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: John Smith\n  - Evidence: \"As CEO and Co-Founder of TechCorp\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: Sarah Johnson\n  - Evidence: \"serves as the Chief Technology Officer at TechCorp\"\n  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only CTO role).\n- Item: Michael Brown\n  - Evidence: \"As Co-Founder and Chief Operating Officer of TechCorp\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: Emily Davis\n  - Evidence: \"serves as the Head of Marketing at TechCorp\"\n  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only Head of Marketing role).\n- Item: David Martinez\n  - Evidence: \"As Co-Founder and Chief Product Officer of TechCorp\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: Lisa Anderson\n  - Evidence: \"serves as the Chief Financial Officer at TechCorp\"\n  - Action: [DISCARD] (Reason: Evidence does not mention 'Co-Founder', only CFO role).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechCorp are John Smith, Michael Brown, and David Martinez."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": ENHANCED_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": "Knowledge context: AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 23 David Lara is a strategic powerhouse. As Co-Founder and Chief Operating Officer of LedgerAI, he leads operations. He is also CEO of Petra Capital. AURA VISION AND THE FUTURE OF AI-DRIVEN SOLUTIONS 24 Paul Chou is a renowned leader. As CEO and Co-Founder of LedgerAI, he drives vision. Bob Carella is a financial expert. As Co-Founder and Chief Financial Officer of LedgerAI, he manages finances. Jorge Guinovart is a marketing leader. As Co-Founder and Chief Marketing Officer of LedgerAI, he leads marketing.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: David Lara\n  - Evidence: \"As Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder. Ignore header 'AURA VISION...').\n- Item: Paul Chou\n  - Evidence: \"As CEO and Co-Founder of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder. Ignore header 'AURA VISION...').\n- Item: Bob Carella\n  - Evidence: \"As Co-Founder and Chief Financial Officer of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- Item: Jorge Guinovart\n  - Evidence: \"As Co-Founder and Chief Marketing Officer of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence contains 'Co-Founder', so this person IS a co-founder).\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of LedgerAI are David Lara, Paul Chou, Bob Carella, and Jorge Guinovart."
                }
            ]
        }
    ]
    
    # Add new examples to the beginning of the dataset (high priority)
    dataset[:0] = new_examples
    return len(new_examples)

def main():
    """Main function to strengthen CoT RAG logic."""
    print("=" * 80)
    print("Strengthening CoT RAG Logic in Training Dataset")
    print("=" * 80)
    print()
    
    # Load dataset
    input_file = "rag_cot_training_dataset.json"
    print(f"📂 Loading dataset: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"   Found {len(dataset)} examples")
    print()
    
    # Update system prompts
    print("🔧 Updating system prompts with enhanced rules...")
    updated = update_system_prompts(dataset)
    print(f"   Updated {updated} system prompts")
    print()
    
    # Add explicit examples
    print("➕ Adding explicit co-founder identification examples...")
    added = add_explicit_cofounder_examples(dataset)
    print(f"   Added {added} new examples")
    print()
    
    # Save updated dataset
    output_file = "rag_cot_training_dataset.json"
    print(f"💾 Saving updated dataset: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Dataset updated: {len(dataset)} total examples")
    print()
    
    print("=" * 80)
    print("✅ Strengthening Complete!")
    print("=" * 80)
    print()
    print("Key improvements:")
    print("  ✓ Enhanced system prompt with explicit co-founder rules")
    print("  ✓ Added examples showing 'Co-Founder' in evidence → [KEEP]")
    print("  ✓ Added examples showing FINAL ANSWER completeness")
    print("  ✓ Added examples showing DISCARD enforcement")
    print("  ✓ Added examples with headers/metadata to test extraction accuracy")
    print()

if __name__ == "__main__":
    main()
