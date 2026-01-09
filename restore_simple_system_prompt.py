#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore Simple System Prompt (Original That Worked)
Uses the simple prompt from train_rag_cot_colab.py that yielded accurate results
"""

import json

# Original simple system prompt that worked (from train_rag_cot_colab.py)
SIMPLE_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items."""

if __name__ == "__main__":
    print("=" * 80)
    print("Restoring Simple System Prompt (Original That Worked)")
    print("=" * 80)
    print()
    
    # Load dataset
    try:
        with open("rag_cot_training_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} examples")
    except FileNotFoundError:
        print("❌ Error: rag_cot_training_dataset.json not found!")
        exit(1)
    
    # Update system prompts to simple version
    updated_count = 0
    for example in data:
        messages = example.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                # Check if it's a CoT system prompt (has REASONING:)
                if "REASONING:" in msg.get("content", "") or "Start with REASONING" in msg.get("content", ""):
                    # Restore to simple prompt
                    msg["content"] = SIMPLE_SYSTEM_PROMPT
                    updated_count += 1
    
    # Save updated dataset
    output_file = "rag_cot_training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {updated_count} system prompts to simple version")
    print(f"✅ Total examples: {len(data)}")
    print(f"✅ Saved to: {output_file}")
    print()
    print("📋 Restored system prompt:")
    print("   - Simple, clear structure (5 steps)")
    print("   - Evidence: '[Verbatim quote from context]' implies exact extraction")
    print("   - No extra CRITICAL rules that might confuse the model")
    print("   - Same as train_rag_cot_colab.py that yielded accurate results")
    print()
    print("=" * 80)
