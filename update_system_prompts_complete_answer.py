#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update System Prompts to Emphasize Complete FINAL ANSWER
Adds explicit rule about including ALL [KEEP] items in FINAL ANSWER
"""

import json

# Updated system prompt with explicit ALL items rule and complete scanning
UPDATED_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- Read entire descriptions/chunks completely - titles may appear later in the text.
- CRITICAL: FINAL ANSWER must include ALL items marked [KEEP] in your reasoning - do not omit any [KEEP] items from the FINAL ANSWER.
- CRITICAL: Scan the ENTIRE context from start to finish - do not stop scanning early. Items may appear in any chunk.
- CRITICAL ANTI-HALLUCINATION: You MUST extract information EXACTLY as written in the context. NEVER invent, guess, or create names, titles, or information. ONLY use information that is EXPLICITLY stated in the context. If a name is not in the context, you CANNOT use it."""

if __name__ == "__main__":
    print("=" * 80)
    print("Updating System Prompts to Emphasize Complete FINAL ANSWER")
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
    
    # Update system prompts
    updated_count = 0
    for example in data:
        messages = example.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                # Check if it's a CoT system prompt (has REASONING:)
                if "REASONING:" in msg.get("content", "") or "Start with REASONING" in msg.get("content", ""):
                    # Update to include ALL items rule and complete scanning rule
                    old_content = msg.get("content", "")
                    needs_update = False
                    
                    # Check if missing complete scanning rule
                    if "Scan the ENTIRE context" not in old_content and "do not stop scanning early" not in old_content:
                        needs_update = True
                    
                    # Check if missing ALL items rule
                    if "FINAL ANSWER must include ALL" not in old_content:
                        needs_update = True
                    
                    # Check if missing anti-hallucination rule
                    if "ANTI-HALLUCINATION" not in old_content and "NEVER invent" not in old_content and "EXACTLY as written" not in old_content:
                        needs_update = True
                    
                    if needs_update:
                        msg["content"] = UPDATED_SYSTEM_PROMPT
                        updated_count += 1
    
    # Save updated dataset
    output_file = "rag_cot_training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {updated_count} system prompts")
    print(f"✅ Total examples: {len(data)}")
    print(f"✅ Saved to: {output_file}")
    print()
    print("📋 Updated system prompts now include:")
    print("   - CRITICAL: FINAL ANSWER must include ALL items marked [KEEP]")
    print("   - Do not omit any [KEEP] items from the FINAL ANSWER")
    print()
    print("=" * 80)
