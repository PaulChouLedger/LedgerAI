#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Training Examples Emphasizing Complete Context Scanning
Addresses issue where model stops scanning early and misses items in middle/end chunks
"""

import json
import random

# System prompt (same as training)
COT_SYSTEM_PROMPT = """You are a precise data extraction bot.
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
- CRITICAL: FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.
- CRITICAL: Scan the ENTIRE context from start to finish - do not stop scanning early. Items may appear in any chunk."""

# Examples where co-founders appear in different positions across chunks
COMPLETE_SCAN_EXAMPLES = [
    {
        "context": """TechFlow Systems is a technology company. John Smith is the CEO and Co-Founder of TechFlow Systems. Sarah Johnson is the Chief Technology Officer at TechFlow Systems.
---
Michael Brown is the Co-Founder and Chief Operating Officer of TechFlow Systems. Emily Davis is the Head of Marketing. Robert Wilson is an External Advisor.
---
David Martinez is the Co-Founder and Chief Product Officer of TechFlow Systems. Lisa Anderson is the Chief Financial Officer.""",
        "query": "Who are the co-founders of TechFlow Systems?",
        "reasoning": """REASONING:
- Item: John Smith
- Evidence: "CEO and Co-Founder of TechFlow Systems"
- Action: [KEEP]
- Item: Sarah Johnson
- Evidence: "Chief Technology Officer at TechFlow Systems"
- Action: [DISCARD] (Reason: CTO, not co-founder).
- Item: Michael Brown
- Evidence: "Co-Founder and Chief Operating Officer of TechFlow Systems"
- Action: [KEEP]
- Item: Emily Davis
- Evidence: "Head of Marketing"
- Action: [DISCARD] (Reason: Head of Marketing, not co-founder).
- Item: Robert Wilson
- Evidence: "External Advisor"
- Action: [DISCARD] (Reason: External Advisor, not co-founder).
- Item: David Martinez
- Evidence: "Co-Founder and Chief Product Officer of TechFlow Systems"
- Action: [KEEP]
- Item: Lisa Anderson
- Evidence: "Chief Financial Officer"
- Action: [DISCARD] (Reason: CFO, not co-founder).
- End of scan.""",
        "answer": "The co-founders of TechFlow Systems are John Smith, Michael Brown, and David Martinez."
    },
    {
        "context": """CloudScale Technologies is a cloud infrastructure company. Paul Chen is the CEO and Co-Founder of CloudScale Technologies.
---
Bob Martinez is the Co-Founder and Chief Financial Officer of CloudScale Technologies. David Kim is the Co-Founder and Chief Operating Officer of CloudScale Technologies. Will Johnson is the Head of Engineering.
---
Jorge Rodriguez is the Co-Founder and Chief Marketing Officer of CloudScale Technologies. Albert Smith is the External Counsel.""",
        "query": "Who are the co-founders of CloudScale Technologies?",
        "reasoning": """REASONING:
- Item: Paul Chen
- Evidence: "CEO and Co-Founder of CloudScale Technologies"
- Action: [KEEP]
- Item: Bob Martinez
- Evidence: "Co-Founder and Chief Financial Officer of CloudScale Technologies"
- Action: [KEEP]
- Item: David Kim
- Evidence: "Co-Founder and Chief Operating Officer of CloudScale Technologies"
- Action: [KEEP]
- Item: Will Johnson
- Evidence: "Head of Engineering"
- Action: [DISCARD] (Reason: Head of Engineering, not co-founder).
- Item: Jorge Rodriguez
- Evidence: "Co-Founder and Chief Marketing Officer of CloudScale Technologies"
- Action: [KEEP]
- Item: Albert Smith
- Evidence: "External Counsel"
- Action: [DISCARD] (Reason: External Counsel, not co-founder).
- End of scan.""",
        "answer": "The co-founders of CloudScale Technologies are Paul Chen, Bob Martinez, David Kim, and Jorge Rodriguez."
    },
    {
        "context": """DataFlow Systems is a data analytics company. Michael Thompson is the Co-Founder and Chief Technology Officer of DataFlow Systems.
---
Jennifer Park is the Co-Founder and Chief Financial Officer of DataFlow Systems. Robert Lee is the CEO and Co-Founder of DataFlow Systems. Susan White is the Head of Product.
---
James Wilson is an External Advisor.""",
        "query": "Who are the co-founders of DataFlow Systems?",
        "reasoning": """REASONING:
- Item: Michael Thompson
- Evidence: "Co-Founder and Chief Technology Officer of DataFlow Systems"
- Action: [KEEP]
- Item: Jennifer Park
- Evidence: "Co-Founder and Chief Financial Officer of DataFlow Systems"
- Action: [KEEP]
- Item: Robert Lee
- Evidence: "CEO and Co-Founder of DataFlow Systems"
- Action: [KEEP]
- Item: Susan White
- Evidence: "Head of Product"
- Action: [DISCARD] (Reason: Head of Product, not co-founder).
- Item: James Wilson
- Evidence: "External Advisor"
- Action: [DISCARD] (Reason: External Advisor, not co-founder).
- End of scan.""",
        "answer": "The co-founders of DataFlow Systems are Michael Thompson, Jennifer Park, and Robert Lee."
    },
    {
        "context": """InnovateAI Solutions is an AI company. James Wilson is the Co-Founder and Chief Technology Officer of InnovateAI Solutions.
---
Patricia Martinez is the Co-Founder and Chief Financial Officer of InnovateAI Solutions. Amanda White is the Head of Product.
---
Daniel Lee is the CEO and Co-Founder of InnovateAI Solutions. Kevin Brown is an External Advisor.""",
        "query": "Who are the co-founders of InnovateAI Solutions?",
        "reasoning": """REASONING:
- Item: James Wilson
- Evidence: "Co-Founder and Chief Technology Officer of InnovateAI Solutions"
- Action: [KEEP]
- Item: Patricia Martinez
- Evidence: "Co-Founder and Chief Financial Officer of InnovateAI Solutions"
- Action: [KEEP]
- Item: Amanda White
- Evidence: "Head of Product"
- Action: [DISCARD] (Reason: Head of Product, not co-founder).
- Item: Daniel Lee
- Evidence: "CEO and Co-Founder of InnovateAI Solutions"
- Action: [KEEP]
- Item: Kevin Brown
- Evidence: "External Advisor"
- Action: [DISCARD] (Reason: External Advisor, not co-founder).
- End of scan.""",
        "answer": "The co-founders of InnovateAI Solutions are James Wilson, Patricia Martinez, and Daniel Lee."
    },
    {
        "context": """QuantumTech is a quantum computing company. Alex Chen is the CEO and Co-Founder of QuantumTech. Emily Davis is the Head of Marketing.
---
Lisa Wang is the Co-Founder and Chief Financial Officer of QuantumTech. Tom Brown is the Co-Founder and Chief Operating Officer of QuantumTech.
---
Mark Johnson is an External Advisor.""",
        "query": "Who are the co-founders of QuantumTech?",
        "reasoning": """REASONING:
- Item: Alex Chen
- Evidence: "CEO and Co-Founder of QuantumTech"
- Action: [KEEP]
- Item: Emily Davis
- Evidence: "Head of Marketing"
- Action: [DISCARD] (Reason: Head of Marketing, not co-founder).
- Item: Lisa Wang
- Evidence: "Co-Founder and Chief Financial Officer of QuantumTech"
- Action: [KEEP]
- Item: Tom Brown
- Evidence: "Co-Founder and Chief Operating Officer of QuantumTech"
- Action: [KEEP]
- Item: Mark Johnson
- Evidence: "External Advisor"
- Action: [DISCARD] (Reason: External Advisor, not co-founder).
- End of scan.""",
        "answer": "The co-founders of QuantumTech are Alex Chen, Lisa Wang, and Tom Brown."
    },
    {
        "context": """LedgerAI Quantum Corporation is an AI company. Albert Soler is a top-tier legal strategist and advisor, bringing unparalleled expertise in litigation, intellectual property, and business law to LedgerAI as External Counsel & Advisor. Peter Moeller is a dynamic leader in business development, serving as Business Development Lead at LedgerAI.
---
Payroll & Stock Administration at Binance.US and Sprinklr, Bob managed multi-billion-dollar payroll and equity programs. A passionate educator, he serves as an Adjunct Professor at Drew University. David Lara is a strategic powerhouse in AI-driven governance, fintech, and large-scale financial management. As Co-Founder and Chief Operating Officer of LedgerAI, he leads the execution of AI-powered intelligence solutions. Jorge Guinovart is a visionary leader at the intersection of AI, blockchain, and decentralized finance. As Co-Founder and Chief Marketing Officer of LedgerAI, he is spearheading global adoption, brand strategy, and market expansion.
---
Paul Chou is a renowned leader in AI, blockchain, and institutional finance. As CEO and Co-Founder of LedgerAI, he is driving the development of AI-powered business intelligence. Bob Carella is a driving force in finance, blockchain, and enterprise strategy. As Co-Founder and Chief Financial Officer of LedgerAI, he architects the company's financial strategy.""",
        "query": "Who are the co-founders of LedgerAI?",
        "reasoning": """REASONING:
- Item: Albert Soler
- Evidence: "External Counsel & Advisor"
- Action: [DISCARD] (Reason: External Counsel, not co-founder).
- Item: Peter Moeller
- Evidence: "Business Development Lead at LedgerAI"
- Action: [DISCARD] (Reason: Business Development Lead, not co-founder).
- Item: David Lara
- Evidence: "As Co-Founder and Chief Operating Officer of LedgerAI"
- Action: [KEEP]
- Item: Jorge Guinovart
- Evidence: "As Co-Founder and Chief Marketing Officer of LedgerAI"
- Action: [KEEP]
- Item: Paul Chou
- Evidence: "As CEO and Co-Founder of LedgerAI"
- Action: [KEEP]
- Item: Bob Carella
- Evidence: "As Co-Founder and Chief Financial Officer of LedgerAI"
- Action: [KEEP]
- End of scan.""",
        "answer": "The co-founders of LedgerAI are David Lara, Jorge Guinovart, Paul Chou, and Bob Carella."
    },
]

def create_training_examples():
    """Create training examples emphasizing complete context scanning"""
    examples = []
    
    for scenario in COMPLETE_SCAN_EXAMPLES:
        user_content = f"Knowledge context: {scenario['context']}\n---\nQuestion: {scenario['query']}"
        assistant_content = f"{scenario['reasoning']}\n\nFINAL ANSWER:\n{scenario['answer']}"
        
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": COT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        })
    
    return examples

if __name__ == "__main__":
    print("=" * 80)
    print("Adding Complete Context Scanning Examples")
    print("=" * 80)
    print()
    
    # Load existing dataset
    try:
        with open("rag_cot_training_dataset.json", 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"✅ Loaded {len(existing_data)} existing examples")
    except FileNotFoundError:
        print("❌ Error: rag_cot_training_dataset.json not found!")
        exit(1)
    
    # Create new examples
    new_examples = create_training_examples()
    print(f"✅ Created {len(new_examples)} new examples emphasizing complete context scanning")
    print()
    
    # Add to existing dataset
    existing_data.extend(new_examples)
    
    # Shuffle to mix examples
    random.shuffle(existing_data)
    
    # Save updated dataset
    output_file = "rag_cot_training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added {len(new_examples)} examples to dataset")
    print(f"✅ Total examples: {len(existing_data)}")
    print(f"✅ Saved to: {output_file}")
    print()
    print("📋 New examples emphasize:")
    print("   - Scanning ENTIRE context from start to finish")
    print("   - Co-founders appearing in different chunks (start, middle, end)")
    print("   - Not stopping early - items may appear in any chunk")
    print("   - Complete multi-chunk context handling")
    print()
    print("=" * 80)
