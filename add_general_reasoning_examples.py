#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add General Reasoning Examples for Various Item Types
Ensures the reasoning pattern generalizes beyond co-founders to other query types
"""

import json
import random

# Simple system prompt (same as training)
SIMPLE_SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items."""

# General Examples for Various Item Types
GENERAL_REASONING_EXAMPLES = [
    {
        "context": """TechFlow Systems is a technology company. John Smith is the CEO and Co-Founder of TechFlow Systems. Sarah Johnson is the Chief Technology Officer at TechFlow Systems. Michael Brown is the Co-Founder and Chief Operating Officer of TechFlow Systems. Emily Davis is the Head of Marketing. Robert Wilson is the Chief Financial Officer at TechFlow Systems. David Martinez is the Co-Founder and Chief Product Officer of TechFlow Systems. Lisa Anderson is an External Advisor.""",
        "query": "Who is the CFO of TechFlow Systems?",
        "reasoning": """REASONING:
- Item: John Smith
- Evidence: "CEO and Co-Founder of TechFlow Systems"
- Action: [DISCARD] (Reason: CEO, not CFO).
- Item: Sarah Johnson
- Evidence: "Chief Technology Officer at TechFlow Systems"
- Action: [DISCARD] (Reason: CTO, not CFO).
- Item: Michael Brown
- Evidence: "Co-Founder and Chief Operating Officer of TechFlow Systems"
- Action: [DISCARD] (Reason: COO, not CFO).
- Item: Emily Davis
- Evidence: "Head of Marketing"
- Action: [DISCARD] (Reason: Head of Marketing, not CFO).
- Item: Robert Wilson
- Evidence: "Chief Financial Officer at TechFlow Systems"
- Action: [KEEP]
- Item: David Martinez
- Evidence: "Co-Founder and Chief Product Officer of TechFlow Systems"
- Action: [DISCARD] (Reason: CPO, not CFO).
- Item: Lisa Anderson
- Evidence: "External Advisor"
- Action: [DISCARD] (Reason: External Advisor, not CFO).
- End of scan.""",
        "answer": "The CFO of TechFlow Systems is Robert Wilson."
    },
    {
        "context": """CloudScale Technologies is a cloud infrastructure company. Paul Chen is the CEO and Co-Founder of CloudScale Technologies. Bob Martinez is the Co-Founder and Chief Financial Officer of CloudScale Technologies. David Kim is the Co-Founder and Chief Operating Officer of CloudScale Technologies. Will Johnson is the Chief Technology Officer at CloudScale Technologies. Jorge Rodriguez is the Co-Founder and Chief Marketing Officer of CloudScale Technologies. Albert Smith is the Head of Sales.""",
        "query": "Who is the CTO of CloudScale Technologies?",
        "reasoning": """REASONING:
- Item: Paul Chen
- Evidence: "CEO and Co-Founder of CloudScale Technologies"
- Action: [DISCARD] (Reason: CEO, not CTO).
- Item: Bob Martinez
- Evidence: "Co-Founder and Chief Financial Officer of CloudScale Technologies"
- Action: [DISCARD] (Reason: CFO, not CTO).
- Item: David Kim
- Evidence: "Co-Founder and Chief Operating Officer of CloudScale Technologies"
- Action: [DISCARD] (Reason: COO, not CTO).
- Item: Will Johnson
- Evidence: "Chief Technology Officer at CloudScale Technologies"
- Action: [KEEP]
- Item: Jorge Rodriguez
- Evidence: "Co-Founder and Chief Marketing Officer of CloudScale Technologies"
- Action: [DISCARD] (Reason: CMO, not CTO).
- Item: Albert Smith
- Evidence: "Head of Sales"
- Action: [DISCARD] (Reason: Head of Sales, not CTO).
- End of scan.""",
        "answer": "The CTO of CloudScale Technologies is Will Johnson."
    },
    {
        "context": """DataFlow Analytics was established in 2017 by a team of data scientists and engineers. The company began operations in April 2017 with initial funding of $6 million from venture capital investors. In 2019, DataFlow Analytics expanded its operations and opened a second research facility. The company reached profitability in 2021 and has continued to grow since then. DataFlow Analytics moved to its current headquarters location in 2023, consolidating all operations into a single state-of-the-art facility.""",
        "query": "When was DataFlow Analytics established?",
        "reasoning": """REASONING:
- Item: 2017
- Evidence: "DataFlow Analytics was established in 2017"
- Action: [KEEP]
- Item: April 2017
- Evidence: "began operations in April 2017"
- Action: [DISCARD] (Reason: Operations start date, not establishment year).
- Item: 2019
- Evidence: "In 2019, DataFlow Analytics expanded"
- Action: [DISCARD] (Reason: Expansion date, not establishment).
- Item: 2021
- Evidence: "reached profitability in 2021"
- Action: [DISCARD] (Reason: Profitability date, not establishment).
- Item: 2023
- Evidence: "moved to its current headquarters location in 2023"
- Action: [DISCARD] (Reason: Move date, not establishment).
- End of scan.""",
        "answer": "DataFlow Analytics was established in 2017."
    },
    {
        "context": """InnovateAI Solutions was founded in 2018 by a group of AI researchers and engineers who saw the potential of machine learning technology. The company launched its first product in June 2018 and quickly gained traction in the enterprise market. InnovateAI Solutions received Series A funding in 2020, which allowed it to expand its team and product portfolio. The company opened its second office in 2022 and has continued to grow since then.""",
        "query": "When was InnovateAI Solutions founded?",
        "reasoning": """REASONING:
- Item: 2018
- Evidence: "InnovateAI Solutions was founded in 2018"
- Action: [KEEP]
- Item: June 2018
- Evidence: "launched its first product in June 2018"
- Action: [DISCARD] (Reason: Product launch date, not founding year).
- Item: 2020
- Evidence: "received Series A funding in 2020"
- Action: [DISCARD] (Reason: Funding date, not founding).
- Item: 2022
- Evidence: "opened its second office in 2022"
- Action: [DISCARD] (Reason: Office opening date, not founding).
- End of scan.""",
        "answer": "InnovateAI Solutions was founded in 2018."
    },
    {
        "context": """TechFlow Systems has 120 employees in the engineering team, 50 employees in the sales department, 30 employees in the marketing team, 25 employees in customer support, and 15 employees in the finance department. The company also has 20 employees in operations and 10 employees in human resources. In total, TechFlow Systems employs 270 people across all departments.""",
        "query": "How many employees are in the engineering team at TechFlow Systems?",
        "reasoning": """REASONING:
- Item: 120
- Evidence: "120 employees in the engineering team"
- Action: [KEEP]
- Item: 50
- Evidence: "50 employees in the sales department"
- Action: [DISCARD] (Reason: Sales department, not engineering).
- Item: 30
- Evidence: "30 employees in the marketing team"
- Action: [DISCARD] (Reason: Marketing team, not engineering).
- Item: 25
- Evidence: "25 employees in customer support"
- Action: [DISCARD] (Reason: Customer support, not engineering).
- Item: 15
- Evidence: "15 employees in the finance department"
- Action: [DISCARD] (Reason: Finance department, not engineering).
- Item: 20
- Evidence: "20 employees in operations"
- Action: [DISCARD] (Reason: Operations, not engineering).
- Item: 10
- Evidence: "10 employees in human resources"
- Action: [DISCARD] (Reason: Human resources, not engineering).
- Item: 270
- Evidence: "270 people across all departments"
- Action: [DISCARD] (Reason: Total employees, not engineering team).
- End of scan.""",
        "answer": "The engineering team at TechFlow Systems has 120 employees."
    },
]

def create_training_examples():
    """Create training examples with general reasoning patterns"""
    examples = []
    
    for scenario in GENERAL_REASONING_EXAMPLES:
        user_content = f"Knowledge context: {scenario['context']}\n---\nQuestion: {scenario['query']}"
        assistant_content = f"{scenario['reasoning']}\n\nFINAL ANSWER:\n{scenario['answer']}"
        
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": SIMPLE_SYSTEM_PROMPT
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
    print("Adding General Reasoning Examples for Various Item Types")
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
    
    # Create new examples with general reasoning patterns
    new_examples = create_training_examples()
    print(f"✅ Created {len(new_examples)} new examples with general reasoning patterns")
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
    print("📋 New examples teach general pattern:")
    print("   - Query asks for 'CFO' + Evidence says 'CFO' → Action: [KEEP]")
    print("   - Query asks for 'CTO' + Evidence says 'CTO' → Action: [KEEP]")
    print("   - Query asks for 'established in' + Evidence says 'established in 2017' → Action: [KEEP]")
    print("   - Query asks for 'founded' + Evidence says 'founded in 2018' → Action: [KEEP]")
    print("   - Query asks for 'engineering team' + Evidence says 'engineering team' → Action: [KEEP]")
    print()
    print("   General rule: If query asks for X, and evidence explicitly states X, then KEEP it")
    print()
    print("=" * 80)
