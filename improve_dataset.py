#!/usr/bin/env python3
"""
Improve RAG CoT training dataset to address:
1. DISCARD enforcement: more explicit examples
2. Query intent: more benefits vs drawbacks examples
3. KEEP/DISCARD logic: more diverse examples
4. Reasoning format: ensure consistency
"""

import json

# Get system prompt from first example
def get_system_prompt():
    return """You are a data extraction bot. Extract items from context based on the query.

STEP 1: Start with REASONING:
STEP 2: For EACH item found, write:
   - Item: [name or thing]
   - Evidence: "[exact quote]"
   - Action: [KEEP] or [DISCARD]
STEP 3: Write "End of scan."
STEP 4: Write FINAL ANSWER using ONLY [KEEP] items.

CRITICAL RULES - DO NOT VIOLATE:

RULE 1 - COMPLETE SCANNING:
Scan EVERY chunk from start to finish.
Do NOT stop after finding one match.
You must scan ALL chunks completely.

RULE 2 - DISCARD ITEMS (MOST IMPORTANT):
If you write [DISCARD] for an item, that item MUST NOT appear in FINAL ANSWER.
[DISCARD] items are FORBIDDEN in FINAL ANSWER.
Never write a [DISCARD] item in FINAL ANSWER.

RULE 3 - KEEP ITEMS:
If you write [KEEP] for an item, that item MUST appear in FINAL ANSWER.
Count how many [KEEP] items you have.
Include ALL [KEEP] items in FINAL ANSWER.

RULE 4 - QUERY MATCHING:
Read the query word by word to understand what is being asked.
Extract ONLY items that match what the query asks for.
If the query asks for X, extract only X. Do NOT extract Y if query asks for X.
Opposites or different categories should be marked [DISCARD].

RULE 5 - MULTIPLE ATTRIBUTES:
Some items can have multiple attributes or roles.
Read the ENTIRE description completely before deciding.
If the item has the attribute that matches the query, mark [KEEP].
If the item does NOT have the attribute that matches the query, mark [DISCARD]."""

def create_example(query, context, reasoning, answer):
    """Create a training example."""
    system_prompt = get_system_prompt()
    user_content = f"Knowledge context: {context}\n---\nQuestion: {query}"
    assistant_content = f"{reasoning}\n\nFINAL ANSWER:\n{answer}"
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

# Load dataset
with open('rag_cot_training_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print("=" * 80)
print("IMPROVING DATASET")
print("=" * 80)
print()

new_examples = []

# 1. DISCARD ENFORCEMENT EXAMPLES - Explicit examples showing DISCARD items must NOT appear in FINAL ANSWER
print("1. Adding DISCARD enforcement examples...")

discard_enforcement_examples = [
    # Example: Co-founder query with non-founders explicitly marked DISCARD
    create_example(
        "Who are the co-founders of DataSystems?",
        "DataSystems has a strong leadership team. Sarah Johnson is the Chief Technology Officer at DataSystems. Mark Williams is the Head of Engineering. Robert Kim is the Chief Financial Officer.",
        """REASONING:
- Item: Sarah Johnson
- Evidence: "Sarah Johnson is the Chief Technology Officer at DataSystems"
- Action: [DISCARD] (Reason: CTO, not co-founder)
- Item: Mark Williams
- Evidence: "Mark Williams is the Head of Engineering"
- Action: [DISCARD] (Reason: Head of Engineering, not co-founder)
- Item: Robert Kim
- Evidence: "Robert Kim is the Chief Financial Officer"
- Action: [DISCARD] (Reason: CFO, not co-founder)
- End of scan.""",
        "No co-founders of DataSystems are mentioned in the context."
    ),
    
    # Example: Benefits query with drawbacks explicitly marked DISCARD
    create_example(
        "What are the benefits of cloud computing?",
        "Cloud computing offers several advantages. Scalability allows businesses to scale resources on demand. Cost efficiency reduces infrastructure expenses. However, cloud computing has limitations. Security concerns arise from data being stored remotely. Network dependency can cause downtime if internet connectivity is lost.",
        """REASONING:
- Item: Scalability
- Evidence: "Scalability allows businesses to scale resources on demand"
- Action: [KEEP] (Reason: This is a benefit of cloud computing)
- Item: Cost efficiency
- Evidence: "Cost efficiency reduces infrastructure expenses"
- Action: [KEEP] (Reason: This is a benefit of cloud computing)
- Item: Security concerns
- Evidence: "Security concerns arise from data being stored remotely"
- Action: [DISCARD] (Reason: This is a drawback/limitation, not a benefit)
- Item: Network dependency
- Evidence: "Network dependency can cause downtime if internet connectivity is lost"
- Action: [DISCARD] (Reason: This is a drawback/limitation, not a benefit)
- End of scan.""",
        "The benefits of cloud computing include scalability (allowing businesses to scale resources on demand) and cost efficiency (reducing infrastructure expenses)."
    ),
    
    # Example: Products query with services explicitly marked DISCARD
    create_example(
        "What products does TechCorp offer?",
        "TechCorp offers several solutions. Their main products include TechCorp Analytics and TechCorp Insights. The company also provides consulting services for enterprise clients. TechCorp Support is available for technical assistance.",
        """REASONING:
- Item: TechCorp Analytics
- Evidence: "Their main products include TechCorp Analytics"
- Action: [KEEP] (Reason: This is a product)
- Item: TechCorp Insights
- Evidence: "Their main products include TechCorp Insights"
- Action: [KEEP] (Reason: This is a product)
- Item: Consulting services
- Evidence: "The company also provides consulting services for enterprise clients"
- Action: [DISCARD] (Reason: Service, not product)
- Item: TechCorp Support
- Evidence: "TechCorp Support is available for technical assistance"
- Action: [DISCARD] (Reason: Service, not product)
- End of scan.""",
        "TechCorp offers TechCorp Analytics and TechCorp Insights."
    ),
]

new_examples.extend(discard_enforcement_examples)
print(f"   Added {len(discard_enforcement_examples)} DISCARD enforcement examples")

# 2. BENEFITS VS DRAWBACKS EXAMPLES - More explicit examples
print()
print("2. Adding benefits vs drawbacks examples...")

benefits_examples = [
    # Example: Benefits of remote work
    create_example(
        "What are the benefits of remote work?",
        "Remote work offers numerous advantages. Flexibility allows employees to work from anywhere. Work-life balance improves when commuting is eliminated. However, remote work has challenges. Isolation can lead to feelings of loneliness. Communication barriers may arise without face-to-face interaction.",
        """REASONING:
- Item: Flexibility
- Evidence: "Flexibility allows employees to work from anywhere"
- Action: [KEEP] (Reason: This is a benefit of remote work)
- Item: Work-life balance
- Evidence: "Work-life balance improves when commuting is eliminated"
- Action: [KEEP] (Reason: This is a benefit of remote work)
- Item: Isolation
- Evidence: "Isolation can lead to feelings of loneliness"
- Action: [DISCARD] (Reason: This is a drawback/challenge, not a benefit)
- Item: Communication barriers
- Evidence: "Communication barriers may arise without face-to-face interaction"
- Action: [DISCARD] (Reason: This is a drawback/challenge, not a benefit)
- End of scan.""",
        "The benefits of remote work include flexibility (allowing employees to work from anywhere) and improved work-life balance (when commuting is eliminated)."
    ),
    
    # Example: Drawbacks of remote work
    create_example(
        "What are the drawbacks of remote work?",
        "Remote work offers numerous advantages. Flexibility allows employees to work from anywhere. Work-life balance improves when commuting is eliminated. However, remote work has challenges. Isolation can lead to feelings of loneliness. Communication barriers may arise without face-to-face interaction.",
        """REASONING:
- Item: Flexibility
- Evidence: "Flexibility allows employees to work from anywhere"
- Action: [DISCARD] (Reason: This is a benefit, not a drawback)
- Item: Work-life balance
- Evidence: "Work-life balance improves when commuting is eliminated"
- Action: [DISCARD] (Reason: This is a benefit, not a drawback)
- Item: Isolation
- Evidence: "Isolation can lead to feelings of loneliness"
- Action: [KEEP] (Reason: This is a drawback/challenge of remote work)
- Item: Communication barriers
- Evidence: "Communication barriers may arise without face-to-face interaction"
- Action: [KEEP] (Reason: This is a drawback/challenge of remote work)
- End of scan.""",
        "The drawbacks of remote work include isolation (which can lead to feelings of loneliness) and communication barriers (which may arise without face-to-face interaction)."
    ),
]

new_examples.extend(benefits_examples)
print(f"   Added {len(benefits_examples)} benefits vs drawbacks examples")

# 3. KEEP/DISCARD LOGIC EXAMPLES - More diverse examples for co-founders
print()
print("3. Adding KEEP/DISCARD logic examples...")

keep_discard_examples = [
    # Example: Role query (CFO) with non-CFO roles marked DISCARD
    create_example(
        "Who is the CFO of InnovateCorp?",
        "InnovateCorp has a diverse leadership team. John Smith is the CEO and Co-Founder of InnovateCorp. Mary Johnson is the Chief Financial Officer of InnovateCorp. David Brown is the Chief Technology Officer. Lisa Wilson is the Chief Operating Officer.",
        """REASONING:
- Item: John Smith
- Evidence: "John Smith is the CEO and Co-Founder of InnovateCorp"
- Action: [DISCARD] (Reason: CEO, not CFO)
- Item: Mary Johnson
- Evidence: "Mary Johnson is the Chief Financial Officer of InnovateCorp"
- Action: [KEEP] (Reason: This person is the CFO)
- Item: David Brown
- Evidence: "David Brown is the Chief Technology Officer"
- Action: [DISCARD] (Reason: CTO, not CFO)
- Item: Lisa Wilson
- Evidence: "Lisa Wilson is the Chief Operating Officer"
- Action: [DISCARD] (Reason: COO, not CFO)
- End of scan.""",
        "The CFO of InnovateCorp is Mary Johnson."
    ),
    
    # Example: Co-founder query with compound role (CEO and Co-Founder)
    create_example(
        "Who are the co-founders of StartupCorp?",
        "StartupCorp was founded by a team of entrepreneurs. Alice Chen is the CEO and Co-Founder of StartupCorp. Bob Miller is the CTO and Co-Founder of StartupCorp. Carol Davis is the Head of Marketing. Dan Wilson is the CFO.",
        """REASONING:
- Item: Alice Chen
- Evidence: "Alice Chen is the CEO and Co-Founder of StartupCorp"
- Action: [KEEP] (Reason: Co-Founder, even though also CEO)
- Item: Bob Miller
- Evidence: "Bob Miller is the CTO and Co-Founder of StartupCorp"
- Action: [KEEP] (Reason: Co-Founder, even though also CTO)
- Item: Carol Davis
- Evidence: "Carol Davis is the Head of Marketing"
- Action: [DISCARD] (Reason: Head of Marketing, not co-founder)
- Item: Dan Wilson
- Evidence: "Dan Wilson is the CFO"
- Action: [DISCARD] (Reason: CFO, not co-founder)
- End of scan.""",
        "The co-founders of StartupCorp are Alice Chen and Bob Miller."
    ),
    
    # Example: Co-founder query where person has co-founder role at different company
    create_example(
        "Who are the co-founders of TechVentures?",
        "TechVentures has a strong executive team. Emma Thompson is the CEO of TechVentures. She previously co-founded InnovateLabs. Frank Rodriguez is the CTO of TechVentures. Grace Lee is the COO of TechVentures.",
        """REASONING:
- Item: Emma Thompson
- Evidence: "Emma Thompson is the CEO of TechVentures. She previously co-founded InnovateLabs"
- Action: [DISCARD] (Reason: Co-founder of InnovateLabs, not TechVentures)
- Item: Frank Rodriguez
- Evidence: "Frank Rodriguez is the CTO of TechVentures"
- Action: [DISCARD] (Reason: CTO, not co-founder)
- Item: Grace Lee
- Evidence: "Grace Lee is the COO of TechVentures"
- Action: [DISCARD] (Reason: COO, not co-founder)
- End of scan.""",
        "No co-founders of TechVentures are mentioned in the context."
    ),
]

new_examples.extend(keep_discard_examples)
print(f"   Added {len(keep_discard_examples)} KEEP/DISCARD logic examples")

# Add new examples to the beginning of the dataset (high priority)
print()
print(f"4. Adding {len(new_examples)} new examples to dataset...")
dataset[:0] = new_examples

# Save updated dataset
print()
print("5. Saving updated dataset...")
with open('rag_cot_training_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print()
print("=" * 80)
print("DATASET IMPROVEMENT COMPLETE")
print("=" * 80)
print()
print(f"✅ Added {len(new_examples)} new examples")
print(f"✅ Total examples: {len(dataset)}")
print()
print("Improvements made:")
print("  1. DISCARD enforcement: Added explicit examples showing DISCARD items must NOT appear in FINAL ANSWER")
print("  2. Benefits vs drawbacks: Added explicit examples with drawbacks marked [DISCARD]")
print("  3. KEEP/DISCARD logic: Added diverse examples for role queries and compound roles")
print("  4. Reasoning format: All new examples use consistent 'Item:', 'Evidence:', 'Action:' format")
print()
print(f"New examples added at indices 0-{len(new_examples)-1} (high priority for training)")
