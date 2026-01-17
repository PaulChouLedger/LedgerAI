#!/usr/bin/env python3
"""
Add Training Examples for Strict Matching, FINAL ANSWER Enforcement, and Complete Scanning
===========================================================================================
Addresses three critical issues:
1. Strict matching: Query term MUST appear verbatim in evidence (not inferred)
2. FINAL ANSWER: Must include ALL [KEEP] items, exclude ALL [DISCARD] items
3. Complete scanning: Must scan entire context, especially in complex scenarios
"""

import json
import re

def load_dataset():
    """Load the training dataset."""
    with open('rag_cot_training_dataset_100percent.json', 'r') as f:
        return json.load(f)

def save_dataset(data):
    """Save the training dataset."""
    with open('rag_cot_training_dataset_100percent.json', 'w') as f:
        json.dump(data, f, indent=2)

def get_system_prompt():
    """Get the current system prompt."""
    return """You are a precise data extraction bot.

ALWAYS START WITH REASONING:
Begin every response with "REASONING:" - this is MANDATORY.

1. REASONING: For each relevant item found in the context:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].

2. End scan with: - End of scan.

3. FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES (APPLY TO ALL QUERIES):

EVIDENCE:
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate.
- You MUST evaluate ALL relevant items in the context before ending the scan.

KEEP/DISCARD:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- FINAL ANSWER must include ALL items marked [KEEP] - do not omit any.

MATCHING (PREVENTS HALLUCINATION):
- Entity/attribute must be EXPLICITLY stated in evidence - the query term must appear.
- Similar entities are NOT matches unless explicitly stated (e.g., "Founder" ≠ "Co-Founder").
- DO NOT assume relationships - only use explicitly stated information.

EMPTY RESULTS:
- If ALL items are marked [DISCARD], FINAL ANSWER must indicate no matches found.

OUTPUT FORMAT:
- FINAL ANSWER should include ONLY the requested information, not extra words or role titles."""

def create_strict_matching_examples():
    """Create examples that enforce strict verbatim matching."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Peter Moeller is a dynamic leader in business development, strategic growth, and integrated marketing, serving as Business Development Lead at LedgerAI. He has built a reputation for accelerating business expansion and optimizing market positioning. David Lara is a strategic powerhouse who serves as Co-Founder and Chief Operating Officer of LedgerAI. Liam Hugill serves as LedgerAI's Ambassador of Influence and Engagement, helping build the global community.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Peter Moeller\n  - Evidence: \"serving as Business Development Lead at LedgerAI\"\n  - Action: [DISCARD] (Reason: Evidence says \"Business Development Lead\", not \"co-founder\". Query term \"co-founder\" must appear verbatim in evidence.)\n- Item: David Lara\n  - Evidence: \"serves as Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"Co-Founder\".)\n- Item: Liam Hugill\n  - Evidence: \"serves as LedgerAI's Ambassador of Influence and Engagement\"\n  - Action: [DISCARD] (Reason: Evidence says \"Ambassador\", not \"co-founder\". Query term \"co-founder\" must appear verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of LedgerAI are David Lara."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The headquarters is now located at 123 Tech Street, New York, NY. The company has offices in London, Tokyo, and San Francisco. The London office opened in 2019, and the Tokyo office opened in 2021. The San Francisco office is the largest regional office.\n---\nQuestion: Where is the headquarters of the company?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 123 Tech Street, New York, NY\n  - Evidence: \"The headquarters is now located at 123 Tech Street, New York, NY\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"headquarters\".)\n- Item: London\n  - Evidence: \"The company has offices in London\"\n  - Action: [DISCARD] (Reason: Evidence says \"offices\", not \"headquarters\". Query asks for \"headquarters\" specifically, not offices.)\n- Item: Tokyo\n  - Evidence: \"The company has offices in Tokyo\"\n  - Action: [DISCARD] (Reason: Evidence says \"offices\", not \"headquarters\". Query asks for \"headquarters\" specifically, not offices.)\n- Item: San Francisco\n  - Evidence: \"The San Francisco office is the largest regional office\"\n  - Action: [DISCARD] (Reason: Evidence says \"office\", not \"headquarters\". Query asks for \"headquarters\" specifically, not offices.)\n- End of scan.\n\nFINAL ANSWER:\nThe headquarters is located at 123 Tech Street, New York, NY."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: James Wilson is the Founder of TechCorp, having established the company in 2010. Maria Garcia serves as Co-Founder and Chief Technology Officer. Robert Lee is the Head of Engineering and has been with the company since 2015.\n---\nQuestion: Who are the co-founders of TechCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: James Wilson\n  - Evidence: \"James Wilson is the Founder of TechCorp\"\n  - Action: [DISCARD] (Reason: Evidence says \"Founder\" (singular), not \"co-founder\". Query asks for \"co-founders\" (plural/co-), so \"Founder\" is NOT a match. Query term \"co-founder\" must appear verbatim in evidence.)\n- Item: Maria Garcia\n  - Evidence: \"serves as Co-Founder and Chief Technology Officer\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"Co-Founder\".)\n- Item: Robert Lee\n  - Evidence: \"is the Head of Engineering\"\n  - Action: [DISCARD] (Reason: Evidence says \"Head of Engineering\", not \"co-founder\". Query term \"co-founder\" must appear verbatim in evidence.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechCorp are Maria Garcia."
                }
            ]
        }
    ]
    
    return examples

def create_final_answer_enforcement_examples():
    """Create examples that enforce FINAL ANSWER must include ALL [KEEP] items and exclude ALL [DISCARD] items."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Paul Chou is the CEO and Co-Founder of LedgerAI. Bob Carella is Co-Founder and Chief Financial Officer of LedgerAI. David Lara is Co-Founder and Chief Operating Officer of LedgerAI. Jorge Guinovart is Co-Founder and Chief Marketing Officer of LedgerAI. Will Specht is Head of Engineering at LedgerAI. Peter Moeller is Business Development Lead at LedgerAI.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Paul Chou\n  - Evidence: \"Paul Chou is the CEO and Co-Founder of LedgerAI\"\n  - Action: [KEEP]\n- Item: Bob Carella\n  - Evidence: \"Bob Carella is Co-Founder and Chief Financial Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: David Lara\n  - Evidence: \"David Lara is Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Jorge Guinovart\n  - Evidence: \"Jorge Guinovart is Co-Founder and Chief Marketing Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Will Specht\n  - Evidence: \"Will Specht is Head of Engineering at LedgerAI\"\n  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder.)\n- Item: Peter Moeller\n  - Evidence: \"Peter Moeller is Business Development Lead at LedgerAI\"\n  - Action: [DISCARD] (Reason: Business Development Lead, not co-founder.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of LedgerAI are Paul Chou, Bob Carella, David Lara, and Jorge Guinovart."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: John Smith is CEO and Co-Founder of TechCorp. Michael Brown is Co-Founder and COO of TechCorp. David Martinez is Co-Founder and Chief Product Officer of TechCorp. Sarah Johnson is CTO at TechCorp. Emily Davis is Head of Marketing.\n---\nQuestion: Who are the co-founders of TechCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: John Smith\n  - Evidence: \"John Smith is CEO and Co-Founder of TechCorp\"\n  - Action: [KEEP]\n- Item: Michael Brown\n  - Evidence: \"Michael Brown is Co-Founder and COO of TechCorp\"\n  - Action: [KEEP]\n- Item: David Martinez\n  - Evidence: \"David Martinez is Co-Founder and Chief Product Officer of TechCorp\"\n  - Action: [KEEP]\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is CTO at TechCorp\"\n  - Action: [DISCARD] (Reason: CTO, not co-founder.)\n- Item: Emily Davis\n  - Evidence: \"Emily Davis is Head of Marketing\"\n  - Action: [DISCARD] (Reason: Head of Marketing, not co-founder.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechCorp are John Smith, Michael Brown, and David Martinez."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The engineering team consists of 50 developers working on core products. The marketing team has 20 members focused on brand awareness. The sales team includes 30 people handling customer relationships.\n---\nQuestion: What is the size of the engineering team?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 50 developers\n  - Evidence: \"The engineering team consists of 50 developers working on core products\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"engineering team\" and \"50 developers\".)\n- Item: 20 members\n  - Evidence: \"The marketing team has 20 members\"\n  - Action: [DISCARD] (Reason: Evidence says \"marketing team\", not \"engineering team\".)\n- Item: 30 people\n  - Evidence: \"The sales team includes 30 people\"\n  - Action: [DISCARD] (Reason: Evidence says \"sales team\", not \"engineering team\".)\n- End of scan.\n\nFINAL ANSWER:\nThe engineering team consists of 50 developers."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The annual revenue for 2023 was $50 million. The company raised $10 million in Series A funding in 2022. Operating expenses totaled $35 million in 2023.\n---\nQuestion: What was the annual revenue?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: $50 million\n  - Evidence: \"The annual revenue for 2023 was $50 million\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"annual revenue\".)\n- Item: $10 million\n  - Evidence: \"The company raised $10 million in Series A funding in 2022\"\n  - Action: [DISCARD] (Reason: Evidence says \"funding\", not \"revenue\".)\n- Item: $35 million\n  - Evidence: \"Operating expenses totaled $35 million in 2023\"\n  - Action: [DISCARD] (Reason: Evidence says \"expenses\", not \"revenue\".)\n- End of scan.\n\nFINAL ANSWER:\nThe annual revenue was $50 million."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Sarah Johnson is the Chief Technology Officer at DataFlow. She leads the engineering team and oversees all technical decisions. Mark Williams is Head of Engineering and reports to Sarah. Robert Kim is the Chief Financial Officer.\n---\nQuestion: Who is the CTO of DataFlow?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is the Chief Technology Officer at DataFlow\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"Chief Technology Officer\" (CTO).)\n- Item: Mark Williams\n  - Evidence: \"Mark Williams is Head of Engineering\"\n  - Action: [DISCARD] (Reason: Evidence says \"Head of Engineering\", not \"CTO\". Query asks for \"CTO\" specifically.)\n- Item: Robert Kim\n  - Evidence: \"Robert Kim is the Chief Financial Officer\"\n  - Action: [DISCARD] (Reason: Evidence says \"Chief Financial Officer\" (CFO), not CTO.)\n- End of scan.\n\nFINAL ANSWER:\nSarah Johnson."
                }
            ]
        }
    ]
    
    return examples

def create_complete_scanning_examples():
    """Create examples that enforce complete scanning of complex contexts."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Paul Chou is the CEO and Co-Founder of LedgerAI, driving the development of AI-powered business intelligence. Bob Carella is Co-Founder and Chief Financial Officer of LedgerAI, architecting the company's financial strategy. David Lara is Co-Founder and Chief Operating Officer of LedgerAI, leading the execution of AI-powered intelligence solutions. Jorge Guinovart is Co-Founder and Chief Marketing Officer of LedgerAI, spearheading global adoption. Will Specht is Head of Engineering at LedgerAI, leading the engineering efforts. Peter Moeller is Business Development Lead at LedgerAI, serving in business development. Liam Hugill is Ambassador of Influence and Engagement at LedgerAI, helping build the global community. Albert Soler is External Counsel & Advisor at LedgerAI, bringing legal expertise.\n---\nQuestion: Who are the co-founders of LedgerAI?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Paul Chou\n  - Evidence: \"Paul Chou is the CEO and Co-Founder of LedgerAI\"\n  - Action: [KEEP]\n- Item: Bob Carella\n  - Evidence: \"Bob Carella is Co-Founder and Chief Financial Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: David Lara\n  - Evidence: \"David Lara is Co-Founder and Chief Operating Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Jorge Guinovart\n  - Evidence: \"Jorge Guinovart is Co-Founder and Chief Marketing Officer of LedgerAI\"\n  - Action: [KEEP]\n- Item: Will Specht\n  - Evidence: \"Will Specht is Head of Engineering at LedgerAI\"\n  - Action: [DISCARD] (Reason: Head of Engineering, not co-founder.)\n- Item: Peter Moeller\n  - Evidence: \"Peter Moeller is Business Development Lead at LedgerAI\"\n  - Action: [DISCARD] (Reason: Business Development Lead, not co-founder.)\n- Item: Liam Hugill\n  - Evidence: \"Liam Hugill is Ambassador of Influence and Engagement at LedgerAI\"\n  - Action: [DISCARD] (Reason: Ambassador, not co-founder.)\n- Item: Albert Soler\n  - Evidence: \"Albert Soler is External Counsel & Advisor at LedgerAI\"\n  - Action: [DISCARD] (Reason: External Counsel & Advisor, not co-founder.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of LedgerAI are Paul Chou, Bob Carella, David Lara, and Jorge Guinovart."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: John Smith is the CEO and Co-Founder of TechCorp. Sarah Johnson is the Chief Technology Officer at TechCorp. Michael Brown is the Co-Founder and Chief Operating Officer of TechCorp. Emily Davis is the Head of Marketing. Robert Wilson is an External Advisor. David Martinez is the Co-Founder and Chief Product Officer of TechCorp. Lisa Anderson is the Chief Financial Officer.\n---\nQuestion: Who are the co-founders of TechCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: John Smith\n  - Evidence: \"John Smith is the CEO and Co-Founder of TechCorp\"\n  - Action: [KEEP]\n- Item: Sarah Johnson\n  - Evidence: \"Sarah Johnson is the Chief Technology Officer at TechCorp\"\n  - Action: [DISCARD] (Reason: CTO, not co-founder.)\n- Item: Michael Brown\n  - Evidence: \"Michael Brown is the Co-Founder and Chief Operating Officer of TechCorp\"\n  - Action: [KEEP]\n- Item: Emily Davis\n  - Evidence: \"Emily Davis is the Head of Marketing\"\n  - Action: [DISCARD] (Reason: Head of Marketing, not co-founder.)\n- Item: Robert Wilson\n  - Evidence: \"Robert Wilson is an External Advisor\"\n  - Action: [DISCARD] (Reason: External Advisor, not co-founder.)\n- Item: David Martinez\n  - Evidence: \"David Martinez is the Co-Founder and Chief Product Officer of TechCorp\"\n  - Action: [KEEP]\n- Item: Lisa Anderson\n  - Evidence: \"Lisa Anderson is the Chief Financial Officer\"\n  - Action: [DISCARD] (Reason: CFO, not co-founder.)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of TechCorp are John Smith, Michael Brown, and David Martinez."
                }
            ]
        }
    ]
    
    return examples

def main():
    print("=" * 80)
    print("ADDING STRICT MATCHING, FINAL ANSWER ENFORCEMENT, AND COMPLETE SCANNING EXAMPLES")
    print("=" * 80)
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"\n📊 Current dataset size: {original_count} examples")
    
    # Create new examples
    strict_matching = create_strict_matching_examples()
    final_answer = create_final_answer_enforcement_examples()
    complete_scanning = create_complete_scanning_examples()
    
    new_examples = strict_matching + final_answer + complete_scanning
    
    print(f"\n📝 Adding {len(new_examples)} new examples:")
    print(f"   - {len(strict_matching)} strict matching examples")
    print(f"   - {len(final_answer)} final answer enforcement examples")
    print(f"   - {len(complete_scanning)} complete scanning examples")
    
    # Add new examples to the beginning (so they're seen first during training)
    data = new_examples + data
    
    # Save dataset
    save_dataset(data)
    
    print(f"\n✅ Dataset updated!")
    print(f"   - Original: {original_count} examples")
    print(f"   - New total: {len(data)} examples")
    print(f"   - Added: {len(new_examples)} examples")
    
    print(f"\n📋 What These Examples Enforce:")
    print(f"   1. ✅ Strict Matching: Query term MUST appear verbatim in evidence")
    print(f"      - \"Business Development Lead\" ≠ \"co-founder\"")
    print(f"      - \"Founder\" ≠ \"co-founder\"")
    print(f"      - \"offices\" ≠ \"headquarters\"")
    print(f"   2. ✅ FINAL ANSWER Rules:")
    print(f"      - Must include ALL [KEEP] items (no omissions)")
    print(f"      - Must exclude ALL [DISCARD] items (no false positives)")
    print(f"      - Preserve context from evidence (\"50 developers\" not \"50 employees\")")
    print(f"      - Include only requested info (\"$50 million\" not \"$50 million in 2023\")")
    print(f"      - Name only for role queries (\"Sarah Johnson\" not \"The CTO\")")
    print(f"   3. ✅ Complete Scanning:")
    print(f"      - Must evaluate ALL people/items in context")
    print(f"      - Especially important for complex contexts with many entities")
    print(f"      - Must not stop scanning early")
    
    print(f"\n🎯 Expected Improvements:")
    print(f"   - LedgerAI Co-Founders: Should find all 4, exclude non-co-founders")
    print(f"   - TechCorp Co-Founders: Should find all 3, not miss David Martinez")
    print(f"   - Headquarters: Should only include headquarters, exclude offices")
    print(f"   - Team Size: Should preserve \"50 developers\" from evidence")
    print(f"   - Revenue: Should include only \"$50 million\", not \"2023\"")
    print(f"   - Role Query: Should include only \"Sarah Johnson\", not \"The CTO\"")

if __name__ == "__main__":
    main()
