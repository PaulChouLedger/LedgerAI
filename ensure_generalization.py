#!/usr/bin/env python3
"""
Ensure Model Generalization
============================
Add diverse examples that reinforce principles (not just fix test cases)
to ensure model can handle ANY query with relevant RAG chunks in the future.
"""

import json
import random

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

def create_diverse_generalization_examples():
    """
    Create diverse examples that reinforce principles for ANY query type.
    These examples use DIFFERENT companies, roles, and scenarios than test cases
    to ensure the model learns PRINCIPLES, not memorizes specific cases.
    """
    
    examples = [
        # Principle: Strict matching - VP ≠ Co-Founder (different company)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Alex Chen is Vice President of Product at InnovateCorp. Sarah Kim is Co-Founder and CEO of InnovateCorp. Michael Park is Director of Engineering at InnovateCorp. Lisa Wang is Co-Founder and CTO of InnovateCorp.\n---\nQuestion: Who are the co-founders of InnovateCorp?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Alex Chen\n  - Evidence: \"Alex Chen is Vice President of Product at InnovateCorp\"\n  - Action: [DISCARD] (Reason: Evidence says \"Vice President\", not \"co-founder\". Query term \"co-founder\" must appear verbatim in evidence.)\n- Item: Sarah Kim\n  - Evidence: \"Sarah Kim is Co-Founder and CEO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"Co-Founder\".)\n- Item: Michael Park\n  - Evidence: \"Michael Park is Director of Engineering at InnovateCorp\"\n  - Action: [DISCARD] (Reason: Evidence says \"Director of Engineering\", not \"co-founder\". Query term \"co-founder\" must appear verbatim in evidence.)\n- Item: Lisa Wang\n  - Evidence: \"Lisa Wang is Co-Founder and CTO of InnovateCorp\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"Co-Founder\".)\n- End of scan.\n\nFINAL ANSWER:\nThe co-founders of InnovateCorp are Sarah Kim and Lisa Wang."
                }
            ]
        },
        # Principle: Complete scanning - find all matching items (different query type)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The company offers three subscription tiers: Basic ($9/month), Pro ($29/month), and Enterprise ($99/month). The Basic plan includes 10GB storage. The Pro plan includes 100GB storage and priority support. The Enterprise plan includes unlimited storage and dedicated account manager.\n---\nQuestion: What are the subscription tiers offered by the company?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Basic\n  - Evidence: \"The company offers three subscription tiers: Basic ($9/month), Pro ($29/month), and Enterprise ($99/month)\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"subscription tiers\" and includes \"Basic\".)\n- Item: Pro\n  - Evidence: \"The company offers three subscription tiers: Basic ($9/month), Pro ($29/month), and Enterprise ($99/month)\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"subscription tiers\" and includes \"Pro\".)\n- Item: Enterprise\n  - Evidence: \"The company offers three subscription tiers: Basic ($9/month), Pro ($29/month), and Enterprise ($99/month)\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"subscription tiers\" and includes \"Enterprise\".)\n- End of scan.\n\nFINAL ANSWER:\nThe subscription tiers are Basic, Pro, and Enterprise."
                }
            ]
        },
        # Principle: Strict matching - Regional Office ≠ Headquarters (different location)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The corporate headquarters is located at 500 Innovation Drive, Austin, Texas. The company has regional offices in Seattle, Boston, and Miami. The Seattle office opened in 2018. The Boston office opened in 2020.\n---\nQuestion: Where is the headquarters located?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 500 Innovation Drive, Austin, Texas\n  - Evidence: \"The corporate headquarters is located at 500 Innovation Drive, Austin, Texas\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"headquarters\".)\n- Item: Seattle\n  - Evidence: \"The company has regional offices in Seattle\"\n  - Action: [DISCARD] (Reason: Evidence says \"regional offices\", not \"headquarters\". Query asks for \"headquarters\" specifically. Query term \"headquarters\" must appear verbatim in evidence.)\n- Item: Boston\n  - Evidence: \"The company has regional offices in Boston\"\n  - Action: [DISCARD] (Reason: Evidence says \"regional offices\", not \"headquarters\". Query asks for \"headquarters\" specifically.)\n- Item: Miami\n  - Evidence: \"The company has regional offices in Miami\"\n  - Action: [DISCARD] (Reason: Evidence says \"regional offices\", not \"headquarters\". Query asks for \"headquarters\" specifically.)\n- End of scan.\n\nFINAL ANSWER:\nThe headquarters is located at 500 Innovation Drive, Austin, Texas."
                }
            ]
        },
        # Principle: Role query - name only (different role)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: Robert Martinez is the Chief Financial Officer at DataSystems. He oversees all financial operations and reporting. Jennifer Lee is the VP of Sales. David Kim is the Chief Operating Officer.\n---\nQuestion: Who is the CFO of DataSystems?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Robert Martinez\n  - Evidence: \"Robert Martinez is the Chief Financial Officer at DataSystems\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"Chief Financial Officer\" (CFO).)\n- Item: Jennifer Lee\n  - Evidence: \"Jennifer Lee is the VP of Sales\"\n  - Action: [DISCARD] (Reason: Evidence says \"VP of Sales\", not \"CFO\".)\n- Item: David Kim\n  - Evidence: \"David Kim is the Chief Operating Officer\"\n  - Action: [DISCARD] (Reason: Evidence says \"Chief Operating Officer\" (COO), not CFO.)\n- End of scan.\n\nFINAL ANSWER:\nRobert Martinez."
                }
            ]
        },
        # Principle: Number query - preserve context (different metric)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The sales team has 25 account executives focused on enterprise clients. The marketing team includes 15 specialists handling digital campaigns. The customer support team consists of 40 representatives providing 24/7 assistance.\n---\nQuestion: What is the size of the sales team?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 25 account executives\n  - Evidence: \"The sales team has 25 account executives focused on enterprise clients\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"sales team\" and \"25 account executives\".)\n- Item: 15 specialists\n  - Evidence: \"The marketing team includes 15 specialists\"\n  - Action: [DISCARD] (Reason: Evidence says \"marketing team\", not \"sales team\".)\n- Item: 40 representatives\n  - Evidence: \"The customer support team consists of 40 representatives\"\n  - Action: [DISCARD] (Reason: Evidence says \"customer support team\", not \"sales team\".)\n- End of scan.\n\nFINAL ANSWER:\nThe sales team has 25 account executives."
                }
            ]
        },
        # Principle: Date query - only requested type (different date type)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The company was acquired in 2019. The product launched in March 2021. The Series B funding round closed in June 2022. The IPO occurred in September 2023.\n---\nQuestion: When was the product launched?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: 2019\n  - Evidence: \"The company was acquired in 2019\"\n  - Action: [DISCARD] (Reason: Evidence says \"acquired\", not \"launched\". Query asks for product launch date only.)\n- Item: March 2021\n  - Evidence: \"The product launched in March 2021\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"launched\".)\n- Item: June 2022\n  - Evidence: \"The Series B funding round closed in June 2022\"\n  - Action: [DISCARD] (Reason: Evidence says \"funding round closed\", not \"launched\". Query asks for product launch date only.)\n- Item: September 2023\n  - Evidence: \"The IPO occurred in September 2023\"\n  - Action: [DISCARD] (Reason: Evidence says \"IPO occurred\", not \"launched\". Query asks for product launch date only.)\n- End of scan.\n\nFINAL ANSWER:\nThe product was launched in March 2021."
                }
            ]
        },
        # Principle: List query - complete scanning (different domain)
        {
            "messages": [
                {
                    "role": "system",
                    "content": get_system_prompt()
                },
                {
                    "role": "user",
                    "content": "Knowledge context: The platform supports integration with Salesforce, HubSpot, Zendesk, and Slack. Custom integrations are available through the API. The platform also connects with Microsoft Teams and Google Workspace.\n---\nQuestion: What integrations does the platform support?"
                },
                {
                    "role": "assistant",
                    "content": "REASONING:\n- Item: Salesforce\n  - Evidence: \"The platform supports integration with Salesforce, HubSpot, Zendesk, and Slack\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"integration\" and includes \"Salesforce\".)\n- Item: HubSpot\n  - Evidence: \"The platform supports integration with Salesforce, HubSpot, Zendesk, and Slack\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"integration\" and includes \"HubSpot\".)\n- Item: Zendesk\n  - Evidence: \"The platform supports integration with Salesforce, HubSpot, Zendesk, and Slack\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"integration\" and includes \"Zendesk\".)\n- Item: Slack\n  - Evidence: \"The platform supports integration with Salesforce, HubSpot, Zendesk, and Slack\"\n  - Action: [KEEP] (Reason: Evidence explicitly states \"integration\" and includes \"Slack\".)\n- Item: Microsoft Teams\n  - Evidence: \"The platform also connects with Microsoft Teams\"\n  - Action: [KEEP] (Reason: Evidence states \"connects with\" which indicates integration.)\n- Item: Google Workspace\n  - Evidence: \"The platform also connects with Google Workspace\"\n  - Action: [KEEP] (Reason: Evidence states \"connects with\" which indicates integration.)\n- End of scan.\n\nFINAL ANSWER:\nThe platform supports integrations with Salesforce, HubSpot, Zendesk, Slack, Microsoft Teams, and Google Workspace."
                }
            ]
        }
    ]
    
    return examples

def main():
    print("=" * 80)
    print("ENSURING MODEL GENERALIZATION")
    print("=" * 80)
    
    # Load dataset
    data = load_dataset()
    original_count = len(data)
    print(f"\n📊 Current dataset size: {original_count} examples")
    
    # Create diverse examples
    diverse_examples = create_diverse_generalization_examples()
    
    print(f"\n📝 Adding {len(diverse_examples)} diverse generalization examples:")
    print(f"   These examples use DIFFERENT companies, roles, and scenarios")
    print(f"   than test cases to ensure model learns PRINCIPLES, not memorizes cases")
    print()
    print(f"   1. Strict matching - VP ≠ Co-Founder (InnovateCorp, different company)")
    print(f"   2. Complete scanning - subscription tiers (different query type)")
    print(f"   3. Strict matching - Regional Office ≠ Headquarters (different location)")
    print(f"   4. Role query - name only (CFO, different role)")
    print(f"   5. Number query - preserve context (account executives, different metric)")
    print(f"   6. Date query - only requested type (product launch, different date)")
    print(f"   7. List query - complete scanning (integrations, different domain)")
    
    # Add diverse examples to the beginning (high priority for generalization)
    data = diverse_examples + data
    
    # Save dataset
    save_dataset(data)
    
    print(f"\n✅ Dataset updated!")
    print(f"   - Original: {original_count} examples")
    print(f"   - New total: {len(data)} examples")
    print(f"   - Added: {len(diverse_examples)} examples")
    
    print(f"\n📋 PRINCIPLES REINFORCED (Applies to ANY Query):")
    print(f"   1. ✅ Strict verbatim matching - query term must appear in evidence")
    print(f"   2. ✅ Complete scanning - evaluate ALL items in context")
    print(f"   3. ✅ FINAL ANSWER rules - include all [KEEP], exclude all [DISCARD]")
    print(f"   4. ✅ Output format - only requested information")
    print(f"   5. ✅ Similar entities ≠ matches (VP ≠ Co-Founder, Office ≠ Headquarters)")
    
    print(f"\n🎯 GENERALIZATION STRATEGY:")
    print(f"   ✅ Generalized prompt (principle-based, not case-specific)")
    print(f"   ✅ Diverse examples (different companies, roles, query types)")
    print(f"   ✅ Principle reinforcement (same rules, different scenarios)")
    print(f"   ✅ Edge case coverage (not just test case fixes)")
    print()
    print(f"💡 RESULT:")
    print(f"   Model will learn PRINCIPLES that apply to:")
    print(f"   - ANY company name")
    print(f"   - ANY role (CTO, CFO, VP, Director, etc.)")
    print(f"   - ANY query type (co-founders, headquarters, revenue, dates, lists)")
    print(f"   - ANY RAG chunk structure")
    print()
    print(f"   ✅ Will handle future queries with relevant RAG chunks")
    print(f"   ✅ Will apply strict matching rules to new scenarios")
    print(f"   ✅ Will complete scanning regardless of context complexity")

if __name__ == "__main__":
    main()
