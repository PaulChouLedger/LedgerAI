#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Dataset Enhancement Script
===============================

1. Reviews existing dataset for relevance score outputs (HIGH/LOW/MEDIUM)
2. Adds targeted training examples for failing categories:
   - Role filtering (CEO vs Co-Founder)
   - Cross-company filtering
   - Multi-chunk extraction
   - "Not found" cases
   - Process queries
   - Relationship queries
   - Comparison queries
   - Analytical queries
"""

import json
import random
import re
from typing import List, Dict, Any, Tuple

# ============================================================================
# Review Existing Dataset
# ============================================================================

def review_dataset_for_relevance_scores(dataset_path: str) -> Dict[str, Any]:
    """Review dataset for relevance score outputs in assistant responses"""
    
    print("=" * 80)
    print("REVIEWING DATASET FOR RELEVANCE SCORE OUTPUTS")
    print("=" * 80)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    problematic_patterns = [
        r'\bHIGH\b',
        r'\bLOW\b',
        r'\bMEDIUM\b',
        r'RELEVANCE',
        r'Score:\s*\d+',
        r'score\s*=\s*\d+',
        r'HIGH RELEVANCE',
        r'LOW RELEVANCE',
        r'MEDIUM RELEVANCE',
    ]
    
    issues = []
    total_examples = len(dataset)
    
    for idx, example in enumerate(dataset):
        messages = example.get('messages', [])
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                for pattern in problematic_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append({
                            'index': idx,
                            'pattern': pattern,
                            'content': content[:200] + '...' if len(content) > 200 else content
                        })
                        break
    
    print(f"\nTotal examples: {total_examples}")
    print(f"Examples with relevance score outputs: {len(issues)}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} problematic examples:")
        for issue in issues[:10]:  # Show first 10
            print(f"  Example {issue['index']}: Pattern '{issue['pattern']}'")
            print(f"    Content: {issue['content'][:100]}...")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\n✅ No relevance score outputs found in assistant responses!")
    
    return {
        'total': total_examples,
        'issues': issues,
        'issue_count': len(issues)
    }

# ============================================================================
# System Prompt (Same as training)
# ============================================================================

def get_system_prompt() -> str:
    """Get the standard system prompt"""
    return """You are an AI assistant trained to analyze RAG chunks and extract relevant information.

CORE PRINCIPLES (SYSTEMATIC EVALUATION PROCESS):

STEP 1: UNDERSTAND THE QUERY
- Identify what information is being requested
- Note any specific filtering requirements (role, entity, attribute, relationship, etc.)
- Understand the scope and context of what needs to be extracted

STEP 2: READ EACH CHUNK COMPLETELY
- Read the entire chunk from start to finish
- Do not stop at keywords - read for full context and meaning
- Understand the complete context before making extraction decisions

STEP 3: ANALYZE CHUNK MEANING
- Understand the semantic meaning, not just surface-level keywords
- Identify entities, relationships, attributes, and concepts mentioned
- Recognize how information relates to the query

STEP 4: EVALUATE RELEVANCE
- Determine if information directly answers or addresses the query
- Apply query-specific filtering (match role, entity, attribute, etc. as requested)
- CRITICAL: For role queries, match the EXACT role (e.g., "co-founders" ≠ "CEO" ≠ "CTO" - extract ONLY the exact role requested)
- CRITICAL: For company queries, extract information ONLY about the company that matches the query. Use the company name EXACTLY as it appears in the chunks (RAG handles fuzzy matching at retrieval - if chunk says "TechCorp", extract "TechCorp" even if query said "Tech Corp"). Do NOT extract information about other companies
- Ignore information that is similar but does NOT answer the query

STEP 5: EXTRACT MATCHING INFORMATION
- Extract only information that passes the relevance evaluation
- Apply exact matching - use information exactly as it appears in chunks
- Track all matching items across all chunks

STEP 6: VERIFY COMPLETENESS
- Ensure you have read ALL chunks completely
- Verify you extracted ALL matching items (do not stop after first match)
- Confirm extraction is complete before finalizing response

STEP 7: SYNTHESIZE RESPONSE
- Combine information from all chunks into coherent answer
- Format naturally and directly address the query
- CRITICAL: If after reading ALL chunks completely you find NO information that matches the query (wrong role, wrong company, or missing entirely), you MUST respond with exactly: "I don't have that information in the provided documents"
- DO NOT infer, guess, or make up information - if it's not explicitly in the chunks, say "I don't have that information in the provided documents"

CRITICAL: Follow these steps in order for EVERY query. Chunk order does not change the answer - read all chunks before responding.

KEY RULES:
1. NEVER hallucinate - if information doesn't exist, say "I don't have that information in the provided documents"
2. NEVER make up names or entities - ONLY use information that appears in the provided chunks
3. CRITICAL: If EXACT match not found (wrong role, wrong company, or missing), respond with "I don't have that information in the provided documents"
4. EXACT MATCHING: Use EXACT names, terms, and information from chunks - NEVER substitute or modify
5. FILTERING: Apply the query's specific requirements - exclude information that doesn't match what is asked (e.g., "co-founders" ≠ "CEO", "TechCorp" ≠ "Tech Corp")
6. COMPLETE EXTRACTION: Extract ALL matching items - read ALL chunks completely before responding
7. ORDER-INDEPENDENT: Extract same results regardless of chunk order

RELEVANCE PRIORITIZATION:
- Prioritize HIGH relevance chunks (score ≥0.70) over LOW relevance chunks (score <0.50)
- Extract ONLY information that directly answers the query
- IGNORE similar information that does NOT answer the query

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""

# ============================================================================
# Helper Functions
# ============================================================================

def generate_name() -> str:
    """Generate random name"""
    first = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Sage", "River",
             "Blake", "Cameron", "Dakota", "Emery", "Finley", "Harper", "Hayden", "Jamie", "Kendall", "Logan",
             "John", "Sarah", "Mike", "David", "Lisa", "Robert", "Emma", "Tom", "Sue", "Frank", "Grace"]
    last = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Harris",
            "Chen", "Wang", "Kim", "Lee", "Garcia", "Martinez", "Lopez", "Gonzalez", "Wilson", "Anderson"]
    return f"{random.choice(first)} {random.choice(last)}"

def generate_company() -> str:
    """Generate random company name"""
    prefixes = ["Tech", "Data", "Cloud", "AI", "Digital", "Smart", "Global", "Next", "Future", "Prime",
                "Quantum", "Nexus", "Vertex", "Apex", "Catalyst", "Synergy", "Pinnacle", "Summit", "Zenith", "Aurora",
                "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Theta", "Iota", "Kappa", "Lambda"]
    suffixes = ["Corp", "Systems", "Solutions", "Labs", "Group", "Industries", "Ventures", "Partners", "Works", "Co",
                "Technologies", "Enterprises", "Holdings", "Dynamics", "Innovations", "Networks", "Services", "Capital"]
    return f"{random.choice(prefixes)}{random.choice(suffixes)}"

def format_chunks(chunks: List[Dict]) -> str:
    """Format chunks for user message"""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0.0)
        file_name = chunk.get('file', 'document.pdf')
        text = chunk['text']
        text_escaped = text.replace("'", "\\'")
        parts.append(f"[Chunk {i}] Score: {score:.2f}, File: {file_name}")
        parts.append(f"FULL CHUNK TEXT: '{text_escaped}'")
        parts.append("")
    return "\n".join(parts)

# ============================================================================
# Generate Targeted Examples
# ============================================================================

def generate_role_filtering_examples(count: int = 50) -> List[Dict[str, Any]]:
    """Generate examples for role filtering (CEO vs Co-Founder, etc.)"""
    examples = []
    roles_to_exclude = ["CEO", "CTO", "CFO", "CMO", "VP of Engineering", "President", "Director"]
    
    for _ in range(count):
        company = generate_company()
        co_founder = generate_name()
        other_role_person = generate_name()
        other_role = random.choice(roles_to_exclude)
        
        # Create chunk with both roles
        chunk_text = f"{other_role_person} is {other_role} of {company}. {co_founder} is Co-Founder of {company}."
        if random.random() < 0.3:  # Sometimes add more context
            chunk_text += f" The company was founded in {random.randint(2015, 2023)}. Market analysts have observed significant shifts in consumer behavior patterns."
        
        query = "who are the co-founders of " + company + "?"
        expected_response = co_founder
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n[Chunk 1] Score: {random.uniform(0.85, 0.95):.2f}, File: document.pdf\nFULL CHUNK TEXT: '{chunk_text}'\n"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

def generate_cross_company_examples(count: int = 30) -> List[Dict[str, Any]]:
    """Generate examples for cross-company filtering"""
    examples = []
    
    for _ in range(count):
        company1 = generate_company()
        company2 = generate_company()
        while company2 == company1:
            company2 = generate_company()
        
        co_founder1 = generate_name()
        co_founder2 = generate_name()
        
        # Create chunk with both companies
        chunk_text = f"{co_founder1} is Co-Founder of {company1}. {co_founder2} is Co-Founder of {company2}."
        if random.random() < 0.4:
            chunk_text += f" Market analysts have observed significant shifts. Economic indicators suggest growth."
        
        query = f"who are the co-founders of {company1}?"
        expected_response = co_founder1
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n[Chunk 1] Score: {random.uniform(0.85, 0.95):.2f}, File: document.pdf\nFULL CHUNK TEXT: '{chunk_text}'\n"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

def generate_multi_chunk_examples(count: int = 30) -> List[Dict[str, Any]]:
    """Generate examples requiring extraction from multiple chunks"""
    examples = []
    
    for _ in range(count):
        company = generate_company()
        co_founders = [generate_name() for _ in range(random.randint(2, 4))]
        
        # Split co-founders across chunks
        chunks = []
        for i, founder in enumerate(co_founders):
            chunk_text = f"{founder} is Co-Founder of {company}."
            if i < len(co_founders) - 1:  # Add context to all but last
                chunk_text += f" Market analysts have observed significant shifts. Economic indicators suggest growth."
            chunks.append({
                "text": chunk_text,
                "score": random.uniform(0.85, 0.95),
                "file": "document.pdf"
            })
        
        query = f"who are the co-founders of {company}?"
        if len(co_founders) == 2:
            expected_response = f"{co_founders[0]} and {co_founders[1]}"
        else:
            expected_response = ", ".join(co_founders[:-1]) + f", and {co_founders[-1]}"
        
        chunks_text = format_chunks(chunks)
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n{chunks_text}"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

def generate_not_found_examples(count: int = 20) -> List[Dict[str, Any]]:
    """Generate examples for 'not found' cases"""
    examples = []
    
    for _ in range(count):
        company = generate_company()
        
        # Case 1: Company mentioned but no co-founders (50%)
        if random.random() < 0.5:
            chunk_text = f"{company} is a technology company. The company has {random.randint(50, 500)} employees. Market analysts have observed significant shifts."
            query = f"who are the co-founders of {company}?"
        # Case 2: Wrong role (CEO/CTO when asked for co-founders) (30%)
        elif random.random() < 0.8:
            person = generate_name()
            role = random.choice(["CEO", "CTO", "CFO"])
            chunk_text = f"{person} is {role} of {company}. The company focuses on innovation. Market analysts have observed significant shifts."
            query = f"who are the co-founders of {company}?"
        # Case 3: Wrong company (20%)
        else:
            other_company = generate_company()
            while other_company == company:
                other_company = generate_company()
            person = generate_name()
            chunk_text = f"{person} is Co-Founder of {other_company}. Market analysts have observed significant shifts."
            query = f"who are the co-founders of {company}?"
        
        expected_response = "I don't have that information in the provided documents."
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n[Chunk 1] Score: {random.uniform(0.70, 0.85):.2f}, File: document.pdf\nFULL CHUNK TEXT: '{chunk_text}'\n"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

def generate_process_examples(count: int = 30) -> List[Dict[str, Any]]:
    """Generate examples for process queries"""
    examples = []
    
    processes = [
        ("authentication system", "first verifying user credentials, then generating a token, and finally granting access based on permissions"),
        ("payment processing", "validating the payment method, checking available funds, processing the transaction, and sending confirmation"),
        ("data processing", "first collecting inputs, then cleaning and validating, next transforming the format, and finally storing in the database"),
        ("machine learning models", "training on data, learning patterns, making predictions, and improving through feedback loops"),
        ("deployment pipeline", "building the code, running tests, creating containers, deploying to staging, running integration tests, and finally deploying to production"),
    ]
    
    for _ in range(count):
        process_name, steps = random.choice(processes)
        query = f"how does the {process_name} work?"
        
        chunk_text = f"The {process_name} works by {steps}."
        if random.random() < 0.5:
            chunk_text += f" Market analysts have observed significant shifts. Economic indicators suggest growth."
        
        expected_response = f"The {process_name} works by {steps}."
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n[Chunk 1] Score: {random.uniform(0.85, 0.95):.2f}, File: document.pdf\nFULL CHUNK TEXT: '{chunk_text}'\n"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

def generate_relationship_examples(count: int = 30) -> List[Dict[str, Any]]:
    """Generate examples for relationship queries"""
    examples = []
    
    relationships = [
        ("strategic partners", "collaborating on joint product development"),
        ("parent-subsidiary", "owns as a subsidiary and work together on integrated solutions"),
        ("alliance", "formed an alliance to share technology resources and market access"),
        ("connected", "connected through a shared technology platform and mutual customers"),
        ("joint venture", "established a joint venture to develop new products in emerging markets"),
    ]
    
    for _ in range(count):
        company1 = generate_company()
        company2 = generate_company()
        while company2 == company1:
            company2 = generate_company()
        
        rel_type, rel_desc = random.choice(relationships)
        
        if rel_type == "parent-subsidiary":
            chunk_text = f"{company1} owns {company2} as a subsidiary. They work together on integrated solutions."
        elif rel_type == "alliance":
            chunk_text = f"{company1} and {company2} formed an alliance to share technology resources and market access."
        elif rel_type == "connected":
            chunk_text = f"{company1} and {company2} are connected through a shared technology platform and mutual customers."
        elif rel_type == "joint venture":
            chunk_text = f"{company1} and {company2} established a joint venture to develop new products in emerging markets."
        else:  # strategic partners
            chunk_text = f"{company1} and {company2} are strategic partners collaborating on joint product development."
        
        if random.random() < 0.4:
            chunk_text += f" Market analysts have observed significant shifts. Economic indicators suggest growth."
        
        query = f"how are {company1} and {company2} related?"
        expected_response = chunk_text.split('.')[0] + '.'  # Just the relationship part
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n[Chunk 1] Score: {random.uniform(0.85, 0.95):.2f}, File: document.pdf\nFULL CHUNK TEXT: '{chunk_text}'\n"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

def generate_comparison_examples(count: int = 30) -> List[Dict[str, Any]]:
    """Generate examples for comparison queries"""
    examples = []
    
    comparisons = [
        ("ProductA", "ProductB", "enterprise solutions", "small businesses"),
        ("ServiceX", "ServiceY", "cloud infrastructure", "on-premise deployment"),
        ("Platform1", "Platform2", "extensive customization options", "simplicity"),
        ("SystemA", "SystemB", "security", "performance and speed"),
    ]
    
    for _ in range(count):
        entity1, entity2, attr1, attr2 = random.choice(comparisons)
        
        chunk_text = f"{entity1} focuses on {attr1} while {entity2} targets {attr2}."
        if random.random() < 0.4:
            chunk_text += f" Market analysts have observed significant shifts. Economic indicators suggest growth."
        
        query = f"compare {entity1} and {entity2}"
        expected_response = f"{entity1} focuses on {attr1} while {entity2} targets {attr2}."
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n[Chunk 1] Score: {random.uniform(0.85, 0.95):.2f}, File: document.pdf\nFULL CHUNK TEXT: '{chunk_text}'\n"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

def generate_analytical_examples(count: int = 30) -> List[Dict[str, Any]]:
    """Generate examples for analytical queries"""
    examples = []
    
    analytical_templates = [
        ("the company expanded internationally", "because of increasing global demand and market opportunities"),
        ("the product was launched early", "due to competitive pressures and customer requests"),
        ("the system failure occurred", "was caused by overloaded servers and insufficient capacity planning"),
        ("sales increased", "improved marketing strategies led to increased sales and customer engagement"),
        ("the merger happened", "to achieve market dominance, reduce costs, and expand product offerings"),
    ]
    
    for _ in range(count):
        action, reason = random.choice(analytical_templates)
        
        if "because" in reason or "due to" in reason or "led to" in reason or "caused" in reason:
            chunk_text = f"{action.capitalize()} {reason}."
        else:
            chunk_text = f"{action.capitalize()} {reason}."
        
        if random.random() < 0.4:
            chunk_text += f" Market analysts have observed significant shifts. Economic indicators suggest growth."
        
        query = f"why did {action}?"
        expected_response = chunk_text
        
        example = {
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRAG Chunks:\n[Chunk 1] Score: {random.uniform(0.85, 0.95):.2f}, File: document.pdf\nFULL CHUNK TEXT: '{chunk_text}'\n"
                },
                {"role": "assistant", "content": expected_response}
            ]
        }
        examples.append(example)
    
    return examples

# ============================================================================
# Main Enhancement Function
# ============================================================================

def enhance_dataset(dataset_path: str, output_path: str, add_count: int = 250):
    """Enhance dataset by reviewing and adding targeted examples"""
    
    print("=" * 80)
    print("RAG DATASET ENHANCEMENT")
    print("=" * 80)
    
    # Step 1: Review existing dataset
    review_results = review_dataset_for_relevance_scores(dataset_path)
    
    # Step 2: Load existing dataset
    print("\n" + "=" * 80)
    print("LOADING EXISTING DATASET")
    print("=" * 80)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        existing_dataset = json.load(f)
    
    print(f"Loaded {len(existing_dataset)} existing examples")
    
    # Step 3: Generate new targeted examples
    print("\n" + "=" * 80)
    print("GENERATING TARGETED EXAMPLES")
    print("=" * 80)
    
    new_examples = []
    
    # Distribute examples across categories
    counts = {
        'role_filtering': 50,
        'cross_company': 30,
        'multi_chunk': 30,
        'not_found': 20,
        'process': 30,
        'relationship': 30,
        'comparison': 30,
        'analytical': 30,
    }
    
    print(f"Generating {counts['role_filtering']} role filtering examples...")
    new_examples.extend(generate_role_filtering_examples(counts['role_filtering']))
    
    print(f"Generating {counts['cross_company']} cross-company filtering examples...")
    new_examples.extend(generate_cross_company_examples(counts['cross_company']))
    
    print(f"Generating {counts['multi_chunk']} multi-chunk extraction examples...")
    new_examples.extend(generate_multi_chunk_examples(counts['multi_chunk']))
    
    print(f"Generating {counts['not_found']} 'not found' examples...")
    new_examples.extend(generate_not_found_examples(counts['not_found']))
    
    print(f"Generating {counts['process']} process query examples...")
    new_examples.extend(generate_process_examples(counts['process']))
    
    print(f"Generating {counts['relationship']} relationship query examples...")
    new_examples.extend(generate_relationship_examples(counts['relationship']))
    
    print(f"Generating {counts['comparison']} comparison query examples...")
    new_examples.extend(generate_comparison_examples(counts['comparison']))
    
    print(f"Generating {counts['analytical']} analytical query examples...")
    new_examples.extend(generate_analytical_examples(counts['analytical']))
    
    print(f"\n✅ Generated {len(new_examples)} new examples")
    
    # Step 4: Combine datasets
    print("\n" + "=" * 80)
    print("COMBINING DATASETS")
    print("=" * 80)
    
    enhanced_dataset = existing_dataset + new_examples
    
    # Shuffle to mix new examples with existing ones
    random.shuffle(enhanced_dataset)
    
    print(f"Total examples: {len(enhanced_dataset)}")
    print(f"  - Existing: {len(existing_dataset)}")
    print(f"  - New: {len(new_examples)}")
    
    # Step 5: Save enhanced dataset
    print("\n" + "=" * 80)
    print("SAVING ENHANCED DATASET")
    print("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved enhanced dataset to: {output_path}")
    print(f"   Total examples: {len(enhanced_dataset)}")
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("ENHANCEMENT SUMMARY")
    print("=" * 80)
    print(f"✅ Reviewed {review_results['total']} existing examples")
    if review_results['issue_count'] > 0:
        print(f"⚠️  Found {review_results['issue_count']} examples with relevance score outputs")
        print(f"   (These should be manually reviewed and fixed)")
    else:
        print(f"✅ No relevance score outputs found")
    print(f"✅ Added {len(new_examples)} targeted examples:")
    for category, count in counts.items():
        print(f"   - {category}: {count} examples")
    print(f"✅ Enhanced dataset saved with {len(enhanced_dataset)} total examples")

if __name__ == "__main__":
    import sys
    
    dataset_path = "rag_analysis_dataset_v2.json"
    output_path = "rag_analysis_dataset_v2_enhanced.json"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    enhance_dataset(dataset_path, output_path)
