#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Analysis Dataset Generator V2 - Clean Implementation
==========================================================

Generates 6000 training examples for SLM to analyze RAG chunks:
- Each chunk: 6-8 sentences with relevant and irrelevant information
- Multiple instances of relevant information across chunks
- Varied system prompts (full, medium, short) with 7-step core principles
- Pattern-based distribution to teach general RAG skills
"""

import json
import random
from typing import List, Dict, Any

# ============================================================================
# System Prompt Variations (7-Step Core Principles)
# ============================================================================

def get_system_prompt_variation(variation_type="full"):
    """Generate system prompt with 7-step core principles"""
    
    core_principles = """CORE PRINCIPLES (SYSTEMATIC EVALUATION PROCESS):

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
- If no matching information found, state "I don't have that information in the provided documents"

CRITICAL: Follow these steps in order for EVERY query. Chunk order does not change the answer - read all chunks before responding."""
    
    if variation_type == "full":
        return f"""You are an AI assistant trained to analyze RAG chunks and extract relevant information.

{core_principles}

ESSENTIAL GUIDELINES:
- NEVER hallucinate - only use information that appears in the provided chunks
- NEVER make up names, entities, or information - if information doesn't exist, say "I don't have that information in the provided documents"
- Use EXACT information from chunks - never substitute or modify names, terms, or entities
- Apply query-specific filtering during Step 4 (evaluate relevance) - match what the query specifically asks for
- Extract ALL matching items - complete Step 6 (verify completeness) ensures nothing is missed
- Relevance scores guide prioritization but do not override the evaluation steps

QUERY TYPE HANDLING (applied during Step 4 - Evaluate Relevance):
- Role/entity queries: Filter by the specific role or entity mentioned in the query
- Comparison queries: Extract information comparing the entities mentioned
- Relationship queries: Extract connection information between entities
- Analytical queries: Extract reasoning, causation, or explanation
- Process queries: Extract step-by-step information
- List queries: Extract all items that match the query criteria

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""
    
    elif variation_type == "medium":
        return f"""You are an AI assistant trained to analyze RAG chunks and extract relevant information.

{core_principles}

KEY RULES:
1. NEVER hallucinate - if information doesn't exist, say "I don't have that information in the provided documents"
2. NEVER make up names or entities - ONLY use information that appears in the provided chunks
3. EXACT MATCHING: Use EXACT names, terms, and information from chunks - NEVER substitute or modify
4. FILTERING: Apply the query's specific requirements - exclude information that doesn't match what is asked
5. COMPLETE EXTRACTION: Extract ALL matching items - read ALL chunks completely before responding
6. ORDER-INDEPENDENT: Extract same results regardless of chunk order

RELEVANCE PRIORITIZATION:
- Prioritize HIGH relevance chunks (score ≥0.70) over LOW relevance chunks (score <0.50)
- Extract ONLY information that directly answers the query
- IGNORE similar information that does NOT answer the query

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""
    
    else:  # short
        return f"""You are an AI assistant that analyzes RAG chunks to extract relevant information.

{core_principles}

ESSENTIAL RULES:
- NEVER hallucinate - if information doesn't exist, say "I don't have that information in the provided documents"
- Use EXACT information from chunks - NEVER invent or modify
- Apply query-specific filtering - exclude information that doesn't match what is asked
- Extract ALL matching items - read ALL chunks completely before responding
- ORDER-INDEPENDENT: Extract same results regardless of chunk order

Return the final answer in natural language. Do not include reasoning steps in the response."""

# ============================================================================
# Helper Functions
# ============================================================================

def generate_random_name():
    """Generate random person name"""
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Sage", "River"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_random_company():
    """Generate random company name"""
    prefixes = ["Tech", "Data", "Cloud", "AI", "Digital", "Smart", "Global", "Next", "Future", "Prime"]
    suffixes = ["Corp", "Systems", "Solutions", "Labs", "Group", "Industries", "Ventures", "Partners", "Works", "Co"]
    return f"{random.choice(prefixes)}{random.choice(suffixes)}"

def generate_random_concept():
    """Generate random concept/term"""
    concepts = ["innovation", "strategy", "growth", "efficiency", "optimization", "transformation", 
                "scalability", "automation", "integration", "collaboration"]
    return random.choice(concepts)

# ============================================================================
# Chunk Generation (6-8 sentences with relevant and irrelevant info)
# ============================================================================

def create_chunk(relevant_sentences: List[str], irrelevant_sentences: List[str], 
                 num_sentences: int = 7) -> str:
    """
    Create a chunk with 6-8 sentences mixing relevant and irrelevant information
    
    Args:
        relevant_sentences: Sentences that answer the query
        irrelevant_sentences: Sentences that are similar but don't answer the query
        num_sentences: Target number of sentences (6-8)
    
    Returns:
        Chunk text with mixed relevant and irrelevant sentences
    """
    # Determine how many relevant vs irrelevant sentences
    # At least 1-2 relevant if available, rest can be irrelevant
    if relevant_sentences:
        num_relevant = random.randint(1, min(3, len(relevant_sentences)))
        selected_relevant = random.sample(relevant_sentences, num_relevant)
    else:
        num_relevant = 0
        selected_relevant = []
    
    num_irrelevant = num_sentences - num_relevant
    
    # Select irrelevant sentences
    if irrelevant_sentences:
        num_to_select = min(num_irrelevant, len(irrelevant_sentences))
        selected_irrelevant = random.sample(irrelevant_sentences, num_to_select)
    else:
        selected_irrelevant = []
    
    # Generate contextual sentences to fill to target
    contextual_templates = [
        "The organization has been working on various initiatives to improve operations.",
        "Recent developments in the industry have influenced strategic decisions.",
        "The team has been focusing on key priorities and objectives.",
        "Several factors contribute to the overall performance and outcomes.",
        "The approach involves multiple stakeholders and collaborative efforts.",
        "Various considerations are taken into account when making decisions.",
        "The context includes both historical data and current trends.",
        "Multiple perspectives are evaluated to ensure comprehensive understanding.",
        "The situation involves complex interactions between different elements.",
        "Several key factors play important roles in the overall process.",
    ]
    
    # Combine relevant and irrelevant
    all_sentences = selected_relevant + selected_irrelevant
    
    # Fill to target number of sentences with contextual ones
    while len(all_sentences) < num_sentences:
        all_sentences.append(random.choice(contextual_templates))
    
    # Shuffle to mix relevant and irrelevant
    random.shuffle(all_sentences)
    
    # Join into chunk (ensure proper sentence formatting)
    chunk_sentences = []
    for sent in all_sentences:
        sent = sent.strip()
        if not sent.endswith('.'):
            sent += '.'
        chunk_sentences.append(sent)
    
    chunk_text = " ".join(chunk_sentences)
    return chunk_text

# ============================================================================
# Query Templates and Generation
# ============================================================================

QUERY_TEMPLATES = [
    # Entity extraction queries
    {"type": "entity", "template": "who are the {role} of {company}?", "domain": "business"},
    {"type": "entity", "template": "what are the {items} of {entity}?", "domain": "general"},
    {"type": "entity", "template": "who is the {role} at {company}?", "domain": "business"},
    
    # List extraction queries
    {"type": "list", "template": "what are the {items} of {entity}?", "domain": "general"},
    {"type": "list", "template": "list the {items} related to {concept}.", "domain": "general"},
    {"type": "list", "template": "what {items} does {company} offer?", "domain": "business"},
    
    # Comparison queries
    {"type": "comparison", "template": "compare {entity1} and {entity2}.", "domain": "general"},
    {"type": "comparison", "template": "what is the difference between {entity1} and {entity2}?", "domain": "general"},
    
    # Analytical queries
    {"type": "analytical", "template": "why did {entity} {action}?", "domain": "general"},
    {"type": "analytical", "template": "what caused {event}?", "domain": "general"},
    
    # Relationship queries
    {"type": "relationship", "template": "how are {entity1} and {entity2} related?", "domain": "general"},
    {"type": "relationship", "template": "what is the connection between {entity1} and {entity2}?", "domain": "general"},
    
    # Process queries
    {"type": "process", "template": "how does {process} work?", "domain": "general"},
    {"type": "process", "template": "what is the process for {action}?", "domain": "general"},
]

def generate_query(template: Dict[str, Any]) -> str:
    """Generate a query from template"""
    query = template["template"]
    
    # Replace placeholders
    if "{role}" in query:
        query = query.replace("{role}", random.choice(["leaders", "members", "directors", "managers"]))
    if "{company}" in query:
        query = query.replace("{company}", generate_random_company())
    if "{items}" in query:
        query = query.replace("{items}", random.choice(["features", "benefits", "components", "advantages"]))
    if "{entity}" in query:
        query = query.replace("{entity}", random.choice([generate_random_company(), generate_random_concept()]))
    if "{concept}" in query:
        query = query.replace("{concept}", generate_random_concept())
    if "{entity1}" in query:
        entity1 = random.choice([generate_random_company(), generate_random_concept()])
        query = query.replace("{entity1}", entity1)
    if "{entity2}" in query:
        entity2 = random.choice([generate_random_company(), generate_random_concept()])
        query = query.replace("{entity2}", entity2)
    if "{action}" in query:
        query = query.replace("{action}", random.choice(["expand", "grow", "change", "improve"]))
    if "{event}" in query:
        query = query.replace("{event}", random.choice(["the expansion", "the growth", "the change"]))
    if "{process}" in query:
        query = query.replace("{process}", random.choice(["the system", "the process", "the workflow"]))
    
    return query

# ============================================================================
# Response Generation
# ============================================================================

def generate_response(query: str, relevant_info: List[str], query_type: str) -> str:
    """Generate expected response from relevant information"""
    
    if not relevant_info:
        return "I don't have that information in the provided documents."
    
    if query_type == "list":
        # Format as list
        items = [info.strip() for info in relevant_info if info.strip()]
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"
    
    elif query_type == "comparison":
        # Format with contrast words
        if len(relevant_info) >= 2:
            return f"{relevant_info[0]} while {relevant_info[1]}"
        return " ".join(relevant_info)
    
    elif query_type == "analytical":
        # Format with reasoning words
        response = " ".join(relevant_info)
        if "because" not in response.lower() and "due to" not in response.lower():
            response = f"because {response}"
        return response
    
    else:
        # Default: join all relevant info
        return " ".join(relevant_info)

# ============================================================================
# Dataset Generation
# ============================================================================

def generate_example(pattern_type: str = "mixed_content") -> Dict[str, Any]:
    """Generate a single training example"""
    
    # Select query template
    template = random.choice(QUERY_TEMPLATES)
    query = generate_query(template)
    query_type = template["type"]
    
    # Select system prompt variation
    prompt_type = random.choices(
        ["full", "medium", "short"],
        weights=[0.2, 0.6, 0.2]
    )[0]
    system_prompt = get_system_prompt_variation(prompt_type)
    
    # Generate relevant information (what should be extracted)
    num_relevant_items = random.randint(2, 4)
    relevant_info = []
    
    if query_type == "entity":
        # Generate entity names
        for _ in range(num_relevant_items):
            relevant_info.append(generate_random_name())
    elif query_type == "list":
        # Generate list items
        items = ["feature A", "feature B", "benefit C", "component D", "advantage E"]
        relevant_info = random.sample(items, num_relevant_items)
    else:
        # Generate factual statements
        for _ in range(num_relevant_items):
            relevant_info.append(f"Fact {random.randint(1, 100)} about the topic.")
    
    # Generate irrelevant information (similar but doesn't answer query)
    num_irrelevant = random.randint(3, 6)
    irrelevant_info = []
    for _ in range(num_irrelevant):
        if query_type == "entity":
            # Similar entities but wrong role/company
            irrelevant_info.append(f"{generate_random_name()} is {random.choice(['CEO', 'CTO', 'Manager'])} of {generate_random_company()}.")
        else:
            # Similar but unrelated facts
            irrelevant_info.append(f"Information about {generate_random_concept()} is discussed here.")
    
    # Generate chunks (3-4 chunks, 6-8 sentences each)
    num_chunks = random.randint(3, 4)
    chunks = []
    
    # Distribute relevant info across chunks (multiple instances)
    relevant_per_chunk = {}
    for i, info in enumerate(relevant_info):
        chunk_idx = i % num_chunks
        if chunk_idx not in relevant_per_chunk:
            relevant_per_chunk[chunk_idx] = []
        relevant_per_chunk[chunk_idx].append(info)
    
    # Also add some relevant info to multiple chunks (to teach complete extraction)
    for info in relevant_info[:2]:  # First 2 items appear in multiple chunks
        additional_chunks = random.sample(range(num_chunks), random.randint(1, 2))
        for chunk_idx in additional_chunks:
            if chunk_idx not in relevant_per_chunk:
                relevant_per_chunk[chunk_idx] = []
            if info not in relevant_per_chunk[chunk_idx]:
                relevant_per_chunk[chunk_idx].append(info)
    
    # Create chunks
    for chunk_idx in range(num_chunks):
        chunk_relevant = relevant_per_chunk.get(chunk_idx, [])
        # Select irrelevant info for this chunk
        num_irrelevant_needed = random.randint(2, 4)
        num_irrelevant_available = min(num_irrelevant_needed, len(irrelevant_info))
        if num_irrelevant_available > 0:
            chunk_irrelevant = random.sample(irrelevant_info, num_irrelevant_available)
        else:
            chunk_irrelevant = []
        
        # Convert to sentences
        relevant_sentences = [f"{info}." for info in chunk_relevant]
        irrelevant_sentences = [f"{info}." for info in chunk_irrelevant]
        
        # Create chunk with 6-8 sentences
        chunk_text = create_chunk(relevant_sentences, irrelevant_sentences, 
                                  num_sentences=random.randint(6, 8))
        
        # Format chunk with relevance score
        relevance_score = random.uniform(0.65, 0.95) if chunk_relevant else random.uniform(0.30, 0.60)
        chunks.append({
            "text": chunk_text,
            "score": round(relevance_score, 2),
            "file": "document.pdf"
        })
    
    # Generate expected response
    response = generate_response(query, relevant_info, query_type)
    
    # Format as training example
    user_content = f"Query: {query}\n\n"
    user_content += "RAG Chunks:\n"
    for i, chunk in enumerate(chunks, 1):
        user_content += f"[Chunk {i}] Score: {chunk['score']:.2f}, File: {chunk['file']}\n"
        user_content += f"FULL CHUNK TEXT: '{chunk['text']}'\n\n"
    
    example = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": response}
        ]
    }
    
    return example

# ============================================================================
# Main Generation
# ============================================================================

def main():
    """Generate 6000 training examples"""
    
    print("="*80)
    print("RAG Analysis Dataset Generator V2")
    print("="*80)
    print()
    print("Generating 6000 training examples...")
    print()
    
    # Pattern distribution (6000 examples)
    patterns = {
        "mixed_content": 900,      # 15% - Extract relevant, ignore irrelevant
        "multi_chunk": 1200,       # 20% - Extract from multiple chunks
        "role_filtering": 900,     # 15% - Filter by role/entity
        "cross_entity": 900,       # 15% - Filter by specific entity
        "synthesis": 600,          # 10% - Combine info from chunks
        "not_found": 600,          # 10% - Recognize missing info
        "comparison": 450,         # 7.5% - Compare entities
        "relationship": 450,       # 7.5% - Extract relationships
    }
    
    dataset = []
    total = sum(patterns.values())
    
    for pattern, count in patterns.items():
        print(f"Generating {count} examples for pattern: {pattern}...")
        for i in range(count):
            example = generate_example(pattern)
            dataset.append(example)
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{count} ({100*(i+1)/count:.1f}%)")
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Save dataset
    output_file = "rag_analysis_dataset_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print()
    print("="*80)
    print("✅ DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"Total examples: {len(dataset)}")
    print(f"Output file: {output_file}")
    print()
    
    # Verify distribution
    prompt_types = {"full": 0, "medium": 0, "short": 0}
    for example in dataset:
        system_content = example["messages"][0]["content"]
        if "ESSENTIAL GUIDELINES:" in system_content:
            prompt_types["full"] += 1
        elif "KEY RULES:" in system_content:
            prompt_types["medium"] += 1
        else:
            prompt_types["short"] += 1
    
    print("System Prompt Distribution:")
    for ptype, count in prompt_types.items():
        print(f"  {ptype}: {count} ({100*count/len(dataset):.1f}%)")
    print()
    print("="*80)

if __name__ == "__main__":
    main()

