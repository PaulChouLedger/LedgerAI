#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 2 Examples from Each Category
=================================
Demonstrates how the dataset handles different query types
"""

import json
import re
from collections import defaultdict

def categorize_query(query):
    """Categorize query into one of the 6 main categories."""
    query_lower = query.lower()
    
    # 1. Entity extraction queries
    if any(w in query_lower for w in ["who are", "list", "enumerate", "identify"]):
        return "entity_extraction"
    
    # 2. Single entity queries
    elif "who is" in query_lower:
        return "single_entity"
    elif "what is" in query_lower and "the" not in query_lower.split()[:3]:
        return "single_entity"
    
    # 3. Analytical queries
    elif "why did" in query_lower or "why" in query_lower:
        return "analytical"
    elif "how does" in query_lower or ("how" in query_lower and "work" in query_lower):
        return "analytical"
    elif "what caused" in query_lower:
        return "analytical"
    elif "what are the implications" in query_lower:
        return "analytical"
    
    # 4. Comparison/relationship queries
    elif "compare" in query_lower:
        return "comparison"
    elif "differences between" in query_lower:
        return "comparison"
    elif "similarities between" in query_lower:
        return "comparison"
    elif "related" in query_lower or "relationship" in query_lower:
        return "comparison"
    
    # 5. Temporal/location queries
    elif "when did" in query_lower or "when" in query_lower:
        return "temporal_location"
    elif "where is" in query_lower or "where" in query_lower:
        return "temporal_location"
    
    # 6. Attribute/description queries
    else:
        return "attribute_description"

def extract_chunks(user_content):
    """Extract chunks from user content."""
    chunks = []
    chunk_pattern = r'\[Chunk (\d+)\] Score: ([\d.]+), File: (.+?)\n\[(\d+)\] FULL CHUNK TEXT: \'(.+?)\''
    chunk_matches = re.finditer(chunk_pattern, user_content, re.DOTALL)
    
    for match in chunk_matches:
        chunk_num = int(match.group(1))
        score = float(match.group(2))
        file_name = match.group(3)
        chunk_text = match.group(5).replace("\\'", "'")
        
        chunks.append({
            "num": chunk_num,
            "score": score,
            "file": file_name,
            "text": chunk_text,
            "preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        })
    
    return chunks

def main():
    print("=" * 80)
    print("Running 2 Examples from Each Category")
    print("=" * 80)
    
    # Load dataset
    with open('rag_analysis_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Categorize all examples
    examples_by_category = defaultdict(list)
    for i, example in enumerate(dataset):
        query = example['messages'][1]['content'].split('Query:')[1].split('\n')[0] if 'Query:' in example['messages'][1]['content'] else ''
        category = categorize_query(query)
        examples_by_category[category].append(i)
    
    # Category names for display
    category_names = {
        "entity_extraction": "1. Entity Extraction",
        "single_entity": "2. Single Entity",
        "analytical": "3. Analytical",
        "comparison": "4. Comparison/Relationship",
        "temporal_location": "5. Temporal/Location",
        "attribute_description": "6. Attribute/Description"
    }
    
    # Sample 2 from each category
    import random
    for category in ["entity_extraction", "single_entity", "analytical", "comparison", "temporal_location", "attribute_description"]:
        if category not in examples_by_category:
            print(f"\n⚠️  Category '{category}' not found")
            continue
        
        indices = examples_by_category[category]
        if len(indices) < 2:
            print(f"\n⚠️  Category '{category}' has only {len(indices)} examples")
            sample_indices = indices
        else:
            sample_indices = random.sample(indices, 2)
        
        print(f"\n{'='*80}")
        print(f"{category_names[category]}")
        print(f"{'='*80}")
        
        for example_num, idx in enumerate(sample_indices, 1):
            example = dataset[idx]
            query = example['messages'][1]['content'].split('Query:')[1].split('\n')[0] if 'Query:' in example['messages'][1]['content'] else ''
            user_content = example['messages'][1]['content']
            response = example['messages'][2]['content']
            
            chunks = extract_chunks(user_content)
            
            print(f"\n--- Example {example_num} (Index {idx}) ---")
            print(f"\n📝 Query: {query}")
            print(f"\n📦 Chunks ({len(chunks)} total):")
            
            for chunk in chunks:
                print(f"\n  Chunk {chunk['num']} (Score: {chunk['score']:.2f}, File: {chunk['file']}):")
                print(f"    {chunk['preview']}")
            
            print(f"\n🤖 Response:")
            print(f"  {response}")
            
            # Check for cross-company co-founders (for entity extraction)
            if category == "entity_extraction" and "co-founder" in query.lower():
                org_match = re.search(r'(?:of|in|for)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
                if org_match:
                    target_org = org_match.group(1).strip()
                    other_company_cofounders = []
                    for chunk in chunks:
                        cofounder_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)\s+is\s+a\s+(?:co-?)?founder\s+of\s+([A-Z][a-zA-Z\s]+?)(?:,|\.)'
                        matches = re.finditer(cofounder_pattern, chunk['text'], re.IGNORECASE)
                        for m in matches:
                            person = m.group(1)
                            company = m.group(2).strip()
                            if company.lower() != target_org.lower():
                                other_company_cofounders.append((person, company))
                    
                    if other_company_cofounders:
                        print(f"\n  ✅ Cross-company filtering test:")
                        for person, company in other_company_cofounders:
                            if person in response:
                                print(f"    ❌ {person} (co-founder of {company}) incorrectly included!")
                            else:
                                print(f"    ✅ {person} (co-founder of {company}) correctly excluded")
            
            print()

if __name__ == "__main__":
    main()

