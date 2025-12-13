#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed Examples from Each Category
====================================
Shows 2 complete examples from each of the 6 categories with full details
"""

import json
import re
from collections import defaultdict

def categorize_query(query):
    """Categorize query into one of the 6 main categories."""
    query_lower = query.lower()
    
    if any(w in query_lower for w in ["who are", "list", "enumerate", "identify"]):
        return "entity_extraction"
    elif "who is" in query_lower:
        return "single_entity"
    elif "what is" in query_lower and "the" not in query_lower.split()[:3]:
        return "single_entity"
    elif "why did" in query_lower or "why" in query_lower or "how does" in query_lower or ("how" in query_lower and "work" in query_lower) or "what caused" in query_lower or "what are the implications" in query_lower:
        return "analytical"
    elif "compare" in query_lower or "differences between" in query_lower or "similarities between" in query_lower or "related" in query_lower or "relationship" in query_lower:
        return "comparison"
    elif "when did" in query_lower or "when" in query_lower or "where is" in query_lower or "where" in query_lower:
        return "temporal_location"
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
            "text": chunk_text
        })
    
    return chunks

def analyze_entity_extraction(query, chunks, response, target_org=None):
    """Analyze entity extraction example."""
    analysis = []
    
    if target_org:
        # Check for cross-company filtering
        other_company_entities = []
        target_company_entities = []
        
        for chunk in chunks:
            # Find co-founder/leader mentions
            pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)\s+is\s+a\s+(?:co-?)?(?:founder|leader|director|member)\s+of\s+([A-Z][a-zA-Z\s]+?)(?:,|\.)'
            matches = re.finditer(pattern, chunk['text'], re.IGNORECASE)
            
            for m in matches:
                person = m.group(1)
                company = m.group(2).strip()
                if company.lower() == target_org.lower():
                    target_company_entities.append((person, company, chunk['num']))
                else:
                    other_company_entities.append((person, company, chunk['num']))
        
        if target_company_entities:
            analysis.append(f"✅ Found {len(target_company_entities)} entity/entities of target company '{target_org}':")
            for person, company, chunk_num in target_company_entities:
                analysis.append(f"   - {person} (in Chunk {chunk_num})")
        
        if other_company_entities:
            analysis.append(f"⚠️  Found {len(other_company_entities)} entity/entities of OTHER companies (should be filtered):")
            for person, company, chunk_num in other_company_entities:
                if person in response:
                    analysis.append(f"   ❌ {person} (co-founder of {company}, Chunk {chunk_num}) - INCORRECTLY INCLUDED")
                else:
                    analysis.append(f"   ✅ {person} (co-founder of {company}, Chunk {chunk_num}) - CORRECTLY EXCLUDED")
    
    # Check response format
    if "The entities are:" in response:
        entities = [e.strip().rstrip('.') for e in response.split("The entities are:")[-1].split(',')]
        analysis.append(f"✅ Response lists {len(entities)} entities in correct format")
    elif "The items are:" in response:
        items = [i.strip().rstrip('.') for i in response.split("The items are:")[-1].split(',')]
        analysis.append(f"✅ Response lists {len(items)} items in correct format")
    
    return analysis

def main():
    print("=" * 80)
    print("DETAILED EXAMPLES FROM EACH CATEGORY")
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
    
    # Category names
    category_names = {
        "entity_extraction": "1. ENTITY EXTRACTION",
        "single_entity": "2. SINGLE ENTITY",
        "analytical": "3. ANALYTICAL",
        "comparison": "4. COMPARISON/RELATIONSHIP",
        "temporal_location": "5. TEMPORAL/LOCATION",
        "attribute_description": "6. ATTRIBUTE/DESCRIPTION"
    }
    
    # Sample 2 from each category
    import random
    for category in ["entity_extraction", "single_entity", "analytical", "comparison", "temporal_location", "attribute_description"]:
        if category not in examples_by_category:
            continue
        
        indices = examples_by_category[category]
        if len(indices) < 2:
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
            
            print(f"\n{'─'*80}")
            print(f"EXAMPLE {example_num} (Dataset Index: {idx})")
            print(f"{'─'*80}")
            
            print(f"\n📝 QUERY:")
            print(f"   {query}")
            
            # Extract target organization if applicable
            target_org = None
            if "co-founder" in query.lower() or "founder" in query.lower() or "leader" in query.lower():
                org_match = re.search(r'(?:of|in|for)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
                if org_match:
                    target_org = org_match.group(1).strip()
                    print(f"   Target: {target_org}")
            
            print(f"\n📦 RAG CHUNKS PROVIDED ({len(chunks)} total):")
            print()
            
            for chunk in chunks:
                print(f"   ┌─ Chunk {chunk['num']} ─────────────────────────────────────────────────────")
                print(f"   │ Score: {chunk['score']:.2f} | File: {chunk['file']}")
                print(f"   │")
                # Format text with proper line breaks
                text_lines = chunk['text'].split('. ')
                for i, line in enumerate(text_lines):
                    if line.strip():
                        print(f"   │ {line.strip()}{'.' if not line.endswith('.') else ''}")
                print(f"   └─────────────────────────────────────────────────────────────────────")
                print()
            
            print(f"🤖 MODEL RESPONSE:")
            print(f"   {response}")
            
            # Category-specific analysis
            print(f"\n🔍 ANALYSIS:")
            
            if category == "entity_extraction":
                analysis = analyze_entity_extraction(query, chunks, response, target_org)
                for line in analysis:
                    print(f"   {line}")
            
            elif category == "single_entity":
                if "who is" in query.lower():
                    # Check if response has person name
                    name_match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', response)
                    if name_match:
                        print(f"   ✅ Response contains person name: {name_match.group(1)}")
                    else:
                        print(f"   ⚠️  Response doesn't contain a person name")
                elif "what is" in query.lower():
                    if len(response.strip()) > 20:
                        print(f"   ✅ Response provides description ({len(response)} chars)")
                    else:
                        print(f"   ⚠️  Response seems too short")
            
            elif category == "analytical":
                if "I don't have" in response or "I couldn't find" in response:
                    print(f"   ✅ Correctly returns 'not found' - analytical reasoning not in chunks")
                else:
                    print(f"   ✅ Response attempts to provide analytical answer")
            
            elif category == "comparison":
                # Extract entities from query
                entity1_match = re.search(r'compare (.+?) and', query.lower())
                entity2_match = re.search(r'and (.+?)[\?\.]', query.lower())
                if not entity1_match:
                    entity1_match = re.search(r'between (.+?) and', query.lower())
                
                if entity1_match and entity2_match:
                    entity1 = entity1_match.group(1).strip()
                    entity2 = entity2_match.group(1).strip()
                    if entity1.lower() in response.lower() and entity2.lower() in response.lower():
                        print(f"   ✅ Response mentions both entities: {entity1} and {entity2}")
                    else:
                        print(f"   ⚠️  Response may not mention both entities")
            
            elif category == "temporal_location":
                if "I don't have" in response or "I couldn't find" in response:
                    print(f"   ✅ Correctly returns 'not found' - temporal/location info not in chunks")
                else:
                    # Check if response has date or location
                    date_match = re.search(r'\d{4}|\d{1,2}/\d{1,2}/\d{2,4}', response)
                    if date_match:
                        print(f"   ✅ Response contains date/time information")
                    else:
                        print(f"   ⚠️  Response doesn't contain clear temporal/location info")
            
            elif category == "attribute_description":
                if "I don't have" in response or "I couldn't find" in response:
                    print(f"   ✅ Correctly returns 'not found' - attribute info not in chunks")
                else:
                    print(f"   ✅ Response attempts to provide attribute/description")
            
            # Check if it's a failed query
            is_failed = any(phrase in response for phrase in [
                "I don't have that information",
                "I couldn't find",
                "I don't have information to answer"
            ])
            
            if is_failed:
                print(f"\n   📌 This is a FAILED QUERY example (teaches model to say 'not found')")
            else:
                print(f"\n   📌 This is a SUCCESSFUL QUERY example (model extracts correct information)")
            
            print()

if __name__ == "__main__":
    main()

