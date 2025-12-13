#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Category-Based Dataset Validation
================================================
Validates 10 examples from each query category to ensure:
1. Proper formatting
2. Response quality
3. Chunk structure
4. Category-specific requirements
"""

import json
import re
from collections import Counter, defaultdict

def categorize_query(query, response):
    """Categorize query into one of the 6 main categories."""
    query_lower = query.lower()
    
    # 1. Entity extraction queries
    if any(w in query_lower for w in ["who are", "list", "enumerate", "identify"]):
        if "co-founder" in query_lower or "founder" in query_lower:
            return "entity_extraction", "co-founder"
        elif "leader" in query_lower:
            return "entity_extraction", "leader"
        elif "director" in query_lower:
            return "entity_extraction", "director"
        elif "list" in query_lower:
            return "entity_extraction", "list"
        else:
            return "entity_extraction", "other"
    
    # 2. Single entity queries
    elif "who is" in query_lower:
        return "single_entity", "who_is"
    elif "what is" in query_lower and "the" not in query_lower.split()[:3]:
        return "single_entity", "what_is"
    
    # 3. Analytical queries
    elif "why did" in query_lower or "why" in query_lower:
        return "analytical", "why"
    elif "how does" in query_lower or ("how" in query_lower and "work" in query_lower):
        return "analytical", "how"
    elif "what caused" in query_lower:
        return "analytical", "what_caused"
    elif "what are the implications" in query_lower:
        return "analytical", "implications"
    
    # 4. Comparison/relationship queries
    elif "compare" in query_lower:
        return "comparison", "compare"
    elif "differences between" in query_lower:
        return "comparison", "differences"
    elif "similarities between" in query_lower:
        return "comparison", "similarities"
    elif "related" in query_lower or "relationship" in query_lower:
        return "comparison", "relationship"
    
    # 5. Temporal/location queries
    elif "when did" in query_lower or "when" in query_lower:
        return "temporal_location", "when"
    elif "where is" in query_lower or "where" in query_lower:
        return "temporal_location", "where"
    
    # 6. Attribute/description queries
    elif "what are the" in query_lower and "in" in query_lower:
        return "attribute_description", "what_are_items"
    elif "what properties" in query_lower:
        return "attribute_description", "properties"
    elif "what are the characteristics" in query_lower:
        return "attribute_description", "characteristics"
    elif "describe" in query_lower:
        return "attribute_description", "describe"
    else:
        return "attribute_description", "other"

def analyze_example(example, index):
    """Analyze a single example for quality and issues."""
    messages = example.get("messages", [])
    
    if len(messages) < 3:
        return {
            "index": index,
            "valid": False,
            "error": "Missing messages"
        }
    
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    
    if not all([system_msg, user_msg, assistant_msg]):
        return {
            "index": index,
            "valid": False,
            "error": "Missing required message roles"
        }
    
    # Extract query
    user_content = user_msg.get("content", "")
    query_match = re.search(r'Query:\s*(.+?)(?:\n|$)', user_content)
    query = query_match.group(1).strip() if query_match else ""
    
    # Extract chunks
    chunks = []
    chunk_pattern = r'\[Chunk (\d+)\] Score: ([\d.]+), File: (.+?)\n\[(\d+)\] FULL CHUNK TEXT: \'(.+?)\''
    chunk_matches = re.finditer(chunk_pattern, user_content, re.DOTALL)
    
    for match in chunk_matches:
        chunk_num = int(match.group(1))
        score = float(match.group(2))
        file_name = match.group(3)
        chunk_text = match.group(5).replace("\\'", "'")
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', chunk_text) if s.strip() and len(s.strip()) > 10]
        
        chunks.append({
            "num": chunk_num,
            "score": score,
            "file": file_name,
            "text_length": len(chunk_text),
            "sentence_count": len(sentences),
            "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        })
    
    # Analyze assistant response
    assistant_response = assistant_msg.get("content", "")
    
    # Categorize
    category, subcategory = categorize_query(query, assistant_response)
    
    # Check for issues
    issues = []
    
    # Check for filler sentences
    filler_phrases = [
        "The discussion encompasses",
        "This information is part of",
        "Additional details provide",
        "Further exploration reveals",
        "Understanding this requires"
    ]
    has_filler = any(phrase in assistant_response for phrase in filler_phrases)
    if has_filler:
        issues.append("Contains filler sentence")
    
    # Check if response is too generic
    is_generic = any(phrase in assistant_response for phrase in [
        "I don't have that information",
        "I couldn't find",
        "I don't have information to answer"
    ])
    
    # Category-specific checks
    if category == "entity_extraction":
        if not is_generic:
            # Check for both "The entities are:" (for person queries) and "The items are:" (for list queries)
            if "The entities are:" in assistant_response:
                entities_part = assistant_response.split("The entities are:")[-1].strip()
                entities = [e.strip().rstrip('.') for e in entities_part.split(',')]
                if len(entities) == 0:
                    issues.append("Entity extraction query but no entities listed")
                else:
                    # Check if entities are proper names
                    non_names = [e for e in entities if len(e.split()) != 2 or 
                                not (e.split()[0][0].isupper() and e.split()[1][0].isupper())]
                    if non_names:
                        issues.append(f"Non-name entities: {non_names}")
            elif "The items are:" in assistant_response:
                # This is correct for "list" queries
                items_part = assistant_response.split("The items are:")[-1].strip()
                items = [e.strip().rstrip('.') for e in items_part.split(',')]
                if len(items) == 0:
                    issues.append("List query but no items listed")
            elif "list" not in query.lower():
                # Only flag if it's not a list query
                issues.append("Entity extraction query but response doesn't list entities")
    
    elif category == "single_entity":
        if not is_generic:
            if "who is" in query.lower():
                # Should have a person name
                name_match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', assistant_response)
                if not name_match:
                    issues.append("'who is' query but no person name in response")
            elif "what is" in query.lower():
                # Should have a description
                if len(assistant_response.strip()) < 20:
                    issues.append("'what is' query but response too short")
    
    elif category == "analytical":
        if not is_generic:
            if len(assistant_response.strip()) < 30:
                issues.append("Analytical query but response too short")
    
    elif category == "comparison":
        if not is_generic:
            # Should mention both entities
            entity1_match = re.search(r'compare (.+?) and', query.lower())
            entity2_match = re.search(r'and (.+?)[\?\.]', query.lower())
            if entity1_match and entity2_match:
                entity1 = entity1_match.group(1).strip()
                entity2 = entity2_match.group(1).strip()
                if entity1.lower() not in assistant_response.lower() or entity2.lower() not in assistant_response.lower():
                    issues.append("Comparison query but doesn't mention both entities")
    
    # Check chunk quality
    chunk_issues = []
    if len(chunks) < 3:
        chunk_issues.append(f"Only {len(chunks)} chunks")
    
    for chunk in chunks:
        if chunk["sentence_count"] < 6:
            chunk_issues.append(f"Chunk {chunk['num']}: {chunk['sentence_count']} sentences")
        if chunk["score"] < 0.50:
            chunk_issues.append(f"Chunk {chunk['num']}: low score {chunk['score']}")
    
    return {
        "index": index,
        "valid": True,
        "query": query,
        "category": category,
        "subcategory": subcategory,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "response": assistant_response,
        "response_length": len(assistant_response),
        "is_generic": is_generic,
        "has_filler": has_filler,
        "issues": issues,
        "chunk_issues": chunk_issues,
        "has_issues": len(issues) > 0 or len(chunk_issues) > 0
    }

def main():
    print("=" * 80)
    print("Comprehensive Category-Based Dataset Validation")
    print("=" * 80)
    
    # Load dataset
    with open('rag_analysis_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"\nTotal examples in dataset: {len(dataset)}")
    print("Analyzing examples by category...\n")
    
    # First pass: categorize all examples
    examples_by_category = defaultdict(list)
    for i, example in enumerate(dataset):
        query = example['messages'][1]['content'].split('Query:')[1].split('\n')[0] if 'Query:' in example['messages'][1]['content'] else ''
        response = example['messages'][2]['content']
        category, subcategory = categorize_query(query, response)
        examples_by_category[category].append(i)
    
    print("Category distribution:")
    for category, indices in sorted(examples_by_category.items()):
        print(f"  {category}: {len(indices)} examples ({100*len(indices)/len(dataset):.1f}%)")
    
    # Sample 10 from each category
    print("\n" + "=" * 80)
    print("Sampling 10 examples from each category for validation")
    print("=" * 80)
    
    category_results = {}
    
    for category in ["entity_extraction", "single_entity", "analytical", "comparison", "temporal_location", "attribute_description"]:
        if category not in examples_by_category:
            print(f"\n⚠️  Category '{category}' not found in dataset")
            continue
        
        indices = examples_by_category[category]
        if len(indices) < 10:
            print(f"\n⚠️  Category '{category}' has only {len(indices)} examples (need 10)")
            sample_indices = indices
        else:
            import random
            sample_indices = random.sample(indices, 10)
        
        print(f"\n{'='*80}")
        print(f"Category: {category.upper().replace('_', ' ')} ({len(sample_indices)} examples)")
        print(f"{'='*80}")
        
        results = []
        for idx in sample_indices:
            result = analyze_example(dataset[idx], idx)
            results.append(result)
        
        category_results[category] = results
        
        # Summary for this category
        valid_count = sum(1 for r in results if r.get("valid", False))
        issues_count = sum(1 for r in results if r.get("has_issues", False) and r.get("valid", False))
        filler_count = sum(1 for r in results if r.get("has_filler", False) and r.get("valid", False))
        generic_count = sum(1 for r in results if r.get("is_generic", False) and r.get("valid", False))
        
        print(f"\nSummary:")
        print(f"  Valid: {valid_count}/{len(results)}")
        print(f"  With issues: {issues_count}/{len(results)}")
        print(f"  With filler: {filler_count}/{len(results)}")
        print(f"  Generic (not found): {generic_count}/{len(results)}")
        
        # Show examples with issues
        if issues_count > 0:
            print(f"\n⚠️  Examples with issues:")
            for r in results:
                if r.get("has_issues", False):
                    print(f"\n  Index {r['index']}: {r['query']}")
                    if r.get("issues"):
                        for issue in r["issues"]:
                            print(f"    - {issue}")
                    if r.get("chunk_issues"):
                        for issue in r["chunk_issues"]:
                            print(f"    - {issue}")
        
        # Show sample responses
        print(f"\nSample responses (first 3):")
        for i, r in enumerate(results[:3], 1):
            if r.get("valid", False):
                print(f"\n  {i}. Query: {r['query']}")
                print(f"     Response: {r['response'][:150]}...")
                if r.get("has_issues", False):
                    print(f"     ⚠️  Has issues")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("Overall Validation Summary")
    print("=" * 80)
    
    all_results = []
    for results in category_results.values():
        all_results.extend(results)
    
    total_valid = sum(1 for r in all_results if r.get("valid", False))
    total_issues = sum(1 for r in all_results if r.get("has_issues", False) and r.get("valid", False))
    total_filler = sum(1 for r in all_results if r.get("has_filler", False) and r.get("valid", False))
    
    print(f"\nTotal examples analyzed: {len(all_results)}")
    print(f"Valid examples: {total_valid}/{len(all_results)}")
    print(f"Examples with issues: {total_issues}/{len(all_results)} ({100*total_issues/len(all_results):.1f}%)")
    print(f"Examples with filler: {total_filler}/{len(all_results)} ({100*total_filler/len(all_results):.1f}%)")
    
    # Issue breakdown
    all_issues = []
    for r in all_results:
        if r.get("valid", False) and r.get("issues"):
            all_issues.extend(r["issues"])
    
    if all_issues:
        issue_counts = Counter(all_issues)
        print(f"\nMost common issues:")
        for issue, count in issue_counts.most_common(5):
            print(f"  {issue}: {count} occurrences")
    
    # Final verdict
    print("\n" + "=" * 80)
    if total_issues == 0 and total_filler == 0:
        print("✅ ALL CATEGORIES PASSED VALIDATION!")
        print("✅ Dataset is ready for training across all categories!")
    elif total_issues / len(all_results) < 0.1:  # Less than 10% have issues
        print("✅ Dataset is mostly ready for training!")
        print(f"⚠️  {total_issues} examples have minor issues (acceptable threshold)")
    else:
        print("⚠️  Dataset needs fixes before training")
        print(f"❌ {total_issues} examples have issues (>10% threshold)")
    print("=" * 80)

if __name__ == "__main__":
    main()

