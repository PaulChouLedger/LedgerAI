#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Simulation and Validation Script
========================================
Simulates 50 random queries from the dataset to verify:
1. Proper formatting
2. Response quality
3. Chunk structure
4. Training compatibility
"""

import json
import random
import re
from collections import Counter

def simulate_query(example, index):
    """Simulate a single query and analyze its structure."""
    messages = example.get("messages", [])
    
    if len(messages) < 3:
        return {
            "index": index,
            "valid": False,
            "error": "Missing messages (need system, user, assistant)"
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
        
        # Count sentences in chunk
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
    
    # Check response quality
    response_issues = []
    
    # Check if response is too generic
    generic_responses = [
        "I don't have that information",
        "I couldn't find",
        "I don't have information to answer"
    ]
    is_generic = any(phrase in assistant_response for phrase in generic_responses)
    
    # Check if response contains entity names (for entity queries)
    query_lower = query.lower()
    is_entity_query = any(w in query_lower for w in ["who are", "list", "identify"])
    
    if is_entity_query:
        # Check if response has proper entity format
        if "The entities are:" in assistant_response:
            entities_part = assistant_response.split("The entities are:")[-1].strip()
            entities = [e.strip().rstrip('.') for e in entities_part.split(',')]
            
            # Check if entities are proper names
            proper_names = []
            for entity in entities:
                if len(entity.split()) == 2:
                    parts = entity.split()
                    if parts[0][0].isupper() and parts[1][0].isupper():
                        proper_names.append(entity)
            
            if len(proper_names) < len(entities):
                response_issues.append(f"Some entities are not proper names: {entities}")
        elif not is_generic:
            response_issues.append("Entity query but response doesn't list entities properly")
    
    # Check chunk quality
    chunk_issues = []
    if len(chunks) < 3:
        chunk_issues.append(f"Only {len(chunks)} chunks (expected 3-4)")
    
    for chunk in chunks:
        if chunk["sentence_count"] < 6:
            chunk_issues.append(f"Chunk {chunk['num']} has only {chunk['sentence_count']} sentences (target: 6-8)")
        if chunk["score"] < 0.50:
            chunk_issues.append(f"Chunk {chunk['num']} has low score {chunk['score']} (might be irrelevant)")
    
    return {
        "index": index,
        "valid": True,
        "query": query,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "response": assistant_response,
        "response_length": len(assistant_response),
        "is_generic": is_generic,
        "response_issues": response_issues,
        "chunk_issues": chunk_issues,
        "has_issues": len(response_issues) > 0 or len(chunk_issues) > 0
    }

def main():
    print("=" * 80)
    print("Dataset Simulation and Validation")
    print("=" * 80)
    
    # Load dataset
    with open('rag_analysis_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"\nTotal examples in dataset: {len(dataset)}")
    print(f"Simulating 50 random queries...\n")
    
    # Sample 50 random examples
    sample_indices = random.sample(range(len(dataset)), min(50, len(dataset)))
    results = []
    
    for idx in sample_indices:
        result = simulate_query(dataset[idx], idx)
        results.append(result)
    
    # Analyze results
    valid_count = sum(1 for r in results if r.get("valid", False))
    invalid_count = len(results) - valid_count
    
    print("=" * 80)
    print("Validation Results")
    print("=" * 80)
    print(f"Valid examples: {valid_count}/50")
    print(f"Invalid examples: {invalid_count}/50")
    
    if invalid_count > 0:
        print("\n❌ Invalid examples found:")
        for r in results:
            if not r.get("valid", False):
                print(f"  Index {r['index']}: {r.get('error', 'Unknown error')}")
    
    # Analyze valid examples
    valid_results = [r for r in results if r.get("valid", False)]
    
    if valid_results:
        # Chunk statistics
        chunk_counts = Counter(r["num_chunks"] for r in valid_results)
        print(f"\n📊 Chunk Distribution:")
        for count, num in sorted(chunk_counts.items()):
            print(f"  {count} chunks: {num} examples")
        
        # Sentence count statistics
        all_sentence_counts = []
        for r in valid_results:
            for chunk in r["chunks"]:
                all_sentence_counts.append(chunk["sentence_count"])
        
        if all_sentence_counts:
            avg_sentences = sum(all_sentence_counts) / len(all_sentence_counts)
            min_sentences = min(all_sentence_counts)
            max_sentences = max(all_sentence_counts)
            print(f"\n📊 Sentence Count Statistics:")
            print(f"  Average: {avg_sentences:.1f} sentences per chunk")
            print(f"  Min: {min_sentences}, Max: {max_sentences}")
            print(f"  Target: 6-8 sentences")
            
            in_range = sum(1 for c in all_sentence_counts if 6 <= c <= 8)
            print(f"  In target range: {in_range}/{len(all_sentence_counts)} ({100*in_range/len(all_sentence_counts):.1f}%)")
        
        # Response statistics
        generic_count = sum(1 for r in valid_results if r.get("is_generic", False))
        print(f"\n📊 Response Statistics:")
        print(f"  Generic 'not found' responses: {generic_count}/{len(valid_results)} ({100*generic_count/len(valid_results):.1f}%)")
        
        avg_response_length = sum(r["response_length"] for r in valid_results) / len(valid_results)
        print(f"  Average response length: {avg_response_length:.1f} characters")
        
        # Issues
        issues_count = sum(1 for r in valid_results if r.get("has_issues", False))
        print(f"\n⚠️  Examples with issues: {issues_count}/{len(valid_results)}")
        
        if issues_count > 0:
            print("\nIssues found:")
            issue_types = Counter()
            for r in valid_results:
                if r.get("has_issues", False):
                    if r.get("response_issues"):
                        for issue in r["response_issues"]:
                            issue_types[issue.split(':')[0] if ':' in issue else issue] += 1
                    if r.get("chunk_issues"):
                        for issue in r["chunk_issues"]:
                            issue_types[issue.split(':')[0] if ':' in issue else issue] += 1
            
            for issue, count in issue_types.most_common(10):
                print(f"  {issue}: {count} occurrences")
    
    # Show sample examples
    print("\n" + "=" * 80)
    print("Sample Examples (First 5)")
    print("=" * 80)
    
    for i, r in enumerate(valid_results[:5], 1):
        print(f"\nExample {i} (Index {r['index']}):")
        print(f"  Query: {r['query']}")
        print(f"  Chunks: {r['num_chunks']}")
        print(f"  Response: {r['response'][:150]}...")
        if r.get("has_issues", False):
            print(f"  ⚠️  Issues:")
            for issue in r.get("response_issues", []):
                print(f"    - {issue}")
            for issue in r.get("chunk_issues", []):
                print(f"    - {issue}")
    
    # Check training compatibility
    print("\n" + "=" * 80)
    print("Training Compatibility Check")
    print("=" * 80)
    
    # Check if format matches training script expectations
    compatibility_issues = []
    
    for r in valid_results[:10]:  # Check first 10
        example = dataset[r["index"]]
        messages = example.get("messages", [])
        
        # Check message structure
        roles = [m.get("role") for m in messages]
        if "system" not in roles:
            compatibility_issues.append("Missing system message")
        if "user" not in roles:
            compatibility_issues.append("Missing user message")
        if "assistant" not in roles:
            compatibility_issues.append("Missing assistant message")
        
        # Check if can be formatted with chat template
        try:
            # This simulates what training script does
            if len(messages) >= 2:
                # Check if messages have required fields
                for msg in messages:
                    if "role" not in msg or "content" not in msg:
                        compatibility_issues.append("Message missing role or content")
        except Exception as e:
            compatibility_issues.append(f"Formatting error: {e}")
    
    if compatibility_issues:
        unique_issues = Counter(compatibility_issues)
        print("⚠️  Compatibility issues found:")
        for issue, count in unique_issues.items():
            print(f"  {issue}: {count} occurrences")
    else:
        print("✅ All checked examples are compatible with training script format")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if invalid_count == 0 and issues_count == 0 and not compatibility_issues:
        print("✅ Dataset appears to be properly formatted and ready for training!")
    else:
        print("⚠️  Dataset has some issues that may affect training:")
        if invalid_count > 0:
            print(f"  - {invalid_count} invalid examples")
        if issues_count > len(valid_results) * 0.1:  # More than 10% have issues
            print(f"  - {issues_count} examples with quality issues (>10% threshold)")
        if compatibility_issues:
            print(f"  - {len(set(compatibility_issues))} compatibility issues found")
        print("\n💡 Consider fixing these issues before training to improve results.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

