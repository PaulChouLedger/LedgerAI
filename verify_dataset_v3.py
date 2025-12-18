#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Dataset v3 Format
========================
Checks for bugs in:
1. "not_found" cases - should NOT have matching entities in chunks
2. Multiple entities cases - should have entities distributed across chunks
3. JSON format validity
"""

import json
import re
from typing import Dict, Any, List

def extract_entities_from_chunk(chunk_text: str, query: str) -> List[str]:
    """Extract entity names from chunk text (similar to training monitor)"""
    entities = set()
    
    # Check if query is asking for entities
    is_entity_query = any(phrase in query.lower() for phrase in [
        "who are the", "who is the", "list the", "what are the"
    ])
    
    if not is_entity_query:
        return []
    
    # Extract role from query
    role_pattern = r"(leaders|members|directors|managers|executives|founders|co-founders)"
    role_match = re.search(role_pattern, query.lower())
    role = role_match.group(1) if role_match else None
    
    # Extract company from query
    company_pattern = r"(?:of|at)\s+([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*)"
    company_match = re.search(company_pattern, query)
    company = company_match.group(1) if company_match else None
    
    # Entity extraction patterns
    name_patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+) serves as',
        r'As [^,]+, ([A-Z][a-z]+ [A-Z][a-z]+)',
        r'([A-Z][a-z]+ [A-Z][a-z]+) holds the position',
        r'([A-Z][a-z]+ [A-Z][a-z]+) is (?:executive|manager|director|founder|co-founder|leader|member)',
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, chunk_text, re.IGNORECASE)
        entities.update(matches)
    
    # Simple pattern: Capitalized First Last name
    simple_names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', chunk_text)
    false_positives = {'Smart Systems', 'Data Systems', 'Cloud Systems', 'AI Systems', 'Tech Systems'}
    for name in simple_names:
        if name not in false_positives and len(name.split()) == 2:
            name_lower = name.lower()
            chunk_lower = chunk_text.lower()
            name_idx = chunk_lower.find(name_lower)
            if name_idx >= 0:
                context_start = max(0, name_idx - 50)
                context_end = min(len(chunk_lower), name_idx + len(name) + 50)
                context = chunk_lower[context_start:context_end]
                role_words = ['executive', 'manager', 'director', 'founder', 'co-founder', 'leader', 'member', 'serves', 'holds', 'position']
                if any(role in context for role in role_words):
                    entities.add(name)
    
    return sorted(list(entities))

def verify_not_found_examples(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify that not_found examples don't have matching entities in chunks"""
    not_found_examples = []
    bugs = []
    
    for i, example in enumerate(dataset):
        messages = example.get("messages", [])
        if len(messages) < 3:
            continue
        
        assistant_msg = messages[2].get("content", "")
        user_msg = messages[1].get("content", "")
        
        # Parse assistant response
        try:
            response_json = json.loads(assistant_msg)
            answer_type = response_json.get("answer_type", "")
            
            if answer_type == "not_found":
                not_found_examples.append(i)
                
                # Extract query
                query_match = re.search(r'Query:\s*(.+?)(?:\n\nRAG|$)', user_msg, re.DOTALL)
                if not query_match:
                    continue
                query = query_match.group(1).strip()
                
                # Extract chunks
                chunk_pattern = r'\[Chunk (\d+)\] Score: ([\d.]+).*?FULL CHUNK TEXT: [\'"](.+?)[\'"]'
                chunks = re.findall(chunk_pattern, user_msg, re.DOTALL)
                
                # Check each chunk for entities
                found_entities = []
                for chunk_num, chunk_score, chunk_text in chunks:
                    chunk_text_clean = chunk_text.replace("\\'", "'").replace('\\"', '"')
                    entities = extract_entities_from_chunk(chunk_text_clean, query)
                    if entities:
                        found_entities.extend(entities)
                
                # BUG: If entities found in chunks but answer_type is "not_found"
                if found_entities:
                    bugs.append({
                        "index": i,
                        "query": query,
                        "found_entities": found_entities,
                        "chunks_checked": len(chunks)
                    })
        except (json.JSONDecodeError, ValueError):
            continue
    
    return {
        "total_not_found": len(not_found_examples),
        "bugs": bugs,
        "bug_count": len(bugs)
    }

def verify_multi_entity_examples(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify that multi-entity examples have entities distributed across chunks"""
    multi_entity_examples = []
    issues = []
    
    for i, example in enumerate(dataset):
        messages = example.get("messages", [])
        if len(messages) < 3:
            continue
        
        assistant_msg = messages[2].get("content", "")
        user_msg = messages[1].get("content", "")
        
        # Parse assistant response
        try:
            response_json = json.loads(assistant_msg)
            answer_type = response_json.get("answer_type", "")
            items = response_json.get("items", [])
            chunks_used = response_json.get("chunks_used", [])
            
            # Check multi-entity examples (entities or list with multiple items)
            if answer_type in ["entities", "list"] and len(items) >= 2:
                multi_entity_examples.append(i)
                
                # Extract query
                query_match = re.search(r'Query:\s*(.+?)(?:\n\nRAG|$)', user_msg, re.DOTALL)
                if not query_match:
                    continue
                query = query_match.group(1).strip()
                
                # Extract chunks
                chunk_pattern = r'\[Chunk (\d+)\] Score: ([\d.]+).*?FULL CHUNK TEXT: [\'"](.+?)[\'"]'
                chunks = re.findall(chunk_pattern, user_msg, re.DOTALL)
                
                # Check if entities are distributed across chunks
                entities_per_chunk = {}
                for chunk_num, chunk_score, chunk_text in chunks:
                    chunk_num = int(chunk_num)
                    chunk_text_clean = chunk_text.replace("\\'", "'").replace('\\"', '"')
                    entities = extract_entities_from_chunk(chunk_text_clean, query)
                    if entities:
                        entities_per_chunk[chunk_num] = entities
                
                # Check if all expected items are found in chunks
                found_items = set()
                for chunk_entities in entities_per_chunk.values():
                    found_items.update(chunk_entities)
                
                missing_items = set(items) - found_items
                if missing_items:
                    issues.append({
                        "index": i,
                        "query": query,
                        "expected_items": items,
                        "found_items": list(found_items),
                        "missing_items": list(missing_items),
                        "chunks_with_entities": len(entities_per_chunk),
                        "chunks_used": chunks_used
                    })
                
                # Check if entities are distributed (for multi_chunk pattern)
                if len(entities_per_chunk) == 1 and len(items) >= 3:
                    # All entities in one chunk - might be okay, but note it
                    issues.append({
                        "index": i,
                        "query": query,
                        "issue": "all_entities_in_one_chunk",
                        "expected_items": len(items),
                        "chunks_with_entities": 1,
                        "chunks_used": chunks_used
                    })
        except (json.JSONDecodeError, ValueError):
            continue
    
    return {
        "total_multi_entity": len(multi_entity_examples),
        "issues": issues,
        "issue_count": len(issues)
    }

def verify_json_format(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify JSON format validity"""
    invalid_json = []
    invalid_structure = []
    
    for i, example in enumerate(dataset):
        messages = example.get("messages", [])
        if len(messages) < 3:
            invalid_structure.append(i)
            continue
        
        assistant_msg = messages[2].get("content", "")
        
        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            clean_msg = assistant_msg.strip()
            if clean_msg.startswith('```json'):
                clean_msg = clean_msg[7:]
            if clean_msg.startswith('```'):
                clean_msg = clean_msg[3:]
            if clean_msg.endswith('```'):
                clean_msg = clean_msg[:-3]
            clean_msg = clean_msg.strip()
            
            response_json = json.loads(clean_msg)
            
            # Check required fields
            if "answer_type" not in response_json:
                invalid_structure.append(i)
            if "items" not in response_json:
                invalid_structure.append(i)
            if "text" not in response_json:
                invalid_structure.append(i)
            if "chunks_used" not in response_json:
                invalid_structure.append(i)
                
        except json.JSONDecodeError:
            invalid_json.append(i)
    
    return {
        "total_examples": len(dataset),
        "invalid_json": len(invalid_json),
        "invalid_structure": len(invalid_structure),
        "valid": len(dataset) - len(invalid_json) - len(invalid_structure)
    }

def main():
    """Main verification function"""
    print("=" * 80)
    print("Dataset v3 Verification")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading dataset...")
    with open("rag_analysis_dataset_v3_json.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} examples")
    print()
    
    # Verify JSON format
    print("1. Verifying JSON format...")
    json_results = verify_json_format(dataset)
    print(f"   Total examples: {json_results['total_examples']}")
    print(f"   ✅ Valid JSON: {json_results['valid']}")
    print(f"   ❌ Invalid JSON: {json_results['invalid_json']}")
    print(f"   ❌ Invalid structure: {json_results['invalid_structure']}")
    if json_results['invalid_json'] > 0 or json_results['invalid_structure'] > 0:
        print(f"   ⚠️  WARNING: Found format issues!")
    print()
    
    # Verify not_found examples
    print("2. Verifying 'not_found' examples...")
    not_found_results = verify_not_found_examples(dataset)
    print(f"   Total 'not_found' examples: {not_found_results['total_not_found']}")
    print(f"   ❌ Bugs found: {not_found_results['bug_count']}")
    if not_found_results['bug_count'] > 0:
        print(f"   ⚠️  WARNING: Found {not_found_results['bug_count']} 'not_found' examples with entities in chunks!")
        print(f"   First 5 bugs:")
        for bug in not_found_results['bugs'][:5]:
            print(f"      Index {bug['index']}: Query='{bug['query'][:50]}...'")
            print(f"         Found entities: {bug['found_entities'][:3]}")
    else:
        print(f"   ✅ All 'not_found' examples are correct (no entities in chunks)")
    print()
    
    # Verify multi-entity examples
    print("3. Verifying multi-entity examples...")
    multi_entity_results = verify_multi_entity_examples(dataset)
    print(f"   Total multi-entity examples: {multi_entity_results['total_multi_entity']}")
    print(f"   ⚠️  Issues found: {multi_entity_results['issue_count']}")
    if multi_entity_results['issue_count'] > 0:
        print(f"   First 5 issues:")
        for issue in multi_entity_results['issues'][:5]:
            if "missing_items" in issue:
                print(f"      Index {issue['index']}: Missing items: {issue['missing_items'][:3]}")
            elif "all_entities_in_one_chunk" in issue.get("issue", ""):
                print(f"      Index {issue['index']}: All {issue['expected_items']} entities in one chunk")
    else:
        print(f"   ✅ All multi-entity examples are correct")
    print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    total_issues = (json_results['invalid_json'] + json_results['invalid_structure'] + 
                   not_found_results['bug_count'] + multi_entity_results['issue_count'])
    
    if total_issues == 0:
        print("✅ DATASET IS CORRECT - No bugs found!")
    else:
        print(f"⚠️  DATASET HAS ISSUES - {total_issues} total issues found")
        print(f"   - JSON format issues: {json_results['invalid_json'] + json_results['invalid_structure']}")
        print(f"   - 'not_found' bugs: {not_found_results['bug_count']}")
        print(f"   - Multi-entity issues: {multi_entity_results['issue_count']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
