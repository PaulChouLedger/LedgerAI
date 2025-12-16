#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix CoT Structure - Remove Redundant Steps
==========================================

Updates the CoT structure to:
1. Understand the query
2. Read each chunk completely (with relevance scores)
3. Analyze chunk meaning (describe what each chunk contains and how it relates)
4. Extract matching information (skip redundant "evaluate relevance")
5. Verify completeness
6. Synthesize response

Removes the redundant STEP 4 (EVALUATE RELEVANCE) that was causing contradictions.
"""

import json
import re
from typing import List, Dict, Any

def extract_query_from_user_content(content: str) -> str:
    """Extract query from user message content"""
    match = re.search(r'Query:\s*(.+?)(?:\n\nRAG|$)', content, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_chunks_from_user_content(content: str) -> List[Dict]:
    """Extract chunks from user message content"""
    chunks = []
    pattern = r'\[Chunk (\d+)\]\s+Score:\s+([\d.]+),\s+File:\s+([^\n]+)\nFULL CHUNK TEXT:\s+\'(.+?)\''
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        chunk_num, score, file_name, text = match
        chunks.append({
            "num": int(chunk_num),
            "text": text,
            "score": float(score),
            "file": file_name.strip()
        })
    
    return chunks

def generate_step1_understand_query(query: str) -> str:
    """STEP 1: Understand the query"""
    query_lower = query.lower()
    
    if "co-founder" in query_lower:
        company = extract_company_name(query)
        return f"The query asks for co-founders of {company}. I need to extract ONLY people explicitly labeled as 'Co-Founder' of {company}, not other roles like CEO, CTO, CFO, or VP. I must also ensure I only extract co-founders of {company}, not other companies."
    
    elif "compare" in query_lower or "difference" in query_lower:
        # Extract entities being compared
        entities = extract_comparison_entities(query)
        return f"The query asks: {query}. I need to find relevant information for {entities} and determine how the entities differ."
    
    elif "how" in query_lower and "work" in query_lower:
        return f"The query asks: {query}. I need to extract step-by-step information showing how something works, including sequential words like 'first', 'then', 'finally'."
    
    elif "why" in query_lower or "what caused" in query_lower:
        return f"The query asks: {query}. I need to extract information explaining why something happened, including causation words like 'because', 'due to', 'led to', or 'caused'."
    
    elif "related" in query_lower or "relationship" in query_lower:
        return f"The query asks: {query}. I need to extract information about how entities are connected, including words like 'partners', 'alliance', 'owns', 'connected', or 'joint venture'."
    
    elif "what are" in query_lower or "list" in query_lower:
        return f"The query asks: {query}. I need to extract all items that match this query from all chunks."
    
    else:
        return f"The query asks: {query}. I need to extract information that directly answers this question from the provided chunks."

def extract_company_name(query: str) -> str:
    """Extract company name from query"""
    match = re.search(r'(?:of|at)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
    if match:
        return match.group(1).strip()
    return "the company"

def extract_comparison_entities(query: str) -> str:
    """Extract entities being compared"""
    # Try to find "between X and Y" or "X and Y"
    match = re.search(r'between\s+([A-Z][a-zA-Z\s]+?)\s+and\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query, re.IGNORECASE)
    if match:
        return f"both {match.group(1).strip()} and {match.group(2).strip()}"
    
    # Try "X vs Y" or "X and Y"
    match = re.search(r'([A-Z][a-zA-Z\s]+?)\s+(?:vs|and)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query, re.IGNORECASE)
    if match:
        return f"both {match.group(1).strip()} and {match.group(2).strip()}"
    
    return "the entities mentioned"

def generate_step2_read_chunks(chunks: List[Dict]) -> str:
    """STEP 2: Read each chunk completely with relevance scores"""
    summaries = []
    for chunk in chunks:
        num = chunk['num']
        score = chunk['score']
        text = chunk['text']
        
        # Get first sentence or two as summary
        sentences = text.split('.')
        summary = '. '.join(sentences[:2]).strip()
        if len(summary) > 150:
            summary = summary[:150] + "..."
        
        relevance = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.50 else "LOW"
        summaries.append(f"Chunk {num} (Score: {score:.2f}, {relevance} relevance): {summary}")
    
    return '\n'.join(summaries)

def generate_step3_analyze_meaning(chunks: List[Dict], query: str, final_answer: str) -> str:
    """STEP 3: Analyze chunk meaning - describe what each chunk contains and how it relates"""
    analyses = []
    query_lower = query.lower()
    
    # Extract entities from query
    query_entities = []
    if "co-founder" in query_lower:
        company = extract_company_name(query)
        query_entities.append(company)
    elif "compare" in query_lower or "difference" in query_lower:
        # Extract individual entity names
        entity_matches = re.findall(r'([A-Z][a-zA-Z]+)', query)
        query_entities.extend(entity_matches[:2])
    
    # Determine which chunks were used in final answer
    used_chunks = set()
    if final_answer and not "don't have that information" in final_answer.lower():
        for chunk in chunks:
            text = chunk['text']
            answer_words = set(final_answer.lower().split()[:10])  # First 10 words
            chunk_words = set(text.lower().split())
            overlap = len(answer_words.intersection(chunk_words))
            if overlap >= 3:  # Significant overlap
                used_chunks.add(chunk['num'])
    
    for chunk in chunks:
        num = chunk['num']
        text = chunk['text']
        score = chunk['score']
        
        analysis = f"Chunk {num}: "
        
        # Check for specific query types
        if "compare" in query_lower or "difference" in query_lower:
            # Check which entities are mentioned
            mentioned_entities = []
            for entity in query_entities:
                if entity.lower() in text.lower():
                    mentioned_entities.append(entity)
            
            if len(mentioned_entities) == 2:
                # Both entities mentioned
                if num in used_chunks:
                    analysis += f"provides descriptive information regarding both {mentioned_entities[0]} and {mentioned_entities[1]} sufficient for comparison."
                else:
                    analysis += f"mentions both {mentioned_entities[0]} and {mentioned_entities[1]} but does not provide descriptive information useful for a comparison."
            elif len(mentioned_entities) == 1:
                # Only one entity mentioned
                if num in used_chunks:
                    analysis += f"provides descriptive information regarding {mentioned_entities[0]} sufficient to compare against {query_entities[1] if len(query_entities) > 1 else 'the other entity'}."
                else:
                    analysis += f"briefly mentions {mentioned_entities[0]} but does not provide descriptive information useful for a comparison."
            else:
                # Check for other entities
                other_entities = re.findall(r'([A-Z][a-zA-Z]+)', text)
                if other_entities and other_entities[0] not in query_entities:
                    analysis += f"mentions {other_entities[0]}, not useful for query."
                else:
                    analysis += f"does not contain information directly relevant to the query."
        
        elif "co-founder" in query_lower:
            company = query_entities[0] if query_entities else "the company"
            if "Co-Founder" in text and company.lower() in text.lower():
                if num in used_chunks:
                    analysis += f"describes {company} and contains co-founder information."
                else:
                    analysis += f"mentions {company} but does not provide co-founder information."
            elif any(role in text for role in ["CEO", "CTO", "CFO", "President", "VP"]):
                analysis += f"contains other roles but not co-founders."
            else:
                analysis += f"does not contain information directly relevant to the query."
        
        else:
            # Generic analysis
            if num in used_chunks:
                analysis += f"provides descriptive information useful for the query."
            else:
                analysis += f"does not contain information directly relevant to the query."
        
        analyses.append(analysis.strip())
    
    return '\n'.join(analyses)

def generate_step4_extract_matching(chunks: List[Dict], final_answer: str, query: str) -> str:
    """STEP 4: Extract matching information (skip redundant evaluate relevance)"""
    if "don't have that information" in final_answer.lower():
        return "No matching information found in any chunk. The query cannot be answered from the provided documents."
    
    # Extract entities/items from final answer
    if " and " in final_answer:
        items = [item.strip() for item in final_answer.split(" and ")]
    elif ", " in final_answer:
        items = [item.strip() for item in final_answer.split(", ")]
    else:
        items = [final_answer.strip()]
    
    extraction = f"Extract information from "
    
    # Identify which chunks contained the information
    chunk_sources = []
    for chunk in chunks:
        text = chunk['text']
        for item in items:
            item_words = set(item.lower().split()[:3])  # First 3 words
            chunk_words = set(text.lower().split())
            if len(item_words.intersection(chunk_words)) >= 2:
                chunk_sources.append(f"Chunk {chunk['num']}")
                break
    
    if chunk_sources:
        extraction += f"{' and '.join(set(chunk_sources))}"
    else:
        extraction += "relevant chunks"
    
    return extraction

def generate_step5_verify_completeness(chunks: List[Dict], final_answer: str) -> str:
    """STEP 5: Verify completeness"""
    num_chunks = len(chunks)
    
    if "don't have that information" in final_answer.lower():
        return f"Ensuring all relevant information was extracted. Read all {num_chunks} chunk(s) completely. No matching information found in any chunk."
    
    # Count items in answer
    if " and " in final_answer:
        num_items = len(final_answer.split(" and "))
    elif ", " in final_answer:
        num_items = len(final_answer.split(", "))
    else:
        num_items = 1
    
    return f"Ensuring all relevant information was extracted. Read all {num_chunks} chunk(s) completely. Extracted {num_items} matching item(s) across all chunks. All relevant information has been identified."

def generate_fixed_cot_response(query: str, chunks: List[Dict], final_answer: str) -> str:
    """Generate fixed CoT response with correct structure"""
    
    step1 = generate_step1_understand_query(query)
    step2 = generate_step2_read_chunks(chunks)
    step3 = generate_step3_analyze_meaning(chunks, query, final_answer)
    step4 = generate_step4_extract_matching(chunks, final_answer, query)
    step5 = generate_step5_verify_completeness(chunks, final_answer)
    step6 = final_answer
    
    cot_response = f"""STEP 1: UNDERSTAND THE QUERY
{step1}

STEP 2: READ EACH CHUNK COMPLETELY
{step2}

STEP 3: ANALYZE CHUNK MEANING
{step3}

STEP 4: EXTRACT MATCHING INFORMATION
{step4}

STEP 5: VERIFY COMPLETENESS
{step5}

STEP 6: SYNTHESIZE RESPONSE
{step6}"""
    
    return cot_response

def fix_example_cot(example: Dict[str, Any]) -> Dict[str, Any]:
    """Fix a single example's CoT structure"""
    messages = example.get('messages', [])
    
    user_msg = None
    assistant_msg = None
    system_msg = None
    
    for msg in messages:
        if msg.get('role') == 'system':
            system_msg = msg
        elif msg.get('role') == 'user':
            user_msg = msg
        elif msg.get('role') == 'assistant':
            assistant_msg = msg
    
    if not user_msg or not assistant_msg:
        return example
    
    # Extract query and chunks
    user_content = user_msg.get('content', '')
    query = extract_query_from_user_content(user_content)
    chunks = extract_chunks_from_user_content(user_content)
    
    if not query or not chunks:
        return example
    
    # Extract final answer (STEP 7 or last part of CoT)
    cot_content = assistant_msg.get('content', '')
    if 'STEP 7: SYNTHESIZE RESPONSE' in cot_content:
        final_answer = cot_content.split('STEP 7: SYNTHESIZE RESPONSE')[-1].strip()
    elif 'STEP 6: SYNTHESIZE RESPONSE' in cot_content:
        final_answer = cot_content.split('STEP 6: SYNTHESIZE RESPONSE')[-1].strip()
    else:
        # Try to extract from end
        final_answer = cot_content.split('\n')[-1].strip()
    
    # Generate fixed CoT response
    fixed_cot = generate_fixed_cot_response(query, chunks, final_answer)
    
    # Update assistant message
    new_messages = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            new_messages.append({"role": "assistant", "content": fixed_cot})
        else:
            new_messages.append(msg)
    
    return {"messages": new_messages}

def fix_dataset_cot(input_path: str, output_path: str):
    """Fix CoT structure in entire dataset"""
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} examples. Fixing CoT structure...")
    
    fixed_dataset = []
    for i, example in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} examples...")
        
        fixed_example = fix_example_cot(example)
        fixed_dataset.append(fixed_example)
    
    print(f"Saving fixed dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Fixed dataset saved! ({len(fixed_dataset)} examples)")

if __name__ == "__main__":
    import sys
    
    input_path = "rag_analysis_dataset_v2.json"
    output_path = "rag_analysis_dataset_v2.json"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    fix_dataset_cot(input_path, output_path)
