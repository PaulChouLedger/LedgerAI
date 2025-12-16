#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Chain-of-Thought Reasoning to RAG Dataset
=============================================

Takes existing dataset and adds explicit reasoning steps to assistant responses.
This teaches the model to think through the problem step-by-step.
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
    
    # Find all chunk blocks
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

def analyze_query_for_cot(query: str) -> str:
    """Generate Step 1: Understand the query"""
    query_lower = query.lower()
    
    if "co-founder" in query_lower:
        company = extract_company_name(query)
        return f"The query asks for co-founders of {company}. I need to extract ONLY people explicitly labeled as 'Co-Founder' of {company}, not other roles like CEO, CTO, CFO, or VP. I must also ensure I only extract co-founders of {company}, not other companies."
    
    elif "ceo" in query_lower or "chief executive" in query_lower:
        company = extract_company_name(query)
        return f"The query asks for the CEO of {company}. I need to find the person with the CEO role at {company}."
    
    elif "how" in query_lower and "work" in query_lower:
        return f"The query asks about a process: {query}. I need to extract step-by-step information showing how something works, including sequential words like 'first', 'then', 'finally'."
    
    elif "why" in query_lower or "what caused" in query_lower:
        return f"The query asks for reasoning or causation: {query}. I need to extract information explaining why something happened, including causation words like 'because', 'due to', 'led to', or 'caused'."
    
    elif "compare" in query_lower or "difference" in query_lower:
        return f"The query asks for a comparison: {query}. I need to extract information comparing entities, including contrast words like 'while', 'whereas', 'versus', or 'in contrast'."
    
    elif "related" in query_lower or "relationship" in query_lower:
        return f"The query asks about relationships: {query}. I need to extract information about how entities are connected, including words like 'partners', 'alliance', 'owns', 'connected', or 'joint venture'."
    
    elif "what are" in query_lower or "list" in query_lower:
        return f"The query asks for a list: {query}. I need to extract all items that match this query from all chunks."
    
    else:
        return f"The query asks: {query}. I need to extract information that directly answers this question from the provided chunks."

def extract_company_name(query: str) -> str:
    """Extract company name from query"""
    # Look for "of X" or "at X"
    match = re.search(r'(?:of|at)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
    if match:
        return match.group(1).strip()
    return "the company"

def read_chunks_for_cot(chunks: List[Dict]) -> str:
    """Generate Step 2: Read each chunk completely"""
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

def analyze_meaning_for_cot(chunks: List[Dict], query: str) -> str:
    """Generate Step 3: Analyze chunk meaning"""
    analyses = []
    query_lower = query.lower()
    
    for chunk in chunks:
        num = chunk['num']
        text = chunk['text']
        score = chunk['score']
        
        # Identify key entities
        entities = []
        if "co-founder" in query_lower:
            # Find Co-Founder mentions
            cofounder_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+is\s+Co-Founder', text)
            entities.extend(cofounder_matches)
        
        # Identify concepts
        concepts = []
        if "how" in query_lower and "work" in query_lower:
            if any(word in text.lower() for word in ["first", "then", "finally", "step"]):
                concepts.append("process steps")
        if "why" in query_lower:
            if any(word in text.lower() for word in ["because", "due to", "led to"]):
                concepts.append("causation/reasoning")
        if "compare" in query_lower:
            if any(word in text.lower() for word in ["while", "whereas", "versus"]):
                concepts.append("comparison")
        if "related" in query_lower:
            if any(word in text.lower() for word in ["partners", "alliance", "owns"]):
                concepts.append("relationship")
        
        analysis = f"Chunk {num}: "
        if entities:
            analysis += f"Contains entities: {', '.join(entities[:3])}. "
        if concepts:
            analysis += f"Relevant concepts: {', '.join(concepts)}. "
        analysis += f"Score {score:.2f} indicates {'high' if score >= 0.70 else 'medium' if score >= 0.50 else 'low'} relevance."
        analyses.append(analysis)
    
    return '\n'.join(analyses) if analyses else "Analyzed semantic meaning of all chunks."

def evaluate_relevance_for_cot(chunks: List[Dict], query: str, final_answer: str) -> str:
    """Generate Step 4: Evaluate relevance"""
    evaluations = []
    query_lower = query.lower()
    
    for chunk in chunks:
        num = chunk['num']
        text = chunk['text']
        score = chunk['score']
        
        # Check if chunk contains information used in final answer
        answer_words = set(final_answer.lower().split())
        chunk_words = set(text.lower().split())
        overlap = len(answer_words.intersection(chunk_words))
        
        relevance_level = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.50 else "LOW"
        
        if overlap > 3:  # Significant overlap with answer
            evaluation = f"Chunk {num} (Score: {score:.2f}, {relevance_level} relevance): Directly answers the query. Contains information that matches the query requirements."
        else:
            evaluation = f"Chunk {num} (Score: {score:.2f}, {relevance_level} relevance): Does not directly answer the query. Information should be ignored."
        
        evaluations.append(evaluation)
    
    return '\n'.join(evaluations)

def extract_matching_info_for_cot(chunks: List[Dict], final_answer: str, query: str) -> str:
    """Generate Step 5: Extract matching information"""
    if "don't have that information" in final_answer.lower():
        return "No matching information found in any chunk. The query cannot be answered from the provided documents."
    
    # Extract entities/items from final answer
    if " and " in final_answer:
        items = [item.strip() for item in final_answer.split(" and ")]
    elif ", " in final_answer:
        items = [item.strip() for item in final_answer.split(", ")]
    else:
        items = [final_answer.strip()]
    
    extraction = f"Found {len(items)} matching item(s):\n"
    for i, item in enumerate(items, 1):
        extraction += f"  {i}. {item}\n"
    
    # Identify which chunks contained the information
    chunk_sources = []
    for chunk in chunks:
        text = chunk['text']
        for item in items:
            # Check if item appears in chunk
            item_words = set(item.lower().split()[:3])  # First 3 words
            chunk_words = set(text.lower().split())
            if len(item_words.intersection(chunk_words)) >= 2:
                chunk_sources.append(f"Chunk {chunk['num']}")
                break
    
    if chunk_sources:
        extraction += f"\nInformation found in: {', '.join(set(chunk_sources))}"
    
    return extraction

def verify_completeness_for_cot(chunks: List[Dict], final_answer: str) -> str:
    """Generate Step 6: Verify completeness"""
    num_chunks = len(chunks)
    
    if "don't have that information" in final_answer.lower():
        return f"Read all {num_chunks} chunk(s) completely.\nNo matching information found in any chunk.\nExtraction is complete - query cannot be answered from the provided documents."
    
    # Count items in answer
    if " and " in final_answer:
        num_items = len(final_answer.split(" and "))
    elif ", " in final_answer:
        num_items = len(final_answer.split(", "))
    else:
        num_items = 1
    
    return f"Read all {num_chunks} chunk(s) completely.\nExtracted {num_items} matching item(s) across all chunks.\nExtraction is complete - all relevant information has been identified."

def generate_cot_response(query: str, chunks: List[Dict], final_answer: str) -> str:
    """Generate complete chain-of-thought response"""
    
    step1 = analyze_query_for_cot(query)
    step2 = read_chunks_for_cot(chunks)
    step3 = analyze_meaning_for_cot(chunks, query)
    step4 = evaluate_relevance_for_cot(chunks, query, final_answer)
    step5 = extract_matching_info_for_cot(chunks, final_answer, query)
    step6 = verify_completeness_for_cot(chunks, final_answer)
    step7 = final_answer
    
    cot_response = f"""STEP 1: UNDERSTAND THE QUERY
{step1}

STEP 2: READ EACH CHUNK COMPLETELY
{step2}

STEP 3: ANALYZE CHUNK MEANING
{step3}

STEP 4: EVALUATE RELEVANCE
{step4}

STEP 5: EXTRACT MATCHING INFORMATION
{step5}

STEP 6: VERIFY COMPLETENESS
{step6}

STEP 7: SYNTHESIZE RESPONSE
{step7}"""
    
    return cot_response

def convert_example_to_cot(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single example to chain-of-thought format"""
    messages = example.get('messages', [])
    
    # Find user and assistant messages
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
        return None
    
    # Extract query and chunks
    user_content = user_msg.get('content', '')
    query = extract_query_from_user_content(user_content)
    chunks = extract_chunks_from_user_content(user_content)
    
    if not query or not chunks:
        return None
    
    # Get final answer
    final_answer = assistant_msg.get('content', '').strip()
    
    # Generate chain-of-thought response
    cot_response = generate_cot_response(query, chunks, final_answer)
    
    # Create new example
    new_example = {
        "messages": [
            system_msg or {"role": "system", "content": "You are an AI assistant trained to analyze RAG chunks using systematic reasoning."},
            user_msg,
            {"role": "assistant", "content": cot_response}
        ]
    }
    
    return new_example

def convert_dataset_to_cot(input_path: str, output_path: str, convert_all: bool = False, sample_size: int = 2000):
    """
    Convert dataset to chain-of-thought format.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save CoT dataset
        convert_all: If True, convert all examples. If False, convert sample_size examples
        sample_size: Number of examples to convert if convert_all=False
    """
    print("=" * 80)
    print("CONVERTING DATASET TO CHAIN-OF-THOUGHT FORMAT")
    print("=" * 80)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples from {input_path}")
    
    # Determine which examples to convert
    if convert_all:
        indices_to_convert = range(len(dataset))
        print(f"Converting ALL {len(dataset)} examples to CoT format...")
    else:
        import random
        indices_to_convert = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        print(f"Converting {len(indices_to_convert)} examples to CoT format...")
    
    cot_examples = []
    failed = 0
    
    for idx in indices_to_convert:
        example = dataset[idx]
        cot_example = convert_example_to_cot(example)
        
        if cot_example:
            cot_examples.append(cot_example)
        else:
            failed += 1
        
        if len(cot_examples) % 100 == 0:
            print(f"  Converted {len(cot_examples)} examples...")
    
    print(f"\n✅ Successfully converted {len(cot_examples)} examples")
    if failed > 0:
        print(f"⚠️  Failed to convert {failed} examples (missing query/chunks)")
    
    # Save converted examples
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cot_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(cot_examples)} CoT examples to: {output_path}")
    print(f"   File size: {len(json.dumps(cot_examples)) / (1024*1024):.1f} MB")
    
    return len(cot_examples)

if __name__ == "__main__":
    import sys
    
    input_path = "rag_analysis_dataset_v2.json"
    output_path = "rag_analysis_dataset_v2_cot.json"
    convert_all = False
    sample_size = 2000
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        convert_all = sys.argv[3].lower() == 'true'
    if len(sys.argv) > 4:
        sample_size = int(sys.argv[4])
    
    convert_dataset_to_cot(input_path, output_path, convert_all, sample_size)
