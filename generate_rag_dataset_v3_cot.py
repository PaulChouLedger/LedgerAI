#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Analysis Dataset Generator V3 - Chain-of-Thought Reasoning
==============================================================

Generates training examples with explicit reasoning steps showing:
1. Understanding the query
2. Reading each chunk completely
3. Analyzing chunk meaning
4. Evaluating relevance
5. Extracting matching information
6. Synthesizing final answer

This teaches the model to think through the problem, not just output answers.
"""

import json
import random
from typing import List, Dict, Any, Tuple

# Import from existing generator
import sys
sys.path.append('.')
from generate_rag_dataset_v2 import (
    get_system_prompt_variation,
    generate_random_name,
    generate_random_company,
    generate_entity_sentence,
    generate_contextual_sentence,
    create_realistic_chunk,
    QUERY_TEMPLATES
)

# ============================================================================
# Chain-of-Thought Response Generation
# ============================================================================

def generate_cot_response(query: str, chunks: List[Dict], relevant_info: List[str], 
                         query_type: str, context: Dict[str, Any]) -> str:
    """
    Generate chain-of-thought response with explicit reasoning steps.
    
    Format:
    STEP 1: UNDERSTAND THE QUERY
    [Analysis of what's being asked]
    
    STEP 2: READ EACH CHUNK COMPLETELY
    [Summary of each chunk]
    
    STEP 3: ANALYZE CHUNK MEANING
    [Semantic analysis of each chunk]
    
    STEP 4: EVALUATE RELEVANCE
    [Relevance evaluation for each chunk]
    
    STEP 5: EXTRACT MATCHING INFORMATION
    [What information matches the query]
    
    STEP 6: VERIFY COMPLETENESS
    [Check that all chunks were read]
    
    STEP 7: SYNTHESIZE RESPONSE
    [Final answer]
    """
    
    # Step 1: Understand the query
    query_analysis = analyze_query(query, query_type)
    
    # Step 2: Read each chunk completely
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0.0)
        text = chunk.get('text', '')
        # Extract key information from chunk
        summary = summarize_chunk(text, i, score)
        chunk_summaries.append(summary)
    
    # Step 3: Analyze chunk meaning
    meaning_analysis = analyze_chunk_meanings(chunks, query)
    
    # Step 4: Evaluate relevance
    relevance_evaluation = evaluate_relevance(chunks, query, query_type, context)
    
    # Step 5: Extract matching information
    extraction = extract_matching_info(chunks, relevant_info, query_type, context)
    
    # Step 6: Verify completeness
    completeness_check = verify_completeness(chunks, relevant_info)
    
    # Step 7: Synthesize final answer
    final_answer = synthesize_final_answer(relevant_info, query_type, query)
    
    # Combine into chain-of-thought response
    cot_response = f"""STEP 1: UNDERSTAND THE QUERY
{query_analysis}

STEP 2: READ EACH CHUNK COMPLETELY
{chr(10).join(chunk_summaries)}

STEP 3: ANALYZE CHUNK MEANING
{meaning_analysis}

STEP 4: EVALUATE RELEVANCE
{relevance_evaluation}

STEP 5: EXTRACT MATCHING INFORMATION
{extraction}

STEP 6: VERIFY COMPLETENESS
{completeness_check}

STEP 7: SYNTHESIZE RESPONSE
{final_answer}"""
    
    return cot_response

def analyze_query(query: str, query_type: str) -> str:
    """Step 1: Analyze what the query is asking for"""
    query_lower = query.lower()
    
    if query_type == "entity":
        if "co-founder" in query_lower:
            company = extract_company_from_query(query)
            return f"The query asks for co-founders of {company}. I need to extract only people explicitly labeled as 'Co-Founder' of {company}, not other roles like CEO, CTO, or CFO."
        elif "ceo" in query_lower or "chief executive" in query_lower:
            company = extract_company_from_query(query)
            return f"The query asks for the CEO of {company}. I need to find the person with the CEO role at {company}."
        else:
            return f"The query asks for entity information: {query}. I need to extract the specific entities mentioned in the query."
    
    elif query_type == "list":
        return f"The query asks for a list: {query}. I need to extract all items that match this query from the chunks."
    
    elif query_type == "analytical":
        return f"The query asks for reasoning or causation: {query}. I need to extract information explaining why or what caused something."
    
    elif query_type == "relationship":
        return f"The query asks about relationships: {query}. I need to extract information about how entities are connected."
    
    elif query_type == "comparison":
        return f"The query asks for a comparison: {query}. I need to extract information comparing the entities mentioned."
    
    elif query_type == "process":
        return f"The query asks about a process: {query}. I need to extract step-by-step information about how something works."
    
    else:
        return f"The query asks: {query}. I need to extract information that directly answers this question."

def extract_company_from_query(query: str) -> str:
    """Extract company name from query"""
    # Simple extraction - look for patterns like "of X" or "at X"
    import re
    patterns = [
        r'of\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)',
        r'at\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()
    return "the company"

def summarize_chunk(text: str, chunk_num: int, score: float) -> str:
    """Step 2: Summarize what's in each chunk"""
    # Extract key sentences (first 2-3 sentences)
    sentences = text.split('.')
    key_sentences = [s.strip() + '.' for s in sentences[:3] if s.strip()]
    summary = ' '.join(key_sentences)
    
    if len(summary) > 200:
        summary = summary[:200] + "..."
    
    relevance = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.50 else "LOW"
    return f"Chunk {chunk_num} (Score: {score:.2f}, {relevance} relevance): {summary}"

def analyze_chunk_meanings(chunks: List[Dict], query: str) -> str:
    """Step 3: Analyze semantic meaning of chunks"""
    analyses = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        score = chunk.get('score', 0.0)
        
        # Identify key entities and concepts
        entities = extract_entities(text)
        concepts = identify_concepts(text, query)
        
        analysis = f"Chunk {i}: Contains information about {', '.join(entities[:3]) if entities else 'various topics'}. "
        if concepts:
            analysis += f"Relevant concepts: {', '.join(concepts[:2])}. "
        analysis += f"Score {score:.2f} suggests {'high' if score >= 0.70 else 'medium' if score >= 0.50 else 'low'} relevance."
        analyses.append(analysis)
    
    return '\n'.join(analyses)

def extract_entities(text: str) -> List[str]:
    """Extract entity names from text"""
    import re
    # Look for "X is Y" or "X of Z" patterns
    patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+is\s+',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+of\s+',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+Co-Founder',
    ]
    entities = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    return list(set(entities))[:5]  # Limit to 5 unique entities

def identify_concepts(text: str, query: str) -> List[str]:
    """Identify relevant concepts in text"""
    concepts = []
    query_lower = query.lower()
    
    if "co-founder" in query_lower:
        if "Co-Founder" in text:
            concepts.append("co-founder information")
    if "how" in query_lower and "work" in query_lower:
        if any(word in text.lower() for word in ["first", "then", "finally", "step"]):
            concepts.append("process steps")
    if "why" in query_lower or "what caused" in query_lower:
        if any(word in text.lower() for word in ["because", "due to", "led to", "caused"]):
            concepts.append("causation/reasoning")
    if "compare" in query_lower or "difference" in query_lower:
        if any(word in text.lower() for word in ["while", "whereas", "versus", "contrast"]):
            concepts.append("comparison")
    if "related" in query_lower or "relationship" in query_lower:
        if any(word in text.lower() for word in ["partners", "alliance", "owns", "connected"]):
            concepts.append("relationship")
    
    return concepts

def evaluate_relevance(chunks: List[Dict], query: str, query_type: str, context: Dict[str, Any]) -> str:
    """Step 4: Evaluate relevance of each chunk"""
    evaluations = []
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        score = chunk.get('score', 0.0)
        
        # Determine if chunk directly answers query
        directly_answers = check_direct_answer(text, query, query_type, context)
        
        relevance_level = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.50 else "LOW"
        
        evaluation = f"Chunk {i} (Score: {score:.2f}, {relevance_level} relevance): "
        if directly_answers:
            evaluation += "Directly answers the query. Information should be extracted."
        else:
            evaluation += "Does not directly answer the query. Information should be ignored."
        
        evaluations.append(evaluation)
    
    return '\n'.join(evaluations)

def check_direct_answer(text: str, query: str, query_type: str, context: Dict[str, Any]) -> bool:
    """Check if chunk directly answers the query"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    if query_type == "entity":
        # Check if query asks for co-founders
        if "co-founder" in query_lower:
            # Must have "Co-Founder" in text, not just CEO/CTO
            if "co-founder" in text_lower and "co-founder" not in query_lower.replace("co-founder", ""):
                # Check company match
                company = extract_company_from_query(query)
                if company.lower() in text_lower or any(word in text_lower for word in company.lower().split()):
                    return True
        # Check if query asks for CEO
        elif "ceo" in query_lower:
            if "ceo" in text_lower:
                company = extract_company_from_query(query)
                if company.lower() in text_lower:
                    return True
        return False
    
    elif query_type in ["analytical", "relationship", "comparison", "process"]:
        # Check for relevant keywords
        if query_type == "analytical":
            return any(word in text_lower for word in ["because", "due to", "led to", "caused"])
        elif query_type == "relationship":
            return any(word in text_lower for word in ["partners", "alliance", "owns", "connected", "related"])
        elif query_type == "comparison":
            return any(word in text_lower for word in ["while", "whereas", "versus", "contrast", "differs"])
        elif query_type == "process":
            return any(word in text_lower for word in ["first", "then", "finally", "step", "process"])
    
    # Default: check if query keywords appear in text
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    overlap = len(query_words.intersection(text_words))
    return overlap >= 2  # At least 2 words overlap

def extract_matching_info(chunks: List[Dict], relevant_info: List[str], query_type: str, context: Dict[str, Any]) -> str:
    """Step 5: Extract matching information"""
    if not relevant_info:
        return "No matching information found in any chunk. The query cannot be answered from the provided documents."
    
    extraction = f"Found {len(relevant_info)} matching item(s):\n"
    for i, info in enumerate(relevant_info, 1):
        extraction += f"  {i}. {info}\n"
    
    # Identify which chunks contained the information
    chunk_sources = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        for info in relevant_info:
            if info in text or any(word in text.lower() for word in info.lower().split()[:2]):
                chunk_sources.append(f"Chunk {i}")
                break
    
    if chunk_sources:
        extraction += f"\nInformation found in: {', '.join(set(chunk_sources))}"
    
    return extraction

def verify_completeness(chunks: List[Dict], relevant_info: List[str]) -> str:
    """Step 6: Verify all chunks were read"""
    num_chunks = len(chunks)
    verification = f"Read all {num_chunks} chunk(s) completely.\n"
    
    if relevant_info:
        verification += f"Extracted {len(relevant_info)} matching item(s) across all chunks.\n"
        verification += "Extraction is complete."
    else:
        verification += "No matching information found in any chunk.\n"
        verification += "Extraction is complete - query cannot be answered."
    
    return verification

def synthesize_final_answer(relevant_info: List[str], query_type: str, query: str) -> str:
    """Step 7: Synthesize final answer"""
    if not relevant_info:
        return "I don't have that information in the provided documents."
    
    # Format based on query type
    if query_type == "entity":
        names = [info.strip() for info in relevant_info if info.strip()]
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return ", ".join(names[:-1]) + f", and {names[-1]}"
    
    elif query_type == "list":
        items = [info.strip() for info in relevant_info if info.strip()]
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"
    
    elif query_type == "comparison":
        if len(relevant_info) >= 2:
            sentences = [info.strip().rstrip('.') for info in relevant_info]
            if any("while" in s.lower() for s in sentences):
                return ". ".join(sentences) + "."
            else:
                return f"{sentences[0]}. {sentences[1]}."
        return " ".join([info.strip().rstrip('.') for info in relevant_info])
    
    elif query_type == "analytical":
        response = " ".join([info.strip() for info in relevant_info])
        if "because" not in response.lower() and "due to" not in response.lower():
            response = f"because {response}"
        return response
    
    elif query_type == "relationship":
        return " ".join([info.strip() for info in relevant_info])
    
    else:
        return " ".join([info.strip() for info in relevant_info])

# ============================================================================
# Modified System Prompt (Simpler - reasoning is in response)
# ============================================================================

def get_cot_system_prompt() -> str:
    """System prompt for chain-of-thought training"""
    return """You are an AI assistant trained to analyze RAG chunks and extract relevant information using systematic reasoning.

Follow these steps for every query:
1. Understand what the query is asking
2. Read each chunk completely
3. Analyze the meaning of each chunk
4. Evaluate relevance of each chunk to the query
5. Extract matching information from relevant chunks
6. Verify you've read all chunks completely
7. Synthesize the final answer

Show your reasoning process step-by-step, then provide the final answer."""

# ============================================================================
# Generate CoT Examples
# ============================================================================

def generate_cot_example(pattern_type: str = "mixed_content") -> Dict[str, Any]:
    """Generate a single training example with chain-of-thought reasoning"""
    
    # Select query template
    template = random.choice(QUERY_TEMPLATES)
    query, context = generate_query_from_template(template)
    query_type = template["type"]
    
    # Generate chunks (simplified - you'd import from generate_rag_dataset_v2)
    # For now, create a simple example structure
    chunks = generate_chunks_for_example(query, query_type, context, pattern_type)
    
    # Extract relevant information
    relevant_info = extract_relevant_info_from_chunks(chunks, query, query_type, context)
    
    # Generate chain-of-thought response
    cot_response = generate_cot_response(query, chunks, relevant_info, query_type, context)
    
    # Format chunks for user message
    chunks_text = format_chunks_for_user(chunks)
    
    example = {
        "messages": [
            {"role": "system", "content": get_cot_system_prompt()},
            {
                "role": "user",
                "content": f"Query: {query}\n\nRAG Chunks:\n{chunks_text}"
            },
            {"role": "assistant", "content": cot_response}
        ]
    }
    
    return example

# Helper functions (simplified - would need full implementation)
def generate_query_from_template(template: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Generate query from template"""
    # Simplified - would use full implementation from generate_rag_dataset_v2
    if template["type"] == "entity":
        company = generate_random_company()
        role = template.get("role", "co-founders")
        query = f"who are the {role} of {company}?"
        return query, {"company": company, "role": role}
    # ... other types
    return "test query", {}

def generate_chunks_for_example(query: str, query_type: str, context: Dict[str, Any], pattern_type: str) -> List[Dict]:
    """Generate chunks for example"""
    # Simplified - would use full implementation
    return [{"text": "Sample chunk text", "score": 0.85, "file": "document.pdf"}]

def extract_relevant_info_from_chunks(chunks: List[Dict], query: str, query_type: str, context: Dict[str, Any]) -> List[str]:
    """Extract relevant information from chunks"""
    # Simplified - would parse chunks to find relevant info
    return ["Sample relevant info"]

def format_chunks_for_user(chunks: List[Dict]) -> str:
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
# Main: Convert Existing Dataset to CoT Format
# ============================================================================

def convert_dataset_to_cot(input_path: str, output_path: str, sample_size: int = 1000):
    """
    Convert existing dataset to chain-of-thought format.
    Takes a sample of examples and adds reasoning steps.
    """
    print("=" * 80)
    print("CONVERTING DATASET TO CHAIN-OF-THOUGHT FORMAT")
    print("=" * 80)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples")
    print(f"Converting {sample_size} examples to CoT format...")
    
    # Sample examples to convert
    sample_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    cot_examples = []
    
    for idx in sample_indices:
        example = dataset[idx]
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
            continue
        
        # Extract query and chunks from user message
        user_content = user_msg.get('content', '')
        query = extract_query_from_content(user_content)
        chunks = extract_chunks_from_content(user_content)
        
        # Extract final answer from assistant message
        final_answer = assistant_msg.get('content', '')
        
        # Generate chain-of-thought response
        # (This is simplified - would need full implementation)
        cot_response = generate_cot_response_simple(query, chunks, final_answer)
        
        # Create new example
        new_example = {
            "messages": [
                system_msg or {"role": "system", "content": get_cot_system_prompt()},
                user_msg,
                {"role": "assistant", "content": cot_response}
            ]
        }
        cot_examples.append(new_example)
        
        if len(cot_examples) % 100 == 0:
            print(f"  Converted {len(cot_examples)}/{sample_size} examples...")
    
    # Save converted examples
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cot_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(cot_examples)} examples to CoT format")
    print(f"✅ Saved to: {output_path}")

def extract_query_from_content(content: str) -> str:
    """Extract query from user content"""
    import re
    match = re.search(r'Query:\s*(.+?)(?:\n\nRAG|$)', content, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_chunks_from_content(content: str) -> List[Dict]:
    """Extract chunks from user content"""
    import re
    chunks = []
    
    # Find all chunk blocks
    pattern = r'\[Chunk (\d+)\]\s+Score:\s+([\d.]+).*?FULL CHUNK TEXT:\s+\'(.+?)\''
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        chunk_num, score, text = match
        chunks.append({
            "text": text,
            "score": float(score),
            "file": "document.pdf"
        })
    
    return chunks

def generate_cot_response_simple(query: str, chunks: List[Dict], final_answer: str) -> str:
    """Generate simplified CoT response from existing answer"""
    # This is a simplified version - would need full reasoning generation
    
    # Step 1: Understand query
    query_analysis = f"The query asks: {query}. I need to extract information that directly answers this question."
    
    # Step 2: Read chunks
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0.0)
        text = chunk.get('text', '')
        summary = text[:150] + "..." if len(text) > 150 else text
        relevance = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.50 else "LOW"
        chunk_summaries.append(f"Chunk {i} (Score: {score:.2f}, {relevance} relevance): {summary}")
    
    # Step 3-6: Simplified reasoning
    reasoning = f"""After reading all {len(chunks)} chunk(s) completely, I analyzed the meaning and evaluated relevance. I extracted the matching information that directly answers the query."""
    
    # Step 7: Final answer
    return f"""STEP 1: UNDERSTAND THE QUERY
{query_analysis}

STEP 2: READ EACH CHUNK COMPLETELY
{chr(10).join(chunk_summaries)}

STEP 3-6: ANALYZE, EVALUATE, EXTRACT, VERIFY
{reasoning}

STEP 7: SYNTHESIZE RESPONSE
{final_answer}"""

if __name__ == "__main__":
    import sys
    
    input_path = "rag_analysis_dataset_v2.json"
    output_path = "rag_analysis_dataset_v3_cot.json"
    sample_size = 2000  # Convert 2000 examples to CoT format
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        sample_size = int(sys.argv[3])
    
    convert_dataset_to_cot(input_path, output_path, sample_size)
