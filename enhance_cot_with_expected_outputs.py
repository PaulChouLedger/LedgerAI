#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhance CoT Dataset with Explicit Expected Output Formats
==========================================================

Updates the dataset to include explicit expected output formats for each CoT step.
This helps the model learn what the expected output structure should be for each step.
"""

import json
import re
from typing import List, Dict, Any

def enhance_cot_step_with_expected_format(step_num: int, step_content: str, query: str, chunks: List[Dict] = None, final_answer: str = None) -> str:
    """
    Enhance a CoT step to include explicit expected output format.
    
    Each step should show:
    1. What to do (instruction)
    2. Expected output format (template)
    3. Actual output (example)
    """
    
    if step_num == 1:
        # STEP 1: UNDERSTAND THE QUERY
        # Expected format: "The query asks for [type]: [query]. I need to [action]."
        query_lower = query.lower()
        
        if "co-founder" in query_lower:
            company = extract_company_name(query)
            expected_format = f"The query asks for co-founders of {company}. I need to extract ONLY people explicitly labeled as 'Co-Founder' of {company}, not other roles like CEO, CTO, CFO, or VP. I must also ensure I only extract co-founders of {company}, not other companies."
        elif "how" in query_lower and "work" in query_lower:
            expected_format = f"The query asks about a process: {query}. I need to extract step-by-step information showing how something works, including sequential words like 'first', 'then', 'finally'."
        elif "why" in query_lower or "what caused" in query_lower:
            expected_format = f"The query asks for reasoning or causation: {query}. I need to extract information explaining why something happened, including causation words like 'because', 'due to', 'led to', or 'caused'."
        elif "compare" in query_lower or "difference" in query_lower:
            expected_format = f"The query asks for a comparison: {query}. I need to extract information comparing entities, including contrast words like 'while', 'whereas', 'versus', or 'in contrast'."
        elif "related" in query_lower or "relationship" in query_lower:
            expected_format = f"The query asks about relationships: {query}. I need to extract information about how entities are connected, including words like 'partners', 'alliance', 'owns', 'connected', or 'joint venture'."
        elif "what are" in query_lower or "list" in query_lower:
            expected_format = f"The query asks for a list: {query}. I need to extract all items that match this query from all chunks."
        else:
            expected_format = f"The query asks: {query}. I need to extract information that directly answers this question from the provided chunks."
        
        return expected_format
    
    elif step_num == 2:
        # STEP 2: READ EACH CHUNK COMPLETELY
        # Expected format: "Chunk X (Score: Y.YY, RELEVANCE relevance): [first 1-2 sentences]..."
        if not chunks:
            return ""
        
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
    
    elif step_num == 3:
        # STEP 3: ANALYZE CHUNK MEANING
        # Expected format: "Chunk X: Contains entities: [...]. Relevant concepts: [...]. Score Y.YY indicates [level] relevance."
        if not chunks:
            return ""
        
        analyses = []
        query_lower = query.lower()
        
        for chunk in chunks:
            num = chunk['num']
            text = chunk['text']
            score = chunk['score']
            
            # Identify key entities
            entities = []
            if "co-founder" in query_lower:
                cofounder_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+is\s+Co-Founder', text)
                entities.extend(cofounder_matches[:3])
            
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
                analysis += f"Contains entities: {', '.join(entities)}. "
            if concepts:
                analysis += f"Relevant concepts: {', '.join(concepts)}. "
            analysis += f"Score {score:.2f} indicates {'high' if score >= 0.70 else 'medium' if score >= 0.50 else 'low'} relevance."
            analyses.append(analysis)
        
        return '\n'.join(analyses) if analyses else "Analyzed semantic meaning of all chunks."
    
    elif step_num == 4:
        # STEP 4: EVALUATE RELEVANCE
        # Expected format: "Chunk X (Score: Y.YY, RELEVANCE relevance): [Directly answers/Does not directly answer] the query. [Contains/Information should be ignored]."
        if not chunks or not final_answer:
            return ""
        
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
    
    elif step_num == 5:
        # STEP 5: EXTRACT MATCHING INFORMATION
        # Expected format: "Found X matching item(s):\n  1. [item1]\n  2. [item2]\n\nInformation found in: Chunk X, Chunk Y"
        if not chunks or not final_answer:
            return ""
        
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
                item_words = set(item.lower().split()[:3])  # First 3 words
                chunk_words = set(text.lower().split())
                if len(item_words.intersection(chunk_words)) >= 2:
                    chunk_sources.append(f"Chunk {chunk['num']}")
                    break
        
        if chunk_sources:
            extraction += f"\nInformation found in: {', '.join(set(chunk_sources))}"
        
        return extraction
    
    elif step_num == 6:
        # STEP 6: VERIFY COMPLETENESS
        # Expected format: "Read all X chunk(s) completely.\nExtracted Y matching item(s) across all chunks.\nExtraction is complete - [all relevant information has been identified/query cannot be answered]."
        if not chunks or not final_answer:
            return ""
        
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
    
    elif step_num == 7:
        # STEP 7: SYNTHESIZE RESPONSE
        # Expected format: Just the final answer (no prefix)
        return final_answer if final_answer else ""
    
    return step_content

def extract_company_name(query: str) -> str:
    """Extract company name from query"""
    match = re.search(r'(?:of|at)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
    if match:
        return match.group(1).strip()
    return "the company"

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

def parse_cot_response(cot_content: str) -> Dict[int, str]:
    """Parse existing CoT response into steps"""
    steps = {}
    current_step = None
    current_content = []
    
    lines = cot_content.split('\n')
    for line in lines:
        if line.startswith('STEP '):
            if current_step is not None:
                steps[current_step] = '\n'.join(current_content).strip()
            # Extract step number
            match = re.search(r'STEP (\d+):', line)
            if match:
                current_step = int(match.group(1))
                current_content = []
        elif current_step is not None:
            current_content.append(line)
    
    if current_step is not None:
        steps[current_step] = '\n'.join(current_content).strip()
    
    return steps

def enhance_example_cot(example: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance a single example's CoT with explicit expected output formats"""
    messages = example.get('messages', [])
    
    user_msg = None
    assistant_msg = None
    
    for msg in messages:
        if msg.get('role') == 'user':
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
    
    # Parse existing CoT response
    cot_content = assistant_msg.get('content', '')
    parsed_steps = parse_cot_response(cot_content)
    
    # Extract final answer (STEP 7)
    final_answer = parsed_steps.get(7, cot_content.split('STEP 7: SYNTHESIZE RESPONSE')[-1].strip() if 'STEP 7' in cot_content else cot_content)
    
    # Enhance each step with expected output format
    enhanced_steps = {}
    for step_num in range(1, 8):
        existing_content = parsed_steps.get(step_num, "")
        enhanced_content = enhance_cot_step_with_expected_format(
            step_num, existing_content, query, chunks, final_answer
        )
        enhanced_steps[step_num] = enhanced_content
    
    # Reconstruct CoT response with enhanced steps
    enhanced_cot = f"""STEP 1: UNDERSTAND THE QUERY
{enhanced_steps[1]}

STEP 2: READ EACH CHUNK COMPLETELY
{enhanced_steps[2]}

STEP 3: ANALYZE CHUNK MEANING
{enhanced_steps[3]}

STEP 4: EVALUATE RELEVANCE
{enhanced_steps[4]}

STEP 5: EXTRACT MATCHING INFORMATION
{enhanced_steps[5]}

STEP 6: VERIFY COMPLETENESS
{enhanced_steps[6]}

STEP 7: SYNTHESIZE RESPONSE
{enhanced_steps[7]}"""
    
    # Update assistant message
    new_messages = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            new_messages.append({"role": "assistant", "content": enhanced_cot})
        else:
            new_messages.append(msg)
    
    return {"messages": new_messages}

def enhance_dataset_cot(input_path: str, output_path: str):
    """Enhance entire dataset with explicit expected output formats"""
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} examples. Enhancing CoT steps with expected output formats...")
    
    enhanced_dataset = []
    for i, example in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} examples...")
        
        enhanced_example = enhance_example_cot(example)
        enhanced_dataset.append(enhanced_example)
    
    print(f"Saving enhanced dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced dataset saved! ({len(enhanced_dataset)} examples)")

if __name__ == "__main__":
    import sys
    
    input_path = "rag_analysis_dataset_v2.json"
    output_path = "rag_analysis_dataset_v2.json"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    enhance_dataset_cot(input_path, output_path)
