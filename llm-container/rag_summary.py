"""
RAG Summary and Advice Module

This module provides functionality for generating summaries and advice based on RAG context.
It uses a two-stage approach:
1. CoT model extracts relevant information from RAG chunks
2. Base model generates a natural, conversational summary from extracted information
"""

import re
import random
from typing import Optional


def extract_information_with_cot(rag_context: str, query: str, llm_chat_simple, extract_llm_response_content) -> str:
    """
    Use CoT model to extract relevant information from RAG context.
    Returns the extracted information (KEEP items) as a string.
    
    Args:
        rag_context: RAG context chunks
        query: Original user query
        llm_chat_simple: Function to call LLM (from container_rest.py)
        extract_llm_response_content: Function to extract response content (from container_rest.py)
    
    Returns:
        Extracted information string (KEEP items from CoT reasoning)
    """
    print(f"[RAG Summary] 🔍 [CoT Extraction] Extracting information using CoT model for summary/advice query")
    
    # Use CoT system prompt for extraction
    cot_system_prompt = (
        "You are a precise data extraction bot.\n\n"
        "ALWAYS START WITH REASONING:\n"
        "Begin every response with \"REASONING:\" - this is MANDATORY.\n\n"
        "1. REASONING: For each relevant item found in the context:\n"
        "   - Item: [What you found]\n"
        "   - Evidence: \"[Verbatim quote from context]\"\n"
        "   - Action: [KEEP] if it matches the query, otherwise [DISCARD].\n\n"
        "2. End scan with: - End of scan.\n\n"
        "3. FINAL ANSWER: based ONLY on [KEEP] items.\n\n"
        "CRITICAL RULES:\n"
        "- Extract ALL relevant information that matches the query.\n"
        "- Use verbatim quotes from context as evidence.\n"
        "- Mark items as [KEEP] if they are relevant to the query.\n"
        "- Mark items as [DISCARD] if they are not relevant.\n"
        "- FINAL ANSWER should include ALL [KEEP] items.\n"
    )
    
    user_content = f"Knowledge context: {rag_context}\n---\nQuestion: {query}"
    
    messages = [
        {"role": "system", "content": cot_system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Use CoT model to extract information
    llm_response = llm_chat_simple(
        messages,
        max_tokens=2048,
        temperature=0,
        stream=False,  # Non-streaming for extraction
        use_cot_model=True,
        top_p=1.0,
        top_k=-1,
        seed=42,
        stop=["<|im_end|>"],
    )
    
    # Extract the response text
    response_text = extract_llm_response_content(llm_response)
    
    # Parse the CoT response to extract KEEP items with their evidence
    # IMPORTANT: For summary/advice, we want the extracted information (KEEP items + evidence),
    # NOT the FINAL ANSWER. The base model will generate the summary/advice from this extracted information.
    kept_items_with_evidence = []
    
    # Pattern to match: - Item: [name] - Evidence: "[quote]" - Action: [KEEP/DISCARD]
    item_pattern = r'-\s*Item\s*:\s*(.*?)\s+-\s+Evidence\s*:\s*"(.*?)"\s+-\s+Action\s*:\s*\[(KEEP|DISCARD)\]'
    
    for match in re.finditer(item_pattern, response_text, re.IGNORECASE | re.DOTALL):
        item_name = match.group(1).strip()
        evidence = match.group(2).strip()
        action = match.group(3).upper()
        
        if action == "KEEP":
            kept_items_with_evidence.append({
                'item': item_name,
                'evidence': evidence
            })
    
    # If pattern didn't match, try simpler pattern without Evidence section
    if not kept_items_with_evidence:
        item_pattern_simple = r'-\s*Item\s*:\s*(.*?)\s+-\s+Action\s*:\s*\[(KEEP|DISCARD)\]'
        for match in re.finditer(item_pattern_simple, response_text, re.IGNORECASE):
            item_name = match.group(1).strip()
            action = match.group(2).upper()
            if action == "KEEP":
                kept_items_with_evidence.append({
                    'item': item_name,
                    'evidence': item_name  # Use item name as evidence if no evidence section
                })
    
    if kept_items_with_evidence:
        # Format extracted information for base model
        # Include both item names and evidence for context
        extracted_parts = []
        for item_data in kept_items_with_evidence:
            extracted_parts.append(f"{item_data['item']}: {item_data['evidence']}")
        
        extracted_info = "\n".join(extracted_parts)
        print(f"[RAG Summary] ✅ [CoT Extraction] Extracted {len(kept_items_with_evidence)} KEEP items with evidence")
        print(f"[RAG Summary] 📋 Extracted info preview: {extracted_info[:200]}...")
        return extracted_info
    else:
        # Fallback: try to extract FINAL ANSWER if no items found
        final_answer_match = re.search(r'FINAL\s+ANSWER\s*:\s*(.*?)(?=REASONING:|$)', response_text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            extracted_info = final_answer_match.group(1).strip()
            print(f"[RAG Summary] ⚠️ [CoT Extraction] No KEEP items found, using FINAL ANSWER as fallback")
            return extracted_info
        else:
            print(f"[RAG Summary] ⚠️ [CoT Extraction] No KEEP items or FINAL ANSWER found, using reasoning section")
            # Extract reasoning section as fallback
            reasoning_match = re.search(r'REASONING\s*:\s*(.*?)(?=FINAL\s+ANSWER:|$)', response_text, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                return reasoning_match.group(1).strip()
            return response_text


def generate_summary_response(
    extracted_info: str,
    query: str,
    llm_chat_simple,
    extract_llm_response_content,
    stream: bool = False
):
    """
    Generate a natural, conversational summary from extracted information using base model.
    
    Args:
        extracted_info: Extracted information from CoT model
        query: Original user query
        llm_chat_simple: Function to call LLM (from container_rest.py)
        extract_llm_response_content: Function to extract response content (from container_rest.py)
        stream: Whether to stream the response
    
    Returns:
        Summary response (generator if stream=True, string if stream=False)
    """
    print(f"[RAG Summary] 📝 [Summary Generation] Generating summary using base model")
    
    summary_system_prompt = (
        "You are Aura Vision, an AI agent created by Ledger AI Quantum Corporation. "
        "You act as a proactive AI agent guiding users to better outcomes through gentle guidance.\n\n"
        "Your task is to create a natural, conversational summary or provide advice based on the extracted information below.\n\n"
        "CRITICAL RULES:\n"
        "- Use ONLY the extracted information provided below.\n"
        "- Create a natural, flowing summary (not a bullet list).\n"
        "- Be conversational and friendly.\n"
        "- Keep it concise but informative (2-4 sentences).\n"
        "- If the extracted information is incomplete, acknowledge that.\n"
        "- End with a brief, natural question if appropriate.\n"
    )
    
    summary_user_content = (
        f"Extracted information:\n{extracted_info}\n\n"
        f"Based on this information, {query.lower()}\n\n"
        "Provide a natural, conversational summary or advice."
    )
    
    summary_messages = [
        {"role": "system", "content": summary_system_prompt},
        {"role": "user", "content": summary_user_content}
    ]
    
    # Use base model (not CoT) for summary generation
    summary_response = llm_chat_simple(
        summary_messages,
        max_tokens=800,
        temperature=None,  # Use default temperature for natural summary
        stream=stream,
        use_cot_model=False,  # Use base model for summary
    )
    
    if stream:
        # Return generator for streaming
        return summary_response
    else:
        return extract_llm_response_content(summary_response)


def is_summary_query(prompt: str) -> bool:
    """
    Detect if a query is asking for a summary or advice.
    
    This includes:
    - Explicit summary/advice requests ("summarize", "advice", etc.)
    - Statements expressing hopes/goals ("I hope that I improve...", "I want to...")
    - Questions about what we know ("what can you tell me about...")
    
    Args:
        prompt: User query
    
    Returns:
        True if query is a summary/advice query, False otherwise
    """
    normalized_prompt = prompt.lower()
    
    summary_patterns = [
        r'\bsummarize\b', r'\bsummary\b', r'\bsummaries\b',
        r'\badvice\b', r'\badvise\b', r'\brecommend\b', r'\brecommendation\b',
        r'\bsuggestion\b', r'\bsuggest\b', r'\bsuggestions\b',
        r'\bwhat can you tell me about\b', r'\bwhat do we know about\b',
        r'\bwhat is known about\b', r'\boverview\b', r'\boverview of\b',
        r'\bexplain\b', r'\bdescribe\b', r'\bwhat\'s the story\b',
        # Statements expressing hopes/goals/intentions
        r'\bi hope\b', r'\bi want\b', r'\bi would like\b', r'\bi\'d like\b',
        r'\bhow can i\b', r'\bhow do i\b', r'\bhow should i\b',
        r'\bwhat should i\b', r'\bwhat can i\b',
        # Queries asking for suggestions/recommendations
        r'\bgive me\b.*\bsuggestion\b', r'\bgive me\b.*\brecommendation\b',
        r'\bone suggestion\b', r'\bone recommendation\b'
    ]
    
    return any(re.search(pattern, normalized_prompt, re.IGNORECASE) for pattern in summary_patterns)


def handle_summary_advice_query(
    prompt: str,
    rag_context: str,
    llm_chat_simple,
    extract_llm_response_content,
    stream: bool = False
):
    """
    Main function to handle summary/advice queries.
    
    This function:
    1. Uses CoT model to extract relevant information from RAG context
    2. Uses base model to generate a natural summary from extracted information
    
    Args:
        prompt: User query
        rag_context: RAG context chunks
        llm_chat_simple: Function to call LLM (from container_rest.py)
        extract_llm_response_content: Function to extract response content (from container_rest.py)
        stream: Whether to stream the response
    
    Returns:
        Summary response (generator if stream=True, string if stream=False)
    """
    print(f"[RAG Summary] 📝 [Summary Mode] Using CoT extraction + base model summary")
    
    # Step 1: Extract information using CoT model
    extracted_info = extract_information_with_cot(
        rag_context=rag_context,
        query=prompt,
        llm_chat_simple=llm_chat_simple,
        extract_llm_response_content=extract_llm_response_content
    )
    
    # Step 1.5: Yield second filler phrase after CoT extraction completes
    # This provides feedback that processing is happening
    if stream:
        second_filler_phrases = [
            "Alright, extracting the answer now.",
            "Got it, pulling that information together.",
            "One moment, finalizing the answer.",
            "Almost there, extracting the details.",
        ]
        second_filler = random.choice(second_filler_phrases)
        print(f"[RAG Summary] 💭 Yielding SECOND filler phrase (after CoT extraction): '{second_filler}'")
        
        def summary_with_filler():
            # Yield second filler phrase first
            yield "<sentence_start>\n"
            yield f"{second_filler}\n"
            yield "<sentence_end>\n"
            print(f"[RAG Summary] ✅ Second filler phrase yielded")
            
            # Then yield the summary response (normalized and wrapped with sentence tags)
            summary_response = generate_summary_response(
                extracted_info=extracted_info,
                query=prompt,
                llm_chat_simple=llm_chat_simple,
                extract_llm_response_content=extract_llm_response_content,
                stream=stream
            )
            
            # Normalize and wrap summary response with sentence tags
            yield "<sentence_start>\n"
            for chunk in summary_response:
                # Normalize dict chunks to extract content
                if isinstance(chunk, dict):
                    content = None
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                    elif 'content' in chunk:
                        content = chunk.get('content', '')
                    if content:
                        yield content
                elif isinstance(chunk, str):
                    # Check if it's a stringified dict
                    if chunk.strip().startswith('{') and ('id' in chunk or 'choices' in chunk):
                        try:
                            import json
                            parsed = json.loads(chunk)
                            if isinstance(parsed, dict):
                                if 'choices' in parsed and len(parsed['choices']) > 0:
                                    delta = parsed['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                                continue
                        except (json.JSONDecodeError, ValueError):
                            pass
                    # Regular string chunk
                    if chunk:
                        yield chunk
                else:
                    # Other types - convert to string
                    if chunk:
                        yield str(chunk)
            yield "\n<sentence_end>\n"
        
        return summary_with_filler()
    else:
        # Step 2: Generate summary using base model (non-streaming)
        return generate_summary_response(
            extracted_info=extracted_info,
            query=prompt,
            llm_chat_simple=llm_chat_simple,
            extract_llm_response_content=extract_llm_response_content,
            stream=stream
        )
