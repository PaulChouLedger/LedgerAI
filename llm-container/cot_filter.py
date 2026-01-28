"""
Simplified CoT (Chain of Thought) filter for RAG queries.

This filter:
1. Buffers tokens until REASONING: is detected
2. Continues buffering until FINAL ANSWER: is found
3. Extracts KEEP and DISCARD items from reasoning
4. Yields only the FINAL ANSWER with proper sentence tags
5. Handles filler phrases for better UX
"""

import re


def filter_cot_reasoning(generator):
    """
    Filter streaming output to extract FINAL ANSWER from CoT REASONING section.
    
    Only yields content after FINAL ANSWER: marker.
    Filters out items marked [DISCARD] and ensures all [KEEP] items are included.
    
    IMPORTANT: This function buffers all tokens and suppresses REASONING section.
    Only the FINAL ANSWER is yielded with proper sentence tags.
    """
    print("[CoT Filter] 🔍 Starting CoT filter - buffering all tokens (suppressing output)")
    
    # Buffer all tokens to analyze the full response
    # CRITICAL: Do NOT yield any tokens during buffering - suppress everything until we extract FINAL ANSWER
    tokens = []
    full_text = ""
    token_count = 0
    
    # First pass: collect all tokens (SUPPRESS - do not yield)
    for token in generator:
        token_count += 1
        # Handle dict tokens
        if isinstance(token, dict):
            token = token.get('content', '') or token.get('text', '') or str(token)
        
        # Store token but DO NOT YIELD - we're buffering to extract FINAL ANSWER
        tokens.append(token)
        
        # Extract text content (remove sentence tags) for analysis
        text = str(token) if token else ""
        text = text.replace("<sentence_start>", "").replace("<sentence_end>", "").replace("\n", " ").strip()
        if text:
            full_text += text + " "
    
    print(f"[CoT Filter] 📊 Buffered {token_count} tokens, {len(full_text)} characters")
    print(f"[CoT Filter] 📝 Full text preview (first 500 chars): {full_text[:500]}")
    
    # Check if this is a CoT response
    full_text_lower = full_text.lower()
    if "reasoning:" not in full_text_lower:
        # Not a CoT response - pass through all tokens
        for token in tokens:
            yield token
        return
    
    # CoT response detected - extract FINAL ANSWER
    # Find REASONING section
    reasoning_match = re.search(r'REASONING:\s*(.*?)(?=FINAL\s+ANSWER:|$)', full_text, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        # No REASONING section found - pass through
        for token in tokens:
            yield token
        return
    
    reasoning_text = reasoning_match.group(1)
    
    # Extract KEEP and DISCARD items
    kept_items = []
    discarded_items = set()
    
    # Pattern to match: - Item: [Name] ... - Action: [KEEP] or [DISCARD]
    item_pattern = r'- Item:\s*([^-]+?)\s*-\s*(?:Evidence:[^-]*?)?\s*-\s*Action:\s*\[(KEEP|DISCARD)\]'
    
    for match in re.finditer(item_pattern, reasoning_text, re.IGNORECASE):
        item_name = match.group(1).strip()
        action = match.group(2).upper()
        
        # Clean item name
        item_name = re.sub(r'^["\']|["\']$', '', item_name)
        item_name = item_name.strip()
        
        if action == "KEEP":
            if item_name and item_name not in kept_items:
                kept_items.append(item_name)
        elif action == "DISCARD":
            if item_name:
                discarded_items.add(item_name.lower())
    
    # Find FINAL ANSWER
    final_answer_match = re.search(r'FINAL\s+ANSWER:\s*(.*?)(?=REASONING:|$)', full_text, re.IGNORECASE | re.DOTALL)
    if not final_answer_match:
        # No FINAL ANSWER found - construct from KEEP items
        if kept_items:
            if len(kept_items) == 1:
                final_answer = kept_items[0]
            elif len(kept_items) == 2:
                final_answer = f"{kept_items[0]} and {kept_items[1]}"
            else:
                final_answer = ", ".join(kept_items[:-1]) + f", and {kept_items[-1]}"
        else:
            # Fallback
            final_answer = "I don't understand. Could you please repeat or rephrase your question?"
    else:
        final_answer = final_answer_match.group(1).strip()
        
        # Clean up final answer
        final_answer = re.sub(r'\[(KEEP|DISCARD|Action|Result)\]', '', final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r'(?m)^- .*$', '', final_answer).strip()
        final_answer = re.sub(r'- End of scan\.?\s*', '', final_answer, flags=re.IGNORECASE)
        
        # Remove DISCARD items
        if discarded_items:
            for discarded_name in discarded_items:
                name_parts = discarded_name.split()
                if len(name_parts) > 1:
                    pattern = r'\b' + r'\s+'.join([re.escape(part) for part in name_parts]) + r'\b'
                else:
                    pattern = r'\b' + re.escape(discarded_name) + r'\b'
                final_answer = re.sub(pattern, '', final_answer, flags=re.IGNORECASE)
            # Clean up extra spaces
            final_answer = re.sub(r'\s+', ' ', final_answer)
            final_answer = re.sub(r',\s*,', ',', final_answer)
            final_answer = re.sub(r',\s*and\s*,', ' and ', final_answer)
            final_answer = re.sub(r'^\s*,\s*', '', final_answer)
            final_answer = re.sub(r'\s*,\s*$', '', final_answer)
        
        # Ensure all KEEP items are included
        if kept_items:
            final_answer_lower = final_answer.lower()
            missing_items = []
            for kept_item in kept_items:
                kept_lower = kept_item.lower()
                item_parts = kept_lower.split()
                if len(item_parts) > 1:
                    pattern = r'\b' + r'\s+'.join([re.escape(part) for part in item_parts]) + r'\b'
                    if not re.search(pattern, final_answer_lower):
                        missing_items.append(kept_item)
                else:
                    if kept_lower not in final_answer_lower:
                        missing_items.append(kept_item)
            
            if missing_items:
                # Add missing items
                if len(kept_items) == 1:
                    final_answer = kept_items[0]
                elif len(kept_items) == 2:
                    final_answer = f"{kept_items[0]} and {kept_items[1]}"
                else:
                    final_answer = ", ".join(kept_items[:-1]) + f", and {kept_items[-1]}"
    
    # CRITICAL: Only yield the final answer with proper sentence tags
    # DO NOT yield any of the buffered REASONING tokens
    print(f"[CoT Filter] ✅ Extracted final answer: '{final_answer[:100]}...'")
    print(f"[CoT Filter] ✅ KEEP items: {kept_items}")
    print(f"[CoT Filter] ✅ DISCARD items: {list(discarded_items)}")
    
    if final_answer.strip():
        print(f"[CoT Filter] 💭 Yielding final answer with sentence tags")
        yield "<sentence_start>\n"
        words = final_answer.strip().split()
        for i, word in enumerate(words):
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word
        yield "\n<sentence_end>\n"
        print(f"[CoT Filter] ✅ Final answer yielded - filter complete")
    else:
        # Fallback
        fallback = "I don't understand. Could you please repeat or rephrase your question?"
        yield "<sentence_start>\n"
        for word in fallback.split():
            yield word + " "
        yield "\n<sentence_end>\n"
