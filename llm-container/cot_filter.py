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
        # CRITICAL: Don't add spaces - preserve model's original output format
        text = str(token) if token else ""
        text = text.replace("<sentence_start>", "").replace("<sentence_end>", "").replace("\n", " ")
        # Only add space if token doesn't already end with one and next token doesn't start with one
        if text:
            # Preserve the model's spacing - just concatenate tokens as-is
            # The model already includes proper spacing in its tokens
            full_text += text
    
    print(f"[CoT Filter] 📊 Buffered {token_count} tokens, {len(full_text)} characters")
    print(f"[CoT Filter] 📝 Full text preview (first 500 chars): {full_text[:500]}")
    
    # Check if this is a CoT response
    # Handle tokenized text where "REASONING:" might be split as "RE ASON ING :"
    # Normalize by removing spaces for detection
    full_text_lower = full_text.lower()
    full_text_no_spaces = full_text_lower.replace(" ", "")
    
    # Check for "reasoning:" in both normalized and original text
    # In tokenized text, "REASONING:" becomes "REASONING:" (no colon in normalized) or "reasoning:" (with colon)
    has_reasoning = "reasoning:" in full_text_lower or "reasoning:" in full_text_no_spaces or "reasoning" in full_text_no_spaces
    
    if not has_reasoning:
        print(f"[CoT Filter] ⚠️ No REASONING marker found - passing through all tokens")
        print(f"[CoT Filter] 🔍 Debug: full_text_lower contains 'reasoning:': {'reasoning:' in full_text_lower}")
        print(f"[CoT Filter] 🔍 Debug: full_text_no_spaces contains 'reasoning:': {'reasoning:' in full_text_no_spaces}")
        print(f"[CoT Filter] 🔍 Debug: full_text_no_spaces contains 'reasoning': {'reasoning' in full_text_no_spaces}")
        # Not a CoT response - pass through all tokens
        for token in tokens:
            yield token
        return
    
    print(f"[CoT Filter] ✅ REASONING section detected")
    
    # CoT response detected - extract FINAL ANSWER
    # Find REASONING section - handle tokenized text where spaces might be inserted
    # Try multiple patterns to handle tokenized "REASONING:" (e.g., "RE ASON ING :")
    reasoning_patterns = [
        r'REASONING\s*:\s*(.*?)(?=FINAL\s+ANSWER:|$)',
        r'RE\s+ASON\s+ING\s*:\s*(.*?)(?=FINAL\s+ANSWER:|$)',
        r'REASON\s+ING\s*:\s*(.*?)(?=FINAL\s+ANSWER:|$)',
    ]
    
    reasoning_match = None
    for pattern in reasoning_patterns:
        reasoning_match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            break
    
    if not reasoning_match:
        # Try with normalized text (no spaces) - this handles tokenized "REASONING:" as "REASONING"
        normalized_text = full_text.replace(" ", "")
        norm_match = re.search(r'REASONING:\s*(.*?)(?=FINALANSWER:|$)', normalized_text, re.IGNORECASE | re.DOTALL)
        if norm_match:
            # Extract reasoning from original text using position
            # Find "reasoning" in original text (case-insensitive, handle tokenized)
            reasoning_start = -1
            for i in range(len(full_text_lower) - 7):
                if full_text_lower[i:i+8] == "reasoning":
                    reasoning_start = i
                    break
            
            if reasoning_start >= 0:
                # Find where FINAL ANSWER starts
                final_answer_start = full_text_lower.find("final answer", reasoning_start)
                if final_answer_start > reasoning_start:
                    reasoning_text = full_text[reasoning_start:final_answer_start]
                    # Remove "REASONING:" prefix (handle tokenized version)
                    reasoning_text = re.sub(r'^RE\s*ASON\s*ING\s*:\s*', '', reasoning_text, flags=re.IGNORECASE)
                    # Create a match object-like structure
                    class MatchObj:
                        def group(self, n):
                            return reasoning_text if n == 1 else None
                    reasoning_match = MatchObj()
                else:
                    reasoning_match = None
    
    if not reasoning_match:
        print(f"[CoT Filter] ⚠️ REASONING section found but couldn't extract - passing through")
        # No REASONING section found - pass through
        for token in tokens:
            yield token
        return
    
    reasoning_text = reasoning_match.group(1)
    
    # Extract KEEP and DISCARD items
    kept_items = []
    discarded_items = set()
    
    # Pattern to match: - Item: [Name] ... - Action: [KEEP] or [DISCARD]
    # Model outputs correctly, so we can use the standard pattern
    item_pattern = r'-\s*Item\s*:\s*([^-]+?)\s*-\s*(?:Evidence\s*:[^-]*?)?\s*-\s*Action\s*:\s*\[(KEEP|DISCARD)\]'
    
    for match in re.finditer(item_pattern, reasoning_text, re.IGNORECASE):
        item_name = match.group(1).strip()
        action = match.group(2).upper()
        
        # Clean item name - just remove quotes and normalize spaces
        item_name = re.sub(r'^["\']|["\']$', '', item_name)
        item_name = re.sub(r'\s+', ' ', item_name)  # Normalize multiple spaces to single space
        item_name = item_name.strip()
        
        if action == "KEEP":
            if item_name and item_name not in kept_items:
                kept_items.append(item_name)
        elif action == "DISCARD":
            if item_name:
                discarded_items.add(item_name.lower())
    
    # Find FINAL ANSWER - handle tokenized text
    final_answer_patterns = [
        r'FINAL\s+ANSWER\s*:\s*(.*?)(?=REASONING:|$)',
        r'FINAL\s+ANSW\s+ER\s*:\s*(.*?)(?=REASONING:|$)',
        r'FINAL\s+ANSW\s*ER\s*:\s*(.*?)(?=REASONING:|$)',
    ]
    
    final_answer_match = None
    for pattern in final_answer_patterns:
        final_answer_match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            break
    
    # If not found, try with normalized text (no spaces)
    if not final_answer_match:
        normalized_text = full_text.replace(" ", "")
        norm_match = re.search(r'FINALANSWER:\s*(.*?)(?=REASONING:|$)', normalized_text, re.IGNORECASE | re.DOTALL)
        if norm_match:
            # Extract from original text using position
            # Find "final answer" in original text (case-insensitive, handle tokenized)
            final_start = -1
            for i in range(len(full_text_lower) - 10):
                if full_text_lower[i:i+11] == "final answer":
                    final_start = i
                    break
            
            if final_start >= 0:
                final_text = full_text[final_start:]
                # Remove "FINAL ANSWER:" prefix (handle tokenized version)
                final_text = re.sub(r'^FINAL\s*ANSW\s*ER\s*:\s*', '', final_text, flags=re.IGNORECASE)
                # Create a match object-like structure
                class MatchObj:
                    def group(self, n):
                        return final_text.strip() if n == 1 else None
                final_answer_match = MatchObj()
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
        
        # Clean up final answer - remove markers and formatting
        # Model outputs correctly, so we just need to remove CoT markers
        final_answer = re.sub(r'\[(KEEP|DISCARD|Action|Result)\]', '', final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r'(?m)^- .*$', '', final_answer).strip()
        final_answer = re.sub(r'- End of scan\.?\s*', '', final_answer, flags=re.IGNORECASE)
        
        # Normalize whitespace (model should already have correct spacing, but clean up any artifacts)
        final_answer = re.sub(r'\s+', ' ', final_answer)
        final_answer = final_answer.strip()
        
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
