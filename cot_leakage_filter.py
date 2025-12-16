#!/usr/bin/env python3
"""
CoT Leakage Post-Processing Filter
Removes intermediate reasoning steps from model outputs
Use this in production to clean model responses
"""

import re
from typing import Optional

def clean_cot_leakage(text: str, aggressive: bool = True) -> str:
    """
    Remove CoT (Chain of Thought) leakage from model output.
    
    Args:
        text: Raw model output that may contain CoT leakage
        aggressive: If True, more aggressively removes patterns
    
    Returns:
        Cleaned text with only final answer
    """
    if not text:
        return text
    
    original_text = text
    
    # Pattern 1: Remove lines that are ONLY extraction instructions
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are only extraction instructions
        if re.match(r'^Extract information from Chunk\s*\d+', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^Extract information from\s*$', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^Chunk\s*\d+[:\-]?\s*$', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^STEP\s*[1-6][:\-]?\s*$', line_stripped, re.IGNORECASE):
            continue
        if re.match(r'^Step\s*[1-6][:\-]?\s*$', line_stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Pattern 2: Remove extraction instructions at the start
    text = re.sub(r'^Extract information from Chunk\s*\d+.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Extract information from Chunk\s*\d+\s*\[and Chunk\s*\d+\].*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Extract information from Chunk\s*\d+\s*and Chunk\s*\d+.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 3: Remove standalone extraction phrases
    text = re.sub(r'Extract information from Chunk\s*\d+\s*\[and Chunk\s*\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Extract information from Chunk\s*\d+\s*and Chunk\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Extract information from Chunk\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Extract information from\s*', '', text, flags=re.IGNORECASE)
    
    # Pattern 4: Remove STEP markers
    text = re.sub(r'STEP\s*[1-6][:\-]?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Step\s*[1-6][:\-]?\s*', '', text, flags=re.IGNORECASE)
    
    # Pattern 5: Remove "Final Answer:" markers but keep content after
    text = re.sub(r'Final Answer:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Final Answer\]\s*', '', text, flags=re.IGNORECASE)
    
    # Pattern 6: Remove standalone "Chunk X:" lines
    text = re.sub(r'^Chunk\s*\d+[:\-]?\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 7: Remove analysis sections that are just instructions
    if aggressive:
        text = re.sub(r'Step \d+:\s*.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'STEP \d+:\s*.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        # Remove "Understanding the Query" sections
        text = re.sub(r'Step 1:.*?Understanding.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Pattern 8: Remove repetitive extraction lines
    lines = text.split('\n')
    filtered_lines = []
    prev_was_extraction = False
    for line in lines:
        is_extraction = bool(re.search(r'Extract information|Chunk\s*\d+', line, re.IGNORECASE))
        if is_extraction and prev_was_extraction:
            continue
        prev_was_extraction = is_extraction
        if not is_extraction or len(line.strip()) > 50:  # Keep if substantial content
            filtered_lines.append(line)
    
    text = '\n'.join(filtered_lines)
    
    # Pattern 9: If text is mostly extraction instructions, try to find actual content
    if len(text.strip()) < 20:
        # Try to extract from original if it had more content
        final_answer_match = re.search(r'Final Answer[:\-]?\s*(.+?)(?:\n\n|\Z)', original_text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            text = final_answer_match.group(1).strip()
        
        # Look for content after extraction instructions (more flexible pattern)
        # Match: "Extract information from Chunk X" followed by actual content
        after_extraction = re.search(
            r'Extract information from Chunk\s*\d+(?:\s*\[and Chunk\s*\d+\])?.*?\n+([A-Z].+?)(?:\n\n|Extract|STEP|\Z)',
            original_text,
            re.IGNORECASE | re.DOTALL
        )
        if after_extraction:
            candidate = after_extraction.group(1).strip()
            # Make sure it's not just another extraction instruction
            if not re.match(r'^Extract information', candidate, re.IGNORECASE) and len(candidate) > 10:
                text = candidate
        
        # Look for content after "Extract information from Chunk X and Chunk Y" on same line
        inline_extraction = re.search(
            r'Extract information from Chunk\s*\d+(?:\s*\[?and Chunk\s*\d+\]?)?[.\s]*([A-Z][^E].+?)(?:\n|Extract|STEP|\Z)',
            original_text,
            re.IGNORECASE | re.DOTALL
        )
        if inline_extraction:
            candidate = inline_extraction.group(1).strip()
            if not re.match(r'^Extract information', candidate, re.IGNORECASE) and len(candidate) > 10:
                text = candidate
        
        # Look for content in brackets like [Final Answer]
        bracket_match = re.search(r'\[Final Answer\]\s*(.+?)(?:\n\n|\Z)', original_text, re.IGNORECASE | re.DOTALL)
        if bracket_match:
            text = bracket_match.group(1).strip()
        
        # Look for names/entities that might be the answer (for co-founder queries)
        # Pattern: "Extract information from Chunk X" followed by names
        name_pattern = r'Extract information from Chunk\s*\d+.*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)'
        name_match = re.search(name_pattern, original_text, re.IGNORECASE)
        if name_match:
            text = name_match.group(1).strip()
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    text = text.strip()
    
    # Final check: if result is empty or too short, try one more extraction method
    if len(text.strip()) < 10:
        # Last resort: look for any substantial content after extraction patterns
        substantial_content = re.search(
            r'Extract information from Chunk\s*\d+.*?([A-Z][^E].{20,})',
            original_text,
            re.IGNORECASE | re.DOTALL
        )
        if substantial_content:
            candidate = substantial_content.group(1).strip()
            # Remove any remaining extraction patterns
            candidate = re.sub(r'Extract information from Chunk\s*\d+', '', candidate, flags=re.IGNORECASE)
            candidate = candidate.strip()
            if len(candidate) > 10:
                text = candidate
    
    # If result is still empty or too short, return a message
    if len(text.strip()) < 10:
        return "[Output contained only extraction instructions - no actual answer extracted]"
    
    return text

def has_cot_leakage(text: str) -> bool:
    """Check if text contains CoT leakage patterns"""
    patterns = [
        r'STEP\s*[1-6]',
        r'Step\s*[1-6]',
        r'Extract information from Chunk',
        r'Chunk\s*\d+[:\-]?\s*$',  # Standalone chunk references
        r'Extract information from\s*$',  # Just extraction instruction
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def extract_final_answer(text: str) -> Optional[str]:
    """
    Try to extract the final answer from text that may contain CoT leakage.
    Returns None if no clear answer found.
    """
    # Method 1: Look for "Final Answer:" marker
    match = re.search(r'Final Answer[:\-]?\s*(.+?)(?:\n\n|$)', text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        if len(answer) > 10:  # Substantial content
            return clean_cot_leakage(answer)
    
    # Method 2: Look for content after extraction instructions
    match = re.search(r'Extract information from Chunk\s*\d+.*?\n\n(.+?)(?:\n\n|$)', text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        if len(answer) > 10 and not has_cot_leakage(answer):
            return clean_cot_leakage(answer)
    
    # Method 3: If no leakage detected, return cleaned version
    if not has_cot_leakage(text):
        cleaned = clean_cot_leakage(text)
        if len(cleaned) > 10:
            return cleaned
    
    return None

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Extract information from Chunk 1 [and Chunk 2]",
        "Extract information from Chunk 1 and Chunk 2\n\nTaylor Brown, Hayden Martinez, Dakota Miller, and Alex Jackson",
        "Step 1: Understanding the query\nStep 2: Extract information\nFinal Answer: The answer is X",
        "Extract information from Chunk 1\nExtract information from Chunk 2\n\nThe managers are: John, Jane, Bob",
    ]
    
    print("Testing CoT Leakage Filter:")
    print("=" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input:  {test[:60]}...")
        cleaned = clean_cot_leakage(test)
        print(f"Output: {cleaned[:60]}...")
        print(f"Has leakage: {has_cot_leakage(test)}")
