#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to Natural Language Converter
==================================

Converts JSON output from fine-tuned model to natural language for user display.
This allows model to learn structured extraction (easier) while still providing
natural language responses to users.
"""

import json
from typing import Dict, Any, Optional

def json_to_natural_language(json_output: str, query: Optional[str] = None) -> str:
    """
    Convert JSON output to natural language for user display.
    
    Args:
        json_output: JSON string from model output
        query: Original query (optional, for context)
    
    Returns:
        Natural language response
    """
    try:
        # Try to parse JSON
        data = json.loads(json_output.strip())
        
        # Handle not_found case
        if data.get("answer_type") == "not_found":
            return data.get("text", "I don't have that information in the provided documents")
        
        # Handle entity/list queries
        if data.get("answer_type") in ["entities", "list"]:
            items = data.get("items", [])
            
            if len(items) == 0:
                return "I don't have that information in the provided documents"
            elif len(items) == 1:
                return items[0]
            elif len(items) == 2:
                return f"{items[0]} and {items[1]}"
            else:
                # Format: "item1, item2, item3, and item4"
                return ", ".join(items[:-1]) + f", and {items[-1]}"
        
        # Handle comparison/analytical/relationship/process queries
        elif data.get("answer_type") in ["comparison", "analytical", "relationship", "process"]:
            text = data.get("text", "")
            if text:
                return text
            else:
                return "I don't have that information in the provided documents"
        
        # Fallback: return text if available
        if data.get("text"):
            return data["text"]
        
        # Last resort: return items if available
        if data.get("items"):
            items = data["items"]
            if len(items) == 1:
                return items[0]
            elif len(items) == 2:
                return f"{items[0]} and {items[1]}"
            else:
                return ", ".join(items[:-1]) + f", and {items[-1]}"
        
        return "I don't have that information in the provided documents"
    
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract natural language from output
        # Sometimes model outputs JSON with extra text
        json_output_clean = json_output.strip()
        
        # Try to find JSON object in output
        start_idx = json_output_clean.find('{')
        end_idx = json_output_clean.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            try:
                json_str = json_output_clean[start_idx:end_idx]
                return json_to_natural_language(json_str, query)
            except:
                pass
        
        # If all else fails, return as-is (might be natural language already)
        return json_output.strip()
    
    except Exception as e:
        # Unexpected error - return original output
        return json_output.strip()

def extract_json_from_output(model_output: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from model output (handles cases where model adds extra text).
    
    Args:
        model_output: Raw model output (may contain JSON + extra text)
    
    Returns:
        Parsed JSON dict or None if not found
    """
    try:
        # Try direct parsing first
        return json.loads(model_output.strip())
    except:
        pass
    
    # Try to find JSON object in output
    start_idx = model_output.find('{')
    end_idx = model_output.rfind('}') + 1
    
    if start_idx >= 0 and end_idx > start_idx:
        try:
            json_str = model_output[start_idx:end_idx]
            return json.loads(json_str)
        except:
            pass
    
    return None

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Entity query
        {
            "json": '{"answer_type": "entities", "items": ["Paul Chou", "David Lara", "Jorge Guinovart", "Bob Carella"], "text": "", "chunks_used": [1, 2, 3, 4]}',
            "query": "who are the co-founders of LedgerAI?",
            "expected": "Paul Chou, David Lara, Jorge Guinovart, and Bob Carella"
        },
        # List query
        {
            "json": '{"answer_type": "list", "items": ["feature1", "feature2", "feature3"], "text": "", "chunks_used": [1, 2]}',
            "query": "list the features",
            "expected": "feature1, feature2, and feature3"
        },
        # Comparison query
        {
            "json": '{"answer_type": "comparison", "items": [], "text": "CompanyA focuses on innovation, while CompanyB emphasizes different aspects.", "chunks_used": [2, 3]}',
            "query": "compare CompanyA and CompanyB",
            "expected": "CompanyA focuses on innovation, while CompanyB emphasizes different aspects."
        },
        # Not found
        {
            "json": '{"answer_type": "not_found", "items": [], "text": "I don\'t have that information in the provided documents", "chunks_used": []}',
            "query": "who are the managers of UnknownCorp?",
            "expected": "I don't have that information in the provided documents"
        },
    ]
    
    print("Testing JSON to Natural Language Converter")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        result = json_to_natural_language(test["json"], test["query"])
        status = "✅" if result == test["expected"] else "❌"
        print(f"\nTest {i}: {status}")
        print(f"  Query: {test['query']}")
        print(f"  JSON: {test['json']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Got: {result}")
        if result != test["expected"]:
            print(f"  ⚠️  Mismatch!")
    
    print("\n" + "=" * 80)
    print("Testing JSON extraction from noisy output")
    print("=" * 80)
    
    noisy_outputs = [
        'Here is the answer: {"answer_type": "entities", "items": ["John", "Jane"], "text": "", "chunks_used": [1]}',
        '{"answer_type": "list", "items": ["item1", "item2"], "text": "", "chunks_used": [1]} Some extra text here',
        'Invalid JSON output that should be returned as-is',
    ]
    
    for i, output in enumerate(noisy_outputs, 1):
        print(f"\nNoisy Output {i}:")
        print(f"  Input: {output[:80]}...")
        result = json_to_natural_language(output)
        print(f"  Result: {result}")
