#!/usr/bin/env python3
"""Inspect a specific example to see what's happening"""

import json
import re

# Load dataset
with open("rag_analysis_dataset_v3_json.json", "r") as f:
    dataset = json.load(f)

# Check example 1 (has missing items)
example = dataset[1]
messages = example["messages"]
user_msg = messages[1]["content"]
assistant_msg = messages[2]["content"]

print("=" * 80)
print("Example 1 (Index 1)")
print("=" * 80)

# Extract query
query_match = re.search(r'Query:\s*(.+?)(?:\n\nRAG|$)', user_msg, re.DOTALL)
query = query_match.group(1).strip() if query_match else "N/A"
print(f"\nQuery: {query}")

# Parse assistant response
response_json = json.loads(assistant_msg)
print(f"\nExpected answer_type: {response_json.get('answer_type')}")
print(f"Expected items: {response_json.get('items')}")

# Extract chunks
chunk_pattern = r'\[Chunk (\d+)\] Score: ([\d.]+).*?FULL CHUNK TEXT: [\'"](.+?)[\'"]'
chunks = re.findall(chunk_pattern, user_msg, re.DOTALL)

print(f"\nChunks ({len(chunks)}):")
for i, (chunk_num, score, text) in enumerate(chunks[:3], 1):  # Show first 3
    text_clean = text.replace("\\'", "'").replace('\\"', '"')
    print(f"\nChunk {chunk_num} (Score: {score}):")
    print(f"  Text (first 200 chars): {text_clean[:200]}...")
    
    # Check if expected items are mentioned
    expected_items = response_json.get('items', [])
    for item in expected_items[:3]:  # Check first 3
        if item.lower() in text_clean.lower():
            print(f"  ✅ '{item}' found in chunk")
        else:
            print(f"  ❌ '{item}' NOT found in chunk")
