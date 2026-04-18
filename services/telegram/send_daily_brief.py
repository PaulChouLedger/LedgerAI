#!/usr/bin/env python3
"""One-shot: generate and send the daily community brief to main channel."""

import os
import sys
import re
import requests
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from rag import rag_context_for
from llm import llm_call
from persona import COMMUNITY_BRIEF_SYSTEM

CHAT_ID = -1003025733750  # Area31

# Gather news from RAG
print("Gathering news from RAG...")
rag_text = ""
queries = [
    "latest crypto news today",
    "AI technology breakthroughs today",
    "global markets stocks economy today",
    "world news geopolitics today",
    "technology science news today",
    "bitcoin ethereum solana price",
]
for q in queries:
    ctx = rag_context_for(q, k=5, max_chars=2000)
    if ctx:
        rag_text += ctx + "\n\n"

print(f"Gathered {len(rag_text)} chars of RAG context")

# Build prompt
system = COMMUNITY_BRIEF_SYSTEM.format(
    rag_context=rag_text or "No news data available.",
    community_context="No recent group context.",
)

now = datetime.now(timezone.utc)
date_str = now.strftime("%A, %B %d, %Y")
prompt = (
    f"Today is {date_str}. "
    "Deliver your daily community brief. Cover the biggest stories "
    "across all categories. Be sharp, be opinionated, connect the dots."
)

print("Calling LLM...")
response = llm_call(prompt, system, 1500)

if not response:
    print("LLM returned nothing!")
    sys.exit(1)

# Clean up
response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
response = re.sub(r'\*\*(.+?)\*\*', r'\1', response)
response = re.sub(r'^#+\s*', '', response, flags=re.MULTILINE)
response = response.strip()

print("=== BRIEF ===")
print(response)
print("=== END ===")

# Send to Telegram with typing cadence
token = os.environ.get("TELEGRAM_BOT_TOKEN")
if not token:
    print("No TELEGRAM_BOT_TOKEN!")
    sys.exit(1)

import time as _time

base_url = f"https://api.telegram.org/bot{token}"

# Split into sentence chunks (same logic as bot.py _split_into_chunks)
import re as _re
_SENTENCE_RE = _re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
chunks = _SENTENCE_RE.split(response)
# Merge short chunks
merged = []
for c in chunks:
    if merged and len(merged[-1]) < 80:
        merged[-1] += " " + c
    else:
        merged.append(c)

print(f"Sending in {len(merged)} chunks...")
for i, chunk in enumerate(merged):
    # Typing indicator
    type_s = len(chunk) * 0.04
    type_s = max(type_s, 0.8)
    type_s = min(type_s, 6.0)
    elapsed = 0.0
    while elapsed < type_s:
        requests.post(f"{base_url}/sendChatAction",
                      json={"chat_id": CHAT_ID, "action": "typing"})
        wait = min(3.0, type_s - elapsed)
        _time.sleep(wait)
        elapsed += wait

    r = requests.post(f"{base_url}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": chunk})
    data = r.json()
    if data.get("ok"):
        print(f"  Chunk {i+1}/{len(merged)}: sent (msg_id={data['result']['message_id']})")
    else:
        print(f"  Chunk {i+1}/{len(merged)}: FAILED {data}")

    # Pause between chunks
    if i < len(merged) - 1:
        _time.sleep(1.0)
