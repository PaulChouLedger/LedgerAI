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

# Send to Telegram
token = os.environ.get("TELEGRAM_BOT_TOKEN")
if not token:
    print("No TELEGRAM_BOT_TOKEN!")
    sys.exit(1)

url = f"https://api.telegram.org/bot{token}/sendMessage"
r = requests.post(url, json={"chat_id": CHAT_ID, "text": response})
data = r.json()
if data.get("ok"):
    print(f"Sent! message_id={data['result']['message_id']}")
else:
    print(f"Failed: {data}")
