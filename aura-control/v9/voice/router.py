"""
voice.router -- Intent routing and LLM model selection.

Decides fast-path (phi3:mini) vs deep-path (qwen2.5) and picks
the emotional style for TTS.

Extracted from carbon_demo.py's phi_then_qwen_ollama + choose_style.
"""

from __future__ import annotations

ESCALATE_PHRASES = [
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "cannot", "can't", "unable", "as an ai", "i can't help",
    "it depends", "need more information", "not enough information",
    "too complex", "complex", "beyond", "no context",
]


def needs_escalation(user_text: str, fast_reply: str) -> bool:
    u = (user_text or "").lower().strip()
    r = (fast_reply or "").lower().strip()

    if len(r) < 25:
        return True
    if any(p in r for p in ESCALATE_PHRASES):
        return True
    if any(k in u for k in [
        "why", "how", "explain", "analyze", "analysis",
        "plan", "strategy", "compare", "tradeoff", "design", "architecture",
    ]) and len(r) < 140:
        return True
    if "user:" in r or "aura:" in r:
        return True
    return False


def choose_style(user_text: str, reply_text: str, model: str) -> str:
    u = (user_text or "").lower()
    r = (reply_text or "").lower()

    if any(k in u for k in [
        "error", "debug", "stack", "trace", "cuda", "latency",
        "vad", "systemd", "linux", "jetson", "orin",
    ]):
        return "technical"
    if any(k in u for k in ["sorry", "anxious", "overwhelmed", "hard day", "stressed"]) or \
       any(k in r for k in ["i hear you", "that sounds hard"]):
        return "empathy"
    if any(k in u for k in ["urgent", "now", "stop", "immediately", "asap"]) or \
       any(k in r for k in ["do this", "right now", "one thing at a time"]):
        return "assertive"
    if any(k in u for k in ["haha", "joke", "funny"]):
        return "playful"
    if any(k in u for k in ["good morning", "hello", "hi", "good evening"]):
        return "warm"
    if model == "qwen":
        return "technical"
    return "neutral"
