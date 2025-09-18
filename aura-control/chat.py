import os
import re
import nltk
from collections import deque
from llama_cpp import Llama
from nltk.tokenize.punkt import PunktSentenceTokenizer
from core.context import retrieve_context
from core.speaker import enqueue_tts_chunk

# === NLTK Punkt setup ===
nltk.data.path.append("/home/aura/nltk_data")
nltk.download("punkt", quiet=True)
tokenizer = PunktSentenceTokenizer()

# === LLaMA.cpp Setup ===
llm = Llama(
    model_path="/models/mistral.gguf",  # ✅ Adjust path as needed
    n_ctx=4096,
    n_threads=4,
    use_mlock=True,
    use_mmap=True,
)

# === Filler phrases ===
FILLER_PHRASES = [
    "I'm happy to help!",
    "Let me think about that...",
    "Sure, one moment...",
    "Alright, let's take a look.",
    "Hold on while I check.",
    "Great question! Let me see..."
]
recent_fillers = deque(maxlen=3)

SHORT_PROMPTS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ok", "cool", "interesting"
]

def is_short_prompt(text):
    return any(text.lower().strip() == sp for sp in SHORT_PROMPTS)

def get_filler():
    for phrase in FILLER_PHRASES:
        if phrase not in recent_fillers:
            recent_fillers.append(phrase)
            return phrase
    return "One sec..."

def generate_response(user_text):
    print(f"🧠 LLM Prompt: {user_text}")

    # === Context-aware prompt ===
    context = retrieve_context(user_text)
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": user_text}
    ]

    # === Stream output from LLaMA.cpp ===
    stream = llm.create_chat_completion(messages=messages, stream=True)

    buffer = ""
    for chunk in stream:
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        buffer += content

        if any(p in buffer for p in ".!?") and len(buffer.strip()) > 12:
            sentences = tokenizer.tokenize(buffer)
            if len(sentences) > 1:
                to_speak = sentences[:-1]
                buffer = sentences[-1]
                for sentence in to_speak:
                    enqueue_tts_chunk(sentence.strip())

    if buffer.strip():
        enqueue_tts_chunk(buffer.strip())
