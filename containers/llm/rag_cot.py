"""
RAG CoT (Chain-of-Thought) Module

This module handles regular RAG queries using the CoT model.
It provides filler phrase logic and CoT filtering.
"""

import random
from typing import Generator, Union


def handle_rag_cot_query(
    prompt: str,
    rag_context: str,
    messages: list,
    llm_chat_simple,
    _normalize_stream_chunks,
    filter_cot_reasoning,
    stream: bool = False,
    max_tokens: int = 2048
):
    """
    Handle regular RAG queries using CoT model with filler phrase.
    
    This function:
    1. Yields filler phrase immediately
    2. Calls CoT model and applies filter to extract FINAL ANSWER from reasoning
    
    Args:
        prompt: User query
        rag_context: RAG context chunks (for logging)
        messages: Chat messages for LLM (already includes RAG context)
        llm_chat_simple: Function to call LLM (from container_rest.py)
        _normalize_stream_chunks: Function to normalize stream chunks (from container_rest.py)
        filter_cot_reasoning: CoT filter function (from cot_filter.py)
        stream: Whether to stream the response
        max_tokens: Maximum tokens for CoT model
    
    Returns:
        Response (generator if stream=True, string if stream=False)
    """
    print(f"[RAG CoT] 🔍 [RAG CoT Mode] Handling regular RAG query with CoT model")
    
    if stream:
        # Filler phrases for regular RAG queries
        filler_phrases = [
            "Hmm, let me think about that.",
            "Oh, good question. Hang on.",
            "Let me think for a sec.",
            "Hmm, one second.",
            "Yeah, give me a moment on that.",
            "Let me pull that together.",
            "Ooh, that's a good one. One sec.",
            "Hmm, let me see.",
        ]
        filler_phrase = random.choice(filler_phrases)
        print(f"[RAG CoT] 💭 Yielding filler phrase and starting CoT model generation")
        
        def rag_cot_with_filler():
            # Yield filler phrase FIRST (provides immediate feedback)
            yield "<sentence_start>\n"
            yield f"{filler_phrase}\n"
            yield "<sentence_end>\n"
            print(f"[RAG CoT] ✅ Filler phrase yielded - starting CoT model generation")
            
            try:
                # Get the generator (this starts the generation)
                llm_response_generator = llm_chat_simple(
                    messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    stream=True,
                    use_cot_model=True,
                    top_p=1.0,
                    top_k=-1,
                    seed=42,
                    stop=["<|im_end|>"],
                )
                
                # Normalize stream chunks before passing to CoT filter
                normalized_stream = _normalize_stream_chunks(llm_response_generator)
                
                # Apply CoT filter to extract FINAL ANSWER from reasoning
                print(f"[RAG CoT] ✅ [CoT Filter] Applying CoT filter to extract final answer")
                yield from filter_cot_reasoning(normalized_stream, query=prompt)
                
            except Exception as e:
                print(f"[RAG CoT] ⚠️ Error during CoT generation/filtering: {e}")
                import traceback
                traceback.print_exc()
                # Yield fallback message
                yield "<sentence_start>\n"
                yield "Hmm, that one tripped me up. Want to try asking a different way?"
                yield "\n<sentence_end>\n"
        
        return rag_cot_with_filler()
    else:
        # Non-streaming: Call CoT model directly
        try:
            llm_response = llm_chat_simple(
                messages,
                max_tokens=max_tokens,
                temperature=0,
                stream=False,
                use_cot_model=True,
                top_p=1.0,
                top_k=-1,
                seed=42,
                stop=["<|im_end|>"],
            )
            return llm_response
        except Exception as e:
            print(f"[RAG CoT] ⚠️ Error during CoT generation: {e}")
            import traceback
            traceback.print_exc()
            return "Hmm, that one tripped me up. Want to try asking a different way?"
