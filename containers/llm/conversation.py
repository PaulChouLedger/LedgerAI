"""
Conversation Module

This module handles regular conversational queries using the base model.
No filler phrases, filters, or RAG - just direct base model responses.
"""

from typing import Generator, Union


def handle_conversation_query(
    prompt: str,
    messages: list,
    llm_chat_simple,
    _normalize_stream_chunks,
    stream: bool = False,
    max_tokens: int = 800,
    rag_attempted_but_no_results: bool = False
):
    """
    Handle regular conversational queries using base model directly.
    
    This function:
    1. Calls base model directly (no filters, no filler phrases)
    2. Streams response immediately for low latency
    3. Handles both truly conversational queries and information-seeking queries with no RAG results
    
    Args:
        prompt: User query
        messages: Chat messages for LLM (already includes appropriate system prompt)
        llm_chat_simple: Function to call LLM (from container_rest.py)
        _normalize_stream_chunks: Function to normalize stream chunks (from container_rest.py)
        stream: Whether to stream the response
        max_tokens: Maximum tokens for base model
        rag_attempted_but_no_results: True if RAG search was attempted but found no results
    
    Returns:
        Response (generator if stream=True, string if stream=False)
    """
    if rag_attempted_but_no_results:
        print(f"[Conversation] 💬 [Conversation Mode] Handling information-seeking query with no RAG results (using base model)")
    else:
        print(f"[Conversation] 💬 [Conversation Mode] Handling conversational query with base model")
    
    if stream:
        def conversation_stream():
            try:
                # Call base model directly (no filters, no filler phrases)
                llm_response = llm_chat_simple(
                    messages,
                    max_tokens=max_tokens,
                    temperature=None,  # Use default temperature for natural conversation
                    stream=True,
                    use_cot_model=False,  # Use base model for conversations
                )
                
                # Normalize stream chunks (handle dicts, strings, etc.)
                normalized_stream = _normalize_stream_chunks(llm_response)
                
                # Stream response directly (no filtering, no buffering)
                print(f"[Conversation] ✅ Streaming base model response directly")
                for chunk in normalized_stream:
                    yield chunk
                    
            except Exception as e:
                print(f"[Conversation] ⚠️ Error during conversation generation: {e}")
                import traceback
                traceback.print_exc()
                # Yield fallback message
                yield "<sentence_start>\n"
                yield "Hmm, that one tripped me up. Want to try asking a different way?"
                yield "\n<sentence_end>\n"
        
        return conversation_stream()
    else:
        # Non-streaming: Call base model directly
        try:
            llm_response = llm_chat_simple(
                messages,
                max_tokens=max_tokens,
                temperature=None,  # Use default temperature for natural conversation
                stream=False,
                use_cot_model=False,  # Use base model for conversations
            )
            return llm_response
        except Exception as e:
            print(f"[Conversation] ⚠️ Error during conversation generation: {e}")
            import traceback
            traceback.print_exc()
            return "Hmm, that one tripped me up. Want to try asking a different way?"
