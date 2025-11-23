#!/usr/bin/env python3
"""
Proactive Analyzer - Analyzes conversations and generates proactive suggestions
"""

import time
import logging
from typing import List, Dict, Optional
from memory_manager import MemoryManager
import requests

logger = logging.getLogger(__name__)

class ProactiveAnalyzer:
    """
    Analyzes conversations and generates proactive suggestions
    """
    
    def __init__(self, memory_manager: MemoryManager, 
                 llm_service_url: str = "http://localhost:11434",
                 rag_client=None):
        """
        Initialize proactive analyzer
        
        Args:
            memory_manager: MemoryManager instance
            llm_service_url: URL of LLM service for generating suggestions
            rag_client: Optional RAG client for knowledge base search
        """
        self.memory_manager = memory_manager
        self.llm_service_url = llm_service_url
        self.rag_client = rag_client
        
        # Configuration
        self.analysis_interval = 30.0  # Analyze every 30 seconds
        self.similarity_threshold = 0.65  # Minimum similarity for relevant matches
        self.min_conversations_for_analysis = 3  # Need at least 3 conversations to analyze
        
        # State
        self.last_analysis_time = 0
        self.last_suggestion_time = 0
        self.suggestion_cooldown = 60.0  # Don't suggest more than once per minute
        self.recent_suggestions = []  # Track recent suggestions to avoid duplicates
    
    def analyze_and_suggest(self, current_conversation: str) -> Optional[str]:
        """
        Analyze current conversation against stored memory and generate suggestion
        
        Args:
            current_conversation: Current conversation text to analyze
        
        Returns:
            Suggestion text if found, None otherwise
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_suggestion_time < self.suggestion_cooldown:
            return None
        
        # Need minimum conversations to analyze
        if len(self.memory_manager.conversations) < self.min_conversations_for_analysis:
            return None
        
        # Search for similar conversations
        similar = self.memory_manager.search_similar(
            current_conversation, 
            k=5, 
            threshold=self.similarity_threshold
        )
        
        if not similar:
            return None
        
        # Get recent conversations for context
        recent = self.memory_manager.search_recent(hours=24, limit=10)
        
        # Analyze patterns and generate suggestion
        suggestion = self._generate_suggestion(current_conversation, similar, recent)
        
        if suggestion:
            self.last_suggestion_time = current_time
            self.recent_suggestions.append({
                "suggestion": suggestion,
                "timestamp": current_time
            })
            # Keep only last 5 suggestions
            self.recent_suggestions = self.recent_suggestions[-5:]
        
        return suggestion
    
    def _generate_suggestion(self, current: str, similar: List[Dict], 
                            recent: List[Dict]) -> Optional[str]:
        """
        Generate proactive suggestion using LLM
        
        Args:
            current: Current conversation
            similar: Similar conversations found
            recent: Recent conversations for context
        
        Returns:
            Suggestion text or None
        """
        try:
            # Build context from similar conversations
            similar_texts = []
            for item in similar[:3]:  # Top 3 similar
                conv = item.get("conversation", {})
                similar_texts.append(conv.get("text", ""))
            
            # Build context from recent conversations
            recent_texts = [conv.get("text", "") for conv in recent[:5]]
            
            # Create analysis prompt
            prompt = self._build_analysis_prompt(current, similar_texts, recent_texts)
            
            # Call LLM to generate suggestion
            suggestion = self._call_llm_for_suggestion(prompt)
            
            # Validate suggestion
            if suggestion and self._is_valid_suggestion(suggestion):
                return suggestion
            
        except Exception as e:
            logger.error(f"[Analyzer] Failed to generate suggestion: {e}")
        
        return None
    
    def _build_analysis_prompt(self, current: str, similar: List[str], 
                              recent: List[str]) -> str:
        """Build prompt for LLM analysis"""
        
        similar_context = "\n".join([f"- {text}" for text in similar]) if similar else "None"
        recent_context = "\n".join([f"- {text}" for text in recent]) if recent else "None"
        
        prompt = f"""You are an AI assistant analyzing a conversation to provide helpful, proactive suggestions.

CURRENT CONVERSATION:
"{current}"

SIMILAR PAST CONVERSATIONS:
{similar_context}

RECENT CONVERSATIONS (for context):
{recent_context}

TASK:
Analyze the current conversation in context of similar past conversations and recent context. 
If you identify a useful insight, pattern, or suggestion that could help the user solve a problem or arrive at a solution, generate a brief, natural suggestion.

GUIDELINES:
- Only suggest if you have a genuinely useful insight
- Be concise (1-2 sentences max)
- Be natural and conversational
- Reference specific information from past conversations if relevant
- Don't be repetitive or obvious
- If no useful insight, respond with "NONE"

FORMAT:
Start with a polite interjection like "Excuse me" or "I just thought of something", then provide the suggestion.

Example good suggestions:
- "Excuse me, have you thought about trying X? Based on your previous experience with Y, I think X might be beneficial."
- "I just thought of something - in your conversation about Z, you mentioned A. Have you considered B as a potential solution?"

RESPONSE:"""
        
        return prompt
    
    def _call_llm_for_suggestion(self, prompt: str) -> Optional[str]:
        """Call LLM service to generate suggestion"""
        try:
            response = requests.post(
                f"{self.llm_service_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "stop": ["NONE", "\n\n"]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                suggestion = result.get("response", "").strip()
                
                # Check if LLM said "NONE" or similar
                if suggestion.upper().startswith("NONE") or len(suggestion) < 10:
                    return None
                
                return suggestion
            else:
                logger.warning(f"[Analyzer] LLM returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning("[Analyzer] LLM request timeout")
            return None
        except Exception as e:
            logger.error(f"[Analyzer] LLM request failed: {e}")
            return None
    
    def _is_valid_suggestion(self, suggestion: str) -> bool:
        """Validate that suggestion is useful and not a duplicate"""
        if not suggestion or len(suggestion) < 20:
            return False
        
        # Check if too similar to recent suggestions
        for recent in self.recent_suggestions:
            if self._text_similarity(suggestion, recent["suggestion"]) > 0.8:
                return False
        
        return True
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity (word overlap)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def analyze_recent_activity(self) -> Optional[str]:
        """
        Periodically analyze recent activity for patterns
        
        Returns:
            Suggestion if pattern found, None otherwise
        """
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_analysis_time < self.analysis_interval:
            return None
        
        self.last_analysis_time = current_time
        
        # Get recent conversations
        recent = self.memory_manager.search_recent(hours=1, limit=20)
        
        if len(recent) < 3:
            return None
        
        # Look for patterns in recent conversations
        # This is a simplified pattern detection - can be enhanced
        topics = {}
        for conv in recent:
            text = conv.get("text", "").lower()
            # Simple keyword extraction (can be enhanced with NLP)
            words = text.split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    topics[word] = topics.get(word, 0) + 1
        
        # Find most common topics
        if topics:
            top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
            # If a topic appears multiple times, might be worth suggesting
            if top_topics[0][1] >= 3:
                # Could generate suggestion based on recurring topic
                pass
        
        return None

