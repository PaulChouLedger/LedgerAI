#!/usr/bin/env python3
"""
Fix RAG Integration in Adaptive Diagnostic Engine
Replace the _match_to_guidelines_rag method to use actual RAG search
"""

import re

def fix_rag_integration():
    """Fix the RAG integration by replacing the method"""
    
    file_path = "llm-medical-container/adaptive_diagnostic_engine.py"
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the method start and end
    method_start = content.find("    def _match_to_guidelines_rag(self, complaint: str) -> List[Dict]:")
    if method_start == -1:
        print("❌ Method not found!")
        return False
    
    # Find the next method to determine where this one ends
    next_method_pattern = r"\n    def [a-zA-Z_][a-zA-Z0-9_]*\("
    next_method_match = re.search(next_method_pattern, content[method_start + 1:])
    
    if next_method_match:
        method_end = method_start + 1 + next_method_match.start()
    else:
        # If no next method found, find the end of the class
        class_end_pattern = r"\nclass [A-Z]"
        class_end_match = re.search(class_end_pattern, content[method_start + 1:])
        if class_end_match:
            method_end = method_start + 1 + class_end_match.start()
        else:
            print("❌ Could not find method end!")
            return False
    
    # New method implementation
    new_method = '''    def _match_to_guidelines_rag(self, complaint: str) -> List[Dict]:
        """
        Match chief complaint to guidelines using RAG API for semantic search
        
        Strategy:
        1. Use RAG client for semantic similarity search
        2. Extract guideline names from RAG results
        3. Fallback to category filtering if RAG fails
        
        Returns:
            List of matched guidelines with initial scores
        """
        complaint_lower = complaint.lower()
        
        # Apply smart normalization (LLM or synonyms) to normalize patient language
        complaint_expanded = self._smart_oldcarts_normalization(complaint_lower)
        self._capture_debug(f"[Engine] 🔄 Smart normalization: '{complaint_lower}' → '{complaint_expanded}'")
        
        matched = []
        matched_guideline_names = set()
        
        self._capture_debug(f"\\n[Engine] 🔍 MATCHING TO GUIDELINES (RAG SEARCH MODE)...")
        self._capture_debug(f"[Engine] 🎯 Strategy: RAG semantic search → category fallback")
        self._capture_debug(f"[Engine] ---")
        
        # PHASE 1: RAG semantic search
        try:
            if self.rag_api and self.use_rag_api:
                # Create search query for medical guidelines
                search_query = f"DIAGNOSTIC GUIDELINE {complaint_expanded}"
                self._capture_debug(f"[Engine] 🔍 RAG search query: '{search_query}'")
                
                # Use RAG client for semantic search
                rag_client = get_rag_client()
                rag_results = rag_client.search(
                    query=search_query,
                    k=30,  # Get more results for better coverage
                    threshold=0.2  # Lower threshold for broader matching
                )
                
                if rag_results:
                    self._capture_debug(f"[Engine] 📚 RAG returned {len(rag_results)} chunks")
                    
                    # Extract guideline names from RAG results
                    rag_guideline_names = set()
                    for result in rag_results:
                        if 'guideline_name' in result:
                            rag_guideline_names.add(result['guideline_name'])
                    
                    self._capture_debug(f"[Engine] 📋 RAG found {len(rag_guideline_names)} unique guidelines")
                    
                    # Add RAG matches with scores based on prevalence
                    for name in rag_guideline_names:
                        if name in self.all_guidelines:
                            guideline = self.all_guidelines[name]
                            prevalence = guideline.get('prevalence', 'uncommon')
                            prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                            initial_score = prevalence_scores.get(prevalence, 0.50)
                            matched.append({'name': name, 'score': initial_score, 'data': guideline})
                            matched_guideline_names.add(name)
                            self._capture_debug(f"[Engine]   ✓ {name} (RAG semantic match, prevalence: {prevalence})")
                else:
                    self._capture_debug(f"[Engine] ⚠️ RAG search returned no results")
            else:
                self._capture_debug(f"[Engine] ⚠️ RAG API not available, using fallback")
                
        except Exception as e:
            self._capture_debug(f"[Engine] ❌ RAG search failed: {e}")
        
        self._capture_debug(f"[Engine] 📊 Phase 1 (RAG semantic): {len(matched)} matches")
        
        # PHASE 2: Category-based fallback if RAG didn't find enough
        if len(matched) < 3:
            self._capture_debug(f"[Engine] 🔍 Phase 2: Category-based fallback...")
            
            # PERFORMANCE OPTIMIZATION: Synonym normalization + substring matching
            normalized_complaint = self._normalize_complaint_with_synonyms(complaint)
            category = self._categorize_complaint_by_substring(normalized_complaint)
            relevant_guidelines = self._get_guidelines_by_category(category)
            
            self._capture_debug(f"[Engine] 🎯 Category filtering: {category} → {len(relevant_guidelines)}/{len(self.all_guidelines)} guidelines")
            
            # Add category-based matches that weren't already found by RAG
            for name, guideline in relevant_guidelines.items():
                if name not in matched_guideline_names:
                    triggers = guideline.get('chief_complaint_triggers', [])
                    
                    # Check for trigger matches
                    for trigger in triggers:
                        trigger_lower = trigger.lower()
                        
                        # Exact or subset match
                        if trigger_lower in complaint_lower or complaint_lower in trigger_lower:
                            prevalence = guideline.get('prevalence', 'uncommon')
                            prevalence_scores = {'common': 0.60, 'uncommon': 0.50, 'rare': 0.40}
                            initial_score = prevalence_scores.get(prevalence, 0.50)
                            matched.append({'name': name, 'score': initial_score, 'data': guideline})
                            matched_guideline_names.add(name)
                            self._capture_debug(f"[Engine]   ✓ {name} (category fallback, trigger: '{trigger}', prevalence: {prevalence})")
                            break  # Found match, move to next guideline
            
            self._capture_debug(f"[Engine] 📊 Phase 2 (category fallback): {len(matched)} total matches")
        
        # Sort by score (prevalence-based) and return top matches
        matched.sort(key=lambda x: x['score'], reverse=True)
        
        self._capture_debug(f"[Engine] ✅ Final result: {len(matched)} guidelines matched")
        for i, match in enumerate(matched[:5]):  # Show top 5
            self._capture_debug(f"[Engine]   {i+1}. {match['name']} (score: {match['score']:.2f})")
        
        return matched
'''
    
    # Replace the method
    new_content = content[:method_start] + new_method + content[method_end:]
    
    # Write the updated file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ RAG integration fixed!")
    print("✅ Method _match_to_guidelines_rag now uses actual RAG search")
    return True

if __name__ == "__main__":
    fix_rag_integration()
