#!/usr/bin/env python3
"""
Verbatim Evidence Extraction Helper
===================================
Utility functions to ensure all evidence in training examples is verbatim from context.
"""

import re
from typing import Optional, List, Tuple


class VerbatimEvidenceExtractor:
    """Extracts verbatim evidence from chunks."""
    
    @staticmethod
    def find_verbatim_quote(search_phrase: str, context: str, min_length: int = 10, max_length: int = 200) -> Optional[str]:
        """
        Find exact verbatim quote containing search_phrase in context.
        Returns the longest matching phrase found.
        """
        search_lower = search_phrase.lower().strip()
        context_lower = context.lower()
        
        # Try exact match first
        if search_lower in context_lower:
            idx = context_lower.find(search_lower)
            # Try to expand to sentence or meaningful phrase
            start = max(0, idx - 50)
            end = min(len(context), idx + len(search_phrase) + 100)
            expanded = context[start:end]
            
            # Find sentence boundaries
            sentence_start = expanded.rfind('.', 0, expanded.find(search_phrase))
            sentence_end = expanded.find('.', expanded.find(search_phrase) + len(search_phrase))
            
            if sentence_start != -1 and sentence_end != -1:
                quote = expanded[sentence_start+1:sentence_end].strip()
                if len(quote) >= min_length and len(quote) <= max_length:
                    return quote
            elif sentence_end != -1:
                quote = expanded[:sentence_end].strip()
                if len(quote) >= min_length and len(quote) <= max_length:
                    return quote
            
            # Fallback to exact match with some context
            start_idx = max(0, idx - 20)
            end_idx = min(len(context), idx + len(search_phrase) + 50)
            quote = context[start_idx:end_idx].strip()
            
            # Clean up
            if quote.startswith('.'):
                quote = quote[1:].strip()
            if len(quote) > max_length:
                period = quote.find('.', 30)
                if period != -1:
                    quote = quote[:period+1]
            
            if len(quote) >= min_length and len(quote) <= max_length:
                return quote
        
        # Try word-by-word matching for partial phrases
        words = search_phrase.split()
        if len(words) >= 3:
            # Try progressively shorter phrases
            for length in range(len(words), 2, -1):
                for start_idx in range(len(words) - length + 1):
                    phrase = ' '.join(words[start_idx:start_idx+length])
                    if phrase.lower() in context_lower:
                        idx = context_lower.find(phrase.lower())
                        # Extract surrounding context
                        start = max(0, idx - 30)
                        end = min(len(context), idx + len(phrase) + 80)
                        quote = context[start:end].strip()
                        
                        # Clean up
                        if quote.startswith('.'):
                            quote = quote[1:].strip()
                        if len(quote) > max_length:
                            period = quote.find('.', 50)
                            if period != -1:
                                quote = quote[:period+1]
                        
                        if len(quote) >= min_length and len(quote) <= max_length:
                            return quote
        
        return None
    
    @staticmethod
    def extract_person_role_evidence(name: str, role: str, company: str, context: str) -> Optional[str]:
        """Extract verbatim evidence for person-role association. Ensures name is in quote."""
        # Try various patterns in order of specificity
        patterns = [
            f"{name} is {role} of {company}",
            f"{name} serves as {role} at {company}",
            f"{name} is the {role} of {company}",
            f"As {role} of {company}, {name}",
            f"{role} of {company}, {name}",
            f"{name}, {role} of {company}",
            f"{name} is {role}",
            f"{name} serves as {role}",
            f"{name} is the {role}",
        ]
        
        for pattern in patterns:
            quote = VerbatimEvidenceExtractor.find_verbatim_quote(pattern, context)
            if quote:
                # CRITICAL: Verify the person's name is actually in the quote
                first_name = name.split()[0]
                last_name = name.split()[-1] if len(name.split()) > 1 else ""
                if first_name in quote and (last_name in quote or len(name.split()) == 1):
                    return quote
                # If name not found, try to find a quote that contains both name and role
                if first_name.lower() in context.lower():
                    # Find context around the name
                    name_idx = context.lower().find(first_name.lower())
                    if name_idx != -1:
                        # Look for role near the name
                        start = max(0, name_idx - 100)
                        end = min(len(context), name_idx + len(name) + 200)
                        name_context = context[start:end]
                        if role.lower() in name_context.lower():
                            # Extract sentence containing both
                            sentence_start = name_context.rfind('.', 0, name_context.find(first_name))
                            sentence_end = name_context.find('.', name_context.find(first_name) + len(name))
                            if sentence_start != -1 and sentence_end != -1:
                                quote = name_context[sentence_start+1:sentence_end].strip()
                                if len(quote) > 20 and len(quote) < 200:
                                    return quote
        
        return None
    
    @staticmethod
    def extract_cofounder_evidence(name: str, company: str, context: str) -> Optional[str]:
        """Extract verbatim evidence for co-founder. CRITICAL: Ensures name is in quote."""
        patterns = [
            f"{name} is Co-Founder of {company}",
            f"{name} is the Co-Founder of {company}",
            f"Co-Founder of {company}, {name}",
            f"As Co-Founder of {company}, {name}",
            f"{name}, Co-Founder of {company}",
            f"Co-Founder and CEO of {company}, {name}",
            f"{name} is Co-Founder and",
            f"As Co-Founder and {name}",
        ]
        
        for pattern in patterns:
            quote = VerbatimEvidenceExtractor.find_verbatim_quote(pattern, context)
            if quote:
                # CRITICAL: Verify the person's name AND company name are in the quote
                first_name = name.split()[0]
                last_name = name.split()[-1] if len(name.split()) > 1 else ""
                has_name = first_name in quote and (last_name in quote or len(name.split()) == 1)
                has_company = company in quote
                if has_name and has_company:
                    return quote
        
        # Fallback: Search for name + "Co-Founder" + company in context
        first_name = name.split()[0]
        last_name = name.split()[-1] if len(name.split()) > 1 else ""
        
        if first_name.lower() in context.lower():
            # Find all occurrences of the name
            import re as re_module
            name_pattern = rf'\b{re_module.escape(first_name)}\s+{re_module.escape(last_name)}\b' if last_name else rf'\b{re_module.escape(first_name)}\b'
            matches = list(re_module.finditer(name_pattern, context, re_module.IGNORECASE))
            
            for match in matches:
                # Look for "Co-Founder" AND company name near this name
                start = max(0, match.start() - 200)
                end = min(len(context), match.end() + 200)
                name_context = context[start:end]
                
                # CRITICAL: Must have both "Co-Founder" and company name
                has_cofounder = 'Co-Founder' in name_context or 'co-founder' in name_context.lower()
                has_company = company in name_context
                
                if has_cofounder and has_company:
                    # Extract sentence containing name, Co-Founder, and company
                    # First, find where company name appears
                    company_idx = name_context.find(company)
                    if company_idx != -1:
                        # Extract from sentence start to after company name (or next sentence)
                        sentence_start = name_context.rfind('.', 0, name_context.find(first_name))
                        # Find sentence end after company name
                        sentence_end = name_context.find('.', company_idx + len(company))
                        if sentence_end == -1:
                            # If no period after company, extend a bit more
                            sentence_end = min(len(name_context), company_idx + len(company) + 50)
                        
                        if sentence_start != -1:
                            quote = name_context[sentence_start+1:sentence_end].strip()
                            # CRITICAL: Verify quote contains name, Co-Founder, AND company
                            if (len(quote) > 20 and len(quote) < 300 and  # Increased max length
                                first_name in quote and 
                                ('Co-Founder' in quote or 'co-founder' in quote.lower()) and
                                company in quote):
                                return quote
        
        return None
    
    @staticmethod
    def extract_education_evidence(name: str, degree: str, university: str, context: str) -> Optional[str]:
        """Extract verbatim evidence for education."""
        first_name = name.split()[0]
        patterns = [
            f"{name} holds a {degree} from {university}",
            f"{first_name} holds a {degree} from {university}",
            f"holds a {degree} from {university}",
            f"{degree} from {university}",
        ]
        
        for pattern in patterns:
            quote = VerbatimEvidenceExtractor.find_verbatim_quote(pattern, context)
            if quote:
                return quote
        
        return None
    
    @staticmethod
    def extract_number_evidence(number: str, context: str) -> Optional[str]:
        """Extract verbatim evidence containing a number."""
        # Try to find the number in context
        if number in context:
            idx = context.find(number)
            start = max(0, idx - 30)
            end = min(len(context), idx + len(number) + 50)
            quote = context[start:end].strip()
            
            # Try to get full sentence
            sentence_start = quote.rfind('.', 0, quote.find(number))
            sentence_end = quote.find('.', quote.find(number) + len(number))
            
            if sentence_start != -1 and sentence_end != -1:
                return quote[sentence_start+1:sentence_end].strip()
            elif sentence_end != -1:
                return quote[:sentence_end].strip()
            
            return quote if len(quote) < 150 else quote[:150]
        
        return None
    
    @staticmethod
    def extract_product_evidence(product: str, company: str, context: str) -> Optional[str]:
        """Extract verbatim evidence for product."""
        patterns = [
            f"{company} offers {product}",
            f"{company} provides {product}",
            f"{product} from {company}",
            f"{product} by {company}",
        ]
        
        for pattern in patterns:
            quote = VerbatimEvidenceExtractor.find_verbatim_quote(pattern, context)
            if quote:
                return quote
        
        return None
    
    @staticmethod
    def extract_benefit_evidence(benefit_keyword: str, context: str) -> Optional[str]:
        """Extract verbatim evidence for benefits."""
        # Look for bullet points or benefit markers
        if "●" in context or "Benefits" in context:
            # Try to find the benefit line
            lines = context.split('\n')
            for line in lines:
                if benefit_keyword.lower() in line.lower() and ("●" in line or "benefit" in line.lower()):
                    return line.strip()
        
        # Fallback to regular search
        return VerbatimEvidenceExtractor.find_verbatim_quote(benefit_keyword, context)


def fix_reasoning_with_verbatim_evidence(reasoning: str, context: str) -> Tuple[str, List[str]]:
    """
    Fix reasoning to ensure all evidence is verbatim from context.
    Returns (fixed_reasoning, warnings)
    """
    warnings = []
    extractor = VerbatimEvidenceExtractor()
    
    # Extract all evidence quotes
    evidence_pattern = r'- Evidence:\s*"([^"]+)"'
    evidences = re.findall(evidence_pattern, reasoning)
    
    if not evidences:
        return reasoning, warnings
    
    fixed_reasoning = reasoning
    
    for evidence in evidences:
        evidence_clean = evidence.strip()
        
        # Check if verbatim
        if evidence_clean.lower() not in context.lower():
            # Try to find verbatim match
            verbatim_match = extractor.find_verbatim_quote(evidence_clean, context)
            
            if verbatim_match:
                # Replace with verbatim
                old_evidence = f'- Evidence: "{evidence}"'
                new_evidence = f'- Evidence: "{verbatim_match}"'
                fixed_reasoning = fixed_reasoning.replace(old_evidence, new_evidence, 1)
            else:
                warnings.append(f"Could not find verbatim match for: '{evidence_clean[:50]}...'")
                # Try to find partial match
                words = evidence_clean.split()
                if len(words) >= 3:
                    phrase = ' '.join(words[:3])
                    partial = extractor.find_verbatim_quote(phrase, context)
                    if partial:
                        old_evidence = f'- Evidence: "{evidence}"'
                        new_evidence = f'- Evidence: "{partial}"'
                        fixed_reasoning = fixed_reasoning.replace(old_evidence, new_evidence, 1)
    
    return fixed_reasoning, warnings


def validate_evidence_verbatim(example: dict) -> Tuple[bool, List[str]]:
    """
    Validate that all evidence in an example is verbatim from context.
    Returns (is_valid, warnings)
    """
    warnings = []
    user_content = example['messages'][1]['content']
    assistant_content = example['messages'][2]['content']
    
    # Extract context
    context = user_content.split('Question:')[0].replace('Knowledge context:', '').strip()
    
    # Extract evidence
    evidence_pattern = r'- Evidence:\s*"([^"]+)"'
    evidences = re.findall(evidence_pattern, assistant_content)
    
    if not evidences:
        return True, warnings
    
    all_verbatim = True
    for evidence in evidences:
        evidence_clean = evidence.strip()
        if evidence_clean.lower() not in context.lower():
            all_verbatim = False
            warnings.append(f"Non-verbatim evidence: '{evidence_clean[:50]}...'")
    
    return all_verbatim, warnings
