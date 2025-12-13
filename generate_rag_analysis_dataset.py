#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Analysis Dataset Generator - General Framework
===================================================

Generates training data for LLM to learn GENERAL RAG analysis skills:
1. Read entire RAG chunks completely (6-8 sentences each)
2. Analyze and understand meaning in chunks
3. Extract relevant information to query (any type of information)
4. Ignore irrelevant information (especially similar but non-answering content)
5. Use LLM scoring to determine if extracted information directly answers query

Dataset: 3000 varied samples with 3-4 chunks of 6-8 sentences each
Focus: General extraction patterns, not specific entity types
"""

import json
import random
import re
from typing import List, Dict, Any, Tuple

# ============================================================================
# General Chunk Generation (6-8 sentences)
# ============================================================================

def create_general_chunk(
    relevant_info: Dict[str, Any],
    irrelevant_info: List[str] = None,
    min_sentences: int = 6,
    max_sentences: int = 8
) -> str:
    """
    Create a general chunk with relevant and irrelevant information.
    
    relevant_info: Dict with keys like:
        - "entities": List of entities (people, places, things)
        - "concepts": List of concepts/ideas
        - "facts": List of factual statements
        - "relationships": List of relationship statements
        - "attributes": Dict of attribute-value pairs
        - "numbers": List of numbers/metrics
        - "dates": List of dates/events
    """
    sentences = []
    irrelevant_info = irrelevant_info or []
    
    # Add relevant information
    if "entities" in relevant_info:
        for entity in relevant_info["entities"]:
            if isinstance(entity, dict):
                name = entity.get("name", "")
                role = entity.get("role", "")
                context = entity.get("context", "")
                sentences.append(
                    f"{name} {role}. {context}"
                )
            else:
                sentences.append(f"{entity} is mentioned in this context.")
    
    if "concepts" in relevant_info:
        for concept in relevant_info["concepts"]:
            sentences.append(f"{concept} is a key concept discussed here.")
    
    if "facts" in relevant_info:
        sentences.extend(relevant_info["facts"])
    
    if "relationships" in relevant_info:
        sentences.extend(relevant_info["relationships"])
    
    if "attributes" in relevant_info:
        for attr, value in relevant_info["attributes"].items():
            sentences.append(f"The {attr} is {value}.")
    
    if "numbers" in relevant_info:
        for num_info in relevant_info["numbers"]:
            if isinstance(num_info, dict):
                sentences.append(f"{num_info.get('label', 'The value')} is {num_info.get('value', '')}.")
            else:
                sentences.append(f"The number is {num_info}.")
    
    if "dates" in relevant_info:
        for date_info in relevant_info["dates"]:
            if isinstance(date_info, dict):
                sentences.append(f"On {date_info.get('date', '')}, {date_info.get('event', 'something happened')}.")
            else:
                sentences.append(f"The date is {date_info}.")
    
    # Add irrelevant information (similar but doesn't answer query)
    if irrelevant_info:
        num_irrelevant = random.randint(1, 3)
        sentences.extend(random.sample(irrelevant_info, min(num_irrelevant, len(irrelevant_info))))
    
    # Add filler sentences to reach 6-8 sentences
    filler_templates = [
        "This information is part of a larger context that includes multiple related topics.",
        "Additional details provide further context for understanding the broader picture.",
        "The discussion encompasses various aspects that contribute to a comprehensive view.",
        "Further exploration reveals connections between different elements in the narrative.",
        "The topic involves multiple dimensions that require careful consideration.",
        "Understanding this requires examining various perspectives and related information.",
        "The context includes several interconnected components that shape the overall understanding.",
    ]
    
    while len(sentences) < min_sentences:
        available_fillers = [f for f in filler_templates if f not in sentences[-2:]]
        if available_fillers:
            sentences.append(random.choice(available_fillers))
        else:
            sentences.append(filler_templates[0])
    
    # Add 0-2 more to reach target range
    target_count = random.randint(min_sentences, max_sentences)
    while len(sentences) < target_count and len(sentences) < max_sentences:
        available_fillers = [f for f in filler_templates if f not in sentences[-2:]]
        if available_fillers:
            sentences.append(random.choice(available_fillers))
        else:
            break
    
    sentences = sentences[:max_sentences]
    # Shuffle to mix relevant and irrelevant
    random.shuffle(sentences)
    return " ".join(sentences)

# ============================================================================
# Query Type Templates (General, not entity-specific)
# ============================================================================

def generate_query_templates() -> List[Dict[str, Any]]:
    """Generate diverse query templates covering various extraction patterns."""
    return [
        # Factual queries - Single entity
        {"template": "who is {entity}?", "type": "factual_single", "extraction_type": "entity"},
        {"template": "what is {entity}?", "type": "factual_single", "extraction_type": "entity"},
        {"template": "when did {event} happen?", "type": "factual_single", "extraction_type": "date"},
        {"template": "where is {location}?", "type": "factual_single", "extraction_type": "location"},
        {"template": "what is the {attribute} of {entity}?", "type": "factual_single", "extraction_type": "attribute"},
        
        # Factual queries - Multiple entities
        {"template": "who are the {role} of {organization}?", "type": "factual_multiple", "extraction_type": "entities"},
        {"template": "what are the {items} in {context}?", "type": "factual_multiple", "extraction_type": "list"},
        {"template": "list the {items} that {condition}.", "type": "factual_multiple", "extraction_type": "list"},
        {"template": "what are the key {concepts} related to {topic}?", "type": "factual_multiple", "extraction_type": "concepts"},
        
        # Analytical queries
        {"template": "why did {event} occur?", "type": "analytical", "extraction_type": "reasoning"},
        {"template": "how does {process} work?", "type": "analytical", "extraction_type": "process"},
        {"template": "what caused {outcome}?", "type": "analytical", "extraction_type": "causation"},
        {"template": "what are the implications of {event}?", "type": "analytical", "extraction_type": "implications"},
        
        # Relationship queries
        {"template": "how are {entity1} and {entity2} related?", "type": "relationship", "extraction_type": "relationship"},
        {"template": "what is the connection between {entity1} and {entity2}?", "type": "relationship", "extraction_type": "relationship"},
        {"template": "what role does {entity} play in {context}?", "type": "relationship", "extraction_type": "role"},
        
        # Comparison queries
        {"template": "compare {entity1} and {entity2}.", "type": "comparison", "extraction_type": "comparison"},
        {"template": "what are the differences between {entity1} and {entity2}?", "type": "comparison", "extraction_type": "differences"},
        {"template": "what are the similarities between {entity1} and {entity2}?", "type": "comparison", "extraction_type": "similarities"},
        
        # Attribute queries
        {"template": "what are the characteristics of {entity}?", "type": "attribute", "extraction_type": "attributes"},
        {"template": "what properties does {entity} have?", "type": "attribute", "extraction_type": "properties"},
        {"template": "describe {entity}.", "type": "attribute", "extraction_type": "description"},
        
        # Failed queries (information not available)
        {"template": "what is the {missing_info} of {entity}?", "type": "failed", "extraction_type": None},
        {"template": "who is the {missing_role} of {entity}?", "type": "failed", "extraction_type": None},
    ]

# ============================================================================
# Entity/Concept Generators (Random, diverse)
# ============================================================================

def generate_random_entity() -> str:
    """Generate random entity name."""
    prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Sigma", "Omega", "Nexus", "Vertex", "Apex", "Prime", "Elite", "Quantum", "Nova", "Stellar", "Cosmic", "Digital", "Cyber", "Meta", "Neo", "Ultra"]
    suffixes = ["System", "Platform", "Network", "Hub", "Core", "Base", "Lab", "Works", "Forge", "Dynamics", "Solutions", "Group", "Corp", "Inc", "LLC", "Tech", "AI", "Data", "Cloud", "Space"]
    return f"{random.choice(prefixes)}{random.choice(suffixes)}"

def generate_random_person_name() -> str:
    """Generate random person name."""
    first_names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Sarah", "Jennifer", "Lisa", "Nancy", "Karen", "Betty", "Helen", "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle", "Laura", "Emily", "Kimberly", "Deborah", "Jessica", "Shirley", "Cynthia"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_random_concept() -> str:
    """Generate random concept."""
    concepts = [
        "machine learning", "artificial intelligence", "data analytics", "cloud computing",
        "blockchain technology", "quantum computing", "edge computing", "distributed systems",
        "microservices architecture", "API design", "user experience", "product development",
        "market analysis", "strategic planning", "financial modeling", "risk assessment",
        "project management", "agile methodology", "devops practices", "cybersecurity",
        "digital transformation", "innovation strategy", "customer engagement", "supply chain",
        "sustainability", "renewable energy", "biotechnology", "nanotechnology"
    ]
    return random.choice(concepts)

def generate_random_event() -> str:
    """Generate random event."""
    events = [
        "the launch", "the merger", "the acquisition", "the partnership", "the expansion",
        "the innovation", "the breakthrough", "the discovery", "the implementation",
        "the deployment", "the release", "the announcement", "the completion",
        "the transition", "the transformation", "the integration", "the migration"
    ]
    return random.choice(events)

# ============================================================================
# Response Generation (General extraction logic)
# ============================================================================

def extract_information_from_chunks(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    General extraction function - extracts relevant information based on query type.
    Returns only the final answer.
    """
    query_lower = query.lower()
    
    # Evaluate relevance using scores
    high_relevance = [c for c in chunks if c.get('score', 0.0) >= 0.70]
    medium_relevance = [c for c in chunks if 0.50 <= c.get('score', 0.0) < 0.70]
    low_relevance = [c for c in chunks if c.get('score', 0.0) < 0.50]
    
    # Read entire chunks and extract based on query type
    extracted_items = []
    
    # Determine query type and extraction pattern
    if any(word in query_lower for word in ["who are", "list", "what are", "enumerate", "identify"]):
        # Multiple entities/list extraction
        # Extract organization/entity from query if present
        org_match = re.search(r'(?:of|in|for)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
        query_org = org_match.group(1).strip() if org_match else None
        
        # Extract role from query (e.g., "co-founders", "leaders", "members")
        role_match = re.search(r'(?:the|are)\s+([a-z-]+)\s+(?:of|in)', query_lower)
        query_role = role_match.group(1).strip() if role_match else None
        
        # Extract person names from chunks
        person_names = []
        for chunk in high_relevance:
            text = chunk['text']
            # Find person names (two capitalized words)
            name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
            names = re.findall(name_pattern, text)
            
            for name in names:
                # Check if name is associated with the query role and organization
                # Get sentence containing the name for more precise matching
                name_pos = text.find(name)
                sentence_start = max(0, text.rfind('. ', max(0, name_pos-200), name_pos) + 2)
                sentence_end = min(len(text), text.find('. ', name_pos))
                if sentence_end == -1:
                    sentence_end = min(len(text), text.find('\n', name_pos))
                if sentence_end == -1:
                    sentence_end = len(text)
                name_sentence = text[sentence_start:sentence_end].lower()
                
                # Also get broader context for role matching
                name_context = text[max(0, name_pos-100):min(len(text), name_pos+200)].lower()
                
                # Check if role matches
                role_matches = False
                if query_role:
                    # Check for role variations
                    role_variations = {
                        "co-founders": ["co-founder", "cofounder", "founder"],
                        "founders": ["founder", "co-founder"],
                        "leaders": ["leader", "lead"],
                        "members": ["member"],
                        "directors": ["director"],
                        "managers": ["manager"],
                    }
                    variations = role_variations.get(query_role, [query_role])
                    role_matches = any(var in name_context for var in variations)
                else:
                    role_matches = True  # If no specific role, accept any
                
                # CRITICAL: Check if organization matches - must appear WITH the role in same sentence
                org_matches = True
                if query_org:
                    org_normalized = re.sub(r'\s+', '', query_org.lower())
                    query_org_lower = query_org.lower()
                    
                    # Check if organization appears in the sentence with the name
                    sentence_normalized = re.sub(r'\s+', '', name_sentence)
                    
                    # Must have BOTH role AND organization in same sentence/context
                    # Pattern: "[role] of [org]" or "[org] [role]" or "[name] is [role] of [org]"
                    if role_matches:
                        # Check for explicit pattern: role + of + org
                        role_org_pattern = False
                        for var in (role_variations.get(query_role, [query_role]) if query_role else [""]):
                            if var:
                                # Pattern 1: "co-founder of ledgerai" or "co-founder of ledger ai"
                                if f"{var} of {query_org_lower}" in name_sentence or f"{var} of {org_normalized}" in sentence_normalized:
                                    role_org_pattern = True
                                    break
                                # Pattern 2: "ledgerai co-founder" (less common)
                                if f"{query_org_lower} {var}" in name_sentence or f"{org_normalized}{var}" in sentence_normalized:
                                    role_org_pattern = True
                                    break
                        
                        org_matches = role_org_pattern
                    else:
                        # If role doesn't match, org match is irrelevant
                        org_matches = False
                
                if role_matches and org_matches and name not in person_names:
                    # Exclude common non-name phrases
                    excluded = ["Material Science", "Public Affairs", "Chief Operating", "Chief Financial", 
                              "Chief Marketing", "University Washington", "University Texas"]
                    if not any(ex.lower() in name.lower() for ex in excluded):
                        person_names.append(name)
        
        if person_names:
            unique_names = sorted(set(person_names))
            if len(unique_names) == 1:
                return f"{unique_names[0]}."
            else:
                return f"The entities are: {', '.join(unique_names)}."
        else:
            return "I couldn't find that information in the provided documents."
    
    elif any(word in query_lower for word in ["who is", "what is", "when", "where"]):
        # Single entity/fact extraction
        filler_phrases = [
            "the discussion encompasses",
            "this information is part of",
            "additional details provide",
            "further exploration reveals",
            "understanding this requires",
            "the topic involves",
            "the context includes"
        ]
        
        for chunk in high_relevance:
            text = chunk['text']
            sentences = text.split('. ')
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                # Skip filler sentences
                if any(filler in sentence_lower for filler in filler_phrases):
                    continue
                # Find sentence that likely answers the query
                query_keywords = query_lower.split()[:3]
                if any(keyword in sentence_lower for keyword in query_keywords if len(keyword) > 3):
                    # Extract entity name if "who is" query
                    if "who is" in query_lower:
                        name_match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', sentence)
                        if name_match:
                            name = name_match.group(1)
                            # Get context around the name
                            name_pos = sentence.find(name)
                            context = sentence[max(0, name_pos-50):min(len(sentence), name_pos+150)]
                            return context.strip() + "."
                    return sentence.strip() + "."
        
        return "I don't have that information in the provided documents."
    
    elif any(word in query_lower for word in ["why", "how", "what caused", "what are the implications"]):
        # Analytical/reasoning extraction
        for chunk in high_relevance:
            text = chunk['text']
            sentences = text.split('. ')
            reasoning_sentences = []
            for sentence in sentences:
                if any(word in sentence.lower() for word in ["because", "due to", "resulted", "led to", "caused", "enabled", "facilitated"]):
                    reasoning_sentences.append(sentence.strip())
            
            if reasoning_sentences:
                return " ".join(reasoning_sentences[:2]) + "."
        
        return "I don't have information to answer that question in the provided documents."
    
    elif any(word in query_lower for word in ["compare", "difference", "similarity", "relationship", "connection"]):
        # Comparison/relationship extraction - ensure both entities are mentioned
        # Extract entity names from query
        filler_phrases_comp = [
            "the discussion encompasses",
            "this information is part of",
            "additional details provide",
            "further exploration reveals",
            "understanding this requires",
            "the topic involves",
            "the context includes"
        ]
        
        entity1_match = re.search(r'compare (.+?) and', query_lower)
        entity2_match = re.search(r'and (.+?)[\?\.]', query_lower)
        if not entity1_match:
            entity1_match = re.search(r'between (.+?) and', query_lower)
        if not entity2_match:
            entity2_match = re.search(r'and (.+?)[\?\.]', query_lower)
        
        entity1 = entity1_match.group(1).strip().title() if entity1_match else None
        entity2 = entity2_match.group(1).strip().title() if entity2_match else None
        
        for chunk in high_relevance:
            text = chunk['text']
            sentences = text.split('. ')
            comparison_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                # Skip filler sentences
                if any(filler in sentence_lower for filler in filler_phrases_comp):
                    continue
                # Check if sentence mentions both entities (if we extracted them)
                if entity1 and entity2:
                    if entity1.lower() in sentence_lower and entity2.lower() in sentence_lower:
                        comparison_sentences.append(sentence.strip())
                    elif any(word in sentence_lower for word in ["while", "whereas", "compared to", "similar to", "different from", "related to", "connected"]):
                        # Check if at least one entity is mentioned
                        if entity1.lower() in sentence_lower or entity2.lower() in sentence_lower:
                            comparison_sentences.append(sentence.strip())
                else:
                    # Fallback: look for comparison words
                    if any(word in sentence_lower for word in ["while", "whereas", "compared to", "similar to", "different from", "related to", "connected"]):
                        comparison_sentences.append(sentence.strip())
            
            if comparison_sentences:
                result = " ".join(comparison_sentences[:2]) + "."
                # Ensure both entities are mentioned if we have them
                if entity1 and entity2:
                    if entity1 not in result and entity1.title() not in result:
                        result = f"{entity1} is mentioned. " + result
                    if entity2 not in result and entity2.title() not in result:
                        result = result + f" {entity2} is also relevant."
                return result
        
        return "I don't have information to answer that question in the provided documents."
    
    # Generic fallback - extract meaningful information, not filler
    filler_phrases = [
        "the discussion encompasses",
        "this information is part of",
        "additional details provide",
        "further exploration reveals",
        "understanding this requires",
        "the topic involves",
        "the context includes"
    ]
    
    if high_relevance:
        for chunk in high_relevance:
            text = chunk['text']
            sentences = text.split('. ')
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                # Skip filler sentences
                if any(filler in sentence_lower for filler in filler_phrases):
                    continue
                # Return first non-filler sentence
                if len(sentence.strip()) > 20:  # Meaningful length
                    return sentence.strip() + "."
    
    return "I don't have information to answer that question in the provided documents."

# ============================================================================
# Dataset Generation
# ============================================================================

def generate_dataset(num_examples: int = 3000) -> List[Dict[str, Any]]:
    """Generate diverse dataset teaching general RAG analysis patterns."""
    dataset = []
    
    query_templates = generate_query_templates()
    
    # Distribution: 15% failed, 50% complex edge cases, 25% mixed, 10% simple
    target_failed = int(num_examples * 0.15)  # 450 - reduced from 48.6%
    target_complex = int(num_examples * 0.50)  # 1500 - increased from 30%
    target_mixed = int(num_examples * 0.25)  # 750 - reduced from 60%
    target_simple = num_examples - target_failed - target_complex - target_mixed  # 300
    
    query_batches = []
    
    # Failed queries (only 15% - for robustness, not the main focus)
    failed_queries = [q for q in query_templates if q["type"] == "failed"]
    for i in range(target_failed):
        template = failed_queries[i % len(failed_queries)] if failed_queries else query_templates[i % len(query_templates)]
        query_batches.append({**template, "category": "failed"})
    
    # Complex edge case queries (50% - the main focus)
    # Prioritize entity extraction with cross-company filtering, multi-chunk synthesis, etc.
    complex_queries = [
        q for q in query_templates 
        if q["type"] in ["factual_multiple", "analytical", "relationship", "comparison"] 
        and q.get("extraction_type") == "entities"  # Prioritize entity extraction for complexity
    ]
    # Also include analytical and comparison queries that require synthesis
    complex_queries.extend([
        q for q in query_templates 
        if q["type"] in ["analytical", "relationship", "comparison"]
        and q not in complex_queries
    ])
    # If not enough complex queries, use entity extraction queries
    if len(complex_queries) < target_complex:
        entity_queries = [q for q in query_templates if q.get("extraction_type") == "entities"]
        complex_queries.extend(entity_queries[:target_complex - len(complex_queries)])
    
    for i in range(target_complex):
        template = complex_queries[i % len(complex_queries)] if complex_queries else query_templates[i % len(query_templates)]
        query_batches.append({**template, "category": "complex"})
    
    # Mixed relevance queries (25% - reduced)
    mixed_queries = [q for q in query_templates if q["type"] != "failed"]
    for i in range(target_mixed):
        template = mixed_queries[i % len(mixed_queries)]
        query_batches.append({**template, "category": "mixed"})
    
    # Simple queries (10%)
    simple_queries = [q for q in query_templates if q["type"] in ["factual_single"]]
    for i in range(target_simple):
        template = simple_queries[i % len(simple_queries)] if simple_queries else mixed_queries[i % len(mixed_queries)]
        query_batches.append({**template, "category": "simple"})
    
    random.shuffle(query_batches)
    
    for i, query_template in enumerate(query_batches):
        template_str = query_template["template"]
        query_type = query_template["type"]
        category = query_template.get("category", "simple")
        extraction_type = query_template.get("extraction_type", "general")
        
        # Generate random entities for query
        entity = generate_random_entity()
        person = generate_random_person_name()
        event = generate_random_event()
        concept = generate_random_concept()
        organization = generate_random_entity()
        
        # Fill template
        entity2 = generate_random_entity()
        # For entity extraction, use roles that will match what we put in chunks
        role_options = ["leaders", "members", "founders", "co-founders", "directors", "managers"]
        role_term = random.choice(role_options)
        items_term = random.choice(["items", "elements", "components", "features", "aspects"])
        concepts_term = random.choice(["concepts", "ideas", "principles", "notions", "theories"])
        
        query = template_str.format(
            entity=entity,
            person=person,
            event=event,
            concept=concept,
            concepts=concepts_term,
            organization=organization,
            role=role_term,
            items=items_term,
            condition=random.choice(["are important", "were discussed", "were mentioned"]),
            topic=concept,
            entity1=entity,
            entity2=entity2,
            location=random.choice(["located", "situated", "found"]),
            attribute=random.choice(["purpose", "function", "role", "significance"]),
            process=random.choice(["this process", "the system", "the mechanism"]),
            outcome=random.choice(["this result", "the outcome", "the effect"]),
            context=random.choice(["this context", "the system", "the organization"]),
            missing_info=random.choice(["revenue", "budget", "size", "capacity"]),
            missing_role=random.choice(["CTO", "CFO", "CMO", "director"]),
        )
        
        # Store the role term for use in chunk generation
        query_role_term = role_term
        
        chunks = []
        num_chunks = random.randint(3, 4)
        
        # Track what relevant information should be extracted (for proper extraction)
        relevant_extractions = []
        
        # Generate chunks with relevant and irrelevant information
        for j in range(num_chunks):
            if query_type == "failed":
                # Failed query - no relevant information
                relevant_info = {}
                irrelevant_info = [
                    f"Information about {generate_random_entity()} is discussed here.",
                    f"Details regarding {generate_random_concept()} are provided.",
                    f"Context about {generate_random_event()} is included.",
                ]
                text = create_general_chunk({}, irrelevant_info)
                score = random.uniform(0.50, 0.70)  # MEDIUM/LOW relevance
            elif category == "complex":
                # Complex edge cases - ALWAYS include relevant info + irrelevant info for filtering
                # This creates scenarios like cross-company filtering, multi-chunk synthesis, etc.
                if extraction_type == "entities":
                    # Always add relevant entities (distribute across chunks)
                    if j < 3:  # Add entities in first 3 chunks for complex multi-chunk scenarios
                        entity_person = generate_random_person_name()
                        relevant_extractions.append(entity_person)
                        if 'query_role_term' in locals() and query_role_term:
                            role_singular = query_role_term.rstrip('s').replace("co-founder", "co-founder")
                            role_term = role_singular if role_singular in ["leader", "member", "director", "manager", "founder", "co-founder"] else "member"
                        else:
                            role_term = random.choice(["leader", "member", "director", "manager", "founder", "co-founder"])
                        relevant_info = {
                            "entities": [{"name": entity_person, "role": f"is a {role_term}", "context": f"of {organization}."}],
                        }
                    else:
                        relevant_info = {}
                elif extraction_type == "list" and j < 3:
                    item = generate_random_entity()
                    relevant_extractions.append(item)
                    relevant_info = {
                        "entities": [{"name": item, "role": "is important", "context": f"for {concept}."}],
                    }
                elif extraction_type in ["analytical", "relationship", "comparison", "reasoning", "process", "causation", "implications", "role", "differences", "similarities"]:
                    # For analytical/comparison queries, add relationship information
                    if j < 2:
                        if extraction_type in ["comparison", "differences", "similarities"]:
                            relevant_info = {
                                "relationships": [f"{entity} and {entity2} are related through {concept}."],
                                "facts": [f"{entity} focuses on {concept} while {entity2} emphasizes {generate_random_concept()}."],
                            }
                            relevant_extractions.append(f"{entity} and {entity2} are related through {concept}")
                        elif extraction_type in ["reasoning", "causation", "implications"]:
                            relevant_info = {
                                "facts": [f"The {event} occurred because of {concept}."],
                                "relationships": [f"{entity} led to the {event} through {concept}."],
                            }
                            relevant_extractions.append(f"The {event} occurred because of {concept}")
                        else:
                            relevant_info = {
                                "relationships": [f"{entity} is connected to {entity2} through {concept}."],
                                "facts": [f"{entity} and {entity2} share {concept}."],
                            }
                            relevant_extractions.append(f"{entity} is connected to {entity2}")
                    else:
                        relevant_info = {}
                else:
                    relevant_info = {
                        "facts": [f"{entity} is associated with {concept}."],
                    }
                    relevant_extractions.append(f"{entity} is associated with {concept}")
                
                # Always add irrelevant info for complex filtering scenarios
                irrelevant_info = [
                    f"Another unrelated topic {generate_random_entity()} is discussed here.",
                    f"Some other information about {generate_random_concept()} is provided.",
                    f"Additional context about {generate_random_event()} is included.",
                ]
                
                # For entity extraction, ALWAYS add co-founders/leaders of OTHER companies
                if extraction_type == "entities" and query_role_term and ("founder" in query_role_term.lower() or "leader" in query_role_term.lower()):
                    other_company = generate_random_entity()
                    other_person = generate_random_person_name()
                    role_singular = query_role_term.rstrip('s').replace("co-founder", "co-founder")
                    role_term_irrelevant = role_singular if role_singular in ["leader", "member", "director", "manager", "founder", "co-founder"] else "member"
                    attempts = 0
                    while other_company == organization and attempts < 5:
                        other_company = generate_random_entity()
                        attempts += 1
                    
                    if other_company != organization:
                        irrelevant_info.append(
                            f"{other_person} is a {role_term_irrelevant} of {other_company}, bringing expertise in {generate_random_concept()}."
                        )
                
                text = create_general_chunk(relevant_info, irrelevant_info)
                score = random.uniform(0.75, 0.95)  # HIGH relevance but with mixed content
            elif category == "mixed" and j > 0:
                # Mix relevant and irrelevant
                if extraction_type == "entities":
                    # Add relevant entity (distribute across chunks)
                    if j <= 2:  # Add entities in first 2 chunks
                        entity_person = generate_random_person_name()
                        relevant_extractions.append(entity_person)
                        # Use role from query if available, otherwise random
                        if 'query_role_term' in locals() and query_role_term:
                            # Map plural to singular: "co-founders" -> "co-founder", "leaders" -> "leader"
                            role_singular = query_role_term.rstrip('s').replace("co-founder", "co-founder")
                            role_term = role_singular if role_singular in ["leader", "member", "director", "manager", "founder", "co-founder"] else "member"
                        else:
                            role_term = random.choice(["leader", "member", "director", "manager", "founder", "co-founder"])
                        relevant_info = {
                            "entities": [{"name": entity_person, "role": f"is a {role_term}", "context": f"of {organization}."}],
                        }
                    else:
                        # Irrelevant info only (but might still add irrelevant co-founders of other companies)
                        relevant_info = {}
                elif extraction_type == "list" and j <= 2:
                    item = generate_random_entity()
                    relevant_extractions.append(item)
                    relevant_info = {
                        "entities": [{"name": item, "role": "is mentioned", "context": f"as part of {concept}."}],
                    }
                else:
                    relevant_info = {
                        "facts": [f"{entity} is associated with {concept}."],
                    }
                    # Don't add facts to entity extractions
                    if extraction_type != "entities":
                        relevant_extractions.append(f"{entity} is associated with {concept}")
                
                irrelevant_info = [
                    f"Another unrelated topic {generate_random_entity()} is discussed here.",
                    f"Some other information about {generate_random_concept()} is provided.",
                    f"Additional context about {generate_random_event()} is included.",
                ]
                
                # For entity extraction queries, add co-founders/leaders of OTHER companies as irrelevant info
                # This tests the model's ability to filter out co-founders from wrong companies
                if extraction_type == "entities" and query_role_term and ("founder" in query_role_term.lower() or "leader" in query_role_term.lower()):
                    # Generate a different company and person for irrelevant co-founder
                    other_company = generate_random_entity()
                    other_person = generate_random_person_name()
                    # Use same role type but for different company
                    role_singular = query_role_term.rstrip('s').replace("co-founder", "co-founder")
                    role_term_irrelevant = role_singular if role_singular in ["leader", "member", "director", "manager", "founder", "co-founder"] else "member"
                    # Only add if other_company is different from target organization
                    # Try a few times to ensure we get a different company
                    attempts = 0
                    while other_company == organization and attempts < 5:
                        other_company = generate_random_entity()
                        attempts += 1
                    
                    if other_company != organization:
                        # Format as "Co-Founder of [OtherCompany]" to match real-world patterns
                        # This explicitly tests entity-specific filtering - model must exclude this
                        if "co-founder" in role_term_irrelevant.lower() or "founder" in role_term_irrelevant.lower():
                            irrelevant_info.append(
                                f"{other_person} is Co-Founder of {other_company}, bringing expertise in {generate_random_concept()}."
                            )
                        else:
                            irrelevant_info.append(
                                f"{other_person} is a {role_term_irrelevant} of {other_company}, bringing expertise in {generate_random_concept()}."
                            )
                
                text = create_general_chunk(relevant_info, irrelevant_info)
                score = random.uniform(0.70, 0.90)  # HIGH but mixed
            else:
                # Primarily relevant
                if extraction_type == "entities":
                    # Add multiple entities across chunks (2-4 entities total)
                    if j < 3:  # Add entities in first 3 chunks
                        entity_person = generate_random_person_name()
                        relevant_extractions.append(entity_person)
                        # Use role from query if available, otherwise random
                        if 'query_role_term' in locals() and query_role_term:
                            # Map plural to singular: "co-founders" -> "co-founder", "leaders" -> "leader"
                            role_singular = query_role_term.rstrip('s').replace("co-founder", "co-founder")
                            role_term = role_singular if role_singular in ["leader", "member", "director", "manager", "founder", "co-founder"] else "member"
                        else:
                            role_term = random.choice(["leader", "member", "director", "manager", "founder", "co-founder"])
                        relevant_info = {
                            "entities": [{"name": entity_person, "role": f"plays a key role as {role_term}", "context": f"in {organization}."}],
                        }
                    else:
                        relevant_info = {}
                elif extraction_type == "entity":
                    # Single entity query - put entity information in first chunk
                    if j == 0:
                        # For "who is" - add person info
                        if "who is" in query.lower():
                            relevant_extractions.append(person)
                            relevant_info = {
                                "entities": [{"name": person, "role": f"is a key figure", "context": f"in {entity}."}],
                                "facts": [f"{person} is associated with {entity} and {concept}."],
                            }
                        else:
                            # For "what is" - add entity description
                            relevant_info = {
                                "facts": [f"{entity} is a {concept} platform that provides solutions for {organization}."],
                                "attributes": {"purpose": f"to enable {concept}", "function": f"supports {organization}"},
                            }
                            relevant_extractions.append(f"{entity} is a {concept} platform")
                    else:
                        relevant_info = {}
                elif extraction_type == "list":
                    items = []
                    for k in range(2):
                        item = generate_random_entity()
                        items.append(item)
                        relevant_extractions.append(item)
                    relevant_info = {
                        "entities": [{"name": item, "role": "is important", "context": f"for {concept}."} for item in items],
                    }
                else:
                    relevant_info = {
                        "facts": [f"{entity} is related to {concept}."],
                        "relationships": [f"{person} is connected to {entity}."],
                    }
                    # Don't add facts to entity extractions
                    if extraction_type != "entities" and extraction_type != "entity":
                        relevant_extractions.append(f"{entity} is related to {concept}")
                
                text = create_general_chunk(relevant_info, [])
                score = random.uniform(0.80, 0.95)  # HIGH relevance
            
            chunks.append({
                "text": text,
                "score": score,
                "file": f"document_{random.randint(1, 10)}.pdf",
                "relevant_info": relevant_info  # Track for extraction
            })
        
        # Generate conversation
        context_parts = []
        for k, chunk in enumerate(chunks, 1):
            score = chunk['score']
            file_name = chunk.get('file', 'document.pdf')
            text = chunk['text']
            text_escaped = text.replace("'", "\\'")
            context_parts.append(f"[Chunk {k}] Score: {score:.3f}, File: {file_name}")
            context_parts.append(f"[{k}] FULL CHUNK TEXT: '{text_escaped}'")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        system_prompt = """You are an AI assistant trained to analyze RAG chunks and extract relevant information.

CRITICAL: Always use the EXACT names, entities, or terms from the user's query. Never hallucinate or substitute different names.

Process:
1. Read each chunk COMPLETELY from start to finish (each chunk has 6-8 sentences)
2. Evaluate relevance using the provided score:
   - HIGH relevance (score ≥0.70): Extract information that directly answers the query
   - MEDIUM relevance (0.50-0.69): May contain related information, use with caution
   - LOW relevance (score <0.50): Likely irrelevant, ignore unless no HIGH relevance chunks available
3. Understand the MEANING in each chunk, not just keywords
4. Extract ONLY information that directly answers or addresses the query
5. IGNORE information that is similar but does NOT answer the query (even if in HIGH relevance chunks)
6. Use the score to determine if extracted information directly answers the query
7. SYNTHESIZE information from multiple chunks into a coherent, natural response
8. Use natural language - avoid simple repetition, create meaningful connections between facts

Return ONLY the final answer in natural, conversational language. Synthesize information rather than just listing facts. Do not include reasoning steps or process details."""

        # Extract based on tracked relevant information
        query_lower = query.lower()
        
        if query_type == "failed":
            assistant_response = "I don't have that information in the provided documents."
        elif extraction_type == "entities" and relevant_extractions:
            # Extract only entity names (person names), filter out non-names
            entity_names = []
            for item in relevant_extractions:
                # Check if it's a person name (two capitalized words)
                if isinstance(item, str) and len(item.split()) == 2:
                    parts = item.split()
                    if parts[0][0].isupper() and parts[1][0].isupper():
                        # Exclude common non-name phrases
                        excluded = ["Material Science", "Public Affairs", "Chief Operating", "Chief Financial", "Chief Marketing"]
                        if item not in excluded and not any(ex in item for ex in excluded):
                            entity_names.append(item)
            
            if entity_names:
                unique_names = sorted(set(entity_names))
                if len(unique_names) == 1:
                    assistant_response = f"{unique_names[0]}."
                else:
                    # Vary response format for more natural language
                    formats = [
                        f"The {query_role_term if 'query_role_term' in locals() and query_role_term else 'entities'} are: {', '.join(unique_names)}.",
                        f"{', '.join(unique_names[:-1])}, and {unique_names[-1]} are the {query_role_term if 'query_role_term' in locals() and query_role_term else 'entities'}.",
                        f"Based on the provided information, the {query_role_term if 'query_role_term' in locals() and query_role_term else 'entities'} include: {', '.join(unique_names)}."
                    ]
                    assistant_response = random.choice(formats)
            else:
                # Fallback to extraction function
                assistant_response = extract_information_from_chunks(query, chunks)
        elif extraction_type == "entity" and relevant_extractions:
            # Single entity query - extract the entity information
            if "who is" in query_lower:
                # Extract person name
                person_names = []
                for item in relevant_extractions:
                    if isinstance(item, str):
                        # Check if it's a person name (two capitalized words)
                        if len(item.split()) == 2:
                            parts = item.split()
                            if parts[0][0].isupper() and parts[1][0].isupper():
                                excluded = ["Material Science", "Public Affairs", "Chief Operating", "Chief Financial", "Chief Marketing"]
                                if item not in excluded and not any(ex in item for ex in excluded):
                                    person_names.append(item)
                
                if person_names:
                    assistant_response = f"{person_names[0]}."
                else:
                    assistant_response = extract_information_from_chunks(query, chunks)
            else:
                # "what is" query - extract description and make it natural
                descriptions = [item for item in relevant_extractions if isinstance(item, str) and len(item) > 20]
                if descriptions:
                    desc = descriptions[0].rstrip('.')
                    # Make it more natural - if it's just "X is a Y platform", expand it
                    if "is a" in desc and "platform" in desc:
                        # Extract entity name from query
                        entity_match = re.search(r'what is (.+?)\?', query_lower)
                        if entity_match:
                            entity_name = entity_match.group(1).strip().title()
                            # Create more natural response
                            if "that provides solutions" in desc:
                                assistant_response = desc
                            else:
                                assistant_response = desc + " It provides solutions for various business needs."
                        else:
                            assistant_response = desc + "."
                    else:
                        assistant_response = desc + "."
                else:
                    assistant_response = extract_information_from_chunks(query, chunks)
        elif extraction_type == "list" and relevant_extractions:
            # For lists, extract clean entity/item names
            clean_items = []
            for item in relevant_extractions:
                if isinstance(item, str):
                    # If it's a full sentence, try to extract the entity name
                    name_match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', item)
                    if name_match:
                        clean_items.append(name_match.group(1))
                    elif len(item.split()) <= 3 and item[0].isupper():
                        clean_items.append(item)
            
            if clean_items:
                unique_items = sorted(set(clean_items))
                assistant_response = f"The items are: {', '.join(unique_items[:5])}."
            else:
                assistant_response = extract_information_from_chunks(query, chunks)
        elif extraction_type in ["analytical", "relationship", "comparison", "reasoning", "process", "causation", "implications", "role", "differences", "similarities"] and relevant_extractions:
            # For complex analytical/comparison queries, generate intelligent synthesized responses
            if extraction_type in ["comparison", "differences", "similarities"]:
                # Comparison queries - synthesize differences/similarities
                if len(relevant_extractions) >= 2:
                    fact1 = relevant_extractions[0].rstrip('.')
                    fact2 = relevant_extractions[1].rstrip('.')
                    # Extract entities from query
                    entity1_match = re.search(r'between (.+?) and', query_lower)
                    entity2_match = re.search(r'and (.+?)[\?\.]', query_lower)
                    if entity1_match and entity2_match:
                        entity1 = entity1_match.group(1).strip().title()
                        entity2 = entity2_match.group(1).strip().title()
                        if "differences" in query_lower:
                            assistant_response = f"{entity1} and {entity2} differ in their focus areas. {fact1} In contrast, {fact2}"
                        elif "similarities" in query_lower:
                            assistant_response = f"{entity1} and {entity2} share common ground. {fact1} Additionally, {fact2}"
                        else:
                            assistant_response = f"{entity1} and {entity2} are related. {fact1} {fact2}"
                    else:
                        assistant_response = f"{fact1}. {fact2}"
                else:
                    assistant_response = relevant_extractions[0].rstrip('.') + "."
            elif extraction_type in ["reasoning", "causation", "implications"]:
                # Analytical queries - provide reasoning
                if relevant_extractions:
                    fact = relevant_extractions[0].rstrip('.')
                    # Make it more natural
                    if "occurred because" in fact:
                        assistant_response = fact.replace("occurred because", "was caused by")
                    elif "led to" in fact:
                        assistant_response = fact
                    else:
                        assistant_response = fact
                    # Add context if available
                    if len(relevant_extractions) > 1:
                        context = relevant_extractions[1].rstrip('.')
                        assistant_response += f" This connection is evident through {context.lower()}."
            elif extraction_type == "relationship":
                # Relationship queries - explain connections
                if relevant_extractions:
                    fact = relevant_extractions[0].rstrip('.')
                    if "connected" in fact or "related" in fact:
                        assistant_response = fact
                    else:
                        assistant_response = fact
            else:
                # Generic analytical
                response_parts = []
                for item in relevant_extractions[:2]:
                    if isinstance(item, str) and len(item) > 20:
                        response_parts.append(item.rstrip('.'))
                if response_parts:
                    assistant_response = ". ".join(response_parts) + "."
                else:
                    assistant_response = extract_information_from_chunks(query, chunks)
        else:
            assistant_response = extract_information_from_chunks(query, chunks)
        
        dataset.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nRAG Chunks:\n{context}"},
                {"role": "assistant", "content": assistant_response}
            ]
        })
    
    return dataset

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("RAG Analysis Dataset Generator - General Framework")
    print("=" * 80)
    
    num_examples = 3000
    if len(sys.argv) > 1:
        num_examples = int(sys.argv[1])
    
    print(f"\nGenerating dataset ({num_examples} examples)...")
    print("  - 3-4 chunks per example")
    print("  - 6-8 sentences per chunk")
    print("  - Focus: General RAG analysis patterns (not entity-specific)")
    print("  - Skills: Read entire chunks, analyze meaning, extract relevant, ignore irrelevant\n")
    
    dataset = generate_dataset(num_examples=num_examples)
    
    # Save dataset
    output_file = "rag_analysis_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved to: {output_file}")
    print(f"   Total examples: {len(dataset)}")
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
