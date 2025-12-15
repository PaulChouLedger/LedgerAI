#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Analysis Dataset Generator - Pattern-Based Approach
========================================================

Generates training data for LLM to learn GENERAL RAG analysis skills:
1. Read entire RAG chunks completely (6-8 sentences each)
2. Analyze and understand meaning in chunks
3. Extract relevant information to query (any type of information)
4. Ignore irrelevant information (especially similar but non-answering content)
5. Use LLM scoring to determine if extracted information directly answers query

Dataset: 3000 varied samples with 3-4 chunks of 6-8 sentences each

PATTERN-BASED DISTRIBUTION (Not Query-Type Based):
- Mixed Content Filtering (20%): Extract relevant, ignore irrelevant similar info
- Multi-Chunk Extraction (20%): Read ALL chunks completely, extract from multiple chunks
- Role/Entity Filtering (15%): Filter by specific role (co-founder vs CEO) or entity type
- Cross-Entity Filtering (15%): Filter by specific entity (Company A vs Company B)
- Synthesis (15%): Combine information from multiple chunks
- Not Found (10%): Recognize missing information
- Edge Cases (5%): Handle edge cases

Key Principle: Each pattern is taught across MULTIPLE query types to ensure
generalizable RAG skills, not memorization of specific query patterns.

Co-Founder Queries: ~5-6% of dataset - used ONLY as examples to teach patterns, not as a category.
Each co-founder example teaches a general principle (role filtering, complete extraction, cross-entity filtering, mixed content).
The same principles apply to ANY query type - co-founder is just one example among many diverse query types.
"""

import json
import random
import re
from typing import List, Dict, Any, Tuple

# ============================================================================
# System Prompt Variations (with core principles)
# ============================================================================

def get_system_prompt_variation(variation_type="full"):
    """
    Generate system prompt variations while retaining core principles:
    1. Read entire chunk
    2. Analyze entire chunk
    3. Understand context in the entire chunk
    4. Extract relevant information to the query
    5. Ignore similar information that does not address query
    """
    
    # Core principles as logical evaluation steps - applied systematically to RAG chunks
    core_principles = """CORE PRINCIPLES (SYSTEMATIC EVALUATION PROCESS):

STEP 1: UNDERSTAND THE QUERY
- Identify what information is being requested
- Note any specific filtering requirements (role, entity, attribute, relationship, etc.)
- Understand the scope and context of what needs to be extracted

STEP 2: READ EACH CHUNK COMPLETELY
- Read the entire chunk from start to finish
- Do not stop at keywords - read for full context and meaning
- Understand the complete context before making extraction decisions

STEP 3: ANALYZE CHUNK MEANING
- Understand the semantic meaning, not just surface-level keywords
- Identify entities, relationships, attributes, and concepts mentioned
- Recognize how information relates to the query

STEP 4: EVALUATE RELEVANCE
- Determine if information directly answers or addresses the query
- Apply query-specific filtering (match role, entity, attribute, etc. as requested)
- Ignore information that is similar but does NOT answer the query

STEP 5: EXTRACT MATCHING INFORMATION
- Extract only information that passes the relevance evaluation
- Apply exact matching - use information exactly as it appears in chunks
- Track all matching items across all chunks

STEP 6: VERIFY COMPLETENESS
- Ensure you have read ALL chunks completely
- Verify you extracted ALL matching items (do not stop after first match)
- Confirm extraction is complete before finalizing response

STEP 7: SYNTHESIZE RESPONSE
- Combine information from all chunks into coherent answer
- Format naturally and directly address the query
- If no matching information found, state "I don't have that information in the provided documents"

CRITICAL: Follow these steps in order for EVERY query. Chunk order does not change the answer - read all chunks before responding."""
    
    if variation_type == "full":
        # Full detailed prompt (20% of examples)
        return f"""You are an AI assistant trained to analyze RAG chunks and extract relevant information.

{core_principles}

ESSENTIAL GUIDELINES:
- NEVER hallucinate - only use information that appears in the provided chunks
- NEVER make up names, entities, or information - if information doesn't exist, say "I don't have that information in the provided documents"
- Use EXACT information from chunks - never substitute or modify names, terms, or entities
- Apply query-specific filtering during Step 4 (evaluate relevance) - match what the query specifically asks for
- Extract ALL matching items - complete Step 6 (verify completeness) ensures nothing is missed
- Relevance scores guide prioritization but do not override the evaluation steps

QUERY TYPE HANDLING (applied during Step 4 - Evaluate Relevance):
- Role/entity queries: Filter by the specific role or entity mentioned in the query
- Comparison queries: Extract information comparing the entities mentioned
- Relationship queries: Extract connection information between entities
- Analytical queries: Extract reasoning, causation, or explanation
- Process queries: Extract step-by-step information
- List queries: Extract all items that match the query criteria

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""
    
    elif variation_type == "medium":
        # Medium prompt with key rules (60% of examples)
        return f"""You are an AI assistant trained to analyze RAG chunks and extract relevant information.

{core_principles}

KEY RULES:
1. NEVER hallucinate - if information doesn't exist, say "I don't have that information in the provided documents"
2. NEVER make up names or entities - ONLY use information that appears in the provided chunks
3. EXACT MATCHING: Use EXACT names, terms, and information from chunks - NEVER substitute or modify
4. FILTERING: Apply the query's specific requirements - exclude information that doesn't match what is asked
5. COMPLETE EXTRACTION: Extract ALL matching items - read ALL chunks completely before responding
6. ORDER-INDEPENDENT: Extract same results regardless of chunk order

RELEVANCE PRIORITIZATION:
- Prioritize HIGH relevance chunks (score ≥0.70) over LOW relevance chunks (score <0.50)
- Extract ONLY information that directly answers the query
- IGNORE similar information that does NOT answer the query

SYSTEMATIC PROCESS:
1. Understand the query - identify what is being asked and any specific filtering requirements
2. Identify RAG chunks provided - count chunks and note relevance scores
3. Read each chunk completely - scan for matching information, track all relevant instances
4. Evaluate relevance - does information answer the query? Apply query-specific filtering
5. Verify complete extraction - ensure you read ALL chunks and extracted ALL matching items
6. Synthesize and format answer - combine information from all chunks into natural response

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""
    
    elif variation_type == "short":
        # Short prompt with minimal rules (20% of examples)
        return f"""You are an AI assistant that analyzes RAG chunks to extract relevant information.

{core_principles}

ESSENTIAL RULES:
- NEVER hallucinate - if information doesn't exist, say "I don't have that information in the provided documents"
- Use EXACT information from chunks - NEVER invent or modify
- Apply query-specific filtering - exclude information that doesn't match what is asked
- Extract ALL matching items - read ALL chunks completely before responding
- ORDER-INDEPENDENT: Extract same results regardless of chunk order

SYSTEMATIC PROCESS:
1. Understand the query and filtering requirements
2. Identify RAG chunks provided
3. Read each chunk completely, tracking relevant information
4. Evaluate: does information answer the query? Apply filtering
5. Verify complete extraction from all chunks
6. Synthesize and format the answer

Return the final answer in natural language. Do not include reasoning steps in the response."""
    
    else:
        # Default to medium
        return get_system_prompt_variation("medium")

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
    Create a rich, professional chunk with relevant and irrelevant information.
    Generates realistic content similar to professional bios, company descriptions, 
    business documents, personal narratives, etc.
    
    relevant_info: Dict with keys like:
        - "entities": List of entities (people, places, things)
        - "concepts": List of concepts/ideas
        - "facts": List of factual statements
        - "relationships": List of relationship statements
        - "attributes": Dict of attribute-value pairs
        - "numbers": List of numbers/metrics
        - "dates": List of dates/events
    """
    irrelevant_info = irrelevant_info or []
    content_parts = []
    
    # Extract entity information for rich bio generation
    entity_name = None
    company_name = None
    role = None
    
    if "entities" in relevant_info:
        for entity in relevant_info["entities"]:
            if isinstance(entity, dict):
                name = entity.get("name", "")
                role_str = entity.get("role", "")
                context = entity.get("context", "")
                
                # Extract company name from context if available
                if "of " in context:
                    company_name = context.split("of ")[-1].rstrip(".")
                
                # Use first entity for rich bio generation
                if not entity_name:
                    entity_name = name
                    role = role_str
                
                # Generate rich professional bio for this entity
                if name and role_str:
                    universities = ["MIT", "Stanford University", "Harvard University", "University of California", "Carnegie Mellon", "University of Washington", "University of Texas", "Columbia University", "Yale University", "Princeton University"]
                    degrees = ["Computer Science", "Electrical Engineering", "Business Administration", "Mathematics", "Economics", "Engineering", "Data Science", "Finance", "Marketing", "Operations"]
                    fields = ["technology", "finance", "operations", "strategy", "product development", "engineering", "data analytics", "artificial intelligence", "blockchain", "cloud computing"]
                    roles_list = ["CEO", "CTO", "CFO", "CMO", "VP of Engineering", "Director of Operations", "Head of Product", "Chief Strategy Officer", "VP of Sales", "Director of Marketing"]
                    companies = [generate_random_entity() for _ in range(3)]
                    
                    # Generate rich bio sentences
                    bio_sentences = [
                        f"{name} is a {random.choice(['renowned', 'visionary', 'strategic', 'experienced', 'accomplished', 'distinguished'])} leader in {random.choice(fields)}, {random.choice(['shaping', 'driving', 'pioneering', 'transforming', 'revolutionizing'])} the future of {random.choice(['enterprise solutions', 'digital transformation', 'business intelligence', 'technology innovation', 'operational excellence'])}.",
                        f"As {role_str} {context if context else f'of {company_name or generate_random_entity()}'}, {name} leads {random.choice(['the execution of', 'the development of', 'strategic initiatives for', 'the implementation of'])} {random.choice(['AI-powered solutions', 'enterprise platforms', 'data analytics systems', 'cloud infrastructure', 'innovative products'])}, {random.choice(['driving efficiency', 'transforming operations', 'enabling growth', 'optimizing performance', 'scaling businesses'])} and {random.choice(['transforming decision-making', 'improving outcomes', 'enhancing capabilities', 'scaling operations', 'delivering value'])}.",
                        f"A graduate of {random.choice(universities)} with degrees in {random.choice(degrees)} and {random.choice(degrees)}, {name}'s expertise spans {random.choice(fields)}, {random.choice(fields)}, and {random.choice(fields)}.",
                        f"Previously, {name} served as {random.choice(roles_list)} at {random.choice(companies)}, where {random.choice(['they revolutionized', 'they pioneered', 'they led', 'they developed', 'they transformed'])} {random.choice(['institutional trading platforms', 'enterprise software solutions', 'data analytics systems', 'cloud infrastructure', 'customer engagement platforms'])}.",
                        f"Before that, {name} was {random.choice(roles_list)} at {random.choice(companies)}, {random.choice(['mastering complex markets', 'building scalable systems', 'leading strategic initiatives', 'developing innovative products', 'optimizing business processes'])}.",
                        f"In addition, as {random.choice(['Founder', 'CEO', 'Advisor', 'Board Member'])} of {random.choice(companies)}, {name} focuses on {random.choice(fields)}, {random.choice(['providing strategic guidance', 'driving innovation', 'expanding market reach', 'building partnerships', 'fostering growth'])}.",
                        f"{name} has built a reputation for {random.choice(['excellence in', 'innovation in', 'leadership in', 'expertise in'])} {random.choice(fields)}, {random.choice(['delivering measurable results', 'transforming organizations', 'driving growth', 'solving complex challenges', 'creating sustainable value'])}.",
                        f"With a proven track record of {random.choice(['optimizing complex systems', 'integrating advanced technologies', 'leading high-performing teams', 'delivering strategic outcomes', 'scaling businesses globally'])}, {name} is {random.choice(['positioning', 'driving', 'enabling', 'transforming', 'leading'])} {company_name or generate_random_entity()} to {random.choice(['achieve market leadership', 'expand globally', 'innovate continuously', 'excel in the industry', 'deliver exceptional results'])}.",
                    ]
                    # Add 2-3 rich bio sentences
                    content_parts.extend(random.sample(bio_sentences, random.randint(2, 3)))
    
    # Add company/industry context if company name is available
    if company_name:
        industries = ["enterprise software", "financial technology", "healthcare technology", "e-commerce", "cloud services", "data analytics", "artificial intelligence", "cybersecurity"]
        services = ["enterprise solutions", "cloud platforms", "data analytics tools", "AI-powered services", "financial services", "consulting services"]
        company_sentences = [
            f"{company_name} operates in the {random.choice(industries)} sector, providing {random.choice(services)} to {random.choice(['enterprise clients', 'mid-market companies', 'startups', 'government agencies', 'global enterprises'])}.",
            f"Founded in {random.randint(2010, 2023)}, {company_name} has established itself as a leader in {random.choice(['enterprise software', 'financial technology', 'data analytics', 'artificial intelligence'])}, serving {random.choice(['Fortune 500 companies', 'emerging businesses', 'global enterprises', 'innovative startups', 'mid-market organizations'])}.",
            f"{company_name}'s mission is to {random.choice(['redefine', 'transform', 'revolutionize', 'enhance', 'optimize'])} {random.choice(['enterprise intelligence', 'business operations', 'data analytics', 'customer experience', 'operational efficiency'])}, {random.choice(['driving efficiency', 'enabling growth', 'creating value', 'fostering innovation', 'delivering excellence'])} through {random.choice(['AI-powered solutions', 'advanced technology', 'strategic partnerships', 'innovative platforms', 'data-driven insights'])}.",
            f"The company's strategy focuses on {random.choice(['innovation', 'customer success', 'market expansion', 'technology leadership', 'operational excellence'])}, {random.choice(['scalability', 'sustainability', 'excellence', 'growth', 'quality'])}, and {random.choice(['partnerships', 'talent development', 'operational efficiency', 'product development', 'customer engagement'])}, positioning it for {random.choice(['long-term success', 'market leadership', 'sustainable growth', 'industry transformation', 'global expansion'])}.",
        ]
        content_parts.extend(random.sample(company_sentences, random.randint(1, 2)))
    
    # Add facts with rich context
    if "facts" in relevant_info:
        for fact in relevant_info["facts"]:
            # Enhance facts with rich professional context
            enhanced_facts = [
                f"Recent analysis reveals that {fact.lower()}",
                f"Industry research indicates that {fact.lower()}",
                f"Strategic assessments show that {fact.lower()}",
                f"Market intelligence demonstrates that {fact.lower()}",
                f"Comprehensive studies confirm that {fact.lower()}",
            ]
            content_parts.append(random.choice(enhanced_facts))
    
    # Add relationships with rich context
    if "relationships" in relevant_info:
        for rel in relevant_info["relationships"]:
            # Enhance relationships with professional context
            enhanced_rels = [
                f"Strategic analysis shows that {rel.lower()}",
                f"Business intelligence indicates that {rel.lower()}",
                f"Market research reveals that {rel.lower()}",
                f"Industry insights demonstrate that {rel.lower()}",
                f"Comprehensive evaluation confirms that {rel.lower()}",
            ]
            content_parts.append(random.choice(enhanced_rels))
    
    # Add concepts with rich professional descriptions
    if "concepts" in relevant_info:
        for concept in relevant_info["concepts"]:
            concept_descriptions = [
                f"{concept.capitalize()} represents a {random.choice(['critical', 'fundamental', 'transformative', 'innovative', 'strategic'])} approach to {random.choice(['addressing', 'solving', 'optimizing', 'enhancing', 'revolutionizing'])} {random.choice(['business challenges', 'operational inefficiencies', 'market opportunities', 'customer needs', 'industry gaps'])}.",
                f"The {concept} framework has been {random.choice(['widely adopted', 'extensively implemented', 'successfully deployed', 'strategically integrated'])} across {random.choice(['enterprise organizations', 'leading companies', 'innovative startups', 'global enterprises'])} to {random.choice(['drive growth', 'improve efficiency', 'enhance performance', 'create value', 'optimize operations'])}.",
                f"{concept.capitalize()} has emerged as a {random.choice(['key', 'essential', 'critical', 'vital'])} component in {random.choice(['modern business strategy', 'digital transformation', 'operational excellence', 'competitive advantage', 'organizational success'])}.",
            ]
            content_parts.append(random.choice(concept_descriptions))
    
    # Add attributes with rich descriptions
    if "attributes" in relevant_info:
        for attr, value in relevant_info["attributes"].items():
            attr_descriptions = [
                f"The {attr} is {value}, {random.choice(['representing', 'indicating', 'demonstrating', 'reflecting'])} {random.choice(['a strategic advantage', 'operational excellence', 'market leadership', 'innovative thinking', 'competitive strength'])}.",
                f"Analysis of the {attr} reveals {value}, which {random.choice(['positions', 'enables', 'facilitates', 'supports'])} {random.choice(['sustainable growth', 'market expansion', 'competitive advantage', 'operational efficiency', 'strategic success'])}.",
            ]
            content_parts.append(random.choice(attr_descriptions))
    
    # Add numbers/metrics with rich business context
    if "numbers" in relevant_info:
        for num_info in relevant_info["numbers"]:
            if isinstance(num_info, dict):
                label = num_info.get('label', 'The value')
                value = num_info.get('value', '')
                metric_descriptions = [
                    f"Financial analysis shows that {label} reached {value}, {random.choice(['representing', 'indicating', 'demonstrating', 'reflecting'])} {random.choice(['strong performance', 'significant growth', 'market leadership', 'operational excellence', 'strategic success'])}.",
                    f"Market data indicates that {label} stands at {value}, {random.choice(['positioning', 'enabling', 'facilitating', 'supporting'])} {random.choice(['future expansion', 'competitive advantage', 'sustainable growth', 'market dominance', 'strategic positioning'])}.",
                ]
                content_parts.append(random.choice(metric_descriptions))
    
    # Add dates/events with rich narrative context
    if "dates" in relevant_info:
        for date_info in relevant_info["dates"]:
            if isinstance(date_info, dict):
                date = date_info.get('date', '')
                event = date_info.get('event', 'something significant occurred')
                event_descriptions = [
                    f"On {date}, {event}, {random.choice(['marking', 'representing', 'signaling', 'demonstrating'])} {random.choice(['a significant milestone', 'strategic progress', 'market evolution', 'organizational growth', 'industry transformation'])}.",
                    f"Historical records show that on {date}, {event}, which {random.choice(['paved the way for', 'enabled', 'facilitated', 'supported'])} {random.choice(['future developments', 'strategic initiatives', 'market expansion', 'operational improvements', 'competitive advantages'])}.",
                ]
                content_parts.append(random.choice(event_descriptions))
    
    # Add rich generic professional content to fill remaining sentences
    num_sentences = random.randint(min_sentences, max_sentences)
    fields = ["technology", "finance", "operations", "strategy", "product development", "engineering", "data analytics", "artificial intelligence", "blockchain", "cloud computing"]
    
    while len(content_parts) < num_sentences:
        generic_content = [
            f"The organization has been at the forefront of {random.choice(fields)}, pioneering {random.choice(['innovative solutions', 'cutting-edge technology', 'advanced platforms', 'transformative systems', 'revolutionary approaches'])} that have {random.choice(['transformed', 'revolutionized', 'enhanced', 'optimized', 'streamlined'])} {random.choice(['business operations', 'customer experiences', 'market dynamics', 'industry standards', 'operational processes'])}.",
            f"With investments in {random.choice(fields)} and {random.choice(fields)}, the company is positioning itself for {random.choice(['sustainable growth', 'market expansion', 'long-term success', 'industry leadership', 'competitive advantage'])}.",
            f"The team's expertise in {random.choice(fields)} has enabled {random.choice(['significant achievements', 'notable successes', 'measurable outcomes', 'strategic wins', 'operational improvements'])}, resulting in {random.choice(['increased market share', 'enhanced capabilities', 'improved performance', 'expanded reach', 'greater efficiency'])}.",
            f"Through {random.choice(['strategic initiatives', 'innovative programs', 'partnerships', 'investments', 'collaborations'])}, the organization has {random.choice(['achieved', 'realized', 'delivered', 'accomplished', 'attained'])} {random.choice(['remarkable growth', 'outstanding results', 'significant progress', 'notable milestones', 'strategic objectives'])}, demonstrating {random.choice(['commitment to excellence', 'strategic vision', 'operational excellence', 'innovative thinking', 'market leadership'])}.",
            f"The company's approach to {random.choice(fields)} combines {random.choice(['advanced technology', 'strategic thinking', 'data-driven insights', 'innovative solutions', 'operational expertise'])} with {random.choice(['operational excellence', 'customer focus', 'market expertise', 'industry knowledge', 'strategic partnerships'])}, creating {random.choice(['competitive advantages', 'unique value propositions', 'sustainable growth', 'market opportunities', 'strategic differentiation'])}.",
            f"Recent developments include {random.choice(['product launches', 'strategic partnerships', 'market expansions', 'technology innovations', 'operational improvements'])}, {random.choice(['platform enhancements', 'service improvements', 'capability expansions', 'infrastructure upgrades', 'process optimizations'])}, and {random.choice(['organizational growth', 'market recognition', 'industry awards', 'customer achievements', 'strategic milestones'])}, signaling {random.choice(['strong momentum', 'positive trajectory', 'continued success', 'future potential', 'sustainable growth'])}.",
            f"Market analysis reveals that the organization's {random.choice(['strategic positioning', 'operational capabilities', 'market presence', 'competitive advantages', 'innovative solutions'])} have {random.choice(['enabled', 'facilitated', 'supported', 'driven', 'accelerated'])} {random.choice(['significant growth', 'market expansion', 'competitive success', 'operational excellence', 'strategic achievements'])}.",
            f"The organization's commitment to {random.choice(['innovation', 'excellence', 'customer success', 'operational efficiency', 'strategic growth'])} is evident in {random.choice(['its track record', 'recent achievements', 'market performance', 'operational metrics', 'strategic initiatives'])}, which has led to {random.choice(['increased market share', 'enhanced reputation', 'competitive advantages', 'operational improvements', 'strategic success'])}.",
        ]
        content_parts.append(random.choice(generic_content))
    
    # Add irrelevant information with rich formatting (but clearly different topic)
    if irrelevant_info:
        num_irrelevant = random.randint(1, 2)
        irrelevant_samples = random.sample(irrelevant_info, min(num_irrelevant, len(irrelevant_info)))
        for irr in irrelevant_samples:
            # Convert irrelevant info to rich format but keep it clearly different
            if "unrelated topic" in irr.lower():
                entity = generate_random_entity()
                industries = ["enterprise software", "financial technology", "healthcare technology", "e-commerce", "cloud services"]
                content_parts.append(f"{entity} operates in the {random.choice(industries)} sector, focusing on {random.choice(['enterprise solutions', 'data analytics', 'cloud platforms', 'AI-powered services'])} and {random.choice(['market expansion', 'customer engagement', 'operational efficiency', 'strategic growth'])}.")
            elif "is Co-Founder of" in irr or "is CEO" in irr or "is CTO" in irr:
                # Keep irrelevant person info but make it rich
                content_parts.append(irr)
        else:
                # Make other irrelevant info sound professional
                content_parts.append(f"Industry analysis indicates that {irr.lower()}")
    
    # Trim to target length and shuffle
    content_parts = content_parts[:max_sentences]
    random.shuffle(content_parts)
    
    return " ".join(content_parts)


# ============================================================================
# Query Type Templates (General, not entity-specific)
# ============================================================================

def generate_query_templates() -> List[Dict[str, Any]]:
    """Generate diverse query templates covering various extraction patterns."""
    return [
        # ========================================================================
        # BUSINESS & COMPANY QUERIES (30%)
        # ========================================================================
        # Company leadership & roles
        {"template": "who are the {role} of {organization}?", "type": "factual_multiple", "extraction_type": "entities", "domain": "business"},
        {"template": "who is the {role} of {organization}?", "type": "factual_single", "extraction_type": "entity", "domain": "business"},
        {"template": "what is {organization}?", "type": "factual_single", "extraction_type": "entity", "domain": "business"},
        {"template": "what does {organization} do?", "type": "factual_single", "extraction_type": "description", "domain": "business"},
        
        # Business operations & strategy
        {"template": "what are the {items} of {organization}?", "type": "factual_multiple", "extraction_type": "list", "domain": "business"},
        {"template": "what is the {attribute} of {organization}?", "type": "factual_single", "extraction_type": "attribute", "domain": "business"},
        {"template": "what are the key {concepts} of {organization}?", "type": "factual_multiple", "extraction_type": "concepts", "domain": "business"},
        {"template": "how does {organization} {process}?", "type": "analytical", "extraction_type": "process", "domain": "business"},
        {"template": "why did {organization} {event}?", "type": "analytical", "extraction_type": "reasoning", "domain": "business"},
        
        # Business relationships
        {"template": "how are {entity1} and {entity2} related?", "type": "relationship", "extraction_type": "relationship", "domain": "business"},
        {"template": "what is the connection between {entity1} and {entity2}?", "type": "relationship", "extraction_type": "relationship", "domain": "business"},
        {"template": "compare {entity1} and {entity2}.", "type": "comparison", "extraction_type": "comparison", "domain": "business"},
        
        # ========================================================================
        # PERSONAL & SELF-REFLECTION QUERIES (30%)
        # ========================================================================
        # Personal goals & achievements
        {"template": "what are my {items}?", "type": "factual_multiple", "extraction_type": "list", "domain": "personal"},
        {"template": "what {items} did I {action}?", "type": "factual_multiple", "extraction_type": "list", "domain": "personal"},
        {"template": "what are my {concepts}?", "type": "factual_multiple", "extraction_type": "concepts", "domain": "personal"},
        {"template": "what {items} are important to me?", "type": "factual_multiple", "extraction_type": "list", "domain": "personal"},
        
        # Personal reflection & analysis
        {"template": "why did I {event}?", "type": "analytical", "extraction_type": "reasoning", "domain": "personal"},
        {"template": "what caused me to {outcome}?", "type": "analytical", "extraction_type": "causation", "domain": "personal"},
        {"template": "how did I {process}?", "type": "analytical", "extraction_type": "process", "domain": "personal"},
        {"template": "what are the implications of my {event}?", "type": "analytical", "extraction_type": "implications", "domain": "personal"},
        
        # Personal timeline & events
        {"template": "when did I {event}?", "type": "factual_single", "extraction_type": "date", "domain": "personal"},
        {"template": "what happened when I {event}?", "type": "factual_single", "extraction_type": "event", "domain": "personal"},
        {"template": "what {items} happened in {timeframe}?", "type": "factual_multiple", "extraction_type": "list", "domain": "personal"},
        
        # Personal relationships & connections
        {"template": "how are {entity1} and {entity2} related in my life?", "type": "relationship", "extraction_type": "relationship", "domain": "personal"},
        {"template": "what is the connection between my {entity1} and {entity2}?", "type": "relationship", "extraction_type": "relationship", "domain": "personal"},
        {"template": "compare my {entity1} and {entity2}.", "type": "comparison", "extraction_type": "comparison", "domain": "personal"},
        
        # Personal attributes & characteristics
        {"template": "what are my {attributes}?", "type": "factual_multiple", "extraction_type": "attributes", "domain": "personal"},
        {"template": "describe my {entity}.", "type": "attribute", "extraction_type": "description", "domain": "personal"},
        {"template": "what {properties} do I have?", "type": "attribute", "extraction_type": "properties", "domain": "personal"},
        
        # Personal development & growth
        {"template": "what {items} have I learned?", "type": "factual_multiple", "extraction_type": "list", "domain": "personal"},
        {"template": "how have I {process}?", "type": "analytical", "extraction_type": "process", "domain": "personal"},
        {"template": "what {items} helped me {outcome}?", "type": "analytical", "extraction_type": "reasoning", "domain": "personal"},
        
        # ========================================================================
        # GENERAL KNOWLEDGE & INFORMATION QUERIES (25%)
        # ========================================================================
        # General factual queries
        {"template": "who is {entity}?", "type": "factual_single", "extraction_type": "entity", "domain": "general"},
        {"template": "what is {entity}?", "type": "factual_single", "extraction_type": "entity", "domain": "general"},
        {"template": "where is {location}?", "type": "factual_single", "extraction_type": "location", "domain": "general"},
        {"template": "when did {event} happen?", "type": "factual_single", "extraction_type": "date", "domain": "general"},
        {"template": "what are the {items} in {context}?", "type": "factual_multiple", "extraction_type": "list", "domain": "general"},
        {"template": "list the {items} that {condition}.", "type": "factual_multiple", "extraction_type": "list", "domain": "general"},
        {"template": "what are the key {concepts} related to {topic}?", "type": "factual_multiple", "extraction_type": "concepts", "domain": "general"},
        
        # General analytical queries
        {"template": "why did {event} occur?", "type": "analytical", "extraction_type": "reasoning", "domain": "general"},
        {"template": "how does {process} work?", "type": "analytical", "extraction_type": "process", "domain": "general"},
        {"template": "what caused {outcome}?", "type": "analytical", "extraction_type": "causation", "domain": "general"},
        {"template": "what are the implications of {event}?", "type": "analytical", "extraction_type": "implications", "domain": "general"},
        
        # General relationship queries
        {"template": "what role does {entity} play in {context}?", "type": "relationship", "extraction_type": "role", "domain": "general"},
        {"template": "what are the differences between {entity1} and {entity2}?", "type": "comparison", "extraction_type": "differences", "domain": "general"},
        {"template": "what are the similarities between {entity1} and {entity2}?", "type": "comparison", "extraction_type": "similarities", "domain": "general"},
        
        # General attribute queries
        {"template": "what are the characteristics of {entity}?", "type": "attribute", "extraction_type": "attributes", "domain": "general"},
        {"template": "what properties does {entity} have?", "type": "attribute", "extraction_type": "properties", "domain": "general"},
        {"template": "describe {entity}.", "type": "attribute", "extraction_type": "description", "domain": "general"},
        
        # ========================================================================
        # FAILED QUERIES (15% - information not available)
        # ========================================================================
        {"template": "what is the {missing_info} of {entity}?", "type": "failed", "extraction_type": None, "domain": "general"},
        {"template": "who is the {missing_role} of {entity}?", "type": "failed", "extraction_type": None, "domain": "general"},
        {"template": "what are my {missing_items}?", "type": "failed", "extraction_type": None, "domain": "personal"},
        {"template": "why did I {missing_event}?", "type": "failed", "extraction_type": None, "domain": "personal"},
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

def generate_random_concept(domain: str = "general") -> str:
    """Generate random concept based on domain."""
    if domain == "personal":
        concepts = [
            "personal growth", "self-improvement", "life goals", "career development",
            "relationships", "health and wellness", "hobbies", "interests", "values",
            "achievements", "challenges", "learning", "creativity", "mindfulness",
            "work-life balance", "time management", "communication skills", "leadership",
            "financial planning", "travel experiences", "family", "friendships",
            "education", "skills development", "passions", "aspirations"
        ]
    elif domain == "business":
        concepts = [
            "machine learning", "artificial intelligence", "data analytics", "cloud computing",
            "blockchain technology", "quantum computing", "edge computing", "distributed systems",
            "microservices architecture", "API design", "user experience", "product development",
            "market analysis", "strategic planning", "financial modeling", "risk assessment",
            "project management", "agile methodology", "devops practices", "cybersecurity",
            "digital transformation", "innovation strategy", "customer engagement", "supply chain",
            "sustainability", "renewable energy", "biotechnology", "nanotechnology"
        ]
    else:  # general
        concepts = [
            "machine learning", "artificial intelligence", "data analytics", "cloud computing",
            "blockchain technology", "quantum computing", "edge computing", "distributed systems",
            "microservices architecture", "API design", "user experience", "product development",
            "market analysis", "strategic planning", "financial modeling", "risk assessment",
            "project management", "agile methodology", "devops practices", "cybersecurity",
            "digital transformation", "innovation strategy", "customer engagement", "supply chain",
            "sustainability", "renewable energy", "biotechnology", "nanotechnology",
            "personal growth", "self-improvement", "life goals", "career development",
            "relationships", "health and wellness", "hobbies", "interests", "values"
        ]
    return random.choice(concepts)

def generate_random_event(domain: str = "general") -> str:
    """Generate random event based on domain."""
    if domain == "personal":
        events = [
            "decided to change careers", "moved to a new city", "started a new hobby",
            "met someone important", "completed a major project", "faced a challenge",
            "achieved a goal", "learned something new", "traveled somewhere",
            "made a decision", "overcame an obstacle", "started a new chapter",
            "reflected on my life", "set new priorities", "grew as a person"
        ]
    elif domain == "business":
        events = [
            "the launch", "the merger", "the acquisition", "the partnership", "the expansion",
            "the innovation", "the breakthrough", "the discovery", "the implementation",
            "the deployment", "the release", "the announcement", "the completion",
            "the transition", "the transformation", "the integration", "the migration"
        ]
    else:  # general
        events = [
            "the launch", "the merger", "the acquisition", "the partnership", "the expansion",
            "the innovation", "the breakthrough", "the discovery", "the implementation",
            "the deployment", "the release", "the announcement", "the completion",
            "the transition", "the transformation", "the integration", "the migration",
            "decided to change", "moved", "started", "met", "completed", "faced",
            "achieved", "learned", "traveled", "made", "overcame", "started"
        ]
    return random.choice(events)

def generate_random_personal_item() -> str:
    """Generate random personal item (goals, skills, interests, etc.)."""
    items = [
        "goals", "achievements", "skills", "interests", "hobbies", "values",
        "strengths", "weaknesses", "experiences", "lessons", "insights",
        "relationships", "connections", "priorities", "aspirations", "dreams",
        "challenges", "obstacles", "opportunities", "decisions", "reflections",
        "learnings", "growth areas", "passions", "talents", "accomplishments"
    ]
    return random.choice(items)

def generate_random_personal_action() -> str:
    """Generate random personal action verb."""
    actions = [
        "achieve", "learn", "grow", "develop", "improve", "overcome", "face",
        "decide", "choose", "pursue", "explore", "discover", "create", "build",
        "change", "transform", "adapt", "evolve", "succeed", "fail", "try"
    ]
    return random.choice(actions)

def generate_random_timeframe() -> str:
    """Generate random timeframe for personal queries."""
    timeframes = [
        "this year", "last year", "this month", "last month", "recently",
        "in the past", "over time", "during that period", "at that time",
        "when I was younger", "in my career", "in my life", "so far"
    ]
    return random.choice(timeframes)

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
        # Check if this is a list query (features, items, etc.) vs entity query (co-founders, leaders)
        is_list_query = any(word in query_lower for word in ["features", "items", "components", "elements", "aspects", "benefits", "advantages"])
        
        if is_list_query:
            # List extraction - extract all items mentioned (features, benefits, components, etc.)
            list_items = []
            filler_phrases_list = [
                "the discussion encompasses",
                "this information is part of",
                "additional details provide",
                "further exploration reveals",
                "understanding this requires",
                "the topic involves",
                "the context includes"
            ]
            
            # Search across ALL high relevance chunks for list items
            # This ensures we don't miss items spread across multiple chunks
            for chunk in high_relevance:
                text = chunk['text']
                sentences = text.split('. ')
                for sentence in sentences:
                    sentence_lower = sentence.lower().strip()
                    # Skip filler sentences
                    if any(filler in sentence_lower for filler in filler_phrases_list):
                        continue
                    
                    # Look for list indicators - expanded list
                    list_indicators = ["includes", "offers", "provides", "features", "supports", "enables", "has", "contains", 
                                      "consists of", "comprises", "incorporates", "offers features", "provides features"]
                    if any(indicator in sentence_lower for indicator in list_indicators):
                        # Extract items from comma-separated or "and" separated lists
                        # Use original sentence (not lowercase) to preserve capitalization
                        sentence_original = sentence.strip()
                        
                        # Find the part after the list indicator
                        for indicator in list_indicators:
                            if indicator in sentence_lower:
                                # Find indicator in original sentence (case-insensitive)
                                indicator_pos = sentence_lower.find(indicator)
                                if indicator_pos != -1:
                                    # Get text after indicator from original sentence
                                    list_text = sentence_original[indicator_pos + len(indicator):].strip()
                                    # Remove leading "the", "a", "an"
                                    list_text = re.sub(r'^(the|a|an)\s+', '', list_text, flags=re.IGNORECASE)
                                    # Remove trailing period and other punctuation
                                    list_text = list_text.rstrip('.,;')
                                    
                                    # Split by comma and "and"
                                    # Handle "X, Y, and Z" pattern - split on comma, then handle "and"
                                    items_raw = re.split(r',\s*', list_text)
                                    final_items = []
                                    for item_raw in items_raw:
                                        item_raw = item_raw.strip()
                                        # Handle "and" at the end (e.g., "X, Y, and Z")
                                        if ' and ' in item_raw:
                                            and_parts = item_raw.split(' and ', 1)
                                            final_items.append(and_parts[0].strip())
                                            final_items.append(and_parts[1].strip())
                                        else:
                                            final_items.append(item_raw)
                                    
                                    # If no commas, try splitting on "and"
                                    if len(final_items) == 1 and ' and ' in list_text:
                                        final_items = [item.strip() for item in list_text.split(' and ')]
                                    
                                    # Clean and extract items
                                    for item in final_items:
                                        item = item.strip()
                                        # Skip empty items
                                        if not item:
                                            continue
                                        # Remove leading articles again (in case they weren't removed)
                                        item = re.sub(r'^(the|a|an)\s+', '', item, flags=re.IGNORECASE).strip()
                                        # Extract meaningful terms (1-3 words, at least 3 chars)
                                        words = item.split()
                                        if len(words) <= 3 and len(item) >= 3:
                                            # Keep original capitalization
                                            list_items.append(item)
                                    break
                        
                        # Also extract capitalized terms that might be features (like "API", "MFA", etc.)
                        # But only if they appear in context of list indicators
                        capitalized_terms = re.findall(r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\b', sentence_original)
                        # Filter out common non-feature words
                        excluded_caps = ["The", "This", "That", "These", "Those", "Company", "Platform", "System", "Product", "Service"]
                        for term in capitalized_terms:
                            if term not in excluded_caps and len(term.split()) <= 2:
                                # Check if it's a known feature type (API, MFA, etc.) or appears near list indicator
                                if any(word in term.lower() for word in ["api", "mfa", "sso", "sdk", "ui", "ux"]) or \
                                   any(indicator in sentence_lower[:sentence_lower.find(term)+50] for indicator in list_indicators):
                                    if term not in list_items:  # Avoid duplicates
                                        list_items.append(term)
            
            if list_items:
                # Remove duplicates while preserving order
                seen = set()
                unique_items = []
                for item in list_items:
                    item_lower = item.lower()
                    if item_lower not in seen:
                        seen.add(item_lower)
                        unique_items.append(item)
                
                if unique_items:
                    return f"The items are: {', '.join(unique_items[:15])}."  # Limit to 15 items
                else:
                    return "I couldn't find that information in the provided documents."
            else:
                return "I couldn't find that information in the provided documents."
        
        # Entity extraction (co-founders, leaders, etc.)
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
                    
                    # CRITICAL: If query asks for co-founders, explicitly EXCLUDE non-co-founder roles
                    # This prevents including CEO, CTO, CFO, etc. when query asks for co-founders
                    if query_role in ["co-founders", "founders"]:
                        # Exclude these roles when looking for co-founders
                        excluded_roles = ["ceo", "cto", "cfo", "cmo", "chief executive", "chief technology", 
                                         "chief financial", "chief marketing", "president", "vp", "vice president",
                                         "director", "manager", "head of", "lead"]
                        # Check if name is associated with an excluded role in the SAME sentence
                        name_sentence_lower = name_sentence.lower()
                        
                        # Check for excluded roles - must be DIRECTLY associated with the name
                        has_excluded_role = False
                        name_lower = name.lower()
                        name_words = name_lower.split()
                        
                        for excluded in excluded_roles:
                            # Check if excluded role appears in sentence
                            excluded_pos = name_sentence_lower.find(excluded)
                            if excluded_pos != -1:
                                # Check if name appears in same sentence
                                if name_lower in name_sentence_lower:
                                    # CRITICAL: Check if name is DIRECTLY associated with excluded role
                                    # Pattern 1: "Name is CEO" or "Name, CEO" or "CEO Name"
                                    # Pattern 2: "Name, the CEO" or "the CEO, Name"
                                    # Get context around the excluded role
                                    context_start = max(0, excluded_pos - 50)
                                    context_end = min(len(name_sentence_lower), excluded_pos + len(excluded) + 50)
                                    role_context = name_sentence_lower[context_start:context_end]
                                    
                                    # Check if name appears near the excluded role
                                    # Name should be within 30 characters of the role
                                    name_pos_in_context = role_context.find(name_lower)
                                    if name_pos_in_context != -1:
                                        # Check distance between name and role
                                        distance = abs(name_pos_in_context - (excluded_pos - context_start))
                                        if distance < 30:
                                            # Check for direct association patterns
                                            # Pattern: "Name is [role]" or "[role] Name" or "Name, [role]"
                                            name_role_patterns = [
                                                f"{name_lower} is {excluded}",
                                                f"{name_lower}, {excluded}",
                                                f"{excluded} {name_lower}",
                                                f"{name_lower} the {excluded}",
                                                f"the {excluded} {name_lower}",
                                            ]
                                            # Also check for first name only
                                            if len(name_words) >= 2:
                                                first_name = name_words[0]
                                                name_role_patterns.extend([
                                                    f"{first_name} is {excluded}",
                                                    f"{first_name}, {excluded}",
                                                    f"{excluded} {first_name}",
                                                ])
                                            
                                            # Check if any pattern matches in the context
                                            if any(pattern in role_context for pattern in name_role_patterns):
                                                has_excluded_role = True
                                                break
                        
                        # Check if name is explicitly a co-founder in the same sentence
                        is_cofounder_in_sentence = any(var in name_sentence_lower for var in ["co-founder", "cofounder", "founder"])
                        
                        # CRITICAL LOGIC:
                        # 1. If person is explicitly a co-founder in sentence, include them (even if also CEO/CTO)
                        if is_cofounder_in_sentence:
                            role_matches = True  # Include co-founders
                        # 2. If person has excluded role AND is NOT a co-founder, exclude them
                        elif has_excluded_role:
                            role_matches = False  # Exclude non-co-founders
                        # 3. If no explicit role found in sentence, don't match (too ambiguous)
                        else:
                            # Only match if we found role variation in broader context
                            role_matches = any(var in name_context for var in variations)
                else:
                    role_matches = True  # If no specific role, accept any
                
                # CRITICAL: Check if organization matches - must appear WITH the role in same sentence
                org_matches = True
                if query_org:
                    # Normalize organization names (remove spaces, handle variations)
                    # Handle company name variations - remove spaces for matching
                    org_normalized = re.sub(r'\s+', '', query_org.lower())
                    query_org_lower = query_org.lower()
                    # Also try with common suffixes removed
                    org_base = re.sub(r'\s+(inc|corp|llc|ltd|ai|tech|systems|solutions)$', '', query_org_lower, flags=re.IGNORECASE)
                    # Also try removing common suffixes and matching (for "Tech Corp" vs "TechCorp")
                    org_without_ai = re.sub(r'\s*ai\s*$', '', org_base, flags=re.IGNORECASE)
                    org_variations = [query_org_lower, org_normalized, org_base, org_without_ai]
                    
                    # Check if organization appears in the sentence with the name
                    sentence_normalized = re.sub(r'\s+', '', name_sentence)
                    
                    # Must have BOTH role AND organization in same sentence/context
                    # Pattern: "[role] of [org]" or "[org] [role]" or "[name] is [role] of [org]"
                    if role_matches:
                        # Check for explicit pattern: role + of + org
                        role_org_pattern = False
                        for var in (role_variations.get(query_role, [query_role]) if query_role else [""]):
                            if var:
                                # Try all organization variations
                                for org_var in org_variations:
                                    # Pattern 1: "co-founder of [company]" variations
                                    if f"{var} of {org_var}" in name_sentence:
                                        role_org_pattern = True
                                        break
                                    # Pattern 2: "[company] co-founder" (less common)
                                    if f"{org_var} {var}" in name_sentence:
                                        role_org_pattern = True
                                        break
                                    # Pattern 3: Check normalized sentence (no spaces)
                                    if f"{var} of {re.sub(r'\s+', '', org_var)}" in sentence_normalized:
                                        role_org_pattern = True
                                        break
                                    if f"{re.sub(r'\s+', '', org_var)}{var}" in sentence_normalized:
                                        role_org_pattern = True
                                        break
                                if role_org_pattern:
                                    break
                        
                        # CRITICAL: Also check that NO OTHER company is mentioned in the same sentence
                        # This prevents matching "Co-Founder of DataSystems" when query asks for "TechCorp"
                        if role_org_pattern:
                            # Get the original sentence text (not lowercase) to extract company names
                            sentence_original = text[sentence_start:sentence_end]
                            # Extract all company names from the sentence (capitalized multi-word entities)
                            other_companies = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', sentence_original)
                            # Check if any other company appears with the role
                            for other_company in other_companies:
                                other_company_lower = other_company.lower()
                                # Skip if it's the query company (check all variations)
                                other_company_normalized = re.sub(r'\s+', '', other_company_lower)
                                if (other_company_lower == query_org_lower or 
                                    other_company_lower == org_normalized or 
                                    other_company_lower == org_base or
                                    other_company_normalized == org_normalized or
                                    other_company_normalized == org_base):
                                    continue
                                # Check if this other company appears with the role
                                for var in (role_variations.get(query_role, [query_role]) if query_role else [""]):
                                    if var and f"{var} of {other_company_lower}" in name_sentence:
                                        # Another company found with the role - this is wrong company
                                        role_org_pattern = False
                                        break
                                if not role_org_pattern:
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
            # CRITICAL: Return explicit "not found" to prevent hallucination
            # Check if chunks mention the company but no co-founders
            company_mentioned = False
            has_any_roles = False
            if query_org:
                query_org_lower = query_org.lower()
                for chunk in high_relevance:
                    chunk_text_lower = chunk['text'].lower()
                    if query_org_lower in chunk_text_lower:
                        company_mentioned = True
                        # Check if there are any roles mentioned (CEO, CTO, etc.) but NOT co-founders
                        if any(role in chunk_text_lower for role in ["ceo", "cto", "cfo", "cmo", "president", "director"]):
                            # Check if co-founder is NOT mentioned
                            if "co-founder" not in chunk_text_lower and "founder" not in chunk_text_lower:
                                has_any_roles = True
                        break
            
            # Use natural variations to avoid repetitive responses
            if company_mentioned and has_any_roles:
                # Company mentioned but only non-co-founder roles found
                not_found_responses = [
                    "I couldn't find information about co-founders in the provided documents.",
                    "The provided documents don't contain information about co-founders.",
                    "I don't have information about co-founders in the provided documents.",
                ]
            elif company_mentioned:
                not_found_responses = [
                    "I couldn't find information about co-founders in the provided documents.",
                    "The provided documents don't contain information about co-founders.",
                    "I don't have that information in the provided documents.",
                ]
            else:
                not_found_responses = [
                    "I couldn't find that information in the provided documents.",
                    "I don't have that information in the provided documents.",
                    "The provided documents don't contain that information.",
                ]
            return random.choice(not_found_responses)
    
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
        reasoning_sentences = []
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
                # Look for reasoning keywords
                reasoning_keywords = ["because", "due to", "resulted", "led to", "caused", "enabled", 
                                     "facilitated", "driven by", "motivated by", "as a result", "in order to",
                                     "to enable", "to support", "to achieve", "for the purpose"]
                if any(keyword in sentence_lower for keyword in reasoning_keywords):
                    reasoning_sentences.append(sentence.strip())
                # Also look for sentences that explain reasons (even without explicit keywords)
                elif "why" in query_lower and any(word in sentence_lower for word in ["reason", "purpose", "goal", "objective", "aim"]):
                    reasoning_sentences.append(sentence.strip())
        
        if reasoning_sentences:
            # Combine reasoning sentences, prioritizing those with explicit reasoning keywords
            # Remove duplicate sentences
            unique_sentences = []
            seen = set()
            for sent in reasoning_sentences:
                sent_lower = sent.lower()
                if sent_lower not in seen:
                    seen.add(sent_lower)
                    unique_sentences.append(sent)
            
            combined = " ".join(unique_sentences[:3])
            # Ensure proper sentence endings
            if not combined.endswith('.') and not combined.endswith('!') and not combined.endswith('?'):
                combined += "."
            return combined
        
        return "I don't have information to answer that question in the provided documents."
    
    elif any(word in query_lower for word in ["how does", "how do", "how is", "how are"]) and "work" in query_lower:
        # Process query - extract step-by-step process
        process_sentences = []
        filler_phrases_process = [
            "the discussion encompasses",
            "this information is part of",
            "additional details provide",
            "further exploration reveals",
            "understanding this requires",
            "the topic involves",
            "the context includes"
        ]
        
        process_keywords = ["first", "then", "next", "after", "before", "step", "process", "work", "verify", 
                           "generate", "grant", "authenticate", "validate", "check", "confirm"]
        
        for chunk in high_relevance:
            text = chunk['text']
            sentences = text.split('. ')
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                # Skip filler sentences
                if any(filler in sentence_lower for filler in filler_phrases_process):
                    continue
                # Look for process-related sentences
                if any(keyword in sentence_lower for keyword in process_keywords):
                    process_sentences.append(sentence.strip())
                # Also include sentences that explain how something works
                elif "work" in sentence_lower or "process" in sentence_lower:
                    process_sentences.append(sentence.strip())
        
        if process_sentences:
            # Remove duplicates and filler
            unique_process = []
            seen = set()
            for sent in process_sentences:
                sent_lower = sent.lower()
                # Skip if it's a filler sentence
                filler_phrases_process_lower = [p.lower() for p in filler_phrases_process]
                if any(filler in sent_lower for filler in filler_phrases_process_lower):
                    continue
                if sent_lower not in seen:
                    seen.add(sent_lower)
                    unique_process.append(sent)
            
            if unique_process:
                result = " ".join(unique_process[:4])
                if not result.endswith('.'):
                    result += "."
                return result
        
        return "I don't have information to explain how that works in the provided documents."
    
    elif any(word in query_lower for word in ["compare", "difference", "similarity", "relationship", "connection", "how are"]):
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
        
        # Extract entities from query
        entity1_match = re.search(r'compare (.+?) and', query_lower)
        entity2_match = re.search(r'and (.+?)[\?\.]', query_lower)
        if not entity1_match:
            entity1_match = re.search(r'between (.+?) and', query_lower)
        if not entity1_match:
            entity1_match = re.search(r'how are (.+?) and', query_lower)
        if not entity2_match:
            entity2_match = re.search(r'and (.+?)[\?\.]', query_lower)
        
        entity1 = entity1_match.group(1).strip() if entity1_match else None
        entity2 = entity2_match.group(1).strip() if entity2_match else None
        
        # Normalize entity names (remove extra spaces, handle case)
        if entity1:
            entity1 = ' '.join(entity1.split())
        if entity2:
            entity2 = ' '.join(entity2.split())
        
        comparison_sentences = []
        relationship_sentences = []
        
        for chunk in high_relevance:
            text = chunk['text']
            sentences = text.split('. ')
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                # Skip filler sentences
                if any(filler in sentence_lower for filler in filler_phrases_comp):
                    continue
                
                # For comparison queries
                if "compare" in query_lower or "difference" in query_lower or "similarity" in query_lower or "differ" in query_lower:
                    # Look for comparison indicators
                    comparison_keywords = ["while", "whereas", "compared to", "similar to", "different from", 
                                         "versus", "vs", "on the other hand", "in contrast", "however", "but",
                                         "focuses on", "targets", "uses", "emphasizes", "prioritizes"]
                    has_comparison = any(keyword in sentence_lower for keyword in comparison_keywords)
                    
                    # Check if both entities are mentioned
                    if entity1 and entity2:
                        entity1_in_sentence = entity1.lower() in sentence_lower
                        entity2_in_sentence = entity2.lower() in sentence_lower
                        # Both entities must be mentioned AND have comparison keyword
                        if entity1_in_sentence and entity2_in_sentence and has_comparison:
                            comparison_sentences.append(sentence.strip())
                        # Or if sentence has strong comparison indicators
                        elif has_comparison and (entity1_in_sentence or entity2_in_sentence):
                            comparison_sentences.append(sentence.strip())
                    elif has_comparison:
                        # If comparison keyword found, include it
                        comparison_sentences.append(sentence.strip())
                
                # For relationship queries
                elif "relationship" in query_lower or "connection" in query_lower or ("how are" in query_lower and "related" in query_lower):
                    # Look for relationship indicators
                    relationship_keywords = ["related", "connected", "partnership", "collaborate", "collaboration", "alliance",
                                            "work together", "joint", "partner", "partners", "associated", "allied",
                                            "subsidiary", "owns", "owned by", "venture"]
                    has_relationship = any(keyword in sentence_lower for keyword in relationship_keywords)
                    
                    # Check if both entities are mentioned
                    if entity1 and entity2:
                        entity1_in_sentence = entity1.lower() in sentence_lower
                        entity2_in_sentence = entity2.lower() in sentence_lower
                        # Both entities must be mentioned AND have relationship keyword
                        if entity1_in_sentence and entity2_in_sentence and has_relationship:
                            relationship_sentences.append(sentence.strip())
                    elif has_relationship:
                        # If relationship keyword found, include it
                        relationship_sentences.append(sentence.strip())
        
        # Return appropriate response - prioritize sentences with both entities and keywords
        if comparison_sentences:
            # Remove duplicates
            unique_comparisons = []
            seen = set()
            for sent in comparison_sentences:
                sent_lower = sent.lower()
                if sent_lower not in seen:
                    seen.add(sent_lower)
                    unique_comparisons.append(sent)
            result = " ".join(unique_comparisons[:3])
            if not result.endswith('.'):
                result += "."
            return result
        elif relationship_sentences:
            # Remove duplicates
            unique_relationships = []
            seen = set()
            for sent in relationship_sentences:
                sent_lower = sent.lower()
                if sent_lower not in seen:
                    seen.add(sent_lower)
                    unique_relationships.append(sent)
            result = " ".join(unique_relationships[:3])
            if not result.endswith('.'):
                result += "."
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

def generate_dataset(num_examples: int = 5000) -> List[Dict[str, Any]]:
    """
    Generate diverse dataset teaching GENERAL RAG analysis patterns.
    
    CRITICAL PRINCIPLE: Teach core RAG skills (reading, analyzing, extracting, filtering) 
    across diverse query types, NOT memorizing specific query patterns.
    
    The model should learn:
    1. Read entire chunks completely (6-8 sentences each)
    2. Analyze and understand meaning in chunks (not just keywords)
    3. Understand context in entire chunk before extracting
    4. Extract relevant information to query (any type: entities, facts, concepts, lists, etc.)
    5. Ignore similar information that does NOT address query
    
    Pattern-Based Distribution (NOT Query-Type Based):
    Each pattern teaches a GENERAL skill that applies to ANY query type:
    - Mixed Content Filtering: Extract relevant, ignore irrelevant (applies to ANY query)
    - Multi-Chunk Extraction: Read ALL chunks completely (applies to ANY query)
    - Role/Entity Filtering: Filter by specific criteria (applies to ANY query with filtering needs)
    - Cross-Entity Filtering: Filter by specific entity (applies to ANY query with multiple entities)
    - Synthesis: Combine information from multiple chunks (applies to ANY analytical query)
    - Not Found: Recognize missing information (applies to ANY query)
    - Chunk Order: Extract same results regardless of order (applies to ANY query)
    
    Co-Founder Queries: Used ONLY to teach general principles:
    - Complete extraction (extract ALL items, not just first)
    - Role filtering (filter by specific criteria)
    - Cross-company filtering (filter by specific entity)
    - Order-independence (same results regardless of chunk order)
    
    These same principles apply to: features, goals, achievements, products, services, etc.
    """
    dataset = []
    
    query_templates = generate_query_templates()
    
    # PATTERN-BASED DISTRIBUTION (not query-type based)
    # Each pattern is taught across multiple query types
    # Increased failing categories to improve model performance
    # Adjusted percentages to sum to 100% (6000 examples)
    # Updated distribution to address regression (49% pass rate, down from 71%)
    # Priority: Increase failing categories (role_filtering 30%, cofounder_mixed 10%, not_found 40%, comparison 20%)
    target_mixed_content = int(num_examples * 0.15)  # 900 - Mixed content (increased to boost cofounder_mixed)
    target_multi_chunk = int(num_examples * 0.08)    # 480 - Multi-chunk extraction (reduced)
    target_role_filtering = int(num_examples * 0.25)  # 1500 - Role/entity filtering (increased from 20% - 30% pass rate needs more examples)
    target_cross_entity = int(num_examples * 0.20)   # 1200 - Cross-entity filtering (increased from 18% - 60% pass rate needs improvement)
    target_synthesis = int(num_examples * 0.05)      # 300 - Synthesis (reduced to make room)
    target_not_found = int(num_examples * 0.25)     # 1500 - Not found (increased from 20% - 40% pass rate needs more examples)
    target_chunk_order = int(num_examples * 0.04)   # 240 - Chunk order variations (reduced)
    target_relationship = int(num_examples * 0.08)   # 480 - Relationship queries (reduced slightly - 60% pass rate)
    target_comparison = int(num_examples * 0.15)    # 900 - Comparison queries (increased from 8% - 20% pass rate needs significant improvement)
    target_edge_cases = int(num_examples * 0.00)    # 0 - Edge cases (removed to make room for critical patterns)
    
    # Adjust to ensure exact total (handle rounding)
    calculated_total = (target_mixed_content + target_multi_chunk + target_role_filtering + 
                       target_cross_entity + target_synthesis + target_not_found + 
                       target_chunk_order + target_relationship + target_comparison + target_edge_cases)
    adjustment = num_examples - calculated_total
    target_role_filtering += adjustment  # Add/subtract adjustment to largest category
    
    query_batches = []
    
    # Separate queries by domain and type
    business_queries = [q for q in query_templates if q.get("domain") == "business"]
    personal_queries = [q for q in query_templates if q.get("domain") == "personal"]
    general_queries = [q for q in query_templates if q.get("domain") == "general"]
    failed_queries = [q for q in query_templates if q["type"] == "failed"]
    
    # Get role-based templates (for role filtering pattern)
    role_templates = [q for q in business_queries + general_queries if "{role}" in q.get("template", "")]
    
    # PATTERN 1: Mixed Content Filtering (15% = 900 examples)
    # Teach GENERAL skill: Extract relevant info, ignore irrelevant similar info
    # This skill applies to ANY query type - co-founders, features, goals, products, etc.
    # DISTRIBUTE ACROSS DIVERSE QUERY TYPES to teach the general principle, not just co-founders
    
    # Diverse distribution across query types to teach general mixed content filtering:
    # - Co-founder queries: teach filtering co-founders vs non-co-founders
    # - Product/Feature queries: teach filtering relevant products/features vs similar but different ones
    # - List queries: teach extracting relevant items vs similar but irrelevant items
    # - Goal/Achievement queries: teach extracting relevant goals vs similar but different goals
    # - Entity queries: teach extracting relevant entities vs similar but different entities
    
    # Distribution: 5% co-founder (as example of filtering), 95% diverse other types
    # Co-founder is just ONE example of the mixed content filtering pattern - minimal examples to teach the principle
    target_cofounder_mixed = int(target_mixed_content * 0.05)  # 45 co-founder queries (0.75% of total) - minimal to teach the pattern
    target_other_mixed = target_mixed_content - target_cofounder_mixed  # 855 diverse other queries
    
    # Co-founder queries with mixed content - teach filtering co-founders vs non-co-founders
    for i in range(target_cofounder_mixed):
        if role_templates:
            template = role_templates[i % len(role_templates)]
            query_batches.append({
                **template, 
                "category": "complex", 
                "force_cofounder": True,
                "pattern": "mixed_content",
                "requires_mixed_content": True,
                "teaches_principle": "mixed_content_filtering"
            })
    
    # DIVERSE query types with mixed content - distribute across domains and extraction types
    # This teaches the GENERAL skill across many contexts, not just co-founders
    
    # Get diverse templates by extraction type
    list_templates = [q for q in business_queries + personal_queries + general_queries 
                     if q.get("extraction_type") == "list"]
    entity_templates = [q for q in business_queries + general_queries 
                       if q.get("extraction_type") == "entity" and "{role}" not in q.get("template", "")]
    analytical_templates = [q for q in business_queries + general_queries 
                           if q.get("extraction_type") in ["analytical", "reasoning", "causation"]]
    relationship_templates_mixed = [q for q in business_queries + general_queries 
                                  if q.get("extraction_type") == "relationship"]
    
    # Combine all diverse templates
    diverse_mixed_templates = (list_templates + entity_templates + analytical_templates + 
                              relationship_templates_mixed + business_queries + personal_queries + general_queries)
    # Remove duplicates and role-based templates (already covered in co-founder mixed)
    diverse_mixed_templates = [q for q in diverse_mixed_templates 
                              if "{role}" not in q.get("template", "")]
    # Remove duplicates by template string
    seen_templates = set()
    unique_diverse_templates = []
    for q in diverse_mixed_templates:
        template_str = q.get("template", "")
        if template_str and template_str not in seen_templates:
            seen_templates.add(template_str)
            unique_diverse_templates.append(q)
    
    # Distribute other_mixed across diverse query types
    for i in range(target_other_mixed):
        if unique_diverse_templates:
            template = unique_diverse_templates[i % len(unique_diverse_templates)]
        else:
            # Fallback to all non-role templates
            template = (business_queries + personal_queries + general_queries)[i % len(business_queries + personal_queries + general_queries)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "mixed_content",
            "requires_mixed_content": True,
            "teaches_principle": "mixed_content_filtering"  # All teach the same general principle
        })
    
    # PATTERN 2: Multi-Chunk Extraction (20% = 600 examples)
    # Teach: Read ALL chunks completely, extract from multiple chunks
    # Distribute across all query types
    multi_chunk_templates = business_queries + personal_queries + general_queries
    for i in range(target_multi_chunk):
        template = multi_chunk_templates[i % len(multi_chunk_templates)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "multi_chunk",
            "requires_multi_chunk": True
        })
    
    # PATTERN 3: Role/Entity Filtering (25% = 1500 examples)
    # Teach: Filter by specific role (co-founder vs CEO) or entity type
    # Co-founder is just ONE example of role filtering - use diverse roles to teach the general principle
    # Co-founder role filtering: 10% of role filtering (150 examples - 2.5% of total) - minimal to teach the pattern
    # Other role filtering: 90% of role filtering (1350 examples - 22.5% of total) - diverse roles teach general principle
    target_cofounder_role = int(target_role_filtering * 0.10)  # 150 co-founder role filtering - minimal to teach the pattern
    target_other_role = target_role_filtering - target_cofounder_role  # 1350 other role filtering - diverse roles teach general principle
    
    for i in range(target_cofounder_role):
        if role_templates:
            template = role_templates[i % len(role_templates)]
            query_batches.append({
                **template,
                "category": "complex",
                "force_cofounder": True,
                "pattern": "role_filtering",
                "requires_role_filtering": True
            })
    
    for i in range(target_other_role):
        if role_templates:
            template = role_templates[i % len(role_templates)]
            # Use various roles, not just co-founders
            query_batches.append({
                **template,
                "category": "complex",
                "pattern": "role_filtering",
                "requires_role_filtering": True
            })
    
    # PATTERN 4: Cross-Entity Filtering (20% = 1200 examples)
    # Teach: Filter by specific entity (Company A vs Company B, Product X vs Product Y)
    # Co-founder is just ONE example of cross-entity filtering - use diverse entities to teach the general principle
    cross_entity_templates = business_queries + general_queries
    # Co-founder cross-entity: 10% of cross-entity (120 examples - 2% of total) - minimal to teach the pattern
    target_cofounder_cross = int(target_cross_entity * 0.10)  # 120 co-founder cross-company queries - minimal to teach the pattern
    target_other_cross = target_cross_entity - target_cofounder_cross  # 1080 other cross-entity queries - diverse entities teach general principle
    
    # Co-founder cross-company queries
    for i in range(target_cofounder_cross):
        if role_templates:
            template = role_templates[i % len(role_templates)]
            query_batches.append({
                **template,
                "category": "complex",
                "force_cofounder": True,
                "pattern": "cross_entity",
                "requires_cross_entity": True
            })
    
    # Other cross-entity queries
    other_cross_templates = [q for q in cross_entity_templates if "{role}" not in q.get("template", "")]
    for i in range(target_other_cross):
        template = other_cross_templates[i % len(other_cross_templates)] if other_cross_templates else cross_entity_templates[i % len(cross_entity_templates)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "cross_entity",
            "requires_cross_entity": True
        })
    
    # PATTERN 5: Synthesis (12% = 600 examples)
    # Teach: Combine information from multiple chunks into coherent answer
    # Distribute across analytical, relationship, comparison, and process queries
    # Emphasize process and comparison queries which had lower pass rates
    synthesis_templates = [q for q in query_templates 
                          if q.get("type") in ["analytical", "relationship", "comparison", "process"]]
    if not synthesis_templates:
        synthesis_templates = business_queries + personal_queries + general_queries
    
    # Prioritize process and comparison queries (50% of synthesis)
    process_comparison_templates = [q for q in synthesis_templates 
                                   if q.get("type") in ["process", "comparison"]]
    other_synthesis_templates = [q for q in synthesis_templates 
                                if q.get("type") not in ["process", "comparison"]]
    
    target_process_comparison = int(target_synthesis * 0.50)  # 300 process/comparison queries
    target_other_synthesis = target_synthesis - target_process_comparison  # 300 other synthesis queries
    
    for i in range(target_process_comparison):
        if process_comparison_templates:
            template = process_comparison_templates[i % len(process_comparison_templates)]
            query_batches.append({
                **template,
                "category": "complex",
                "pattern": "synthesis",
                "requires_synthesis": True
            })
    
    for i in range(target_other_synthesis):
        template = other_synthesis_templates[i % len(other_synthesis_templates)] if other_synthesis_templates else synthesis_templates[i % len(synthesis_templates)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "synthesis",
            "requires_synthesis": True
        })
    
    # PATTERN 6: Not Found (18% = 900 examples, increased to reduce hallucination)
    # Teach: Recognize when information is not available
    for i in range(target_not_found):
        template = failed_queries[i % len(failed_queries)] if failed_queries else query_templates[i % len(query_templates)]
        query_batches.append({
            **template,
            "category": "failed",
            "pattern": "not_found"
        })
    
    # PATTERN 7: Relationship Queries (10% = 600 examples, NEW - increased from ~5% to address 40% pass rate)
    # Teach: Extract relationship information with explicit keywords (partners, collaborate, connected, alliance, joint venture)
    relationship_templates = [q for q in query_templates if q.get("extraction_type") == "relationship"]
    if not relationship_templates:
        # Fallback: create relationship templates from business and general queries
        relationship_templates = [q for q in business_queries + general_queries if "relationship" in q.get("type", "").lower() or "connection" in q.get("template", "").lower()]
    
    for i in range(target_relationship):
        if relationship_templates:
            template = relationship_templates[i % len(relationship_templates)]
        else:
            template = business_queries[i % len(business_queries)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "relationship",
            "extraction_type": "relationship",
            "requires_relationship_keywords": True  # Ensure relationship keywords are included
        })
    
    # PATTERN 8: Comparison Queries (8% = 480 examples, NEW - increased from ~5% to address 60% pass rate)
    # Teach: Extract comparison information with explicit keywords (while, whereas, versus, in contrast)
    comparison_templates = [q for q in query_templates if q.get("extraction_type") == "comparison" or "compare" in q.get("type", "").lower()]
    if not comparison_templates:
        # Fallback: create comparison templates from business and general queries
        comparison_templates = [q for q in business_queries + general_queries if "compare" in q.get("template", "").lower() or "difference" in q.get("template", "").lower()]
    
    for i in range(target_comparison):
        if comparison_templates:
            template = comparison_templates[i % len(comparison_templates)]
        else:
            template = business_queries[i % len(business_queries)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "comparison",
            "extraction_type": "comparison",
            "requires_comparison_keywords": True  # Ensure comparison keywords are included
        })
    
    # PATTERN 9: Chunk Order Variations (4% = 240 examples)
    # Teach GENERAL skill: Extract same results regardless of chunk order (order-independence)
    # This skill applies to ANY query type - the model must read ALL chunks before responding
    # Co-founder is just ONE example of this GENERAL principle: "read all chunks, extract all matches"
    # This same principle applies to: features lists, goals lists, product lists, etc.
    chunk_order_templates = business_queries + general_queries
    # Co-founder queries: 10% of chunk order variations (24 examples - 0.4% of total) - minimal to teach the pattern
    target_cofounder_chunk_order = int(target_chunk_order * 0.10)  # 24 co-founder chunk order examples - minimal to teach the pattern
    target_other_chunk_order = target_chunk_order - target_cofounder_chunk_order  # 216 other chunk order examples - diverse queries teach general principle
    
    for i in range(target_cofounder_chunk_order):
        if role_templates:
            template = role_templates[i % len(role_templates)]
            query_batches.append({
                **template,
                "category": "complex",
                "force_cofounder": True,
                "pattern": "chunk_order",
                "requires_chunk_order_variation": True,
                "requires_multi_chunk": True  # Must have multiple chunks to vary order
            })
    
    for i in range(target_other_chunk_order):
        template = chunk_order_templates[i % len(chunk_order_templates)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "chunk_order",
            "requires_chunk_order_variation": True,
            "requires_multi_chunk": True  # Must have multiple chunks to vary order
        })
    
    # PATTERN 10: Edge Cases (0% = 0 examples, removed to make room for critical patterns)
    # Teach: Handle edge cases with wide variety:
    # - Very long chunks (1000+ words)
    # - Very short chunks (50 words)
    # - Formatting issues (no punctuation, all caps, etc.)
    # - Mixed chunk sizes in same query
    # - Special characters and unicode
    # - Multiple languages mixed
    # - Tables and structured data
    # - Incomplete sentences
    edge_case_types = [
        "very_long_chunk",      # 25% of edge cases
        "very_short_chunk",     # 25% of edge cases
        "formatting_issues",     # 20% of edge cases
        "mixed_chunk_sizes",    # 15% of edge cases
        "special_characters",    # 10% of edge cases
        "incomplete_sentences", # 5% of edge cases
    ]
    
    edge_case_templates = business_queries + personal_queries + general_queries
    for i in range(target_edge_cases):
        template = edge_case_templates[i % len(edge_case_templates)]
        # Distribute edge case types
        edge_case_type = edge_case_types[i % len(edge_case_types)]
        query_batches.append({
            **template,
            "category": "complex",
            "pattern": "edge_case",
            "is_edge_case": True,
            "edge_case_type": edge_case_type
        })
    
    random.shuffle(query_batches)
    
    for i, query_template in enumerate(query_batches):
        template_str = query_template["template"]
        query_type = query_template["type"]
        category = query_template.get("category", "simple")
        extraction_type = query_template.get("extraction_type", "general")
        
        # Get domain from template
        domain = query_template.get("domain", "general")
        
        # Generate domain-specific entities and concepts
        entity = generate_random_entity()
        person = generate_random_person_name()
        event = generate_random_event(domain)
        concept = generate_random_concept(domain)
        organization = generate_random_entity()
        
        # Fill template with domain-specific terms
        entity2 = generate_random_entity()
        
        # For entity extraction, use roles that will match what we put in chunks
        # Pattern-based approach: Only use co-founders when pattern requires it
        pattern = query_template.get("pattern", "")
        if query_template.get("force_cofounder", False):
            role_term = "co-founders"
        elif pattern == "role_filtering" and extraction_type == "entities":
            # Role filtering pattern: Use diverse roles to teach filtering
            role_weights = {
                "co-founders": 0.33,  # 33% co-founder (for role filtering pattern)
                "founders": 0.10,     # 10% founder
                "leaders": 0.20,      # 20% leader
                "members": 0.15,      # 15% member
                "directors": 0.12,     # 12% director
                "managers": 0.10,     # 10% manager
            }
            # Weighted random choice
            rand = random.random()
            cumulative = 0
            role_term = "leaders"  # default
            for role, weight in role_weights.items():
                cumulative += weight
                if rand <= cumulative:
                    role_term = role
                    break
        else:
            # Default: Use diverse roles, but less emphasis on co-founders
            role_weights = {
                "co-founders": 0.10,  # 10% co-founder (reduced from 40%)
                "founders": 0.05,     # 5% founder
                "leaders": 0.25,      # 25% leader
                "members": 0.20,      # 20% member
                "directors": 0.20,    # 20% director
                "managers": 0.20,     # 20% manager
            }
            # Weighted random choice
            rand = random.random()
            cumulative = 0
            role_term = "leaders"  # default
            for role, weight in role_weights.items():
                cumulative += weight
                if rand <= cumulative:
                    role_term = role
                    break
        
        # Store the role term for use in chunk generation (BEFORE we check should_add_nonfounders)
        query_role_term = role_term
        
        # Domain-specific items and concepts
        if domain == "personal":
            items_term = random.choice(["goals", "achievements", "skills", "interests", "hobbies", "values", 
                                       "strengths", "experiences", "lessons", "insights", "relationships",
                                       "priorities", "aspirations", "challenges", "decisions"])
            concepts_term = random.choice(["personal growth", "self-improvement", "life goals", "career development",
                                          "relationships", "health and wellness", "hobbies", "values"])
            action = generate_random_personal_action()
            timeframe = generate_random_timeframe()
        elif domain == "business":
            items_term = random.choice(["items", "elements", "components", "features", "aspects", "products",
                                       "services", "strategies", "initiatives", "projects"])
            concepts_term = random.choice(["concepts", "ideas", "principles", "strategies", "approaches",
                                          "methodologies", "frameworks", "solutions"])
            action = random.choice(["launch", "implement", "develop", "execute", "manage", "optimize"])
            timeframe = random.choice(["this quarter", "this year", "recently", "in the past"])
        else:  # general
            items_term = random.choice(["items", "elements", "components", "features", "aspects"])
            concepts_term = random.choice(["concepts", "ideas", "principles", "notions", "theories"])
            action = random.choice(["happen", "occur", "take place", "develop"])
            timeframe = random.choice(["recently", "in the past", "over time", "at that time"])
        
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
            attributes=random.choice(["strengths", "weaknesses", "characteristics", "qualities", "traits"]),
            properties=random.choice(["skills", "abilities", "capabilities", "features"]),
            process=random.choice(["this process", "the system", "the mechanism"]),
            outcome=random.choice(["this result", "the outcome", "the effect"]),
            context=random.choice(["this context", "the system", "the organization"]),
            missing_info=random.choice(["revenue", "budget", "size", "capacity"]),
            missing_role=random.choice(["CTO", "CFO", "CMO", "director"]),
            missing_items=random.choice(["goals", "achievements", "skills", "interests"]),
            missing_event=random.choice(["achieve that", "decide that", "learn that"]),
            action=action,
            timeframe=timeframe,
        )
        
        # query_role_term is already set above
        
        chunks = []
        num_chunks = random.randint(3, 4)
        
        # Track what relevant information should be extracted (for proper extraction)
        relevant_extractions = []
        
        # Pattern-based decisions for chunk generation
        pattern = query_template.get("pattern", "")
        requires_mixed_content = query_template.get("requires_mixed_content", False)
        requires_multi_chunk = query_template.get("requires_multi_chunk", False)
        requires_role_filtering = query_template.get("requires_role_filtering", False)
        requires_cross_entity = query_template.get("requires_cross_entity", False)
        requires_synthesis = query_template.get("requires_synthesis", False)
        
        # For role filtering pattern: Always add non-founders when query asks for co-founders
        should_add_nonfounders = False
        should_add_other_company = False
        if 'query_role_term' in locals() and query_role_term and "co-founder" in query_role_term.lower():
            if requires_role_filtering:
                # Role filtering pattern: Always add non-founders (100%)
                should_add_nonfounders = True
            else:
                # Other patterns: 80% chance
                should_add_nonfounders = random.random() < 0.80
            
            if requires_cross_entity:
                # Cross-entity pattern: Always add other companies (100%)
                should_add_other_company = True
            else:
                # Other patterns: 95% chance
                should_add_other_company = random.random() < 0.95
        
        # Track if we've actually added them (to ensure we add in at least one chunk)
        added_nonfounders = False
        added_other_company = False
        
        # Edge case handling
        is_edge_case = query_template.get("is_edge_case", False)
        edge_case_type = query_template.get("edge_case_type", None)
        
        # Generate chunks with relevant and irrelevant information
        # CRITICAL: Initialize text variable to ensure it's always set
        text = None
        for j in range(num_chunks):
            # Reset text for each chunk
            text = None
            if query_type == "failed":
                # Failed query - no relevant information
                # CRITICAL: For co-founder queries, ensure chunks mention company but NO co-founders
                # This explicitly trains model to return "not found" instead of hallucinating
                # Increased examples to address 40% pass rate on not found handling
                relevant_info = {}
                
                # For co-founder queries, mention company but explicitly NO co-founders
                # This tests model's ability to recognize absence of information
                if query_role_term and "co-founder" in query_role_term.lower() and 'organization' in locals():
                    # Mention company but with NO co-founder information
                    # Add non-founder roles to test that model doesn't hallucinate
                    irrelevant_info = [
                        f"{organization} operates in the {generate_random_concept()} sector, focusing on {generate_random_concept()}.",
                        f"The company's strategy focuses on {generate_random_concept()} and {generate_random_concept()}.",
                        f"Information about {generate_random_entity()} is discussed here.",
                        f"Details regarding {generate_random_concept()} are provided.",
                    ]
                    # Add non-co-founder roles to ensure model doesn't confuse them with co-founders
                    non_founder_roles = ["CEO", "CTO", "CFO", "CMO", "VP of Engineering", "Director"]
                    non_founder_role = random.choice(non_founder_roles)
                    non_founder_person = generate_random_person_name()
                    irrelevant_info.append(
                        f"{non_founder_person} is {non_founder_role} of {organization}, leading the company's operations."
                    )
                else:
                    # General failed query - no relevant information
                    irrelevant_info = [
                        f"Information about {generate_random_entity()} is discussed here.",
                        f"Details regarding {generate_random_concept()} are provided.",
                        f"Context about {generate_random_event()} is included.",
                    ]
                
                # If query asks about co-founders, add company mention but NO co-founder info
                if extraction_type == "entities" and query_role_term and "founder" in query_role_term.lower():
                    # Mention the company but explicitly without co-founder information
                    # Vary the format to train robust "not found" detection
                    company_mention_formats = [
                        f"{organization} is a company focused on {generate_random_concept()}. The company has {random.randint(10, 100)} employees.",
                        f"{organization} was founded in {random.randint(2010, 2020)}. The company focuses on {generate_random_concept()}.",
                        f"The company {organization} operates in the {generate_random_concept()} sector. They have offices in multiple locations.",
                    ]
                    company_mention = random.choice(company_mention_formats)
                    irrelevant_info.insert(0, company_mention)
                    # DO NOT add any co-founder information - this trains "not found" response
                    # Sometimes add non-founder roles to test exclusion
                    if random.random() < 0.5:
                        non_founder_roles = ["CEO", "CTO", "CFO"]
                        non_founder_role = random.choice(non_founder_roles)
                        non_founder_person = generate_random_person_name()
                        irrelevant_info.append(
                            f"{non_founder_person} is {non_founder_role} of {organization}."
                        )
                
                text = create_general_chunk({}, irrelevant_info)
                score = random.uniform(0.50, 0.70)  # MEDIUM/LOW relevance
            elif category == "complex":
                # Complex edge cases - ALWAYS include relevant info + irrelevant info for filtering
                # This creates scenarios like cross-company filtering, multi-chunk synthesis, etc.
                # Handle domain-specific content generation
                if domain == "personal" and extraction_type == "list":
                    # Personal list queries (goals, achievements, skills, etc.)
                    # Pattern-based: Multi-chunk pattern distributes across ALL chunks
                    if requires_multi_chunk:
                        max_personal_chunks = num_chunks
                    else:
                        max_personal_chunks = 3
                    
                    if j < max_personal_chunks:
                        # Determine list type from query
                        query_lower = query.lower()
                        list_type = "goals"  # default
                        if "achievement" in query_lower:
                            list_type = "achievements"
                        elif "skill" in query_lower:
                            list_type = "skills"
                        elif "interest" in query_lower:
                            list_type = "interests"
                        elif "hobby" in query_lower:
                            list_type = "hobbies"
                        elif "value" in query_lower:
                            list_type = "values"
                        
                        personal_list_items = {
                            "goals": ["improve my health", "learn a new language", "travel more", "start a business", "write a book"],
                            "achievements": ["completed a marathon", "graduated from university", "got promoted", "learned to play guitar"],
                            "skills": ["communication", "leadership", "problem-solving", "time management", "creativity"],
                            "interests": ["photography", "cooking", "hiking", "reading", "music", "technology"],
                            "hobbies": ["playing guitar", "painting", "gardening", "chess", "running", "yoga"],
                            "values": ["integrity", "family", "growth", "adventure", "creativity", "helping others"]
                        }
                        
                        all_items = personal_list_items.get(list_type, personal_list_items["goals"])
                        num_items = random.randint(2, min(4, len(all_items)))
                        items = random.sample(all_items, num_items)
                        
                        # Distribute items across chunks
                        items_per_chunk = max(1, num_items // max_personal_chunks)
                        chunk_start = j * items_per_chunk
                        chunk_end = min((j + 1) * items_per_chunk, num_items)
                        chunk_items = items[chunk_start:chunk_end]
                        
                        for item in chunk_items:
                            relevant_extractions.append(item)
                        
                        # Format as personal list sentence
                        if chunk_items:
                            if len(chunk_items) == 1:
                                list_text = f"My {list_type} include {chunk_items[0]}."
                            elif len(chunk_items) == 2:
                                list_text = f"My {list_type} include {chunk_items[0]} and {chunk_items[1]}."
                            else:
                                list_text = f"My {list_type} include {', '.join(chunk_items[:-1])}, and {chunk_items[-1]}."
                            
                            relevant_info = {
                                "facts": [list_text],
                            }
                        else:
                            relevant_info = {}
                    else:
                        relevant_info = {}
                elif domain == "personal" and extraction_type in ["reasoning", "causation", "implications"]:
                    # Personal analytical queries - ensure we always have reasoning
                    if requires_synthesis:
                        # Synthesis pattern: distribute across more chunks
                        max_reasoning_chunks = num_chunks
                    else:
                        max_reasoning_chunks = 2
                    
                    if j < max_reasoning_chunks:
                        # Extract the actual event/action from query if possible
                        query_lower = query.lower()
                        if "why did i" in query_lower:
                            # Extract action from "why did I [action]?"
                            action_match = re.search(r'why did i (.+?)\?', query_lower)
                            if action_match:
                                action = action_match.group(1).strip()
                            else:
                                action = event
                        else:
                            action = event
                        
                        personal_reasoning = f"I {action} because of {concept} and {generate_random_concept('personal')}. "
                        personal_reasoning += f"This decision was driven by my {generate_random_personal_item()} and {generate_random_personal_item()}."
                        relevant_info = {
                            "facts": [personal_reasoning],
                        }
                        relevant_extractions.append(personal_reasoning)
                    else:
                        relevant_info = {}
                elif extraction_type == "entities":
                    # Pattern-based: Multi-chunk pattern distributes across ALL chunks
                    # Other patterns: Distribute across first 3 chunks
                    # For co-founder queries, ensure 3-4 co-founders (especially for chunk order variation)
                    requires_chunk_order_variation = query_template.get("requires_chunk_order_variation", False)
                    
                    if query_role_term and "co-founder" in query_role_term.lower():
                        # Co-founder queries: Generate 3-4 co-founders across chunks
                        # For chunk order variation, always generate 4 co-founders
                        num_cofounders = 4 if requires_chunk_order_variation else random.randint(3, 4)
                        cofounders_per_chunk = max(1, num_cofounders // num_chunks)
                        chunk_start = j * cofounders_per_chunk
                        chunk_end = min((j + 1) * cofounders_per_chunk, num_cofounders)
                        
                        if chunk_start < num_cofounders:
                            # Generate co-founders for this chunk range
                            chunk_cofounders = []
                            for k in range(chunk_start, chunk_end):
                                entity_person = generate_random_person_name()
                                relevant_extractions.append(entity_person)
                                
                                # CRITICAL: For role filtering pattern, add non-founders in same chunk to test exclusion
                                if requires_role_filtering:
                                    # Add non-founder roles (CEO, CTO, CFO) that must be excluded
                                    non_founder_roles = ["CEO", "CTO", "CFO", "CMO", "President", "VP of Engineering", "Director of Operations"]
                                    non_founder_role = random.choice(non_founder_roles)
                                    non_founder_person = generate_random_person_name()
                                    # Add non-founder in same chunk (model must exclude)
                                    chunk_cofounders.append({"name": entity_person, "role": "is", "context": f"Co-Founder of {organization}. {non_founder_person} is {non_founder_role} of {organization}."})
                                else:
                                    # Format as "Co-Founder of [Company]" for proper pattern matching
                                    chunk_cofounders.append({"name": entity_person, "role": "is", "context": f"Co-Founder of {organization}."})
                            
                            if chunk_cofounders:
                                relevant_info = {"entities": chunk_cofounders}
                            else:
                                relevant_info = {}
                        else:
                            relevant_info = {}
                    else:
                        # Non-co-founder queries: Use original logic
                        if requires_multi_chunk:
                            # Multi-chunk pattern: Distribute entities across ALL chunks
                            max_entities_chunks = num_chunks
                        else:
                            # Other patterns: First 3 chunks
                            max_entities_chunks = 3
                        
                        if j < max_entities_chunks:  # Add entities across chunks based on pattern
                            entity_person = generate_random_person_name()
                            relevant_extractions.append(entity_person)
                            if 'query_role_term' in locals() and query_role_term:
                                role_singular = query_role_term.rstrip('s').replace("co-founder", "co-founder")
                                role_term = role_singular if role_singular in ["leader", "member", "director", "manager", "founder", "co-founder"] else "member"
                            else:
                                role_term = random.choice(["leader", "member", "director", "manager", "founder", "co-founder"])
                            
                            # CRITICAL: For co-founder queries, ensure 80% have non-founders to exclude (was 70%)
                            # This addresses the 10% pass rate on role filtering
                            if query_role_term and "co-founder" in query_role_term.lower():
                                # If we decided to add non-founders at query level, ALWAYS add them in the SAME chunk where we add the co-founder
                                # This ensures 80% of queries have non-founders (matching should_add_nonfounders decision)
                                # Since we're adding a co-founder in this chunk (j < 3), add non-founders here too if needed
                                # CRITICAL: Always add non-founders in first chunk (j==0) if should_add_nonfounders is True
                                # This guarantees 80% of queries have non-founders
                                will_add_here = should_add_nonfounders and (j == 0 or (not added_nonfounders and j < 3))
                                
                                if will_add_here:
                                    added_nonfounders = True
                                    # Add 1-2 non-founders in same chunk
                                    num_non_founders = random.randint(1, 2)
                                    non_founders = []
                                    non_founder_roles = ["CEO", "CTO", "CFO", "CMO", "President", "VP of Engineering", "Director of Operations"]
                                    for _ in range(num_non_founders):
                                        non_founder_name = generate_random_person_name()
                                        non_founder_role = random.choice(non_founder_roles)
                                        non_founders.append(f"{non_founder_name} is {non_founder_role} of {organization}.")
                                    
                                    # Add co-founder
                                    co_founder_text = f"{entity_person} is Co-Founder of {organization}."
                                    # Combine in same sentence or adjacent sentences
                                    if random.random() < 0.5:
                                        # Same sentence
                                        entity_text = f"{co_founder_text} {' '.join(non_founders)}"
                                    else:
                                        # Adjacent sentences
                                        entity_text = f"{co_founder_text} {' '.join(non_founders)}"
                                    
                                    relevant_info = {
                                        "entities": [{"name": entity_person, "role": "is", "context": entity_text}],
                                    }
                                else:
                                    # No non-founders in this chunk (either not selected, or already added)
                                    relevant_info = {
                                        "entities": [{"name": entity_person, "role": "is", "context": f"Co-Founder of {organization}."}],
                                    }
                            else:
                                # Not a co-founder query, but role_term might be co-founder
                                # Format as "Co-Founder of [Company]" for proper pattern matching (for non-co-founder queries)
                                if "co-founder" in role_term.lower() or "founder" in role_term.lower():
                                    relevant_info = {
                                        "entities": [{"name": entity_person, "role": "is", "context": f"Co-Founder of {organization}."}],
                                    }
                                else:
                                    relevant_info = {
                                        "entities": [{"name": entity_person, "role": f"is a {role_term}", "context": f"of {organization}."}],
                                    }
                        else:
                            relevant_info = {}
                elif extraction_type == "list":
                    # Pattern-based: Multi-chunk pattern distributes across ALL chunks
                    # Other patterns: Distribute across first 3 chunks
                    if requires_multi_chunk:
                        # Multi-chunk pattern: Distribute items across ALL chunks
                        max_list_chunks = num_chunks
                    else:
                        # Other patterns: First 3 chunks
                        max_list_chunks = 3
                    
                    if j < max_list_chunks:
                        # Generate actual list items based on domain
                        # CRITICAL: For list extraction (20% pass rate), ensure ALL items are extracted
                        # Distribute items across chunks to train multi-chunk extraction
                        if domain == "personal":
                        # Personal list items
                        personal_list_items = {
                            "goals": ["improve my health", "learn a new language", "travel more", "start a business", "write a book", "get better at public speaking"],
                            "achievements": ["completed a marathon", "graduated from university", "got promoted", "learned to play guitar", "published an article", "volunteered 100 hours"],
                            "skills": ["communication", "leadership", "problem-solving", "time management", "creativity", "analytical thinking"],
                            "interests": ["photography", "cooking", "hiking", "reading", "music", "technology"],
                            "hobbies": ["playing guitar", "painting", "gardening", "chess", "running", "yoga"],
                            "values": ["integrity", "family", "growth", "adventure", "creativity", "helping others"]
                        }
                        # Determine list type from query
                        list_type = "goals"  # default
                        query_lower = query.lower()
                        for key in personal_list_items.keys():
                            if key in query_lower:
                                list_type = key
                                break
                        
                        # Generate 3-5 items to ensure complete extraction training
                        all_items = personal_list_items.get(list_type, personal_list_items["goals"])
                        num_items = random.randint(3, min(5, len(all_items)))
                        items = random.sample(all_items, num_items)
                        
                        # Distribute items across chunks (j is chunk index)
                        items_per_chunk = max(1, num_items // 3)
                        chunk_start = j * items_per_chunk
                        chunk_end = min((j + 1) * items_per_chunk, num_items)
                        chunk_items = items[chunk_start:chunk_end]
                        
                        for item in chunk_items:
                            relevant_extractions.append(item)
                        
                        # Format as personal list sentence - use explicit list format
                            if chunk_items:
                        if len(chunk_items) == 1:
                            list_text = f"My {list_type} include {chunk_items[0]}."
                        elif len(chunk_items) == 2:
                            list_text = f"My {list_type} include {chunk_items[0]} and {chunk_items[1]}."
                        else:
                            list_text = f"My {list_type} include {', '.join(chunk_items[:-1])}, and {chunk_items[-1]}."
                                
                                relevant_info = {
                                    "facts": [list_text],
                                }
                            else:
                                relevant_info = {}
                    else:
                        # Business/general list items
                        list_item_types = {
                            "features": ["real-time analytics", "secure encryption", "automated reporting", "API integrations", "custom dashboards", "machine learning", "blockchain support", "authentication", "authorization", "logging", "monitoring"],
                            "benefits": ["cost savings", "improved efficiency", "scalability", "reliability", "better performance", "increased productivity"],
                            "components": ["database", "API server", "frontend", "backend", "cache layer", "load balancer"],
                            "advantages": ["24/7 support", "global coverage", "fast deployment", "easy integration", "flexible pricing"]
                        }
                        # Determine list type from query context
                        list_type = "features"  # default
                        query_lower = query.lower()
                        if "benefit" in query_lower:
                            list_type = "benefits"
                        elif "component" in query_lower:
                            list_type = "components"
                        elif "advantage" in query_lower:
                            list_type = "advantages"
                        
                        # Generate 3-5 items to ensure complete extraction training
                        all_items = list_item_types.get(list_type, list_item_types["features"])
                        num_items = random.randint(3, min(5, len(all_items)))
                        items = random.sample(all_items, num_items)
                        
                        # Distribute items across chunks
                        items_per_chunk = max(1, num_items // 3)
                        chunk_start = j * items_per_chunk
                        chunk_end = min((j + 1) * items_per_chunk, num_items)
                        chunk_items = items[chunk_start:chunk_end]
                        
                        for item in chunk_items:
                            relevant_extractions.append(item)
                        
                        # Format as proper list sentence - use explicit list format
                            if chunk_items:  # Only format if we have items
                        list_verb = random.choice(['offers', 'provides', 'includes', 'features'])
                        if len(chunk_items) == 1:
                            list_text = f"{organization} {list_verb} {chunk_items[0]}."
                        elif len(chunk_items) == 2:
                            list_text = f"{organization} {list_verb} {chunk_items[0]} and {chunk_items[1]}."
                        else:
                            list_text = f"{organization} {list_verb} {', '.join(chunk_items[:-1])}, and {chunk_items[-1]}."
                    
                    relevant_info = {
                        "facts": [list_text],
                    }
                            else:
                                # No items for this chunk
                                relevant_info = {}
                    else:
                        # No relevant info for this chunk (j >= max_list_chunks)
                        relevant_info = {}
                elif extraction_type in ["analytical", "relationship", "comparison", "reasoning", "process", "causation", "implications", "role", "differences", "similarities"]:
                    # For analytical/comparison queries, add relationship information
                    if j < 2:
                        if extraction_type in ["comparison", "differences", "similarities"]:
                            # Generate explicit comparison sentences with comparison keywords
                            comparison_templates = [
                                f"{entity} focuses on {concept} while {entity2} emphasizes {generate_random_concept()}.",
                                f"{entity} uses {concept} whereas {entity2} uses {generate_random_concept()}.",
                                f"{entity} targets enterprise clients, in contrast, {entity2} targets small businesses.",
                                f"{entity} offers cloud deployment versus {entity2} which offers on-premise solutions.",
                            ]
                            comparison_sentence = random.choice(comparison_templates)
                            relevant_info = {
                                "facts": [comparison_sentence],
                            }
                            relevant_extractions.append(comparison_sentence)
                        elif extraction_type in ["reasoning", "causation", "implications"]:
                            # Generate explicit reasoning sentences with reasoning keywords
                            reasoning_templates = [
                                f"The {event} occurred because of {concept} and {generate_random_concept()}.",
                                f"The {event} happened due to {concept}, which enabled {generate_random_concept()}.",
                                f"{entity} expanded {generate_random_concept()} as a result of {concept} and market opportunities.",
                                f"The decision was driven by {concept}, competitive pressures, and customer demand.",
                            ]
                            reasoning_sentence = random.choice(reasoning_templates)
                            relevant_info = {
                                "facts": [reasoning_sentence],
                            }
                            relevant_extractions.append(reasoning_sentence)
                        elif extraction_type == "relationship":
                            # Generate explicit relationship sentences with relationship keywords
                            relationship_templates = [
                                f"{entity} and {entity2} are strategic partners collaborating on {concept}.",
                                f"{entity} owns {entity2} as a subsidiary, and they work together on {concept}.",
                                f"{entity} and {entity2} formed an alliance to share {concept} and {generate_random_concept()}.",
                                f"{entity} and {entity2} are connected through a joint venture developing {concept}.",
                            ]
                            relationship_sentence = random.choice(relationship_templates)
                            relevant_info = {
                                "facts": [relationship_sentence],
                            }
                            relevant_extractions.append(relationship_sentence)
                        elif extraction_type == "process":
                            # Generate explicit process sentences with process keywords
                            process_templates = [
                                f"The {concept} works by first {generate_random_concept()}, then {generate_random_concept()}, and finally {generate_random_concept()}.",
                                f"Processing involves validating {generate_random_concept()}, checking {generate_random_concept()}, and sending {generate_random_concept()}.",
                                f"The system processes data by collecting {generate_random_concept()}, cleaning and validating, transforming the format, and storing in the database.",
                            ]
                            process_sentence = random.choice(process_templates)
                            relevant_info = {
                                "facts": [process_sentence],
                            }
                            relevant_extractions.append(process_sentence)
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
                
                # Pattern-based irrelevant info: Mixed content pattern ALWAYS adds irrelevant info
                # Other patterns: Add irrelevant info for filtering scenarios
                if requires_mixed_content or category == "complex":
                    # Mixed content pattern or complex: Always add irrelevant info
                irrelevant_info = [
                    f"Another unrelated topic {generate_random_entity()} is discussed here.",
                    f"Some other information about {generate_random_concept()} is provided.",
                    f"Additional context about {generate_random_event()} is included.",
                ]
                else:
                    # Other patterns: Sometimes add irrelevant info (70% chance)
                    if random.random() < 0.70:
                        irrelevant_info = [
                            f"Another unrelated topic {generate_random_entity()} is discussed here.",
                            f"Some other information about {generate_random_concept()} is provided.",
                        ]
                    else:
                        irrelevant_info = []
                
                # For entity extraction, ALWAYS add co-founders/leaders of OTHER companies
                # CRITICAL: Increase cross-company filtering examples (50% pass rate -> need more)
                if extraction_type == "entities" and query_role_term and ("founder" in query_role_term.lower() or "leader" in query_role_term.lower()):
                    # 90% of the time, add other company co-founders (increased from ~50%)
                    # This addresses the 50% pass rate on cross-company filtering
                    # CRITICAL: For co-founder queries specifically, make it 95% to ensure we hit 90% overall
                    # Use the query-level decision, but ensure we add in at least one chunk
                    if query_role_term and "co-founder" in query_role_term.lower():
                        # If we decided to add other company at query level, ALWAYS add them in at least one chunk
                        # This ensures 95% of queries have other companies (matching should_add_other_company decision)
                        # CRITICAL: Always add in first chunk (j==0) if should_add_other_company is True
                        # This guarantees 95% of queries have other companies
                        will_add_other_here = should_add_other_company and (j == 0 or (not added_other_company and j < num_chunks))
                    else:
                        # For non-co-founder queries, use 90% probability per chunk
                        will_add_other_here = random.random() < 0.90
                    
                    if will_add_other_here:
                        added_other_company = True
                        other_company = generate_random_entity()
                        other_person = generate_random_person_name()
                        role_singular = query_role_term.rstrip('s').replace("co-founder", "co-founder")
                        role_term_irrelevant = role_singular if role_singular in ["leader", "member", "director", "manager", "founder", "co-founder"] else "member"
                        attempts = 0
                        while other_company == organization and attempts < 5:
                            other_company = generate_random_entity()
                            attempts += 1
                        
                        if other_company != organization:
                            # Format as "Co-Founder of [OtherCompany]" to match real-world patterns
                            # This explicitly tests entity-specific filtering - model must exclude this
                            if "co-founder" in role_term_irrelevant.lower() or "founder" in role_term_irrelevant.lower():
                                # Add in same sentence or adjacent sentence to test filtering
                                if random.random() < 0.5:
                                    # Same sentence - harder filtering test
                                    irrelevant_info.append(
                                        f"{other_person} is Co-Founder of {other_company}, bringing expertise in {generate_random_concept()}."
                                    )
                                else:
                                    # Adjacent sentence - still need to filter
                                    irrelevant_info.append(
                                        f"{other_person} is Co-Founder of {other_company}."
                                    )
                        else:
                            irrelevant_info.append(
                                f"{other_person} is a {role_term_irrelevant} of {other_company}, bringing expertise in {generate_random_concept()}."
                            )
                    
                    # CRITICAL: Add role filtering test - include CEO/CTO/CFO of SAME company (must exclude)
                    # This tests that model excludes non-co-founder roles when query asks for co-founders
                    # CRITICAL: For co-founder queries, ALWAYS add non-founders in at least one chunk
                    # This ensures 80%+ of co-founder queries have non-founders to exclude
                    if query_role_term and "co-founder" in query_role_term.lower():
                        # For co-founder queries, add non-founders in 80% of chunks (not just 80% of queries)
                        # This ensures most queries have non-founders
                        if random.random() < 0.80:  # 80% chance per chunk
                            non_founder_roles = ["CEO", "CTO", "CFO", "CMO", "Chief Executive Officer", "Chief Technology Officer", 
                                                "Chief Financial Officer", "President", "VP of Engineering", "Director of Operations"]
                            non_founder_role = random.choice(non_founder_roles)
                            non_founder_person = generate_random_person_name()
                            # Add as irrelevant - model must exclude this person
                            # Sometimes in same sentence, sometimes adjacent
                            if random.random() < 0.5:
                                irrelevant_info.append(
                                    f"{non_founder_person} is {non_founder_role} of {organization}, leading the company's operations."
                                )
                            else:
                                irrelevant_info.append(
                                    f"{non_founder_person} is {non_founder_role} of {organization}."
                                )
                
                # Handle edge cases
                if is_edge_case and edge_case_type:
                text = create_general_chunk(relevant_info, irrelevant_info)
                    # Apply edge case modifications
                    if edge_case_type == "very_long_chunk":
                        # Expand chunk to 15-20 sentences (very long)
                        extra_sentences = [
                            f"Additional context about {generate_random_concept()} is provided here.",
                            f"Further details regarding {generate_random_event()} are discussed.",
                            f"More information about {generate_random_entity()} is included.",
                            f"Extended discussion of {generate_random_concept()} continues.",
                            f"Additional perspectives on {generate_random_event()} are explored.",
                            f"Further analysis of {generate_random_entity()} is presented.",
                            f"More details about {generate_random_concept()} are elaborated.",
                            f"Extended context about {generate_random_event()} is provided.",
                            f"Comprehensive coverage of {generate_random_concept()} is included.",
                            f"In-depth examination of {generate_random_event()} is provided.",
                        ]
                        # Add 7-12 more sentences
                        num_extra = random.randint(7, 12)
                        text += " " + " ".join(random.sample(extra_sentences, min(num_extra, len(extra_sentences))))
                    elif edge_case_type == "very_short_chunk":
                        # Reduce to 2-3 sentences (very short)
                        sentences = text.split('. ')
                        text = '. '.join(sentences[:random.randint(2, 3)]) + '.'
                    elif edge_case_type == "formatting_issues":
                        # Add formatting issues: no punctuation, all caps, etc.
                        if random.random() < 0.3:
                            # Remove some punctuation
                            text = text.replace('. ', ' ').replace(', ', ' ')
                        elif random.random() < 0.5:
                            # Mix case randomly
                            words = text.split()
                            text = ' '.join([w.upper() if random.random() < 0.2 else w for w in words])
                    elif edge_case_type == "special_characters":
                        # Add special characters and unicode
                        special_chars = ["©", "®", "™", "€", "£", "¥", "—", "…", "•"]
                        if random.random() < 0.5:
                            text = text.replace('.', random.choice(special_chars), 1)
                    elif edge_case_type == "incomplete_sentences":
                        # Remove final punctuation from some sentences
                        if random.random() < 0.4:
                            text = text.rstrip('.')
                else:
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
                        # Format as "Co-Founder of [Company]" for proper pattern matching
                        if "co-founder" in role_term.lower() or "founder" in role_term.lower():
                            relevant_info = {
                                "entities": [{"name": entity_person, "role": "is", "context": f"Co-Founder of {organization}."}],
                            }
                        else:
                            relevant_info = {
                                "entities": [{"name": entity_person, "role": f"is a {role_term}", "context": f"of {organization}."}],
                            }
                    else:
                        # Irrelevant info only (but might still add irrelevant co-founders of other companies)
                        relevant_info = {}
                elif extraction_type == "list" and j <= 2:
                    # Generate actual list items (features, benefits, components) not entity names
                    list_item_types = {
                        "features": ["real-time analytics", "secure encryption", "automated reporting", "API integrations", "custom dashboards"],
                        "benefits": ["cost savings", "improved efficiency", "scalability", "reliability"],
                        "components": ["database", "API server", "frontend", "backend"],
                        "advantages": ["24/7 support", "global coverage", "fast deployment"]
                    }
                    list_type = "features"  # default
                    if "benefit" in query.lower():
                        list_type = "benefits"
                    elif "component" in query.lower():
                        list_type = "components"
                    elif "advantage" in query.lower():
                        list_type = "advantages"
                    
                    items = random.sample(list_item_types.get(list_type, list_item_types["features"]), min(2, len(list_item_types.get(list_type, []))))
                    for item in items:
                        relevant_extractions.append(item)
                    
                    if len(items) == 1:
                        list_text = f"{organization} {random.choice(['offers', 'provides', 'includes'])} {items[0]}."
                    else:
                        list_text = f"{organization} {random.choice(['offers', 'provides', 'includes'])} {items[0]} and {items[1]}."
                    
                    relevant_info = {
                        "facts": [list_text],
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
                    
                    # CRITICAL: Add role filtering test - include CEO/CTO/CFO of SAME company (must exclude)
                    # This tests that model excludes non-co-founder roles when query asks for co-founders
                    if query_role_term and "co-founder" in query_role_term.lower():
                        # 70% chance to add non-co-founder roles in same chunk (for role filtering test)
                        # Increased to 70% to ensure enough examples for role filtering (was 50%)
                        if random.random() < 0.70:
                            non_founder_roles = ["CEO", "CTO", "CFO", "CMO", "Chief Executive Officer", "Chief Technology Officer"]
                            non_founder_role = random.choice(non_founder_roles)
                            non_founder_person = generate_random_person_name()
                            # Add as irrelevant - model must exclude this person
                            irrelevant_info.append(
                                f"{non_founder_person} is {non_founder_role} of {organization}, leading the company's operations."
                            )
                
                # Handle edge cases for mixed category
                if is_edge_case and edge_case_type == "mixed_chunk_sizes":
                    # Mix chunk sizes: some long, some short
                    if j == 0:
                        # First chunk: very long
                text = create_general_chunk(relevant_info, irrelevant_info)
                        extra_sentences = [
                            f"Extended discussion of {generate_random_concept()} continues.",
                            f"Additional perspectives on {generate_random_event()} are explored.",
                            f"Further analysis of {generate_random_entity()} is presented.",
                        ]
                        text += " " + " ".join(extra_sentences * 3)
                    elif j == num_chunks - 1:
                        # Last chunk: very short
                        text = create_general_chunk(relevant_info, irrelevant_info)
                        sentences = text.split('. ')
                        text = '. '.join(sentences[:2]) + '.'
                    else:
                        text = create_general_chunk(relevant_info, irrelevant_info)
                else:
                    text = create_general_chunk(relevant_info, irrelevant_info)
                
                score = random.uniform(0.70, 0.90)  # HIGH but mixed
            else:
                # Primarily relevant
                if extraction_type == "entities":
                    # Add multiple entities across chunks
                    # For co-founder queries, ensure 3-4 co-founders (especially for chunk order variation)
                    requires_chunk_order_variation = query_template.get("requires_chunk_order_variation", False)
                    if query_role_term and "co-founder" in query_role_term.lower():
                        # Co-founder queries: Generate 3-4 co-founders across chunks
                        # For chunk order variation, always generate 4 co-founders
                        num_cofounders = 4 if requires_chunk_order_variation else random.randint(3, 4)
                        cofounders_per_chunk = max(1, num_cofounders // num_chunks)
                        chunk_start = j * cofounders_per_chunk
                        chunk_end = min((j + 1) * cofounders_per_chunk, num_cofounders)
                        
                        if chunk_start < num_cofounders:
                            # Generate co-founders for this chunk range
                            chunk_cofounders = []
                            for k in range(chunk_start, chunk_end):
                                entity_person = generate_random_person_name()
                                relevant_extractions.append(entity_person)
                                chunk_cofounders.append({"name": entity_person, "role": "is Co-Founder", "context": f"of {organization}."})
                            
                            if chunk_cofounders:
                                relevant_info = {"entities": chunk_cofounders}
                            else:
                                relevant_info = {}
                        else:
                            relevant_info = {}
                    else:
                        # Non-co-founder queries: Add 2-4 entities across chunks
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
            
            # CRITICAL: Ensure text is always set and never empty before appending to chunks
            if text is None:
                # Text was never set - this should never happen, but create a default chunk
                text = f"This document contains relevant information about {organization if 'organization' in locals() else 'the topic'}."
                score = 0.75  # Default score if not set
            elif not text or len(text.strip()) == 0:
                # Fallback: Generate a default chunk text if empty
                text = f"This document contains relevant information about {organization if 'organization' in locals() else 'the topic'}."
            
            # CRITICAL: Ensure text is a string and not empty
            if not isinstance(text, str):
                text = str(text) if text is not None else "This document contains relevant information about the query topic."
            
            # CRITICAL: Ensure chunks are always appended with valid text
            chunks.append({
                "text": text,
                "score": score if 'score' in locals() else 0.75,
                "file": f"document_{random.randint(1, 10)}.pdf",
                "relevant_info": relevant_info if 'relevant_info' in locals() else {}  # Track for extraction
            })
        
        # Chunk order variation: Shuffle chunks for order-independence training
        requires_chunk_order_variation = query_template.get("requires_chunk_order_variation", False)
        if requires_chunk_order_variation and len(chunks) > 1:
            # Shuffle chunks to teach order-independence
            # Keep same chunks, just change order
            random.shuffle(chunks)
            # Note: This ensures model learns to extract same results regardless of chunk order
        
        # CRITICAL: Ensure chunks list is never empty before formatting
        # This is a safety check to prevent empty chunks from causing formatting issues
        if not chunks or len(chunks) == 0:
            # Chunks list is empty - create at least one default chunk
            chunks = [{
                "text": "This document contains relevant information about the query topic.",
                "score": 0.75,
                "file": "document_1.pdf",
                "relevant_info": {}
            }]
        
        # Generate conversation
        # CRITICAL: Ensure chunks list is never empty and always properly formatted
        # Filter out invalid chunks first
        valid_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict) and 'text' in chunk and chunk['text'] and len(chunk['text'].strip()) > 0:
                valid_chunks.append(chunk)
        
        # If no valid chunks, create a default chunk
        if not valid_chunks:
            valid_chunks = [{
                "text": "This document contains relevant information about the query topic.",
                "score": 0.75,
                "file": "document_1.pdf",
                "relevant_info": {}
            }]
        
        # Now format all valid chunks
        context_parts = []
        for k, chunk in enumerate(valid_chunks, 1):
            score = chunk.get('score', 0.75)
            file_name = chunk.get('file', f'document_{random.randint(1, 10)}.pdf')
            text = chunk['text']
            
            # Text should never be empty at this point (we filtered above)
            text_escaped = text.replace("'", "\\'")
            context_parts.append(f"[Chunk {k}] Score: {score:.3f}, File: {file_name}")
            context_parts.append(f"[{k}] FULL CHUNK TEXT: '{text_escaped}'")
            context_parts.append("")
        
        # Double-check: context_parts should never be empty now
        if not context_parts:
            # Final fallback: Create a default chunk if something went wrong
            context_parts = [
                "[Chunk 1] Score: 0.750, File: document_1.pdf",
                "[1] FULL CHUNK TEXT: 'This document contains relevant information about the query topic.'",
                ""
            ]
        
        context = "\n".join(context_parts)
        
        # Vary system prompts: 20% full, 60% medium, 20% short
        # This prevents format memorization while retaining core principles
        prompt_variation = random.choices(
            ["full", "medium", "short"],
            weights=[0.2, 0.6, 0.2]
        )[0]
        
        system_prompt = get_system_prompt_variation(prompt_variation)

        # Extract based on tracked relevant information
        # For non-failed queries, prioritize tracked relevant_extractions over extraction function
        # This ensures we have valid answers for queries that should have them
        query_lower = query.lower()
        not_found_phrases = ["don't have", "couldn't find", "not found", "don't have that information"]
        
        # For failed queries, always return "not found"
        if query_type == "failed":
                assistant_response = "I don't have that information in the provided documents."
        # For non-failed queries, use tracked relevant_extractions first
        elif relevant_extractions:
            # We have tracked relevant information, use it instead of extraction function
            # This ensures non-failed queries have valid answers
            assistant_response = None  # Will be set below based on extraction_type
        else:
            # No tracked extractions, use extraction function as fallback
            assistant_response = extract_information_from_chunks(query, chunks)
            # If extraction function returns "not found" for non-failed query, that's a problem
            # But we'll use it as fallback
        
        # For entity extraction, use tracked extractions (we know they're correct)
        if extraction_type == "entities" and relevant_extractions:
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
            # If we didn't find entity names in tracked extractions, use extraction function
            if not entity_names:
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
        if extraction_type == "list" and relevant_extractions:
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
                # No clean items found, use extraction function
                assistant_response = extract_information_from_chunks(query, chunks)
        elif extraction_type in ["analytical", "relationship", "comparison", "reasoning", "process", "causation", "implications", "role", "differences", "similarities"] and relevant_extractions:
            # For complex analytical/comparison queries, generate intelligent synthesized responses
            if extraction_type in ["comparison", "differences", "similarities"]:
                # Comparison queries - synthesize differences/similarities with EXPLICIT contrast words
                # Ensure responses include: "whereas", "versus", "in contrast", "while"
                if len(relevant_extractions) >= 2:
                    fact1 = relevant_extractions[0].rstrip('.')
                    fact2 = relevant_extractions[1].rstrip('.')
                    # Extract entities from query
                    entity1_match = re.search(r'between (.+?) and|compare (.+?) and|(.+?) vs|(.+?) versus', query_lower)
                    entity2_match = re.search(r'and (.+?)[\?\.]|vs (.+?)[\?\.]|versus (.+?)[\?\.]', query_lower)
                    
                    entity1 = None
                    entity2 = None
                    if entity1_match:
                        entity1 = (entity1_match.group(1) or entity1_match.group(2) or entity1_match.group(3) or entity1_match.group(4)).strip().title() if entity1_match else None
                    if entity2_match:
                        entity2 = (entity2_match.group(1) or entity2_match.group(2) or entity2_match.group(3)).strip().title() if entity2_match else None
                    
                    if entity1 and entity2:
                        # Use explicit contrast words
                        contrast_words = ["whereas", "while", "in contrast", "versus"]
                        contrast_word = random.choice(contrast_words)
                        
                        if "differences" in query_lower or "differ" in query_lower:
                            if contrast_word == "versus":
                                assistant_response = f"{entity1} {fact1} versus {entity2} which {fact2}"
                            elif contrast_word == "whereas":
                                assistant_response = f"{entity1} {fact1}, whereas {entity2} {fact2}"
                            elif contrast_word == "while":
                                assistant_response = f"{entity1} {fact1}, while {entity2} {fact2}"
                            else:  # "in contrast"
                                assistant_response = f"{entity1} {fact1}. In contrast, {entity2} {fact2}"
                        elif "similarities" in query_lower or "similar" in query_lower:
                            assistant_response = f"{entity1} and {entity2} share common ground. {entity1} {fact1} Additionally, {entity2} {fact2}"
                        else:
                            # General comparison - use contrast word
                            if contrast_word == "versus":
                                assistant_response = f"{entity1} {fact1} versus {entity2} which {fact2}"
                            elif contrast_word == "whereas":
                                assistant_response = f"{entity1} {fact1}, whereas {entity2} {fact2}"
                            elif contrast_word == "while":
                                assistant_response = f"{entity1} {fact1}, while {entity2} {fact2}"
                            else:  # "in contrast"
                                assistant_response = f"{entity1} {fact1}. In contrast, {entity2} {fact2}"
                    else:
                        # No entities extracted, use contrast words in facts
                        contrast_word = random.choice(["whereas", "while", "in contrast"])
                        if contrast_word == "whereas":
                            assistant_response = f"{fact1}, whereas {fact2}"
                        elif contrast_word == "while":
                            assistant_response = f"{fact1}, while {fact2}"
                        else:  # "in contrast"
                            assistant_response = f"{fact1}. In contrast, {fact2}"
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
                        additional_context = relevant_extractions[1].rstrip('.')
                        assistant_response += f" This connection is evident through {additional_context.lower()}."
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
        # If we haven't set assistant_response yet, use extraction function
        if assistant_response is None:
            assistant_response = extract_information_from_chunks(query, chunks)
        
        # Final check: For non-failed queries, if we still have "not found", that's a problem
        # Use relevant_extractions if available - this is critical for non-failed queries
        if query_type != "failed" and any(phrase in assistant_response.lower() for phrase in not_found_phrases):
            if relevant_extractions:
                # We have relevant info but extraction function didn't find it
                # Generate a response from relevant_extractions - this ensures non-failed queries have answers
                if extraction_type == "list":
                    # Format as list
                    clean_items = [str(item) for item in relevant_extractions if isinstance(item, str) and len(str(item)) > 3]
                    if clean_items:
                        unique_items = sorted(set(clean_items))[:10]
                        assistant_response = f"The items are: {', '.join(unique_items)}."
                elif extraction_type in ["analytical", "relationship", "comparison", "reasoning", "causation", "implications", "process"]:
                    # Format as analytical response
                    facts = [str(item) for item in relevant_extractions[:2] if isinstance(item, str) and len(str(item)) > 20]
                    if facts:
                        assistant_response = ". ".join(facts) + "."
                elif extraction_type in ["attributes", "description", "properties"]:
                    # Format as attribute response
                    if relevant_extractions:
                        assistant_response = str(relevant_extractions[0]) + "."
                else:
                    # Generic response - use first relevant extraction
                    if relevant_extractions:
                        first_item = str(relevant_extractions[0])
                        if len(first_item) > 10:  # Only use if meaningful
                            assistant_response = first_item + ("." if not first_item.endswith(".") else "")
        
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
    
    num_examples = 6000  # Increased from 4000 to 5000 with focus on failing categories
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
