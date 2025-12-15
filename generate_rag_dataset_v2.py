#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Analysis Dataset Generator V2 - High Quality Implementation
================================================================

Generates 6000 training examples for SLM to analyze RAG chunks:
- Each chunk: 6-8 sentences with realistic, meaningful content
- Multiple instances of relevant information across chunks
- Varied system prompts (full, medium, short) with 7-step core principles
- Pattern-based distribution to teach general RAG skills
- Realistic business/technical content instead of placeholders
"""

import json
import random
from typing import List, Dict, Any, Tuple

# ============================================================================
# System Prompt Variations (7-Step Core Principles)
# ============================================================================

def get_system_prompt_variation(variation_type="full"):
    """Generate system prompt with 7-step core principles"""
    
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

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""
    
    else:  # short
        return f"""You are an AI assistant that analyzes RAG chunks to extract relevant information.

{core_principles}

ESSENTIAL RULES:
- NEVER hallucinate - if information doesn't exist, say "I don't have that information in the provided documents"
- Use EXACT information from chunks - NEVER invent or modify
- Apply query-specific filtering - exclude information that doesn't match what is asked
- Extract ALL matching items - read ALL chunks completely before responding
- ORDER-INDEPENDENT: Extract same results regardless of chunk order

Return the final answer in natural language. Do not include reasoning steps in the response."""

# ============================================================================
# Realistic Content Generation
# ============================================================================

def generate_random_name():
    """Generate random person name"""
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Sage", "River",
                   "Blake", "Cameron", "Dakota", "Emery", "Finley", "Harper", "Hayden", "Jamie", "Kendall", "Logan"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
                  "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Harris"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_random_company():
    """Generate random company name"""
    prefixes = ["Tech", "Data", "Cloud", "AI", "Digital", "Smart", "Global", "Next", "Future", "Prime",
                "Quantum", "Nexus", "Vertex", "Apex", "Catalyst", "Synergy", "Pinnacle", "Summit", "Zenith", "Aurora"]
    suffixes = ["Corp", "Systems", "Solutions", "Labs", "Group", "Industries", "Ventures", "Partners", "Works", "Co",
                "Technologies", "Enterprises", "Holdings", "Dynamics", "Innovations", "Networks", "Services", "Capital", "Alliance"]
    return f"{random.choice(prefixes)}{random.choice(suffixes)}"

def generate_random_concept():
    """Generate random concept/term"""
    concepts = ["innovation", "strategy", "growth", "efficiency", "optimization", "transformation", 
                "scalability", "automation", "integration", "collaboration", "digitalization",
                "analytics", "cloud computing", "machine learning", "data science", "cybersecurity",
                "blockchain", "IoT", "edge computing", "quantum computing"]
    return random.choice(concepts)

# ============================================================================
# Realistic Sentence Generation
# ============================================================================

def generate_entity_sentence(name: str, role: str, company: str) -> str:
    """Generate realistic sentence about a person in a role at a company"""
    templates = [
        f"{name} serves as {role} at {company}, leading strategic initiatives and overseeing key operations.",
        f"As {role} of {company}, {name} is responsible for driving innovation and managing cross-functional teams.",
        f"{name} holds the position of {role} at {company}, where they focus on expanding market presence and building partnerships.",
        f"In their role as {role} at {company}, {name} has been instrumental in developing new product lines and improving customer satisfaction.",
        f"{company}'s {role}, {name}, has extensive experience in technology leadership and business development.",
    ]
    return random.choice(templates)

def generate_company_feature_sentence(company: str, feature: str) -> str:
    """Generate realistic sentence about a company feature"""
    templates = [
        f"{company} offers {feature} as part of its comprehensive solution suite, designed to meet enterprise needs.",
        f"One of {company}'s key offerings is {feature}, which enables customers to streamline their operations and improve productivity.",
        f"The {feature} provided by {company} has been widely adopted by organizations seeking to modernize their infrastructure.",
        f"{company} has developed {feature} to address the growing demand for scalable and efficient business solutions.",
        f"Customers can leverage {feature} from {company} to enhance their digital transformation initiatives.",
    ]
    return random.choice(templates)

def generate_comparison_sentence(entity1: str, entity2: str, attribute: str) -> str:
    """Generate realistic comparison sentence"""
    comparisons = [
        f"{entity1} focuses on {attribute}, while {entity2} emphasizes different aspects of the same domain.",
        f"While {entity1} excels in {attribute}, {entity2} takes a more comprehensive approach to the market.",
        f"{entity1} and {entity2} differ significantly in their approach to {attribute}, with each offering unique advantages.",
        f"The primary distinction between {entity1} and {entity2} lies in their handling of {attribute}.",
        f"{entity1} prioritizes {attribute} more than {entity2}, which focuses on broader market coverage.",
    ]
    return random.choice(comparisons)

def generate_analytical_sentence(entity: str, action: str, reason: str) -> str:
    """Generate realistic analytical/causal sentence"""
    templates = [
        f"{entity} decided to {action} due to {reason}, which created new opportunities for growth.",
        f"The decision by {entity} to {action} was primarily driven by {reason}, reflecting changing market conditions.",
        f"{entity} chose to {action} because {reason}, enabling the organization to adapt to evolving customer needs.",
        f"Motivated by {reason}, {entity} made the strategic move to {action}, positioning itself for future success.",
        f"{entity}'s move to {action} was a direct response to {reason}, demonstrating proactive market leadership.",
    ]
    return random.choice(templates)

def generate_relationship_sentence(entity1: str, entity2: str, relationship: str) -> str:
    """Generate realistic relationship sentence"""
    templates = [
        f"{entity1} and {entity2} have established a {relationship}, enabling both parties to leverage complementary strengths.",
        f"The {relationship} between {entity1} and {entity2} has resulted in innovative solutions and expanded market reach.",
        f"{entity1} maintains a {relationship} with {entity2}, facilitating knowledge sharing and collaborative development.",
        f"Through their {relationship}, {entity1} and {entity2} have created synergies that benefit their respective customer bases.",
        f"The {relationship} connecting {entity1} and {entity2} has been instrumental in driving mutual growth and success.",
    ]
    return random.choice(templates)

def generate_contextual_sentence(domain: str = "business") -> str:
    """Generate realistic contextual sentence that doesn't directly answer queries - highly diverse pool"""
    business_contexts = [
        # Market and industry context
        "Market analysts have observed significant shifts in consumer behavior patterns over the past quarter.",
        "Industry reports indicate a growing trend toward digital transformation initiatives across multiple sectors.",
        "Economic indicators suggest a period of sustained growth in technology investments.",
        "Regulatory changes in the financial sector have prompted organizations to reassess their compliance frameworks.",
        "Global supply chain disruptions have accelerated the adoption of alternative sourcing strategies.",
        "Customer feedback surveys reveal increasing demand for personalized service experiences.",
        "Competitive analysis shows emerging players entering the market with innovative business models.",
        "Investment patterns demonstrate a strong preference for sustainable and ESG-compliant ventures.",
        
        # Organizational operations
        "The quarterly review process identified several areas requiring immediate attention and resource allocation.",
        "Cross-functional teams have been collaborating on process improvement initiatives throughout the organization.",
        "Performance metrics indicate steady progress toward achieving the annual strategic objectives.",
        "Stakeholder meetings have been scheduled to discuss upcoming initiatives and resource requirements.",
        "Internal audits revealed opportunities for streamlining operational workflows and reducing overhead costs.",
        "The leadership team has been evaluating potential partnerships to expand market reach.",
        "Training programs have been implemented to enhance employee skills in emerging technologies.",
        "Budget allocations reflect a strategic shift toward innovation and research development.",
        
        # Technology and innovation
        "Recent technological advancements have opened new possibilities for product development and service delivery.",
        "The IT department has been upgrading infrastructure to support increased computational demands.",
        "Data analytics capabilities have been enhanced to provide more actionable business insights.",
        "Cloud migration projects are progressing according to schedule with minimal disruption to operations.",
        "Security protocols have been strengthened in response to evolving cybersecurity threats.",
        "Integration of artificial intelligence tools has improved efficiency in routine operational tasks.",
        "Mobile application development has become a priority to meet changing customer expectations.",
        "API integrations have enabled seamless data exchange between different business systems.",
        
        # Strategic planning
        "Long-term strategic planning sessions have been conducted to align organizational goals with market opportunities.",
        "Scenario analysis has been performed to evaluate potential outcomes under different market conditions.",
        "Risk assessment frameworks have been updated to account for emerging business challenges.",
        "Portfolio diversification strategies are being considered to mitigate potential market volatility.",
        "Expansion into new geographic markets requires careful evaluation of regulatory and cultural factors.",
        "Brand positioning strategies have been refined to better communicate value propositions to target audiences.",
        "Customer segmentation analysis has identified new opportunities for targeted marketing campaigns.",
        "Product lifecycle management processes have been optimized to reduce time-to-market for new offerings.",
        
        # Financial and operational
        "Revenue streams have diversified over the past year, reducing dependence on traditional income sources.",
        "Cost optimization initiatives have resulted in improved profit margins without compromising service quality.",
        "Cash flow management has been strengthened through improved forecasting and collection processes.",
        "Vendor relationships have been renegotiated to secure more favorable terms and pricing structures.",
        "Asset utilization rates have improved following the implementation of predictive maintenance programs.",
        "Working capital management has been optimized to support growth initiatives while maintaining liquidity.",
        "Financial reporting systems have been upgraded to provide real-time visibility into business performance.",
        "Investment in employee development programs has shown positive returns in productivity and retention.",
        
        # Customer and market
        "Customer satisfaction scores have remained consistently high despite increased service volume.",
        "Market research indicates growing awareness of the brand among target demographic segments.",
        "Sales pipeline analysis reveals strong potential for revenue growth in the coming quarters.",
        "Customer retention strategies have been successful in maintaining long-term relationships.",
        "Product launch campaigns have generated significant interest and early adoption rates.",
        "Customer support operations have been expanded to handle increased inquiry volumes.",
        "Market penetration strategies are being developed for underserved geographic regions.",
        "Customer journey mapping has identified key touchpoints for improving overall experience.",
        
        # General business context
        "Organizational restructuring efforts have been completed to better align with strategic priorities.",
        "Communication channels have been enhanced to facilitate better information flow across departments.",
        "Quality assurance processes have been standardized to ensure consistent service delivery.",
        "Project management methodologies have been adopted to improve delivery timelines and outcomes.",
        "Knowledge management systems have been implemented to capture and share institutional expertise.",
        "Change management initiatives have been launched to support organizational transformation efforts.",
        "Partnership ecosystems have been developed to create mutually beneficial business relationships.",
        "Innovation labs have been established to explore emerging technologies and their applications.",
    ]
    return random.choice(business_contexts)

# ============================================================================
# Query Templates and Generation
# ============================================================================

QUERY_TEMPLATES = [
    # Entity extraction queries
    {"type": "entity", "template": "who are the {role} of {company}?", "domain": "business"},
    {"type": "entity", "template": "who is the {role} at {company}?", "domain": "business"},
    
    # List extraction queries
    {"type": "list", "template": "what are the {items} of {entity}?", "domain": "general"},
    {"type": "list", "template": "list the {items} related to {concept}.", "domain": "general"},
    {"type": "list", "template": "what {items} does {company} offer?", "domain": "business"},
    
    # Comparison queries
    {"type": "comparison", "template": "compare {entity1} and {entity2}.", "domain": "general"},
    {"type": "comparison", "template": "what is the difference between {entity1} and {entity2}?", "domain": "general"},
    
    # Analytical queries
    {"type": "analytical", "template": "why did {entity} {action}?", "domain": "general"},
    {"type": "analytical", "template": "what caused {event}?", "domain": "general"},
    
    # Relationship queries
    {"type": "relationship", "template": "how are {entity1} and {entity2} related?", "domain": "general"},
    {"type": "relationship", "template": "what is the connection between {entity1} and {entity2}?", "domain": "general"},
    
    # Process queries
    {"type": "process", "template": "how does {process} work?", "domain": "general"},
    {"type": "process", "template": "what is the process for {action}?", "domain": "general"},
]

def generate_query(template: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Generate a query from template and return query with context"""
    query = template["template"]
    context = {}
    
    # Replace placeholders
    if "{role}" in query:
        role = random.choice(["leaders", "members", "directors", "managers", "executives", "founders"])
        query = query.replace("{role}", role)
        context["role"] = role
    if "{company}" in query:
        company = generate_random_company()
        query = query.replace("{company}", company)
        context["company"] = company
    if "{items}" in query:
        items = random.choice(["features", "benefits", "components", "advantages", "capabilities", "services"])
        query = query.replace("{items}", items)
        context["items"] = items
    if "{entity}" in query:
        entity = random.choice([generate_random_company(), generate_random_concept()])
        query = query.replace("{entity}", entity)
        context["entity"] = entity
    if "{concept}" in query:
        concept = generate_random_concept()
        query = query.replace("{concept}", concept)
        context["concept"] = concept
    if "{entity1}" in query:
        entity1 = random.choice([generate_random_company(), generate_random_concept()])
        query = query.replace("{entity1}", entity1)
        context["entity1"] = entity1
    if "{entity2}" in query:
        entity2 = random.choice([generate_random_company(), generate_random_concept()])
        query = query.replace("{entity2}", entity2)
        context["entity2"] = entity2
    if "{action}" in query:
        action = random.choice(["expand", "grow", "change", "improve", "restructure", "diversify"])
        query = query.replace("{action}", action)
        context["action"] = action
    if "{event}" in query:
        event = random.choice(["the expansion", "the growth", "the change", "the restructuring", "the merger"])
        query = query.replace("{event}", event)
        context["event"] = event
    if "{process}" in query:
        process = random.choice(["the system", "the process", "the workflow", "the platform", "the framework"])
        query = query.replace("{process}", process)
        context["process"] = process
    
    return query, context

# ============================================================================
# Realistic Response Generation
# ============================================================================

def generate_response(query: str, relevant_info: List[str], query_type: str, context: Dict[str, Any]) -> str:
    """Generate realistic expected response from relevant information"""
    
    if not relevant_info:
        return "I don't have that information in the provided documents."
    
    if query_type == "entity":
        # Format as list of names
        names = [info.strip() for info in relevant_info if info.strip()]
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            return ", ".join(names[:-1]) + f", and {names[-1]}"
    
    elif query_type == "list":
        # Format as list
        items = [info.strip() for info in relevant_info if info.strip()]
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"
    
    elif query_type == "comparison":
        # Format comparison - sentences already contain comparison words, just join them
        # If we have multiple comparison sentences, join them naturally
        if len(relevant_info) >= 2:
            # Check if sentences already contain "while" - if so, just join with period
            sentences = [info.strip().rstrip('.') for info in relevant_info]  # Remove trailing periods
            if any("while" in s.lower() for s in sentences):
                return ". ".join(sentences) + "."
            else:
                return f"{sentences[0]}. {sentences[1]}."
        return " ".join([info.strip().rstrip('.') for info in relevant_info])
    
    elif query_type == "analytical":
        # Format with reasoning words
        response = " ".join([info.strip() for info in relevant_info])
        if "because" not in response.lower() and "due to" not in response.lower():
            response = f"because {response}"
        return response
    
    elif query_type == "relationship":
        # Format relationship description
        return " ".join([info.strip() for info in relevant_info])
    
    else:
        # Default: join all relevant info
        return " ".join([info.strip() for info in relevant_info])

# ============================================================================
# Realistic Chunk Generation
# ============================================================================

def create_realistic_chunk(relevant_sentences: List[str], irrelevant_sentences: List[str], 
                          num_sentences: int = 7) -> str:
    """
    Create a realistic chunk with 6-8 sentences mixing relevant and irrelevant information.
    Creates natural flow like real document excerpts.
    """
    # Determine how many relevant vs irrelevant sentences
    if relevant_sentences:
        num_relevant = random.randint(1, min(3, len(relevant_sentences)))
        selected_relevant = random.sample(relevant_sentences, num_relevant)
    else:
        num_relevant = 0
        selected_relevant = []
    
    num_irrelevant = num_sentences - num_relevant
    
    # Select irrelevant sentences
    if irrelevant_sentences:
        num_to_select = min(num_irrelevant, len(irrelevant_sentences))
        selected_irrelevant = random.sample(irrelevant_sentences, num_to_select)
    else:
        selected_irrelevant = []
    
    # Fill to target with diverse contextual sentences (avoid repetition)
    all_sentences = selected_relevant + selected_irrelevant
    used_contextual = set()
    max_attempts = 100  # Prevent infinite loop
    
    while len(all_sentences) < num_sentences:
        attempts = 0
        contextual = generate_contextual_sentence()
        # Try to avoid exact duplicates (but allow similar themes for realism)
        while contextual in used_contextual and attempts < max_attempts:
            contextual = generate_contextual_sentence()
            attempts += 1
        used_contextual.add(contextual)
        all_sentences.append(contextual)
    
    # Create natural flow: start with contextual, mix in relevant/irrelevant, end with contextual
    # This makes chunks read more like real document excerpts
    contextual_only = [s for s in all_sentences if s not in selected_relevant and s not in selected_irrelevant]
    mixed = selected_relevant + selected_irrelevant
    
    # Build chunk with natural structure
    chunk_sentences = []
    if contextual_only:
        # Start with 1-2 contextual sentences
        chunk_sentences.extend(random.sample(contextual_only, min(2, len(contextual_only))))
        contextual_only = [s for s in contextual_only if s not in chunk_sentences]
    
    # Add relevant/irrelevant sentences
    random.shuffle(mixed)
    chunk_sentences.extend(mixed)
    
    # Fill remaining with contextual
    while len(chunk_sentences) < num_sentences and contextual_only:
        chunk_sentences.append(contextual_only.pop(0))
    
    # Ensure proper sentence formatting
    formatted_sentences = []
    for sent in chunk_sentences:
        sent = sent.strip()
        if not sent.endswith('.'):
            sent += '.'
        formatted_sentences.append(sent)
    
    chunk_text = " ".join(formatted_sentences)
    return chunk_text

# ============================================================================
# Dataset Generation
# ============================================================================

def generate_example(pattern_type: str = "mixed_content") -> Dict[str, Any]:
    """Generate a single training example with realistic content"""
    
    # Select query template
    template = random.choice(QUERY_TEMPLATES)
    query, context = generate_query(template)
    query_type = template["type"]
    
    # Select system prompt variation
    prompt_type = random.choices(
        ["full", "medium", "short"],
        weights=[0.2, 0.6, 0.2]
    )[0]
    system_prompt = get_system_prompt_variation(prompt_type)
    
    # Handle "not_found" pattern - no relevant information
    if pattern_type == "not_found":
        relevant_info = []
        relevant_sentences_templates = []
    else:
        # Generate relevant information (what should be extracted) - as realistic sentences
        num_relevant_items = random.randint(2, 4)
        relevant_info = []
        relevant_sentences_templates = []
        
        if query_type == "entity":
            # Generate entity names and sentences about them
            role = context.get("role", "leaders")
            company = context.get("company", generate_random_company())
            for _ in range(num_relevant_items):
                name = generate_random_name()
                relevant_info.append(name)
                relevant_sentences_templates.append(generate_entity_sentence(name, role, company))
        
        elif query_type == "list":
            # Generate list items with realistic descriptions
            items_type = context.get("items", "features")
            entity = context.get("entity") or context.get("company") or context.get("concept") or generate_random_company()
            list_items = {
                "features": ["real-time analytics dashboard", "automated reporting system", "cloud-based storage", "API integration", "mobile application"],
                "benefits": ["reduced operational costs", "improved efficiency", "enhanced security", "scalable infrastructure", "better user experience"],
                "components": ["data processing engine", "user interface module", "authentication system", "notification service", "analytics platform"],
                "advantages": ["faster processing speed", "lower maintenance costs", "greater flexibility", "better compatibility", "improved reliability"],
                "capabilities": ["real-time monitoring", "automated backups", "multi-user collaboration", "custom integrations", "advanced analytics"],
                "services": ["consulting", "implementation support", "training programs", "maintenance", "custom development"]
            }
            items_list = list_items.get(items_type, ["feature A", "feature B", "feature C"])
            selected_items = random.sample(items_list, min(num_relevant_items, len(items_list)))
            for item in selected_items:
                relevant_info.append(item)
                if items_type == "features" and "company" in context:
                    relevant_sentences_templates.append(generate_company_feature_sentence(context["company"], item))
                else:
                    relevant_sentences_templates.append(f"The {items_type} include {item}, which provides significant value to users.")
        
        elif query_type == "comparison":
            # Generate comparison sentences
            entity1 = context.get("entity1", generate_random_company())
            entity2 = context.get("entity2", generate_random_company())
            attributes = ["market approach", "technology stack", "customer focus", "pricing strategy", "innovation strategy"]
            for _ in range(num_relevant_items):
                attr = random.choice(attributes)
                sentence = generate_comparison_sentence(entity1, entity2, attr)
                relevant_info.append(sentence)
                relevant_sentences_templates.append(sentence)
        
        elif query_type == "analytical":
            # Generate analytical/causal sentences
            entity = context.get("entity") or context.get("company") or generate_random_company()
            action = context.get("action", "expand")
            reasons = [
                "increasing market demand",
                "competitive pressures",
                "customer feedback",
                "technological advancements",
                "regulatory changes",
                "strategic opportunities"
            ]
            for _ in range(num_relevant_items):
                reason = random.choice(reasons)
                sentence = generate_analytical_sentence(entity, action, reason)
                relevant_info.append(sentence)
                relevant_sentences_templates.append(sentence)
        
        elif query_type == "relationship":
            # Generate relationship sentences
            entity1 = context.get("entity1", generate_random_company())
            entity2 = context.get("entity2", generate_random_company())
            relationships = ["strategic partnership", "joint venture", "supplier relationship", "customer relationship", "technology alliance"]
            for _ in range(num_relevant_items):
                rel = random.choice(relationships)
                sentence = generate_relationship_sentence(entity1, entity2, rel)
                relevant_info.append(sentence)
                relevant_sentences_templates.append(sentence)
        
        else:
            # Default: generate generic but realistic sentences
            for _ in range(num_relevant_items):
                sentence = generate_contextual_sentence()
                relevant_info.append(sentence)
                relevant_sentences_templates.append(sentence)
    
    # Generate irrelevant information (similar but doesn't answer query)
    num_irrelevant = random.randint(3, 6)
    irrelevant_sentences_templates = []
    for _ in range(num_irrelevant):
        if query_type == "entity":
            # Similar entities but wrong role/company
            wrong_company = generate_random_company()
            wrong_role = random.choice(["CEO", "CTO", "Manager", "Director"])
            wrong_name = generate_random_name()
            irrelevant_sentences_templates.append(generate_entity_sentence(wrong_name, wrong_role, wrong_company))
        else:
            # Similar but unrelated contextual sentences
            irrelevant_sentences_templates.append(generate_contextual_sentence())
    
    # Generate chunks (3-4 chunks, 6-8 sentences each)
    num_chunks = random.randint(3, 4)
    chunks = []
    
    # Distribute relevant info across chunks (multiple instances)
    relevant_per_chunk = {}
    if relevant_info and relevant_sentences_templates:
        for i, (info, sentence_template) in enumerate(zip(relevant_info, relevant_sentences_templates)):
            chunk_idx = i % num_chunks
            if chunk_idx not in relevant_per_chunk:
                relevant_per_chunk[chunk_idx] = []
            relevant_per_chunk[chunk_idx].append(sentence_template)
        
        # Also add some relevant info to multiple chunks (to teach complete extraction)
        for info, sentence_template in zip(relevant_info[:2], relevant_sentences_templates[:2]):
            additional_chunks = random.sample(range(num_chunks), random.randint(1, 2))
            for chunk_idx in additional_chunks:
                if chunk_idx not in relevant_per_chunk:
                    relevant_per_chunk[chunk_idx] = []
                if sentence_template not in relevant_per_chunk[chunk_idx]:
                    relevant_per_chunk[chunk_idx].append(sentence_template)
    
    # Create chunks
    for chunk_idx in range(num_chunks):
        chunk_relevant = relevant_per_chunk.get(chunk_idx, [])
        # Select irrelevant info for this chunk
        num_irrelevant_needed = random.randint(2, 4)
        num_irrelevant_available = min(num_irrelevant_needed, len(irrelevant_sentences_templates))
        if num_irrelevant_available > 0:
            chunk_irrelevant = random.sample(irrelevant_sentences_templates, num_irrelevant_available)
        else:
            chunk_irrelevant = []
        
        # Create chunk with 6-8 sentences
        chunk_text = create_realistic_chunk(chunk_relevant, chunk_irrelevant, 
                                            num_sentences=random.randint(6, 8))
        
        # Format chunk with relevance score
        relevance_score = random.uniform(0.65, 0.95) if chunk_relevant else random.uniform(0.30, 0.60)
        chunks.append({
            "text": chunk_text,
            "score": round(relevance_score, 2),
            "file": "document.pdf"
        })
    
    # Generate expected response
    response = generate_response(query, relevant_info, query_type, context)
    
    # Format as training example
    user_content = f"Query: {query}\n\n"
    user_content += "RAG Chunks:\n"
    for i, chunk in enumerate(chunks, 1):
        user_content += f"[Chunk {i}] Score: {chunk['score']:.2f}, File: {chunk['file']}\n"
        user_content += f"FULL CHUNK TEXT: '{chunk['text']}'\n\n"
    
    example = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": response}
        ]
    }
    
    return example

# ============================================================================
# Main Generation
# ============================================================================

def main():
    """Generate 6000 training examples"""
    
    print("="*80)
    print("RAG Analysis Dataset Generator V2 - High Quality")
    print("="*80)
    print()
    print("Generating 6000 training examples with realistic content...")
    print()
    
    # Pattern distribution (6000 examples)
    patterns = {
        "mixed_content": 900,      # 15% - Extract relevant, ignore irrelevant
        "multi_chunk": 1200,       # 20% - Extract from multiple chunks
        "role_filtering": 900,     # 15% - Filter by role/entity
        "cross_entity": 900,       # 15% - Filter by specific entity
        "synthesis": 600,          # 10% - Combine info from chunks
        "not_found": 600,          # 10% - Recognize missing info
        "comparison": 450,         # 7.5% - Compare entities
        "relationship": 450,       # 7.5% - Extract relationships
    }
    
    dataset = []
    total = sum(patterns.values())
    
    for pattern, count in patterns.items():
        print(f"Generating {count} examples for pattern: {pattern}...")
        for i in range(count):
            example = generate_example(pattern)
            dataset.append(example)
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{count} ({100*(i+1)/count:.1f}%)")
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Save dataset
    output_file = "rag_analysis_dataset_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print()
    print("="*80)
    print("✅ DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"Total examples: {len(dataset)}")
    print(f"Output file: {output_file}")
    print()
    
    # Verify distribution
    prompt_types = {"full": 0, "medium": 0, "short": 0}
    for example in dataset:
        system_content = example["messages"][0]["content"]
        if "ESSENTIAL GUIDELINES:" in system_content:
            prompt_types["full"] += 1
        elif "KEY RULES:" in system_content:
            prompt_types["medium"] += 1
        else:
            prompt_types["short"] += 1
    
    print("System Prompt Distribution:")
    for ptype, count in prompt_types.items():
        print(f"  {ptype}: {count} ({100*count/len(dataset):.1f}%)")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
