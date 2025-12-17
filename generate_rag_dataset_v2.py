#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Analysis Dataset Generator - Master Script (Unified)
========================================================

UNIFIED MASTER SCRIPT: Combines generate_rag_dataset_v2.py and generate_rag_dataset_complete.py
- Generates 6250 training examples for SLM to analyze RAG chunks
- Uses 6-step CoT system prompt with expected output formats
- Assistant responses contain ONLY final answer (no CoT steps) - prevents CoT leakage
- Each chunk: 6-8 sentences with realistic, meaningful content
- Multiple instances of relevant information across chunks
- Pattern-based distribution to teach general RAG skills
- Realistic business/technical content instead of placeholders
- Mixed relevance scores for robust training

OUTPUT FORMAT:
- System prompt: 6-step CoT with expected output formats (for instruction)
- Assistant response: ONLY final answer (what model learns to output)
- No intermediate CoT steps in assistant response (prevents CoT leakage)
"""

import json
import random
from typing import List, Dict, Any, Tuple

# ============================================================================
# System Prompt Variations (7-Step Core Principles)
# ============================================================================

# 6-Step CoT System Prompt (unified from generate_rag_dataset_complete.py)
# This is the master system prompt used for all examples
SYSTEM_PROMPT_6_STEP = """You are an AI assistant trained to analyze RAG chunks and extract relevant information.

CORE PRINCIPLES (SYSTEMATIC EVALUATION PROCESS):

STEP 1: UNDERSTAND THE QUERY
- Identify what information is being requested
- Note any specific filtering requirements (role, entity, attribute, relationship, etc.)
- Understand the scope and context of what needs to be extracted

EXPECTED OUTPUT FORMAT FOR STEP 1:
"The query asks: [query]. I need to [action]."

Example outputs:
- "The query asks for co-founders of TechCorp. I need to extract ONLY people explicitly labeled as 'Co-Founder' of TechCorp, not other roles like CEO, CTO, CFO, or VP."
- "The query asks: what is the difference between FutureCapital and AICapital?. I need to find relevant information for both entities and determine how the two entities differ."
- "The query asks: what are the features of blockchain?. I need to extract ALL items that match this query from ALL chunks - this is a list query requiring complete extraction."
- "The query asks: who are the managers of TechCorp?. I need to extract ALL managers (plural indicates multiple), not just the first one I find. I must read ALL chunks and extract EVERY manager mentioned."

STEP 2: READ EACH CHUNK COMPLETELY
- Read the entire chunk from start to finish
- Do not stop at keywords - read for full context and meaning
- Understand the complete context before making extraction decisions
- Provide a relevance score based on how well the chunk applies to the query

EXPECTED OUTPUT FORMAT FOR STEP 2:
"Chunk X (Score: Y.YY, [HIGH/MEDIUM/LOW] relevance): [first 1-2 sentences of chunk]..."

Example output:
"Chunk 1 (Score: 0.85, HIGH relevance): John Smith is Co-Founder of TechCorp. Sarah Jones is Co-Founder of DataSystems.
Chunk 2 (Score: 0.66, MEDIUM relevance): Partnership ecosystems have been developed to create mutually beneficial business relationships..."

STEP 3: ANALYZE CHUNK MEANING
- Understand the semantic meaning, not just surface-level keywords
- Describe what each chunk contains and how it relates to the query
- Identify entities, relationships, attributes, and concepts mentioned
- Determine if the chunk provides useful information for answering the query

EXPECTED OUTPUT FORMAT FOR STEP 3:
"Chunk X: [describes what the chunk contains and whether it's useful for the query]"

Example outputs:
- "Chunk 1: describes FutureCapital providing detailed information but does not mention AICapital."
- "Chunk 2: briefly mentions AICapital but does not provide descriptive information useful for a comparison."
- "Chunk 3: provides descriptive information regarding AICapital sufficient to compare against FutureCapital."
- "Chunk 4: mentions IrrelevantAI, not useful for query."
- "Chunk 1: describes mentions QuantumSystems, contains co-founder information."

STEP 4: EXTRACT MATCHING INFORMATION
- Extract information from chunks identified as relevant in Step 3
- Apply exact matching - use information exactly as it appears in chunks
- Track ALL matching items across ALL chunks - for list queries, count items as you extract them
- CRITICAL: Do NOT stop after finding first match - continue extracting from all relevant chunks

EXPECTED OUTPUT FORMAT FOR STEP 4:
"Extract information from Chunk X [and Chunk Y]"

Example outputs:
- "Extract information from Chunk 1 and Chunk 3" (found 2 managers in Chunk 1, 1 manager in Chunk 3 - total 3 managers)
- "Extract information from Chunk 1" (found all 4 features in Chunk 1)
- "Extract information from Chunk 1, Chunk 2, and Chunk 4" (found items scattered across multiple chunks - must extract from all)
- "No matching information found in any chunk. The query cannot be answered from the provided documents."

STEP 5: VERIFY COMPLETENESS
- Ensure you have read ALL chunks completely
- Verify you extracted ALL matching items (do not stop after first match)
- CRITICAL FOR LIST/MULTIPLE ENTITY QUERIES: 
  * If query asks for multiple items using plural forms (e.g., "who are the managers", "list the features", "what are the services", "who are the directors", "list the components"), you MUST extract ALL matching items from ALL chunks
  * Do NOT stop after finding the first match - continue reading ALL chunks until you have checked every single one
  * If you find 3 managers across chunks, list all 3. If you find 4 features, list all 4. If you find 2 services in Chunk 1 and 2 more in Chunk 3, list all 4 services
  * Count the items as you extract them - if the query asks for "managers" (plural), expect multiple managers and extract ALL of them
  * Read each chunk from start to finish - items may appear anywhere in a chunk, not just at the beginning
  * If a chunk mentions multiple matching items, extract ALL of them, not just the first one
- Confirm extraction is complete before finalizing response

EXPECTED OUTPUT FORMAT FOR STEP 5:
"Ensuring all relevant information was extracted. Read all X chunk(s) completely. [Extracted Y matching item(s) across all chunks. All relevant information has been identified / No matching information found in any chunk.]"

Example outputs:
- "Ensuring all relevant information was extracted. Read all 4 chunk(s) completely. Extracted 2 matching item(s) across all chunks. All relevant information has been identified."
- "Ensuring all relevant information was extracted. Read all 3 chunk(s) completely. No matching information found in any chunk."

STEP 6: SYNTHESIZE RESPONSE
- Combine information from all chunks into coherent answer
- Format naturally and directly address the query
- CRITICAL: If after reading ALL chunks completely you find NO information that matches the query (wrong role, wrong company, or missing entirely), you MUST respond with exactly: "I don't have that information in the provided documents"
- DO NOT infer, guess, or make up information - if it's not explicitly in the chunks, say "I don't have that information in the provided documents"

EXPECTED OUTPUT FORMAT FOR STEP 6:
[Just the final answer - no prefix, no "STEP 6:" marker, just the answer itself]

Example outputs:
- "John Smith and Mike Brown" (for "who are the co-founders" - extracted both from chunks)
- "cloud-based storage, real-time analytics dashboard, automated reporting system, and mobile application" (for "list the features" - extracted ALL 4 features from multiple chunks)
- "Alice Johnson, Bob Smith, and Carol Williams" (for "who are the managers" - extracted ALL 3 managers, not just the first one)
- "The primary distinction between FutureCapital and AICapital lies in their handling of innovation strategy. While FutureCapital excels in pricing strategy, AICapital takes a more comprehensive approach to the market."
- "I don't have that information in the provided documents"

CRITICAL: Follow these steps in order for EVERY query. Chunk order does not change the answer - read all chunks before responding.

ESSENTIAL GUIDELINES:
- NEVER hallucinate - only use information that appears in the provided chunks
- NEVER make up names, entities, or information - if information doesn't exist, say "I don't have that information in the provided documents"
- CRITICAL: If you cannot find the EXACT information requested in ANY chunk, you MUST respond with "I don't have that information in the provided documents" - DO NOT guess, infer, or make up information
- Use EXACT information from chunks - never substitute or modify names, terms, or entities
- Apply query-specific filtering during Step 3 (analyze chunk meaning) - match what the query specifically asks for
- Extract ALL matching items - complete Step 5 (verify completeness) ensures nothing is missed
- CRITICAL FOR LIST QUERIES: When query uses plural forms ("who are the", "list the", "what are the"), you MUST extract ALL matching items. Extracting only 1 item when multiple exist is INCORRECT. Read ALL chunks completely and extract EVERY matching item from EVERY chunk before responding
- Relevance scores guide prioritization but do not override the analysis steps

QUERY TYPE HANDLING (applied during Step 3 - Analyze Chunk Meaning):
- Role/entity queries: Filter by the SPECIFIC role mentioned (e.g., "co-founders" means ONLY co-founders, NOT CEOs, CTOs, or other roles). If the query asks for "co-founders", extract ONLY people explicitly labeled as co-founders, NOT other roles even if they are at the same company
- Company-specific queries: Extract information ONLY about the company that matches the query. If query asks about "TechCorp", extract information ONLY about the matching company in chunks (RAG handles fuzzy matching like "Tech Corp" → "TechCorp" at retrieval level). Use the company name EXACTLY as it appears in the chunks. Do NOT extract information about other companies mentioned in the same chunk
- Comparison queries: Extract information comparing the entities mentioned
- Relationship queries: Extract connection information between entities
- Analytical queries: Extract reasoning, causation, or explanation
- Process queries: Extract step-by-step information
- List queries: CRITICAL - Extract ALL items that match the query criteria. These queries use plural forms ("who are the", "list the", "what are the") and expect multiple items:
  * Read ALL chunks completely from start to finish
  * Extract EVERY matching item from EVERY chunk - do NOT stop after finding first match
  * If query asks "who are the managers" and you find managers in Chunk 1, Chunk 2, and Chunk 3, extract ALL managers from ALL chunks
  * If query asks "list the features" and you find features scattered across multiple chunks, extract ALL features from ALL chunks
  * Count items as you extract: if you find 1 item in Chunk 1, 2 items in Chunk 2, and 1 item in Chunk 3, your final answer must include all 4 items
  * Partial extraction is INCORRECT - extracting only 1 manager when 3 exist is a failure
  * Verify completeness: before finalizing, mentally count all extracted items to ensure you haven't missed any

CRITICAL OUTPUT REQUIREMENT:
- You MUST output ONLY the final answer (STEP 6 content)
- DO NOT output STEP 1, STEP 2, STEP 3, STEP 4, or STEP 5
- DO NOT output "Extract information from Chunk X" or any intermediate reasoning
- DO NOT output "STEP 6: SYNTHESIZE RESPONSE" or any step markers
- Output ONLY the final answer text itself (e.g., "John Smith and Mike Brown" or "I don't have that information in the provided documents")
- The CoT steps (STEP 1-5) are for INTERNAL reasoning only - they should NOT appear in your output
- If you output any intermediate steps, your response is INCORRECT

Return ONLY the final answer in natural, conversational language. Do not include reasoning steps in the response."""

def get_system_prompt_variation(variation_type="full"):
    """
    Get system prompt - now always uses unified 6-step CoT format
    
    NOTE: variation_type parameter is kept for compatibility but always returns
    the same 6-step CoT prompt. This ensures consistent training format.
    """
    # Always use the unified 6-step CoT system prompt
    return SYSTEM_PROMPT_6_STEP

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
    {"type": "entity", "template": "who are the {role} at {company}?", "domain": "business"},
    
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
    # ENHANCED: For multi_chunk and role_filtering patterns, prioritize list/entity queries
    # These patterns are specifically designed to teach complete multi-entity extraction
    if pattern_type in ["multi_chunk", "role_filtering"]:
        # 70% chance of list/entity queries for these patterns
        list_entity_templates = [t for t in QUERY_TEMPLATES if t["type"] in ["list", "entity"]]
        other_templates = [t for t in QUERY_TEMPLATES if t["type"] not in ["list", "entity"]]
        if random.random() < 0.70 and list_entity_templates:
            template = random.choice(list_entity_templates)
        else:
    template = random.choice(QUERY_TEMPLATES)
    else:
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
        # ENHANCED: For list/entity queries, prefer 3-4 items to force complete extraction
        # Default to 2-4, but will be adjusted per query type below
        num_relevant_items = random.randint(2, 4)
        relevant_info = []
        relevant_sentences_templates = []
        
        if query_type == "entity":
            # Generate entity names and sentences about them
            role = context.get("role", "leaders")
            company = context.get("company", generate_random_company())
            # ENHANCED: For entity queries, ensure we generate 3-4 entities (not just 2)
            # This forces model to extract multiple entities, not just first match
            if pattern_type in ["multi_chunk", "role_filtering"]:
                # For multi-chunk patterns, generate more entities to force complete extraction
                num_relevant_items = random.randint(3, 4)
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
            # ENHANCED: For list queries, ensure we generate 3-4 items (not just 2)
            # This forces model to extract ALL items, not stop after first match
            if pattern_type in ["multi_chunk", "mixed_content"]:
                num_relevant_items = random.randint(3, 4)
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
    
    # ENHANCED DISTRIBUTION LOGIC FOR MULTI-ENTITY EXTRACTION
    # Distribute relevant info across chunks (multiple instances)
    # Key improvement: For list/entity queries, explicitly scatter items across multiple chunks
    # This forces model to read ALL chunks to find ALL items, preventing early stopping
    relevant_per_chunk = {}
    if relevant_info and relevant_sentences_templates:
        # For list and entity queries, ensure items are scattered across multiple chunks
        # This forces model to read ALL chunks to find ALL items
        if query_type in ["list", "entity"]:
            # Strategy: Scatter items across chunks, ensuring multiple chunks have items
            # This teaches model to NOT stop after first match
            items_per_chunk = {}
            for chunk_idx in range(num_chunks):
                items_per_chunk[chunk_idx] = []
            
            # Distribute items across chunks (ensure at least 2 chunks have items for multi-item queries)
            if len(relevant_info) >= 2:
                # For 2-3 items: put in 2 different chunks
                # For 4+ items: put in 3+ different chunks
                num_chunks_with_items = min(len(relevant_info), max(2, num_chunks))
                chunks_with_items = random.sample(range(num_chunks), num_chunks_with_items)
                
                # Distribute items across selected chunks
                for i, (info, sentence_template) in enumerate(zip(relevant_info, relevant_sentences_templates)):
                    # Use modulo to distribute, but ensure we use chunks_with_items
                    chunk_idx = chunks_with_items[i % len(chunks_with_items)]
                    items_per_chunk[chunk_idx].append((info, sentence_template))
            else:
                # Single item: can be in any chunk
                chunk_idx = random.randint(0, num_chunks - 1)
                items_per_chunk[chunk_idx].append((relevant_info[0], relevant_sentences_templates[0]))
            
            # Convert to relevant_per_chunk format
            for chunk_idx, items in items_per_chunk.items():
                if items:
                    relevant_per_chunk[chunk_idx] = [sentence for _, sentence in items]
        else:
            # For other query types, use original distribution logic
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
    
    # Create chunks with mixed relevance scores for realistic training
    # Strategy: Ensure variety - some relevant chunks can be MEDIUM, some irrelevant can be MEDIUM
    chunk_relevance_types = []
    
    # Determine which chunks should have relevant info
    for chunk_idx in range(num_chunks):
        chunk_relevant = relevant_per_chunk.get(chunk_idx, [])
        chunk_relevance_types.append({
            'idx': chunk_idx,
            'has_relevant': bool(chunk_relevant),
            'relevant': chunk_relevant
        })
    
    # Assign mixed relevance scores (more realistic distribution)
    # - Relevant chunks: 60% HIGH, 30% MEDIUM, 10% LOW (to teach ignoring low-relevance even if has keywords)
    # - Irrelevant chunks: 20% MEDIUM (to teach ignoring medium-relevance if doesn't answer query), 80% LOW
    for chunk_info in chunk_relevance_types:
        chunk_idx = chunk_info['idx']
        chunk_relevant = chunk_info['relevant']
        
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
        
        # Assign relevance score with mixed distribution
        if chunk_relevant:
            # Relevant chunk: mix of HIGH, MEDIUM, and occasionally LOW
            rand = random.random()
            if rand < 0.60:  # 60% HIGH
                relevance_score = random.uniform(0.70, 0.95)
            elif rand < 0.90:  # 30% MEDIUM
                relevance_score = random.uniform(0.50, 0.69)
            else:  # 10% LOW (teaches model to still extract from low-score chunks if relevant)
                relevance_score = random.uniform(0.40, 0.49)
        else:
            # Irrelevant chunk: mostly LOW, some MEDIUM
            rand = random.random()
            if rand < 0.80:  # 80% LOW
                relevance_score = random.uniform(0.30, 0.49)
            else:  # 20% MEDIUM (teaches model to ignore medium-score chunks if irrelevant)
                relevance_score = random.uniform(0.50, 0.69)
        
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
    """Generate 6250 training examples with 6-step CoT format"""
    
    print("="*80)
    print("RAG Analysis Dataset Generator - Master Script (Unified)")
    print("="*80)
    print()
    print("Generating 6250 training examples with 6-step CoT format...")
    print("✅ System prompt: 6-step CoT with expected output formats")
    print("✅ Assistant response: ONLY final answer (no CoT steps)")
    print("✅ Prevents CoT leakage - model learns to output only final answer")
    print()
    
    # Pattern distribution (6250 examples total)
    # ENHANCED: Increased multi_chunk and role_filtering to emphasize multi-entity extraction
    # Reduced other patterns to maintain 6250 total
    patterns = {
        "mixed_content": 700,      # 11.2% - Extract relevant, ignore irrelevant (reduced from 900)
        "multi_chunk": 1500,       # 24.0% - Extract from multiple chunks (INCREASED from 1200 - more multi-entity examples)
        "role_filtering": 1200,    # 19.2% - Filter by role/entity (INCREASED from 900 - more entity list queries)
        "cross_entity": 800,       # 12.8% - Filter by specific entity (reduced from 900)
        "synthesis": 550,          # 8.8% - Combine info from chunks (reduced from 600)
        "not_found": 550,          # 8.8% - Recognize missing info (reduced from 600)
        "comparison": 400,         # 6.4% - Compare entities (reduced from 450)
        "relationship": 400,        # 6.4% - Extract relationships (reduced from 450)
        "analytical": 150,         # 2.4% - Analytical queries (reduced from 250)
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
    print("✅ Format: 6-step CoT system prompt + final answer only in assistant response")
    print("✅ Prevents CoT leakage - model will learn to output only final answer")
    print()
    
    # Verify format
    cot_steps_in_response = 0
    final_answer_only = 0
    for i, example in enumerate(dataset[:100]):  # Check first 100
        messages = example.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "STEP 1:" in content or "STEP 4:" in content or "Extract information from Chunk" in content:
                    cot_steps_in_response += 1
                else:
                    final_answer_only += 1
    
    print("Format Verification (first 100 examples):")
    print(f"  ✅ Final answer only: {final_answer_only}")
    print(f"  ❌ CoT steps in response: {cot_steps_in_response} (should be 0)")
    if cot_steps_in_response == 0:
        print("  ✅ Dataset format is CORRECT!")
    else:
        print("  ⚠️  WARNING: Some examples still contain CoT steps!")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
