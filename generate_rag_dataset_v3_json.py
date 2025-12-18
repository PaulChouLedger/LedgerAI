#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Analysis Dataset Generator - JSON Output Version (Optimized)
================================================================

OPTIMIZED FOR BETTER FINE-TUNING:
- JSON output format (easier for model to learn structured extraction)
- Simplified system prompt (removes CoT complexity)
- Focus on extraction completeness (all entities, all items)
- Post-processing can convert JSON to natural language

OUTPUT FORMAT:
- System prompt: Simplified instructions focusing on JSON extraction
- Assistant response: JSON object with extracted information
- No CoT steps in output (prevents leakage)
"""

import json
import random
from typing import List, Dict, Any, Tuple

# ============================================================================
# Simplified System Prompt (JSON Output)
# ============================================================================

SYSTEM_PROMPT_JSON = """You are an AI assistant that extracts information from RAG chunks and returns it as JSON.

TASK:
1. Read ALL chunks completely from start to finish
2. Extract ALL matching items (do NOT stop after first match)
3. Return results as valid JSON

OUTPUT FORMAT:
Return a JSON object with this structure:
{
  "answer_type": "entities" | "list" | "comparison" | "analytical" | "relationship" | "process" | "not_found",
  "items": ["item1", "item2", ...],  // For entities/list queries - ALL items found
  "text": "natural language answer",  // For comparison/analytical/relationship/process queries
  "chunks_used": [1, 2, ...]  // Which chunks contained the information
}

CRITICAL RULE #1: EXTRACT ALL ITEMS - PARTIAL EXTRACTION IS WRONG
- For queries asking for multiple items (plural forms like "who are the", "list the", "what are the"):
  * Extract ALL matching items from ALL chunks
  * Do NOT stop after finding the first match
  * Read every chunk completely before responding
  * Count items: if query asks for "co-founders" and you find 4, include all 4 in "items" array
  * If you find 10 services, include all 10 - partial extraction is INCORRECT

EXAMPLES OF CORRECT vs INCORRECT:

Query: "who are the co-founders of TechCorp?"
WRONG (Partial Extraction - INCORRECT):
{
  "items": ["John Smith"]  // ❌ Only 1 of 4 co-founders - WRONG!
}

CORRECT (Complete Extraction):
{
  "answer_type": "entities",
  "items": ["John Smith", "Sarah Jones", "Mike Brown", "Alice White"],  // ✅ All 4 co-founders
  "text": "",
  "chunks_used": [1, 2, 3]
}

Query: "list the features of ProductX"
WRONG (Partial Extraction - INCORRECT):
{
  "items": ["feature1", "feature2"]  // ❌ Only 2 of 5 features - WRONG!
}

CORRECT (Complete Extraction):
{
  "answer_type": "list",
  "items": ["feature1", "feature2", "feature3", "feature4", "feature5"],  // ✅ All 5 features
  "text": "",
  "chunks_used": [1, 3, 4]
}

CRITICAL RULE #2: ANSWER TYPE SELECTION
Match the query pattern to determine answer_type:

- "how are X and Y related?" → answer_type: "relationship"
- "what is the connection between X and Y?" → answer_type: "relationship"
- "why did X [action]?" → answer_type: "analytical"
- "what caused X to [action]?" → answer_type: "analytical"
- "what is the difference between X and Y?" → answer_type: "comparison"
- "compare X and Y" → answer_type: "comparison"
- "how does [process] work?" → answer_type: "process"
- "what is the process for X?" → answer_type: "process"
- "who are the [role] of X?" → answer_type: "entities"
- "list the [items] of X" → answer_type: "list"

CRITICAL RULE #3: ROLE FILTERING
- If query asks for "co-founders", extract ONLY people labeled as "Co-Founder"
- Do NOT include CEOs, CTOs, or other roles even if they're at the same company
- Exact role match required: "co-founders" ≠ "CEO" ≠ "CTO"

EXAMPLES:

Query: "what is the difference between CompanyA and CompanyB?"
Expected JSON:
{
  "answer_type": "comparison",
  "items": [],
  "text": "CompanyA focuses on innovation strategy, while CompanyB emphasizes different aspects of the same domain.",
  "chunks_used": [2, 3]
}

Query: "who are the managers of UnknownCorp?"
Expected JSON:
{
  "answer_type": "not_found",
  "items": [],
  "text": "I don't have that information in the provided documents",
  "chunks_used": []
}

IMPORTANT:
- Output ONLY valid JSON - no explanation, no markdown, no code blocks
- Extract ALL items - partial extraction is incorrect and will be penalized
- Read ALL chunks before responding - don't stop after first chunk
- Use exact information from chunks - never modify or infer
- Match answer_type to query pattern - don't default to "comparison"
"""

# ============================================================================
# Realistic Content Generation (same as v2)
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
# Realistic Sentence Generation (same as v2)
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
    """Generate realistic contextual sentence (same as v2)"""
    business_contexts = [
        "Market analysts have observed significant shifts in consumer behavior patterns over the past quarter.",
        "Industry reports indicate a growing trend toward digital transformation initiatives across multiple sectors.",
        "Economic indicators suggest a period of sustained growth in technology investments.",
        "Regulatory changes in the financial sector have prompted organizations to reassess their compliance frameworks.",
        "Global supply chain disruptions have accelerated the adoption of alternative sourcing strategies.",
        "Customer feedback surveys reveal increasing demand for personalized service experiences.",
        "Competitive analysis shows emerging players entering the market with innovative business models.",
        "Investment patterns demonstrate a strong preference for sustainable and ESG-compliant ventures.",
        "The quarterly review process identified several areas requiring immediate attention and resource allocation.",
        "Cross-functional teams have been collaborating on process improvement initiatives throughout the organization.",
        "Performance metrics indicate steady progress toward achieving the annual strategic objectives.",
        "Stakeholder meetings have been scheduled to discuss upcoming initiatives and resource requirements.",
        "Internal audits revealed opportunities for streamlining operational workflows and reducing overhead costs.",
        "The leadership team has been evaluating potential partnerships to expand market reach.",
        "Training programs have been implemented to enhance employee skills in emerging technologies.",
        "Budget allocations reflect a strategic shift toward innovation and research development.",
        "Recent technological advancements have opened new possibilities for product development and service delivery.",
        "The IT department has been upgrading infrastructure to support increased computational demands.",
        "Data analytics capabilities have been enhanced to provide more actionable business insights.",
        "Cloud migration projects are progressing according to schedule with minimal disruption to operations.",
        "Security protocols have been strengthened in response to evolving cybersecurity threats.",
        "Integration of artificial intelligence tools has improved efficiency in routine operational tasks.",
        "Mobile application development has become a priority to meet changing customer expectations.",
        "API integrations have enabled seamless data exchange between different business systems.",
        "Long-term strategic planning sessions have been conducted to align organizational goals with market opportunities.",
        "Scenario analysis has been performed to evaluate potential outcomes under different market conditions.",
        "Risk assessment frameworks have been updated to account for emerging business challenges.",
        "Portfolio diversification strategies are being considered to mitigate potential market volatility.",
        "Expansion into new geographic markets requires careful evaluation of regulatory and cultural factors.",
        "Brand positioning strategies have been refined to better communicate value propositions to target audiences.",
        "Customer segmentation analysis has identified new opportunities for targeted marketing campaigns.",
        "Product lifecycle management processes have been optimized to reduce time-to-market for new offerings.",
        "Revenue streams have diversified over the past year, reducing dependence on traditional income sources.",
        "Cost optimization initiatives have resulted in improved profit margins without compromising service quality.",
        "Cash flow management has been strengthened through improved forecasting and collection processes.",
        "Vendor relationships have been renegotiated to secure more favorable terms and pricing structures.",
        "Asset utilization rates have improved following the implementation of predictive maintenance programs.",
        "Working capital management has been optimized to support growth initiatives while maintaining liquidity.",
        "Financial reporting systems have been upgraded to provide real-time visibility into business performance.",
        "Investment in employee development programs has shown positive returns in productivity and retention.",
        "Customer satisfaction scores have remained consistently high despite increased service volume.",
        "Market research indicates growing awareness of the brand among target demographic segments.",
        "Sales pipeline analysis reveals strong potential for revenue growth in the coming quarters.",
        "Customer retention strategies have been successful in maintaining long-term relationships.",
        "Product launch campaigns have generated significant interest and early adoption rates.",
        "Customer support operations have been expanded to handle increased inquiry volumes.",
        "Market penetration strategies are being developed for underserved geographic regions.",
        "Customer journey mapping has identified key touchpoints for improving overall experience.",
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
# Query Templates
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
    
    # Replace placeholders (same as v2)
    if "{role}" in query:
        role = random.choice(["leaders", "members", "directors", "managers", "executives", "founders", "co-founders"])
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
        action = random.choice(["expand", "restructure", "diversify", "grow", "change", "improve"])
        query = query.replace("{action}", action)
        context["action"] = action
    if "{event}" in query:
        event = random.choice(["the expansion", "the merger", "the change", "the restructuring"])
        query = query.replace("{event}", event)
        context["event"] = event
    if "{process}" in query:
        process = random.choice(["the system", "the workflow", "the framework", "the platform", "the process"])
        query = query.replace("{process}", process)
        context["process"] = process
    
    return query, context

# ============================================================================
# Chunk Generation (same logic as v2)
# ============================================================================

def create_realistic_chunk(relevant_sentences: List[str], irrelevant_sentences: List[str], 
                          num_sentences: int = 7) -> str:
    """Create realistic chunk with 6-8 sentences (same as v2)"""
    if relevant_sentences:
        num_relevant = random.randint(1, min(3, len(relevant_sentences)))
        selected_relevant = random.sample(relevant_sentences, num_relevant)
    else:
        num_relevant = 0
        selected_relevant = []
    
    num_irrelevant = num_sentences - num_relevant
    
    if irrelevant_sentences:
        num_to_select = min(num_irrelevant, len(irrelevant_sentences))
        selected_irrelevant = random.sample(irrelevant_sentences, num_to_select)
    else:
        selected_irrelevant = []
    
    all_sentences = selected_relevant + selected_irrelevant
    used_contextual = set()
    max_attempts = 100
    
    while len(all_sentences) < num_sentences:
        attempts = 0
        contextual = generate_contextual_sentence()
        while contextual in used_contextual and attempts < max_attempts:
            contextual = generate_contextual_sentence()
            attempts += 1
        used_contextual.add(contextual)
        all_sentences.append(contextual)
    
    contextual_only = [s for s in all_sentences if s not in selected_relevant and s not in selected_irrelevant]
    mixed = selected_relevant + selected_irrelevant
    
    # Natural flow: contextual -> mixed -> contextual
    if contextual_only:
        start_contextual = random.sample(contextual_only, min(1, len(contextual_only)))
        remaining_contextual = [s for s in contextual_only if s not in start_contextual]
        end_contextual = random.sample(remaining_contextual, min(1, len(remaining_contextual))) if remaining_contextual else []
    else:
        start_contextual = []
        end_contextual = []
    
    chunk_sentences = start_contextual + mixed + end_contextual
    
    # Fill to target if needed
    while len(chunk_sentences) < num_sentences:
        chunk_sentences.append(generate_contextual_sentence())
    
    return " ".join(chunk_sentences[:num_sentences])

# ============================================================================
# JSON Response Generation (NEW - replaces natural language)
# ============================================================================

def generate_json_response(query: str, relevant_info: List[str], query_type: str, context: Dict[str, Any], 
                          chunks_used: List[int]) -> str:
    """Generate JSON response from relevant information"""
    
    if not relevant_info:
        return json.dumps({
            "answer_type": "not_found",
            "items": [],
            "text": "I don't have that information in the provided documents",
            "chunks_used": []
        }, ensure_ascii=False)
    
    if query_type in ["entity", "list"]:
        # Extract items (names, features, etc.)
        items = [info.strip() for info in relevant_info if info.strip()]
        answer_type = "entities" if query_type == "entity" else "list"
        
        return json.dumps({
            "answer_type": answer_type,
            "items": items,
            "text": "",
            "chunks_used": chunks_used
        }, ensure_ascii=False)
    
    elif query_type == "comparison":
        # Join comparison sentences
        if len(relevant_info) >= 2:
            sentences = [info.strip().rstrip('.') for info in relevant_info]
            if any("while" in s.lower() for s in sentences):
                text = ". ".join(sentences) + "."
            else:
                text = f"{sentences[0]}. {sentences[1]}."
        else:
            text = " ".join([info.strip().rstrip('.') for info in relevant_info])
        
        return json.dumps({
            "answer_type": "comparison",
            "items": [],
            "text": text,
            "chunks_used": chunks_used
        }, ensure_ascii=False)
    
    elif query_type == "analytical":
        # Format with reasoning
        text = " ".join([info.strip() for info in relevant_info])
        if "because" not in text.lower() and "due to" not in text.lower():
            text = f"because {text}"
        
        return json.dumps({
            "answer_type": "analytical",
            "items": [],
            "text": text,
            "chunks_used": chunks_used
        }, ensure_ascii=False)
    
    elif query_type == "relationship":
        text = " ".join([info.strip() for info in relevant_info])
        return json.dumps({
            "answer_type": "relationship",
            "items": [],
            "text": text,
            "chunks_used": chunks_used
        }, ensure_ascii=False)
    
    elif query_type == "process":
        text = " ".join([info.strip() for info in relevant_info])
        return json.dumps({
            "answer_type": "process",
            "items": [],
            "text": text,
            "chunks_used": chunks_used
        }, ensure_ascii=False)
    
    else:
        # Default
        text = " ".join([info.strip() for info in relevant_info])
        return json.dumps({
            "answer_type": "list",
            "items": [],
            "text": text,
            "chunks_used": chunks_used
        }, ensure_ascii=False)

# ============================================================================
# Example Generation (adapted from v2)
# ============================================================================

def generate_example(pattern_type: str) -> Dict[str, Any]:
    """Generate a training example (adapted from v2 with JSON output)"""
    
    # Select query template based on pattern
    if pattern_type in ["multi_chunk", "role_filtering"]:
        # Prioritize entity/list queries for multi-entity extraction
        if random.random() < 0.7:
            template = random.choice([t for t in QUERY_TEMPLATES if t["type"] in ["entity", "list"]])
        else:
            template = random.choice(QUERY_TEMPLATES)
    else:
        template = random.choice(QUERY_TEMPLATES)
    
    query, context = generate_query(template)
    query_type = template["type"]
    
    # Generate relevant information based on query type
    relevant_info = []
    irrelevant_info = []
    
    if query_type == "entity":
        role = context.get("role", "leaders")
        company = context.get("company", generate_random_company())
        
        # ENHANCED: Generate 3-4 entities for multi-entity extraction
        if pattern_type in ["multi_chunk", "role_filtering"]:
            num_relevant_items = random.randint(3, 4)
        else:
            num_relevant_items = random.randint(1, 3)
        
        for _ in range(num_relevant_items):
            name = generate_random_name()
            relevant_info.append(name)
            irrelevant_info.append(generate_entity_sentence(name, "CEO", generate_random_company()))
    
    elif query_type == "list":
        items = context.get("items", "features")
        entity = context.get("entity") or context.get("company") or generate_random_company()
        
        # ENHANCED: Generate 3-4 items for complete extraction
        if pattern_type in ["multi_chunk", "role_filtering"]:
            num_relevant_items = random.randint(3, 4)
        else:
            num_relevant_items = random.randint(2, 4)
        
        for _ in range(num_relevant_items):
            if items in ["features", "benefits", "components", "advantages", "capabilities", "services"]:
                item = f"{items[:-1] if items.endswith('s') else items} {random.randint(1, 100)}"
                relevant_info.append(item)
            else:
                relevant_info.append(f"{items} item {random.randint(1, 100)}")
    
    elif query_type == "comparison":
        entity1 = context.get("entity1", generate_random_company())
        entity2 = context.get("entity2", generate_random_company())
        attribute = random.choice(["innovation strategy", "pricing strategy", "market approach", "customer focus", "technology stack"])
        relevant_info.append(generate_comparison_sentence(entity1, entity2, attribute))
    
    elif query_type == "analytical":
        entity = context.get("entity", generate_random_company())
        action = context.get("action", "expand")
        reasons = ["competitive pressures", "regulatory changes", "customer feedback", "technological advancements", "increasing market demand", "strategic opportunities"]
        reason = random.choice(reasons)
        relevant_info.append(generate_analytical_sentence(entity, action, reason))
    
    elif query_type == "relationship":
        entity1 = context.get("entity1", generate_random_company())
        entity2 = context.get("entity2", generate_random_company())
        relationships = ["strategic partnership", "joint venture", "supplier relationship", "customer relationship", "technology alliance"]
        relationship = random.choice(relationships)
        relevant_info.append(generate_relationship_sentence(entity1, entity2, relationship))
    
    elif query_type == "process":
        process = context.get("process", "the system")
        relevant_info.append(generate_contextual_sentence())
        relevant_info.append(generate_contextual_sentence())
    
    # CRITICAL FIX: For "not_found" pattern, clear relevant_info so chunks don't contain matching entities
    # This ensures model learns: "if entities are in chunks, extract them; if not, say not_found"
    if pattern_type == "not_found":
        relevant_info = []  # Clear relevant info - chunks should NOT contain matching entities
        chunks_used = []     # No chunks should be marked as used
    
    # Generate chunks with relevant info distributed
    num_chunks = random.randint(3, 5)
    chunks = []
    if pattern_type != "not_found":
        chunks_used = []  # Only initialize if not already cleared above
    else:
        chunks_used = []  # Already cleared above, but ensure it's empty
    
    if pattern_type == "multi_chunk" and query_type in ["entity", "list"]:
        # CRITICAL FIX: Scatter items across multiple chunks, ensuring ALL items are included
        # This forces model to read ALL chunks to get complete answer
        # Track which items have been added to chunks
        items_added_to_chunks = []
        
        # Generate irrelevant entities/items to teach filtering
        irrelevant_entities = []
        if query_type == "entity":
            role = context.get("role", "leaders")
            company = context.get("company", generate_random_company())
            # Generate 2-4 irrelevant entities (different role or different company)
            num_irrelevant = random.randint(2, 4)
            for _ in range(num_irrelevant):
                irrelevant_name = generate_random_name()
                # Use different role or different company to make them irrelevant
                if random.random() < 0.5:
                    # Different role, same company (irrelevant)
                    different_role = random.choice(["CEO", "CTO", "CFO", "VP", "Director", "Head of Engineering"])
                    irrelevant_entities.append((irrelevant_name, different_role, company))
                else:
                    # Same role, different company (irrelevant)
                    different_company = generate_random_company()
                    irrelevant_entities.append((irrelevant_name, role, different_company))
        elif query_type == "list":
            items = context.get("items", "features")
            entity = context.get("entity") or context.get("company") or generate_random_company()
            # Generate 2-4 irrelevant items (different entity)
            num_irrelevant = random.randint(2, 4)
            different_entity = generate_random_company()
            for _ in range(num_irrelevant):
                if items in ["features", "benefits", "components", "advantages", "capabilities", "services"]:
                    irrelevant_item = f"{items[:-1] if items.endswith('s') else items} {random.randint(100, 200)}"
                else:
                    irrelevant_item = f"{items} item {random.randint(100, 200)}"
                irrelevant_entities.append((irrelevant_item, different_entity))
        
        # CRITICAL: Distribute relevant items across chunks - don't put all in first chunk
        # This ensures model must read multiple chunks to get complete answer
        # Minimum 2 chunks must have entities (forces multi-chunk reading)
        if len(relevant_info) >= 2:
            # Ensure at least 2 chunks get entities
            chunks_with_entities = min(num_chunks, max(2, len(relevant_info)))
            items_per_chunk = max(1, len(relevant_info) // chunks_with_entities)
            
            for i, chunk_idx in enumerate(range(num_chunks)):
                chunk_relevant = []
                
                # Calculate which items go in this chunk
                if i < chunks_with_entities and len(relevant_info) > 0:
                    start_idx = i * items_per_chunk
                    end_idx = min(start_idx + items_per_chunk, len(relevant_info)) if i < chunks_with_entities - 1 else len(relevant_info)
                    
                    if start_idx < len(relevant_info):
                        chunk_relevant = relevant_info[start_idx:end_idx]
                        items_added_to_chunks.extend(chunk_relevant)
                        if chunk_relevant:
                            chunks_used.append(chunk_idx + 1)
                else:
                    # Later chunks may not have entities (but model should still check)
                    chunk_relevant = []
        else:
            # Fallback: if only 1 entity, still distribute across 2 chunks to force reading
            for i, chunk_idx in enumerate(range(min(2, num_chunks))):
                if i == 0 and len(relevant_info) > 0:
                    chunk_relevant = relevant_info[:1]
                    items_added_to_chunks.extend(chunk_relevant)
                    chunks_used.append(chunk_idx + 1)
                else:
                    chunk_relevant = []
            
            # Add irrelevant entities/items to this chunk (teach filtering)
            chunk_irrelevant_sentences = [generate_contextual_sentence() for _ in range(3)]  # Fewer generic sentences
            
            # Create sentences for this chunk
            chunk_sentences = []
            if query_type == "entity":
                role = context.get("role", "leaders")
                company = context.get("company", generate_random_company())
                # Add relevant entities
                for name in chunk_relevant:
                    chunk_sentences.append(generate_entity_sentence(name, role, company))
                # Add irrelevant entities (different role/company)
                for irrelevant_name, irrelevant_role, irrelevant_company in irrelevant_entities[:random.randint(1, 2)]:
                    chunk_sentences.append(generate_entity_sentence(irrelevant_name, irrelevant_role, irrelevant_company))
            elif query_type == "list":
                items = context.get("items", "features")
                entity = context.get("entity") or context.get("company") or generate_random_company()
                # Add relevant items
                for item in chunk_relevant:
                    chunk_sentences.append(generate_company_feature_sentence(entity, item))
                # Add irrelevant items (different entity)
                for irrelevant_item, irrelevant_entity in irrelevant_entities[:random.randint(1, 2)]:
                    chunk_sentences.append(generate_company_feature_sentence(irrelevant_entity, irrelevant_item))
            
            chunk_text = create_realistic_chunk(chunk_sentences, chunk_irrelevant_sentences)
            relevance_score = 0.85 if chunk_relevant else random.uniform(0.3, 0.6)
            
            chunks.append({
                "text": chunk_text,
                "score": round(relevance_score, 2),
                "file": "document.pdf"
            })
        
        # CRITICAL FIX: Only include items that were actually added to chunks in expected answer
        relevant_info = items_added_to_chunks
    else:
        # Standard distribution (single chunk or non-entity/list queries)
        # Track which items are actually added to chunks
        items_added_to_chunks = []
        
        # Generate irrelevant entities/items to teach filtering
        irrelevant_entities = []
        if query_type == "entity" and relevant_info:
            role = context.get("role", "leaders")
            company = context.get("company", generate_random_company())
            # Generate 2-3 irrelevant entities
            num_irrelevant = random.randint(2, 3)
            for _ in range(num_irrelevant):
                irrelevant_name = generate_random_name()
                if random.random() < 0.5:
                    different_role = random.choice(["CEO", "CTO", "CFO", "VP", "Director"])
                    irrelevant_entities.append((irrelevant_name, different_role, company))
                else:
                    different_company = generate_random_company()
                    irrelevant_entities.append((irrelevant_name, role, different_company))
        elif query_type == "list" and relevant_info:
            items = context.get("items", "features")
            entity = context.get("entity") or context.get("company") or generate_random_company()
            num_irrelevant = random.randint(2, 3)
            different_entity = generate_random_company()
            for _ in range(num_irrelevant):
                if items in ["features", "benefits", "components", "advantages", "capabilities", "services"]:
                    irrelevant_item = f"{items[:-1] if items.endswith('s') else items} {random.randint(100, 200)}"
                else:
                    irrelevant_item = f"{items} item {random.randint(100, 200)}"
                irrelevant_entities.append((irrelevant_item, different_entity))
        
        for i in range(num_chunks):
            # FIXED: Put ALL relevant items in first chunk (or distribute if many items)
            if i == 0 and relevant_info:
                chunk_relevant = relevant_info  # Put ALL items in first chunk
                items_added_to_chunks.extend(relevant_info)
                chunks_used.append(i + 1)
            else:
                chunk_relevant = []
            
            chunk_irrelevant_sentences = [generate_contextual_sentence() for _ in range(3)]
            
            chunk_sentences = []
            if query_type == "entity" and chunk_relevant:
                role = context.get("role", "leaders")
                company = context.get("company", generate_random_company())
                # Add ALL relevant entities
                for name in chunk_relevant:
                    chunk_sentences.append(generate_entity_sentence(name, role, company))
                # Add irrelevant entities to teach filtering
                for irrelevant_name, irrelevant_role, irrelevant_company in irrelevant_entities[:random.randint(1, 2)]:
                    chunk_sentences.append(generate_entity_sentence(irrelevant_name, irrelevant_role, irrelevant_company))
            elif query_type == "list" and chunk_relevant:
                items = context.get("items", "features")
                entity = context.get("entity") or context.get("company") or generate_random_company()
                # Add ALL relevant items
                for item in chunk_relevant:
                    chunk_sentences.append(generate_company_feature_sentence(entity, item))
                # Add irrelevant items to teach filtering
                for irrelevant_item, irrelevant_entity in irrelevant_entities[:random.randint(1, 2)]:
                    chunk_sentences.append(generate_company_feature_sentence(irrelevant_entity, irrelevant_item))
            elif chunk_relevant:
                chunk_sentences = chunk_relevant
            
            chunk_text = create_realistic_chunk(chunk_sentences, chunk_irrelevant_sentences)
            relevance_score = 0.85 if chunk_relevant else random.uniform(0.3, 0.6)
            
            chunks.append({
                "text": chunk_text,
                "score": round(relevance_score, 2),
                "file": "document.pdf"
            })
        
        # CRITICAL FIX: Only include items that were actually added to chunks in expected answer
        if items_added_to_chunks:
            relevant_info = items_added_to_chunks
    
    # Generate JSON response
    # Note: relevant_info has been updated to only include items actually added to chunks
    response = generate_json_response(query, relevant_info, query_type, context, chunks_used)
    
    # Format as training example
    user_content = f"Query: {query}\n\n"
    user_content += "RAG Chunks:\n"
    for i, chunk in enumerate(chunks, 1):
        user_content += f"[Chunk {i}] Score: {chunk['score']:.2f}, File: {chunk['file']}\n"
        user_content += f"FULL CHUNK TEXT: '{chunk['text']}'\n\n"
    
    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_JSON},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": response}
        ]
    }
    
    return example

# ============================================================================
# Main Generation
# ============================================================================

def main():
    """Generate 6250 training examples with JSON output format"""
    
    print("="*80)
    print("RAG Analysis Dataset Generator - JSON Output Version (Optimized)")
    print("="*80)
    print()
    print("Generating 6250 training examples with JSON output format...")
    print("✅ System prompt: Simplified instructions focusing on JSON extraction")
    print("✅ Assistant response: JSON object with extracted information")
    print("✅ Post-processing can convert JSON to natural language")
    print()
    
    # Pattern distribution (ENHANCED for better multi-entity extraction)
    patterns = {
        "mixed_content": 700,
        "multi_chunk": 2500,       # INCREASED from 1500 - emphasize multi-entity extraction (40% more)
        "role_filtering": 1200,    # Emphasize entity list queries
        "cross_entity": 800,
        "synthesis": 550,
        "not_found": 550,
        "comparison": 400,
        "relationship": 400,
        "analytical": 150,
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
    output_file = "rag_analysis_dataset_v3_json.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print()
    print("="*80)
    print("✅ DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"Total examples: {len(dataset)}")
    print(f"Output file: {output_file}")
    print()
    print("✅ Format: Simplified system prompt + JSON output in assistant response")
    print("✅ JSON format makes extraction easier for model to learn")
    print()
    
    # Verify format
    json_valid = 0
    json_invalid = 0
    for i, example in enumerate(dataset[:100]):  # Check first 100
        messages = example.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                try:
                    json.loads(content)
                    json_valid += 1
                except:
                    json_invalid += 1
    
    print("Format Verification (first 100 examples):")
    print(f"  ✅ Valid JSON: {json_valid}")
    print(f"  ❌ Invalid JSON: {json_invalid} (should be 0)")
    if json_invalid == 0:
        print("  ✅ Dataset format is CORRECT!")
    else:
        print("  ⚠️  WARNING: Some examples have invalid JSON!")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
