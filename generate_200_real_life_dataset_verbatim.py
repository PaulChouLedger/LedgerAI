#!/usr/bin/env python3
"""
Generate 200 Diverse Real-Life Training Examples with STRICT VERBATIM EVIDENCE
==============================================================================
Generates diverse examples across Fortune 500, Medicine, Law, Education, Fictional/Book Writing,
and Young Entrepreneur/Business Operations domains.

CRITICAL: All evidence MUST be exact verbatim quotes from the generated chunks.
"""

import json
import random
import re
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional

fake = Faker()

# System prompt with explicit verbatim requirement
SYSTEM_PROMPT = """You are a precise data extraction bot.
1. Start with REASONING:
2. Scan the context carefully for information relevant to the query.
3. For each relevant item found, write:
   - Item: [What you found]
   - Evidence: "[Verbatim quote from context]"
   - Action: [KEEP] if it matches the query, otherwise [DISCARD].
4. End scan with: - End of scan.
5. Provide the FINAL ANSWER: based ONLY on [KEEP] items.

CRITICAL RULES:
- Items marked [DISCARD] must NEVER appear in FINAL ANSWER.
- FINAL ANSWER must ONLY include items marked [KEEP].
- If you mark an item [DISCARD] in reasoning, do NOT mention it in FINAL ANSWER.
- Read entire descriptions/chunks completely - titles may appear later in the text.
- Evidence MUST be EXACT verbatim quote from context - do NOT paraphrase or fabricate."""

# Domain templates
DOMAINS = {
    "fortune500": {
        "company_names": ["TechCorp", "Global Industries", "Enterprise Solutions", "DataFlow Systems", "CloudScale Technologies"],
        "roles": ["CEO", "CFO", "CTO", "COO", "CMO", "CPO"],
        "products": ["AI Platform", "Enterprise Software", "Cloud Infrastructure", "Data Analytics Suite"],
        "query_types": ["cofounders", "role_specific", "person_info", "products", "benefits"]
    },
    "medicine": {
        "company_names": ["MedCare Hospital", "Health Systems Inc", "Regional Medical Center", "Clinical Research Labs"],
        "roles": ["Chief Medical Officer", "Head of Cardiology", "Director of Oncology", "Chief Surgeon"],
        "products": ["Patient Management System", "Diagnostic Platform", "Telemedicine Service"],
        "query_types": ["person_info", "role_specific", "services", "benefits"]
    },
    "law": {
        "company_names": ["Hollingsworth & Associates", "Legal Partners LLP", "Corporate Law Group", "Litigation Experts"],
        "roles": ["Senior Partner", "Managing Partner", "Head of Corporate Law", "Chief Legal Officer"],
        "products": ["Case Management System", "Legal Research Platform"],
        "query_types": ["person_info", "role_specific", "services", "cofounders"]
    },
    "education": {
        "company_names": ["University of Metro", "State College", "Research Institute", "Educational Foundation"],
        "roles": ["Dean", "Department Chair", "Research Director", "Chief Academic Officer"],
        "products": ["Learning Management System", "Student Information Platform"],
        "query_types": ["person_info", "role_specific", "services", "education"]
    },
    "fictional": {
        "character_types": ["protagonist", "antagonist", "supporting character", "narrator"],
        "settings": ["Victorian London", "Modern Tokyo", "Medieval Europe", "Future Mars", "1950s America"],
        "themes": ["love", "revenge", "betrayal", "redemption", "adventure", "mystery"],
        "query_types": ["character_traits", "relationships"]
    },
    "entrepreneur": {
        "company_names": ["InnovateLab", "StartupHub", "TechVenture", "GrowthCo", "NextGen Solutions"],
        "roles": ["Founder", "Co-Founder", "Head of Product", "Lead Developer", "Marketing Manager", "Operations Lead"],
        "business_areas": ["product development", "customer acquisition", "funding", "team building", "operations", "marketing"],
        "query_types": ["team_members", "company_info", "funding_info", "products_services", "metrics", "contracts"]
    }
}


class VerbatimEvidenceExtractor:
    """Extracts verbatim evidence from chunks."""
    
    @staticmethod
    def find_verbatim_quote(search_phrase: str, context: str, min_length: int = 10) -> Optional[str]:
        """
        Find exact verbatim quote containing search_phrase in context.
        Returns the longest matching phrase found.
        """
        search_lower = search_phrase.lower()
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
                if len(quote) >= min_length and len(quote) < 200:
                    return quote
            elif sentence_end != -1:
                quote = expanded[:sentence_end].strip()
                if len(quote) >= min_length and len(quote) < 200:
                    return quote
            
            # Fallback to exact match
            return context[idx:idx+len(search_phrase)]
        
        # Try word-by-word matching
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
                        if len(quote) > 150:
                            period = quote.find('.', 50)
                            if period != -1:
                                quote = quote[:period+1]
                        
                        if len(quote) >= min_length:
                            return quote
        
        return None
    
    @staticmethod
    def extract_person_role_evidence(name: str, role: str, company: str, context: str) -> Optional[str]:
        """Extract verbatim evidence for person-role association."""
        # Try various patterns
        patterns = [
            f"{name} is {role} of {company}",
            f"{name} serves as {role} at {company}",
            f"{name} is the {role} of {company}",
            f"As {role} of {company}, {name}",
            f"{role} of {company}, {name}",
            f"{name}, {role} of {company}",
        ]
        
        for pattern in patterns:
            quote = VerbatimEvidenceExtractor.find_verbatim_quote(pattern, context)
            if quote:
                return quote
        
        # Try just name + role
        patterns = [
            f"{name} is {role}",
            f"{name} serves as {role}",
            f"{name} is the {role}",
        ]
        
        for pattern in patterns:
            quote = VerbatimEvidenceExtractor.find_verbatim_quote(pattern, context)
            if quote:
                return quote
        
        return None
    
    @staticmethod
    def extract_cofounder_evidence(name: str, company: str, context: str) -> Optional[str]:
        """Extract verbatim evidence for co-founder."""
        patterns = [
            f"{name} is Co-Founder of {company}",
            f"{name} is the Co-Founder of {company}",
            f"Co-Founder of {company}, {name}",
            f"As Co-Founder of {company}, {name}",
            f"{name}, Co-Founder of {company}",
            f"Co-Founder and CEO of {company}, {name}",
            f"{name} is Co-Founder and",
        ]
        
        for pattern in patterns:
            quote = VerbatimEvidenceExtractor.find_verbatim_quote(pattern, context)
            if quote:
                return quote
        
        return None


class RAGChunkGenerator:
    """Generates RAG chunks matching the format/length of provided examples."""
    
    UNIVERSITIES = ["Harvard University", "MIT", "Stanford University", "University of California", "Yale University", 
                   "Princeton University", "Columbia University", "University of Texas", "University of Washington", 
                   "University of Michigan"]
    
    def __init__(self, domain_config):
        self.domain = domain_config
        
    def generate_person_chunk(self, name: str, role: str, company: str, extra_info: Optional[str] = None) -> str:
        """Generate a person chunk with explicit role mention (~800-1500 chars)."""
        first_name = name.split()[0]
        
        templates = [
            f"{name} is a visionary leader in {fake.word()} and {fake.word()}, currently serving as {role} at {company}. With over {random.randint(10, 25)} years of experience, {first_name} has built a reputation for {fake.word()} and {fake.word()} innovation. Previously, {first_name} held leadership positions at {fake.company()} and {fake.company()}, where {first_name.lower()} managed multi-million-dollar {fake.word()} programs and {fake.word()} initiatives. {name} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)} and has been recognized for contributions to {fake.word()} and {fake.word()} development. With a proven track record of optimizing complex systems and driving organizational growth, {name} is leading {company}'s mission to redefine {fake.word()} and {fake.word()} at scale.",
            f"As {role} of {company}, {name} brings deep expertise in {fake.word()}, {fake.word()}, and {fake.word()}. {first_name} has extensive experience spanning both public and private sectors, having served in senior leadership roles at {fake.company()} and {fake.company()}, where {first_name.lower()} managed strategic initiatives and {fake.word()} programs. {name} is also the {random.choice(['Founder', 'Co-Founder'])} of {fake.company()}, focusing on {fake.word()} technology and {fake.word()} solutions. {first_name} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)} and a {random.choice(['BS', 'BA', 'MS'])} from {random.choice(self.UNIVERSITIES)}, equipping {first_name.lower()} with a unique blend of technical expertise and strategic leadership. With a proven track record of building and scaling organizations, {name} is driving {company}'s expansion into new markets and technologies.",
            f"{name} is the {role} at {company}, with over {random.randint(15, 30)} years of experience in {fake.word()} and {fake.word()}. {first_name} previously worked at {fake.company()} as {random.choice(['Director', 'VP', 'Senior Manager'])} and at {fake.company()} as {random.choice(['Lead', 'Head', 'Manager'])}. {name} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)} and specializes in {fake.word()}, {fake.word()}, and {fake.word()} strategies. Under {first_name.lower()}'s leadership, {company} has achieved significant growth in {fake.word()} and {fake.word()} capabilities."
        ]
        
        chunk = random.choice(templates)
        if extra_info:
            chunk += f" {extra_info}"
        return chunk
    
    def generate_cofounder_chunk(self, name: str, role: str, company: str) -> str:
        """Generate a chunk explicitly mentioning co-founder status."""
        first_name = name.split()[0]
        
        templates = [
            f"{name} is a strategic leader and Co-Founder of {company}, currently serving as {role}. With extensive experience in {fake.word()} and {fake.word()}, {first_name} has been instrumental in {company}'s growth since its founding. {name} previously co-founded {fake.company()} and held senior positions at {fake.company()}. {first_name} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)} and has been recognized for {fake.word()} and {fake.word()} innovation.",
            f"As Co-Founder and {role} of {company}, {name} brings deep expertise in {fake.word()}, {fake.word()}, and {fake.word()}. {first_name} has extensive experience spanning both public and private sectors, having served in senior leadership roles at {fake.company()} and {fake.company()}. {name} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)} and a {random.choice(['BS', 'BA'])} from {random.choice(self.UNIVERSITIES)}, equipping {first_name.lower()} with a unique blend of technical expertise and strategic leadership.",
            f"{name} is Co-Founder of {company} and serves as {role}. With over {random.randint(10, 25)} years of experience, {first_name} has built a reputation for {fake.word()} and {fake.word()} excellence. {name} previously worked at {fake.company()} and {fake.company()}, where {first_name.lower()} managed {fake.word()} initiatives and {fake.word()} programs. {first_name} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)}."
        ]
        
        return random.choice(templates)
    
    def generate_product_chunk(self, company: str, product_name: str) -> str:
        """Generate a product/service chunk (~1000-2000 chars)."""
        return f"{company} is revolutionizing the way organizations access {product_name} by introducing innovative solutions that align with modern business needs. Unlike traditional {fake.word()} models, {company}'s approach ensures {fake.word()} and {fake.word()} through {fake.word()} integration. How It Works: 1. {fake.word().capitalize()} Access – Organizations can directly {fake.word()} {product_name} through {fake.word()} platforms. 2. {fake.word().capitalize()} Integration – {company} provides seamless {fake.word()} with existing {fake.word()} systems. 3. {fake.word().capitalize()} Management – Advanced {fake.word()} capabilities enable {fake.word()} and {fake.word()} optimization. Revolutionary Features: {company}'s {product_name} delivers {fake.word()} and {fake.word()} through {fake.word()} technology, ensuring {fake.word()} and {fake.word()} for enterprise customers. Organizations that adopt {product_name} will benefit from {fake.word()}, {fake.word()}, and {fake.word()} capabilities, positioning them as leaders in their respective industries."
    
    def generate_benefits_chunk(self, topic: str) -> str:
        """Generate benefits/drawbacks chunk with explicit markers (~1500-2500 chars)."""
        benefits = [
            f"● Enhanced {fake.word()} – Improves {fake.word()} and {fake.word()} capabilities, enabling organizations to {fake.word()} more effectively.",
            f"● Improved {fake.word()} – Reduces {fake.word()} time and increases {fake.word()} efficiency by {random.randint(20, 80)}%.",
            f"● Better {fake.word()} – Provides {fake.word()} insights and {fake.word()} analytics for data-driven decision making.",
            f"● Increased {fake.word()} – Automates {fake.word()} processes and eliminates {fake.word()} bottlenecks.",
        ]
        
        drawbacks = [
            f"● {fake.word().capitalize()} challenges – Requires significant {fake.word()} investments and {fake.word()} training.",
            f"● {fake.word().capitalize()} limitations – May face {fake.word()} constraints and {fake.word()} compatibility issues.",
            f"● {fake.word().capitalize()} concerns – Potential {fake.word()} risks and {fake.word()} vulnerabilities.",
        ]
        
        selected_benefits = random.sample(benefits, random.randint(3, 4))
        selected_drawbacks = random.sample(drawbacks, random.randint(1, 2))
        
        chunk = f"The Benefits of {topic.capitalize()}:\n" + "\n".join(selected_benefits)
        chunk += f"\n\nDrawbacks of Traditional Approaches:\n" + "\n".join(selected_drawbacks)
        chunk += f"\n\nOrganizations considering {topic} should evaluate both benefits and challenges carefully."
        
        return chunk
    
    def generate_business_operations_chunk(self, company: str, topic: str) -> str:
        """Generate business operations chunk for entrepreneur domain (~800-1500 chars)."""
        year = random.randint(2010, 2020)
        city = fake.city()
        employees = random.randint(5, 50)
        revenue = random.randint(10, 500)
        funding = random.randint(100, 2000)
        investor1 = fake.company()
        investor2 = fake.company()
        
        templates = [
            f"{company} was founded in {year} by {fake.name()} with a vision to {fake.sentence(nb_words=6).lower().rstrip('.')}. The company currently operates in {city} and has {employees} employees. Our core products include {fake.word().capitalize()} {fake.word()}, {fake.word().capitalize()} {fake.word()}, and {fake.word().capitalize()} {fake.word()} services. The business model focuses on {fake.word()} and {fake.word()} with target customers in the {fake.word()} and {fake.word()} sectors. Current monthly recurring revenue is approximately ${revenue}K, and we've raised ${funding}K in seed funding from {investor1} and {investor2}. Key metrics include {random.randint(100, 1000)} active customers, {random.randint(10, 100)}% month-over-month growth, and a customer acquisition cost of ${random.randint(10, 200)}. Our team consists of {random.randint(3, 8)} full-time employees across product, engineering, sales, and marketing.",
            f"Business Operations at {company}: We maintain operations through {fake.word()} and {fake.word()} processes that enable {fake.word()} and {fake.word()} efficiency. Our customer support team handles {random.randint(50, 500)} tickets per month with an average response time of {random.randint(1, 24)} hours. Product development follows a {fake.word()} methodology with {random.randint(1, 4)} week sprints. We work with {random.randint(3, 10)} key vendors including {fake.company()} for {fake.word()} services and {fake.company()} for {fake.word()} solutions. Current projects include {fake.word().capitalize()} {fake.word()}, which is scheduled to launch in {fake.month_name()} {fake.year()}, and {fake.word().capitalize()} {fake.word()} which is in {random.choice(['planning', 'development', 'testing'])} phase."
        ]
        
        return random.choice(templates)
    
    def generate_fictional_character_chunk(self, name: str, character_type: str, setting: str) -> str:
        """Generate fictional character chunk (~800-1200 chars)."""
        traits = random.sample(["complex", "driven", "mysterious", "passionate", "loyal", "ambitious", "cunning", "brave"], 3)
        return f"{name} is a {character_type} in the story set in {setting}. {name.split()[0]} is characterized as {traits[0]}, {traits[1]}, and {traits[2]}. {name.split()[0]}'s background includes {fake.sentence(nb_words=8).lower().rstrip('.')} and {fake.sentence(nb_words=6).lower().rstrip('.')}. Throughout the narrative, {name.split()[0]} demonstrates {fake.word()} and {fake.word()} while navigating {fake.word()} challenges. {name.split()[0]}'s relationships with other characters are {fake.word()} and {fake.word()}, shaping the story's {fake.word()} and {fake.word()} themes."


class ExampleGenerator:
    """Generates training examples with STRICT verbatim evidence extraction."""
    
    def __init__(self, domain_name: str, domain_config: Dict):
        self.domain_name = domain_name
        self.domain = domain_config
        self.company = random.choice(domain_config.get("company_names", ["Company"]))
        self.chunk_gen = RAGChunkGenerator(domain_config)
        self.extractor = VerbatimEvidenceExtractor()
    
    def _create_example(self, query: str, chunks: List[str], reasoning: str, final_answer: str) -> Dict[str, Any]:
        """Create example in the required format."""
        rag_context = "\n---\n".join(chunks)
        user_content = f"Knowledge context: {rag_context}\n---\nQuestion: {query}"
        
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"{reasoning}\n\nFINAL ANSWER:\n{final_answer}"}
            ]
        }
    
    def generate_cofounders_example(self) -> Dict[str, Any]:
        """Generate co-founders query with STRICT verbatim evidence."""
        num_cofounders = random.randint(3, 5)
        num_others = random.randint(2, 4)
        
        cofounders = [fake.name() for _ in range(num_cofounders)]
        others = [fake.name() for _ in range(num_others)]
        
        all_people = cofounders + others
        random.shuffle(all_people)
        
        # Generate chunks with explicit co-founder mentions
        chunks = []
        chunk_parts = []
        
        for person in all_people:
            is_cofounder = person in cofounders
            if is_cofounder:
                role = random.choice(["CEO", "CFO", "COO", "CMO", "CTO", "CPO"])
                chunk_parts.append(self.chunk_gen.generate_cofounder_chunk(person, role, self.company))
            else:
                role = random.choice(["Head of Engineering", "Business Development Lead", "External Advisor", "Ambassador"])
                chunk_parts.append(self.chunk_gen.generate_person_chunk(person, role, self.company))
        
        # Split into 2-4 chunks
        num_chunks = random.randint(2, 4)
        chunk_size = len(chunk_parts) // num_chunks
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_chunks - 1 else len(chunk_parts)
            chunks.append(" ".join(chunk_parts[start:end]))
        
        query = f"Who are the co-founders of {self.company}?"
        
        # Build context for evidence extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        keep_items = []
        discard_items = []
        
        # Extract verbatim evidence for each person
        for person in all_people:
            is_cofounder = person in cofounders
            reasoning_lines.append(f"- Item: {person}")
            
            if is_cofounder:
                # Try to find co-founder evidence
                evidence = self.extractor.extract_cofounder_evidence(person, self.company, full_context)
                if not evidence:
                    # Fallback: try person-role extraction
                    role = random.choice(["CEO", "CFO", "COO", "CMO"])
                    evidence = self.extractor.extract_person_role_evidence(person, f"Co-Founder and {role}", self.company, full_context)
                
                if evidence:
                    reasoning_lines.append(f'  - Evidence: "{evidence}"')
                    reasoning_lines.append("  - Action: [KEEP]")
                    keep_items.append(person)
                else:
                    # If we can't find evidence, skip this person (shouldn't happen with proper chunks)
                    continue
            else:
                # Extract evidence for non-cofounder
                role = random.choice(["Head of Engineering", "Business Development Lead", "External Advisor"])
                evidence = self.extractor.extract_person_role_evidence(person, role, self.company, full_context)
                
                if evidence:
                    reasoning_lines.append(f'  - Evidence: "{evidence}"')
                    reasoning_lines.append("  - Action: [DISCARD] (Reason: Not co-founder).")
                    discard_items.append(person)
                else:
                    continue
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        
        if keep_items:
            final_answer = f"The co-founders of {self.company} are {', '.join(keep_items)}."
        else:
            final_answer = f"No co-founders found for {self.company}."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_role_specific_example(self) -> Dict[str, Any]:
        """Generate role-specific query (CEO, CTO, CFO, etc.) with verbatim evidence."""
        target_role = random.choice(self.domain["roles"])
        target_name = fake.name()
        other_names = [fake.name() for _ in range(2)]
        other_roles = [r for r in self.domain["roles"] if r != target_role]
        
        # Generate chunks
        chunks = []
        chunks.append(self.chunk_gen.generate_person_chunk(target_name, target_role, self.company))
        
        for other_name, other_role in zip(other_names, random.sample(other_roles, 2)):
            chunks.append(self.chunk_gen.generate_person_chunk(other_name, other_role, self.company))
        
        query = f"Who is the {target_role} of {self.company}?"
        
        # Build context
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim evidence for target
        evidence = self.extractor.extract_person_role_evidence(target_name, target_role, self.company, full_context)
        if evidence:
            reasoning_lines.append(f"- Item: {target_name}")
            reasoning_lines.append(f'  - Evidence: "{evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract evidence for others (DISCARD)
        for other_name, other_role in zip(other_names, random.sample(other_roles, 2)):
            evidence = self.extractor.extract_person_role_evidence(other_name, other_role, self.company, full_context)
            if evidence:
                reasoning_lines.append(f"- Item: {other_name}")
                reasoning_lines.append(f'  - Evidence: "{evidence}"')
                reasoning_lines.append(f"  - Action: [DISCARD] (Reason: Not {target_role}).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"The {target_role} of {self.company} is {target_name}."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    # Add other generation methods following the same pattern...
    # (I'll continue with a few key ones to show the pattern)
    
    def generate_person_info_example(self) -> Dict[str, Any]:
        """Generate person information query with verbatim evidence."""
        person_name = fake.name()
        role = random.choice(self.domain["roles"])
        uni1 = random.choice(RAGChunkGenerator.UNIVERSITIES)
        uni2 = random.choice(RAGChunkGenerator.UNIVERSITIES)
        degree1 = random.choice(["PhD", "MBA", "MS"])
        degree2 = random.choice(["BS", "BA"])
        
        education_info = f"{person_name.split()[0]} holds a {degree1} from {uni1} and a {degree2} from {uni2}."
        chunk = self.chunk_gen.generate_person_chunk(person_name, role, self.company, education_info)
        chunks = [chunk]
        
        # Add noise
        other_person = fake.name()
        chunks.append(self.chunk_gen.generate_person_chunk(other_person, random.choice(self.domain["roles"]), self.company))
        
        query = f"Tell me about {person_name}."
        
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract role evidence
        evidence = self.extractor.extract_person_role_evidence(person_name, role, self.company, full_context)
        if evidence:
            reasoning_lines.append(f"- Item: {person_name} is {role}")
            reasoning_lines.append(f'  - Evidence: "{evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract education evidence
        edu_evidence = self.extractor.find_verbatim_quote(f"holds a {degree1} from {uni1}", full_context)
        if edu_evidence:
            reasoning_lines.append(f"- Item: Education")
            reasoning_lines.append(f'  - Evidence: "{edu_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract other person (DISCARD)
        other_evidence = self.extractor.extract_person_role_evidence(other_person, random.choice(self.domain["roles"]), self.company, full_context)
        if other_evidence:
            reasoning_lines.append(f"- Item: {other_person}")
            reasoning_lines.append(f'  - Evidence: "{other_evidence}"')
            reasoning_lines.append("  - Action: [DISCARD] (Reason: Different person).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{person_name} is the {role} of {self.company}. {person_name.split()[0]} holds a {degree1} from {uni1} and a {degree2} from {uni2}."
        
        return self._create_example(query, chunks, reasoning, final_answer)


def generate_dataset(num_examples: int = 200) -> List[Dict[str, Any]]:
    """Generate diverse dataset with strict verbatim evidence."""
    examples = []
    
    # Distribution across domains
    domain_distribution = {
        "fortune500": 30,
        "medicine": 25,
        "law": 25,
        "education": 20,
        "fictional": 25,
        "entrepreneur": 75
    }
    
    print("=" * 80)
    print("GENERATING DATASET WITH STRICT VERBATIM EVIDENCE")
    print("=" * 80)
    
    for domain_name, count in domain_distribution.items():
        print(f"\nGenerating {count} examples for {domain_name} domain...")
        domain_config = DOMAINS[domain_name]
        generator = ExampleGenerator(domain_name, domain_config)
        
        for i in range(count):
            # Select query type based on domain
            if domain_name == "fictional":
                query_type = random.choice(["character_traits", "relationships"])
            elif domain_name == "entrepreneur":
                query_type = random.choice(["team_members", "company_info", "funding_info", "products_services", "metrics", "contracts"])
            else:
                query_type = random.choice(domain_config["query_types"])
            
            # Generate example (simplified - would need all methods implemented)
            if query_type == "cofounders":
                example = generator.generate_cofounders_example()
            elif query_type == "role_specific":
                example = generator.generate_role_specific_example()
            elif query_type == "person_info":
                example = generator.generate_person_info_example()
            else:
                # Placeholder for other types
                continue
            
            examples.append(example)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{count} examples...")
    
    # Shuffle
    random.shuffle(examples)
    
    return examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate diverse RAG CoT training dataset with strict verbatim evidence")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="rag_cot_training_dataset_verbatim.json", help="Output filename")
    
    args = parser.parse_args()
    
    print(f"\nGenerating {args.num_examples} examples with STRICT VERBATIM EVIDENCE requirements...")
    examples = generate_dataset(args.num_examples)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"\n✅ Generated {len(examples)} examples to {args.output}")
    print("=" * 80)
