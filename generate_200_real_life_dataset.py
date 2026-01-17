#!/usr/bin/env python3
"""
Generate 200 Diverse Real-Life Training Examples
================================================
Generates diverse examples across Fortune 500, Medicine, Law, Education, Fictional/Book Writing,
and Young Entrepreneur/Business Operations domains.
"""

import json
import random
import re
from faker import Faker
from verbatim_evidence_helper import VerbatimEvidenceExtractor

fake = Faker()
extractor = VerbatimEvidenceExtractor()

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


class RAGChunkGenerator:
    """Generates RAG chunks matching the format/length of provided examples."""
    
    UNIVERSITIES = ["Harvard University", "MIT", "Stanford University", "University of California", "Yale University", "Princeton University", "Columbia University", "University of Texas", "University of Washington", "University of Michigan"]
    
    def __init__(self, domain_config):
        self.domain = domain_config
        
    def generate_person_chunk(self, name, role, company, extra_info=None):
        """Generate a person chunk (~800-1500 chars, matching example format)."""
        templates = [
            f"{name} is a visionary leader in {fake.word()} and {fake.word()}, currently serving as {role} at {company}. With over {random.randint(10, 25)} years of experience, {name.split()[0]} has built a reputation for {fake.word()} and {fake.word()} innovation. Previously, {name.split()[0]} held leadership positions at {fake.company()} and {fake.company()}, where {name.split()[0].lower()} managed multi-million-dollar {fake.word()} programs and {fake.word()} initiatives. {name} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)} and has been recognized for contributions to {fake.word()} and {fake.word()} development. With a proven track record of optimizing complex systems and driving organizational growth, {name} is leading {company}'s mission to redefine {fake.word()} and {fake.word()} at scale.",
            f"As {role} of {company}, {name} brings deep expertise in {fake.word()}, {fake.word()}, and {fake.word()}. {name.split()[0]} has extensive experience spanning both public and private sectors, having served in senior leadership roles at {fake.company()} and {fake.company()}, where {name.split()[0].lower()} managed strategic initiatives and {fake.word()} programs. {name} is also the {random.choice(['Founder', 'Co-Founder'])} of {fake.company()}, focusing on {fake.word()} technology and {fake.word()} solutions. {name.split()[0]} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {random.choice(self.UNIVERSITIES)} and a {random.choice(['BS', 'BA', 'MS'])} from {random.choice(self.UNIVERSITIES)}, equipping {name.split()[0].lower()} with a unique blend of technical expertise and strategic leadership. With a proven track record of building and scaling organizations, {name} is driving {company}'s expansion into new markets and technologies."
        ]
        chunk = random.choice(templates)
        if extra_info:
            chunk += f" {extra_info}"
        return chunk
    
    def generate_product_chunk(self, company, product_name):
        """Generate a product/service chunk (~1000-2000 chars)."""
        return f"{company} is revolutionizing the way organizations access {product_name} by introducing innovative solutions that align with modern business needs. Unlike traditional {fake.word()} models, {company}'s approach ensures {fake.word()} and {fake.word()} through {fake.word()} integration. How It Works: 1. {fake.word().capitalize()} Access – Organizations can directly {fake.word()} {product_name} through {fake.word()} platforms. 2. {fake.word().capitalize()} Integration – {company} provides seamless {fake.word()} with existing {fake.word()} systems. 3. {fake.word().capitalize()} Management – Advanced {fake.word()} capabilities enable {fake.word()} and {fake.word()} optimization. Revolutionary Features: {company}'s {product_name} delivers {fake.word()} and {fake.word()} through {fake.word()} technology, ensuring {fake.word()} and {fake.word()} for enterprise customers. Organizations that adopt {product_name} will benefit from {fake.word()}, {fake.word()}, and {fake.word()} capabilities, positioning them as leaders in their respective industries."
    
    def generate_benefits_chunk(self, topic):
        """Generate benefits/drawbacks chunk (~1500-2500 chars)."""
        benefits = [
            f"● Enhanced {fake.word()} – Improves {fake.word()} and {fake.word()} capabilities, enabling organizations to {fake.word()} more effectively.",
            f"● Improved {fake.word()} – Reduces {fake.word()} and {fake.word()} costs, resulting in significant {fake.word()} savings.",
            f"● Better {fake.word()} – Provides {fake.word()} and {fake.word()} insights, supporting {fake.word()} decision-making.",
            f"● Increased {fake.word()} – Automates {fake.word()} processes, freeing up resources for {fake.word()} initiatives."
        ]
        drawbacks = [
            f"● {fake.word().capitalize()} challenges – Requires {fake.word()} and {fake.word()} investments, which may impact {fake.word()}.",
            f"● {fake.word().capitalize()} limitations – Some organizations may face {fake.word()} constraints that limit {fake.word()} adoption.",
            f"● {fake.word().capitalize()} concerns – {fake.word()} and {fake.word()} considerations must be addressed before {fake.word()}."
        ]
        selected_benefits = random.sample(benefits, random.randint(2, 4))
        selected_drawbacks = random.sample(drawbacks, random.randint(1, 3))
        return f"The Benefits of {topic}: {' '.join(selected_benefits)} Drawbacks to Consider: {' '.join(selected_drawbacks)}"
    
    def generate_fictional_character_chunk(self, character_name, character_type, setting):
        """Generate fictional character chunk for book writing scenarios (~800-1500 chars)."""
        templates = [
            f"{character_name} is a {random.choice(['complex', 'mysterious', 'driven', 'passionate'])} {character_type} in the story set in {setting}. Born in {fake.year()} to a {random.choice(['wealthy', 'humble', 'noble', 'working-class'])} family, {character_name.split()[0]} developed a {random.choice(['strong sense of justice', 'deep-seated ambition', 'wounded heart', 'curious mind'])} that shapes every decision. {character_name} works as a {fake.job().lower()} but secretly {random.choice(['investigates', 'pursues', 'studies', 'seeks'])} {fake.word()} which becomes central to the plot. The character's relationship with {fake.name()} is marked by {random.choice(['betrayal', 'trust', 'conflict', 'love'])} and {random.choice(['ultimately', 'briefly', 'never truly'])} resolves in the {random.choice(['first act', 'climax', 'denouement'])}. {character_name}'s defining moment comes when {character_name.split()[0]} must choose between {fake.word()} and {fake.word()}, revealing the character's true nature.",
            f"In {setting}, {character_name} serves as the story's {character_type}, a {random.choice(['former', 'aspiring', 'reluctant', 'fallen'])} {fake.job().lower()} with a {random.choice(['troubled past', 'hidden talent', 'dark secret', 'bright future'])}. The character's backstory includes {random.choice(['a tragic loss', 'an unexpected inheritance', 'a life-changing encounter', 'a moment of redemption'])} that occurred in {fake.city()} during {fake.year()}. {character_name} is known for {character_name.split()[0].lower()}'s {random.choice(['sharp wit', 'unwavering loyalty', 'questionable morals', 'extraordinary courage'])} and has a {random.choice(['complicated', 'strained', 'strong', 'broken'])} relationship with {fake.name()}, who serves as {random.choice(['mentor', 'rival', 'love interest', 'antagonist'])}. The character arc follows {character_name.split()[0]}'s journey from {fake.word()} to {fake.word()}, with key turning points at {random.choice(['Chapter 5', 'the midpoint', 'the third act'])}."
        ]
        return random.choice(templates)
    
    def generate_business_operations_chunk(self, company_name, topic):
        """Generate business operations chunk for entrepreneur domain (~800-1500 chars)."""
        templates = [
            f"{company_name} was founded in {fake.year()} by {fake.name()} with a vision to {fake.sentence(nb_words=6).lower().rstrip('.')}. The company currently operates in {fake.city()} and has {random.randint(5, 50)} employees. Our core products include {fake.word().capitalize()} {fake.word()}, {fake.word().capitalize()} {fake.word()}, and {fake.word().capitalize()} {fake.word()} services. The business model focuses on {fake.word()} and {fake.word()} with target customers in the {fake.word()} and {fake.word()} sectors. Current monthly recurring revenue is approximately ${random.randint(10, 500)}K, and we've raised ${random.randint(100, 2000)}K in seed funding from {fake.company()} and {fake.company()}. Key metrics include {random.randint(100, 1000)} active customers, {random.randint(10, 100)}% month-over-month growth, and a customer acquisition cost of ${random.randint(10, 200)}. Our team consists of {random.randint(3, 8)} full-time employees across product, engineering, sales, and marketing functions.",
            f"Business Operations at {company_name}: We maintain operations through {fake.word()} and {fake.word()} processes that enable {fake.word()} and {fake.word()} efficiency. Our customer support team handles {random.randint(50, 500)} tickets per month with an average response time of {random.randint(1, 24)} hours. Product development follows a {fake.word()} methodology with {random.randint(1, 4)} week sprints. We work with {random.randint(3, 10)} key vendors including {fake.company()} for {fake.word()} services and {fake.company()} for {fake.word()} solutions. Current projects include {fake.word().capitalize()} {fake.word()}, which is scheduled to launch in {fake.month_name()} {fake.year()}, and {fake.word().capitalize()} {fake.word()} which is in {random.choice(['planning', 'development', 'testing'])} phase. Our office is located at {fake.address()} and we maintain {random.choice(['remote', 'hybrid', 'in-office'])} work arrangements for our team."
        ]
        return random.choice(templates)


class ExampleGenerator:
    """Generates complete training examples with REASONING and FINAL ANSWER."""
    
    def __init__(self, domain_config, domain_name="default"):
        self.domain = domain_config
        self.domain_name = domain_name
        self.chunk_gen = RAGChunkGenerator(domain_config)
        if domain_name != "fictional":
            if domain_name == "entrepreneur":
                self.company = random.choice(domain_config.get("company_names", []))
            else:
                self.company = random.choice(domain_config.get("company_names", []))
    
    def _extract_cofounder_evidence_simple(self, person_name: str, company: str, context: str) -> str:
        """Simple, robust extraction: find person name, then extract sentence with Co-Founder and company."""
        # Find person name in context
        person_idx = context.find(person_name)
        if person_idx == -1:
            return None
        
        # Look for Co-Founder and company near the person (within 300 chars)
        search_start = max(0, person_idx - 100)
        search_end = min(len(context), person_idx + 300)
        person_snippet = context[search_start:search_end]
        
        # Check if Co-Founder and company are in this snippet
        if 'Co-Founder' not in person_snippet and 'co-founder' not in person_snippet.lower():
            return None
        if company not in person_snippet:
            return None
        
        # Find sentence boundaries
        # Start from person name (or sentence start before it)
        relative_person_idx = person_idx - search_start
        sentence_start = person_snippet.rfind('.', 0, relative_person_idx)
        if sentence_start == -1:
            sentence_start = 0
        else:
            sentence_start += 1  # Start after period
        
        # End at next sentence after company name
        company_idx_in_snippet = person_snippet.find(company, relative_person_idx)
        if company_idx_in_snippet != -1:
            sentence_end = person_snippet.find('.', company_idx_in_snippet + len(company))
            if sentence_end == -1:
                sentence_end = len(person_snippet)
        else:
            sentence_end = len(person_snippet)
        
        evidence = person_snippet[sentence_start:sentence_end].strip()
        
        # Ensure person name is at the start (fix any truncation)
        if not evidence.startswith(person_name):
            # Find person name in evidence
            person_in_evidence = person_name in evidence
            if person_in_evidence:
                # Rebuild from person name
                person_idx_in_evidence = evidence.find(person_name)
                evidence = evidence[person_idx_in_evidence:].strip()
            else:
                # Person not in evidence - extract from actual person position
                actual_start = search_start + sentence_start
                actual_person_idx = context.find(person_name, actual_start, search_start + sentence_end)
                if actual_person_idx != -1:
                    actual_end = context.find('.', actual_person_idx + len(person_name))
                    if actual_end == -1:
                        actual_end = min(len(context), actual_person_idx + 300)
                    evidence = context[actual_person_idx:actual_end].strip()
        
        # Final validation
        if (person_name in evidence and 
            ('Co-Founder' in evidence or 'co-founder' in evidence.lower()) and
            company in evidence):
            return evidence
        
        return None
        
    def generate_cofounders_example(self):
        """Generate co-founders query (multi-person extraction with DISCARD items)."""
        num_cofounders = random.randint(3, 5)
        num_others = random.randint(2, 4)
        
        cofounders = [fake.name() for _ in range(num_cofounders)]
        others = [fake.name() for _ in range(num_others)]
        
        all_people = cofounders + others
        random.shuffle(all_people)
        
        # Generate 2-3 chunks
        num_chunks = random.randint(2, 3)
        chunks = []
        chunk_people = [all_people[i::num_chunks] for i in range(num_chunks)]
        
        for i, people_group in enumerate(chunk_people):
            chunk_parts = []
            for person in people_group:
                is_cofounder = person in cofounders
                role = random.choice(["Co-Founder and CEO", "Co-Founder and CFO", "Co-Founder and COO", "Co-Founder"]) if is_cofounder else random.choice(["Head of Engineering", "Business Development Lead", "External Advisor", "Senior Director"])
                chunk_parts.append(self.chunk_gen.generate_person_chunk(person, role, self.company))
            chunks.append(" ".join(chunk_parts))
        
        query = f"Who are the co-founders of {self.company}?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        # Build REASONING with verbatim evidence extraction
        reasoning_lines = ["REASONING:"]
        keep_items = []
        
        for person in all_people:
            is_cofounder = person in cofounders
            reasoning_lines.append(f"- Item: {person}")
            
            if is_cofounder:
                # Use simple, robust extraction method
                evidence = self._extract_cofounder_evidence_simple(person, self.company, full_context)
                
                if evidence:
                    reasoning_lines.append(f'  - Evidence: "{evidence}"')
                    reasoning_lines.append("  - Action: [KEEP]")
                    keep_items.append(person)
                else:
                    # Skip if no evidence found (shouldn't happen with proper chunks)
                    continue
            else:
                # Extract verbatim evidence for non-cofounder
                role = random.choice(["Head of Engineering", "Business Development Lead", "External Advisor"])
                evidence = extractor.extract_person_role_evidence(person, role, self.company, full_context)
                
                if evidence:
                    reasoning_lines.append(f'  - Evidence: "{evidence}"')
                    reasoning_lines.append("  - Action: [DISCARD] (Reason: Not co-founder).")
                else:
                    continue
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        
        if keep_items:
            final_answer = f"The co-founders of {self.company} are {', '.join(keep_items)}."
        else:
            final_answer = f"No co-founders found for {self.company}."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_role_specific_example(self):
        """Generate role-specific query (e.g., CFO, CTO)."""
        target_role = random.choice(self.domain["roles"])
        target_name = fake.name()
        other_roles = [r for r in self.domain["roles"] if r != target_role]
        other_names = [fake.name() for _ in range(2)]
        
        chunks = []
        chunks.append(self.chunk_gen.generate_person_chunk(target_name, target_role, self.company))
        for other_name, other_role in zip(other_names, random.sample(other_roles, 2)):
            chunks.append(self.chunk_gen.generate_person_chunk(other_name, other_role, self.company))
        
        query = f"Who is the {target_role} of {self.company}?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim evidence for target
        evidence = extractor.extract_person_role_evidence(target_name, target_role, self.company, full_context)
        if evidence:
            reasoning_lines.append(f"- Item: {target_name}")
            reasoning_lines.append(f'  - Evidence: "{evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract verbatim evidence for others (DISCARD)
        for other_name, other_role in zip(other_names, random.sample(other_roles, 2)):
            evidence = extractor.extract_person_role_evidence(other_name, other_role, self.company, full_context)
            if evidence:
                reasoning_lines.append(f"- Item: {other_name}")
                reasoning_lines.append(f'  - Evidence: "{evidence}"')
                reasoning_lines.append(f"  - Action: [DISCARD] (Reason: Not {target_role}).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"The {target_role} of {self.company} is {target_name}."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_person_info_example(self):
        """Generate person information query."""
        person_name = fake.name()
        role = random.choice(self.domain["roles"])
        uni1 = random.choice(RAGChunkGenerator.UNIVERSITIES)
        uni2 = random.choice(RAGChunkGenerator.UNIVERSITIES)
        
        chunk = self.chunk_gen.generate_person_chunk(
            person_name, role, self.company,
            f"{person_name.split()[0]} holds a {random.choice(['PhD', 'MBA', 'MS'])} from {uni1} and a {random.choice(['BS', 'BA'])} from {uni2}."
        )
        chunks = [chunk]
        
        # Add noise chunk
        other_person = fake.name()
        chunks.append(self.chunk_gen.generate_person_chunk(other_person, random.choice(self.domain["roles"]), self.company))
        
        query = f"Tell me about {person_name}."
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim role evidence
        evidence = extractor.extract_person_role_evidence(person_name, role, self.company, full_context)
        if evidence:
            reasoning_lines.append(f"- Item: {person_name} is {role}")
            reasoning_lines.append(f'  - Evidence: "{evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract verbatim education evidence
        degree1 = random.choice(['PhD', 'MBA', 'MS'])
        degree2 = random.choice(['BS', 'BA'])
        edu_evidence = extractor.extract_education_evidence(person_name, degree1, uni1, full_context)
        if not edu_evidence:
            edu_evidence = extractor.find_verbatim_quote(f"holds a {degree1} from {uni1}", full_context)
        if edu_evidence:
            reasoning_lines.append(f"- Item: Education")
            reasoning_lines.append(f'  - Evidence: "{edu_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract other person (DISCARD)
        other_evidence = extractor.extract_person_role_evidence(other_person, random.choice(self.domain['roles']), self.company, full_context)
        if other_evidence:
            reasoning_lines.append(f"- Item: {other_person}")
            reasoning_lines.append(f'  - Evidence: "{other_evidence}"')
            reasoning_lines.append("  - Action: [DISCARD] (Reason: Different person).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{person_name} is the {role} of {self.company}. {person_name.split()[0]} holds a {degree1} from {uni1} and a {degree2} from {uni2}."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_education_example(self):
        """Generate education query."""
        person_name = fake.name()
        uni1 = random.choice(RAGChunkGenerator.UNIVERSITIES)
        uni2 = random.choice(RAGChunkGenerator.UNIVERSITIES)
        degree1 = random.choice(["PhD", "MBA", "MS"])
        degree2 = random.choice(["BS", "BA"])
        
        chunk = self.chunk_gen.generate_person_chunk(
            person_name, random.choice(self.domain["roles"]), self.company,
            f"{person_name.split()[0]} holds a {degree1} from {uni1} and a {degree2} from {uni2}."
        )
        chunks = [chunk]
        
        # Add noise chunk
        other_person = fake.name()
        chunks.append(self.chunk_gen.generate_person_chunk(
            other_person, random.choice(self.domain["roles"]), self.company,
            f"{other_person.split()[0]} holds a {random.choice(['PhD', 'MBA'])} from {random.choice(RAGChunkGenerator.UNIVERSITIES)}."
        ))
        
        query = f"Where did {person_name} go to school?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim education evidence
        edu_evidence1 = extractor.extract_education_evidence(person_name, degree1, uni1, full_context)
        if not edu_evidence1:
            edu_evidence1 = extractor.find_verbatim_quote(f"holds a {degree1} from {uni1}", full_context)
        if edu_evidence1:
            reasoning_lines.append(f"- Item: {uni1}")
            reasoning_lines.append(f'  - Evidence: "{edu_evidence1}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        edu_evidence2 = extractor.find_verbatim_quote(f"and a {degree2} from {uni2}", full_context)
        if not edu_evidence2:
            edu_evidence2 = extractor.find_verbatim_quote(f"{degree2} from {uni2}", full_context)
        if edu_evidence2:
            reasoning_lines.append(f"- Item: {uni2}")
            reasoning_lines.append(f'  - Evidence: "{edu_evidence2}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract other person education (DISCARD)
        other_uni = random.choice(RAGChunkGenerator.UNIVERSITIES)
        other_degree = random.choice(['PhD', 'MBA'])
        other_edu_evidence = extractor.extract_education_evidence(other_person, other_degree, other_uni, full_context)
        if other_edu_evidence:
            reasoning_lines.append(f"- Item: {other_person} education")
            reasoning_lines.append(f'  - Evidence: "{other_edu_evidence}"')
            reasoning_lines.append("  - Action: [DISCARD] (Reason: Different person).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{person_name} went to {uni1} ({degree1}) and {uni2} ({degree2})."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_benefits_example(self):
        """Generate benefits query (with DISCARD for drawbacks)."""
        topic = random.choice(["cloud computing", "AI integration", "data analytics", "remote work"])
        
        chunk = self.chunk_gen.generate_benefits_chunk(topic)
        chunks = [chunk]
        
        query = f"What are the benefits of {topic}?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        # Extract benefits and drawbacks from chunk (they have ● markers)
        reasoning_lines = ["REASONING:"]
        
        # Find benefit lines (they start with ● and are before "Drawbacks")
        lines = chunk.split('\n')
        benefit_lines = []
        drawback_lines = []
        in_benefits = False
        in_drawbacks = False
        
        for line in lines:
            if "Benefits" in line or "benefits" in line.lower():
                in_benefits = True
                in_drawbacks = False
            elif "Drawbacks" in line or "drawbacks" in line.lower():
                in_drawbacks = True
                in_benefits = False
            elif line.strip().startswith('●') and in_benefits:
                benefit_lines.append(line.strip())
            elif line.strip().startswith('●') and in_drawbacks:
                drawback_lines.append(line.strip())
        
        # Extract verbatim benefit evidence (KEEP)
        for i, benefit_line in enumerate(benefit_lines[:4]):  # Max 4 benefits
            if benefit_line:
                # Remove the ● marker for evidence
                evidence = benefit_line.replace('●', '').strip()
                reasoning_lines.append(f"- Item: Benefit {i+1}")
                reasoning_lines.append(f'  - Evidence: "{evidence}"')
                reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract verbatim drawback evidence (DISCARD)
        for i, drawback_line in enumerate(drawback_lines[:2]):  # Max 2 drawbacks
            if drawback_line:
                evidence = drawback_line.replace('●', '').strip()
                reasoning_lines.append(f"- Item: Drawback {i+1}")
                reasoning_lines.append(f'  - Evidence: "{evidence}"')
                reasoning_lines.append("  - Action: [DISCARD] (Reason: This is a drawback, not a benefit).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        
        # Build final answer from kept benefits
        benefit_items = [line.replace('●', '').strip() for line in benefit_lines[:4]]
        if benefit_items:
            final_answer = f"The benefits of {topic} include: {', '.join(benefit_items[:3])}."
        else:
            final_answer = f"The benefits of {topic} include: enhanced capabilities, improved efficiency, and better insights."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_products_example(self):
        """Generate products/services query."""
        product = random.choice(self.domain.get("products", ["Product", "Service"]))
        
        chunk = self.chunk_gen.generate_product_chunk(self.company, product)
        chunks = [chunk]
        
        query = f"What products does {self.company} offer?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim product evidence
        evidence = extractor.extract_product_evidence(product, self.company, full_context)
        if not evidence:
            evidence = extractor.find_verbatim_quote(f"{self.company} is revolutionizing", full_context)
        if evidence:
            reasoning_lines.append(f"- Item: {product}")
            reasoning_lines.append(f'  - Evidence: "{evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{self.company} offers {product}."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_character_traits_example(self):
        """Generate character traits query for fictional/book writing."""
        character_name = fake.name()
        character_type = random.choice(self.domain["character_types"])
        setting = random.choice(self.domain["settings"])
        
        chunk = self.chunk_gen.generate_fictional_character_chunk(character_name, character_type, setting)
        chunks = [chunk]
        
        # Add noise chunk with different character
        other_character = fake.name()
        chunks.append(self.chunk_gen.generate_fictional_character_chunk(
            other_character, random.choice(self.domain["character_types"]), setting
        ))
        
        query = f"What are the key traits of {character_name}?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        traits = ["complex", "driven", "mysterious", "passionate", "loyal", "ambitious"]
        selected_traits = random.sample(traits, random.randint(2, 3))
        
        reasoning_lines = ["REASONING:"]
        keep_traits = []
        
        for trait in selected_traits:
            # Extract verbatim evidence for trait
            evidence = extractor.find_verbatim_quote(f"{character_name} is a {trait}", full_context)
            if not evidence:
                evidence = extractor.find_verbatim_quote(f"{trait} {character_type}", full_context)
            if evidence:
                reasoning_lines.append(f"- Item: {trait}")
                reasoning_lines.append(f'  - Evidence: "{evidence}"')
                reasoning_lines.append("  - Action: [KEEP]")
                keep_traits.append(trait)
        
        # Extract other character (DISCARD)
        other_evidence = extractor.find_verbatim_quote(other_character, full_context)
        if other_evidence:
            reasoning_lines.append(f"- Item: {other_character} traits")
            reasoning_lines.append(f'  - Evidence: "{other_evidence}"')
            reasoning_lines.append("  - Action: [DISCARD] (Reason: Different character).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        
        if keep_traits:
            final_answer = f"{character_name}'s key traits are: {', '.join(keep_traits)}."
        else:
            final_answer = f"{character_name}'s key traits are mentioned in the context."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_relationships_example(self):
        """Generate character relationships query."""
        character_name = fake.name()
        other_characters = [fake.name() for _ in range(3)]
        relationship_types = ["mentor", "rival", "love interest", "antagonist"]
        
        chunk = self.chunk_gen.generate_fictional_character_chunk(character_name, "protagonist", random.choice(self.domain["settings"]))
        chunks = [chunk]
        
        query = f"Who are {character_name}'s relationships with?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        keep_relationships = random.sample(other_characters, 2)
        discard_relationships = [c for c in other_characters if c not in keep_relationships]
        
        reasoning_lines = ["REASONING:"]
        keep_items = []
        
        for rel_char in keep_relationships:
            # Extract verbatim evidence for relationship
            evidence = extractor.find_verbatim_quote(f"{character_name} has", full_context)
            if not evidence:
                evidence = extractor.find_verbatim_quote(rel_char, full_context)
            if evidence:
                reasoning_lines.append(f"- Item: {rel_char}")
                reasoning_lines.append(f'  - Evidence: "{evidence}"')
                reasoning_lines.append("  - Action: [KEEP]")
                keep_items.append(rel_char)
        
        for rel_char in discard_relationships:
            # Extract evidence for non-relationship (DISCARD)
            evidence = extractor.find_verbatim_quote(rel_char, full_context)
            if evidence and character_name not in evidence:
                reasoning_lines.append(f"- Item: {rel_char}")
                reasoning_lines.append(f'  - Evidence: "{evidence}"')
                reasoning_lines.append("  - Action: [DISCARD] (Reason: Not a direct relationship).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        
        if keep_items:
            final_answer = f"{character_name}'s key relationships are with {', '.join(keep_items)}."
        else:
            final_answer = f"{character_name}'s relationships are mentioned in the context."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    # Entrepreneur domain methods
    def generate_team_members_example(self):
        """Generate team members query for entrepreneur domain."""
        num_team = random.randint(3, 6)
        team_members = [fake.name() for _ in range(num_team)]
        others = [fake.name() for _ in range(2)]  # Non-team members
        
        all_people = team_members + others
        random.shuffle(all_people)
        
        chunks = []
        chunk_parts = []
        for person in all_people:
            is_team = person in team_members
            role = random.choice(self.domain["roles"]) if is_team else random.choice(["Advisor", "Consultant", "Contractor"])
            chunk_parts.append(self.chunk_gen.generate_person_chunk(person, role, self.company))
        chunks.append(" ".join(chunk_parts))
        
        query = f"Who are the team members at {self.company}?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        keep_items = []
        
        for person in all_people:
            is_team = person in team_members
            reasoning_lines.append(f"- Item: {person}")
            
            if is_team:
                # Extract verbatim role evidence for team members
                role = random.choice(self.domain['roles'])
                evidence = extractor.extract_person_role_evidence(person, role, self.company, full_context)
                if evidence:
                    reasoning_lines.append(f'  - Evidence: "{evidence}"')
                    reasoning_lines.append("  - Action: [KEEP]")
                    keep_items.append(person)
            else:
                # Extract verbatim evidence for non-team members
                role = random.choice(['Advisor', 'Consultant', 'Contractor'])
                evidence = extractor.extract_person_role_evidence(person, role, self.company, full_context)
                if evidence:
                    reasoning_lines.append(f'  - Evidence: "{evidence}"')
                    reasoning_lines.append("  - Action: [DISCARD] (Reason: Not a team member).")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        
        if keep_items:
            final_answer = f"The team members at {self.company} are {', '.join(keep_items)}."
        else:
            final_answer = f"No team members found for {self.company}."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_company_info_example(self):
        """Generate company information query."""
        chunk = self.chunk_gen.generate_business_operations_chunk(self.company, "company_info")
        chunks = [chunk]
        
        query = f"Tell me about {self.company}."
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim evidence for founded
        founded_evidence = extractor.find_verbatim_quote(f"{self.company} was founded", full_context)
        if founded_evidence:
            reasoning_lines.append(f"- Item: Founded")
            reasoning_lines.append(f'  - Evidence: "{founded_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract verbatim evidence for location
        location_evidence = extractor.find_verbatim_quote("currently operates in", full_context)
        if location_evidence:
            reasoning_lines.append(f"- Item: Location")
            reasoning_lines.append(f'  - Evidence: "{location_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract verbatim evidence for employees
        employees_evidence = extractor.find_verbatim_quote("employees", full_context)
        if employees_evidence:
            reasoning_lines.append(f"- Item: Employees")
            reasoning_lines.append(f'  - Evidence: "{employees_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{self.company} was founded and currently operates with employees."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_funding_info_example(self):
        """Generate funding information query."""
        chunk = self.chunk_gen.generate_business_operations_chunk(self.company, "funding")
        chunks = [chunk]
        
        query = f"What is the funding status of {self.company}?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim evidence for funding
        funding_evidence = extractor.find_verbatim_quote("seed funding", full_context)
        if not funding_evidence:
            funding_evidence = extractor.find_verbatim_quote("raised", full_context)
        if funding_evidence:
            reasoning_lines.append(f"- Item: Seed funding")
            reasoning_lines.append(f'  - Evidence: "{funding_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract verbatim evidence for investors
        investors_evidence = extractor.find_verbatim_quote("from", full_context)
        if investors_evidence:
            reasoning_lines.append(f"- Item: Investors")
            reasoning_lines.append(f'  - Evidence: "{investors_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract verbatim evidence for revenue
        revenue_evidence = extractor.find_verbatim_quote("monthly recurring revenue", full_context)
        if revenue_evidence:
            reasoning_lines.append(f"- Item: Revenue")
            reasoning_lines.append(f'  - Evidence: "{revenue_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{self.company} has raised seed funding from investors and has monthly recurring revenue."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_products_services_example(self):
        """Generate products/services query for entrepreneur domain."""
        chunk = self.chunk_gen.generate_business_operations_chunk(self.company, "products")
        chunks = [chunk]
        
        query = f"What products and services does {self.company} offer?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim evidence for products
        products_evidence = extractor.find_verbatim_quote("core products include", full_context)
        if products_evidence:
            reasoning_lines.append(f"- Item: Products")
            reasoning_lines.append(f'  - Evidence: "{products_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{self.company} offers products and services."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_metrics_example(self):
        """Generate business metrics query."""
        chunk = self.chunk_gen.generate_business_operations_chunk(self.company, "metrics")
        chunks = [chunk]
        
        query = f"What are the key business metrics for {self.company}?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim evidence for metrics
        customers_evidence = extractor.find_verbatim_quote("active customers", full_context)
        if customers_evidence:
            reasoning_lines.append(f"- Item: Active customers")
            reasoning_lines.append(f'  - Evidence: "{customers_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        growth_evidence = extractor.find_verbatim_quote("month-over-month growth", full_context)
        if growth_evidence:
            reasoning_lines.append(f"- Item: Growth rate")
            reasoning_lines.append(f'  - Evidence: "{growth_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        cac_evidence = extractor.find_verbatim_quote("customer acquisition cost", full_context)
        if cac_evidence:
            reasoning_lines.append(f"- Item: CAC")
            reasoning_lines.append(f'  - Evidence: "{cac_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{self.company}'s key metrics include active customers, month-over-month growth, and customer acquisition cost."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def generate_contracts_example(self):
        """Generate contracts/vendors query."""
        chunk = self.chunk_gen.generate_business_operations_chunk(self.company, "contracts")
        chunks = [chunk]
        
        query = f"Who are {self.company}'s key vendors and partners?"
        
        # Build full context for verbatim extraction
        full_context = "\n---\n".join(chunks)
        
        reasoning_lines = ["REASONING:"]
        
        # Extract verbatim evidence for vendors
        vendors_evidence = extractor.find_verbatim_quote("key vendors", full_context)
        if not vendors_evidence:
            vendors_evidence = extractor.find_verbatim_quote("vendors including", full_context)
        if vendors_evidence:
            reasoning_lines.append(f"- Item: Vendors")
            reasoning_lines.append(f'  - Evidence: "{vendors_evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        reasoning_lines.append("- End of scan.")
        reasoning = "\n".join(reasoning_lines)
        final_answer = f"{self.company}'s key vendors and partners are mentioned in the context."
        
        return self._create_example(query, chunks, reasoning, final_answer)
    
    def _create_example(self, query, chunks, reasoning, final_answer):
        """Create training example structure."""
        rag_context = "\n---\n".join(chunks)
        user_content = f"Knowledge context: {rag_context}\n---\nQuestion: {query}"
        assistant_content = f"{reasoning}\n\nFINAL ANSWER:\n{final_answer}"
        
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }


def generate_dataset(num_examples=200):
    """Generate diverse dataset across all domains."""
    examples = []
    
    # Distribution across domains (200 total)
    domain_distribution = {
        "fortune500": 40,
        "medicine": 30,
        "law": 25,
        "education": 25,
        "fictional": 30,
        "entrepreneur": 50  # New domain
    }
    
    query_type_generators = {
        "cofounders": "generate_cofounders_example",
        "role_specific": "generate_role_specific_example",
        "person_info": "generate_person_info_example",
        "education": "generate_education_example",
        "benefits": "generate_benefits_example",
        "products": "generate_products_example",
        "services": "generate_products_example",  # Reuse products generator
        "character_traits": "generate_character_traits_example",
        "relationships": "generate_relationships_example",
        "team_members": "generate_team_members_example",
        "company_info": "generate_company_info_example",
        "funding_info": "generate_funding_info_example",
        "products_services": "generate_products_services_example",
        "metrics": "generate_metrics_example",
        "contracts": "generate_contracts_example"
    }
    
    for domain_name, count in domain_distribution.items():
        domain_config = DOMAINS[domain_name]
        generator = ExampleGenerator(domain_config, domain_name)
        
        # Distribute query types for this domain
        available_types = domain_config["query_types"]
        type_counts = {qt: count // len(available_types) for qt in available_types}
        remainder = count - sum(type_counts.values())
        for i, qt in enumerate(available_types[:remainder]):
            type_counts[qt] += 1
        
        for query_type, type_count in type_counts.items():
            if query_type in query_type_generators:
                generator_method = getattr(generator, query_type_generators[query_type])
                for _ in range(type_count):
                    examples.append(generator_method())
    
    # Shuffle examples
    random.shuffle(examples)
    
    return examples


def main():
    print("Generating 200 diverse real-life training examples...")
    examples = generate_dataset(200)
    
    output_file = "rag_cot_training_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"✅ Generated {len(examples)} examples")
    print(f"✅ Saved to {output_file}")
    
    # Print summary
    domain_distribution = {
        "fortune500": 40,
        "medicine": 30,
        "law": 25,
        "education": 25,
        "fictional": 30,
        "entrepreneur": 50
    }
    print("\n📊 Summary:")
    print(f"   Total examples: {len(examples)}")
    print(f"   Domains: Fortune 500 ({domain_distribution['fortune500']}), Medicine ({domain_distribution['medicine']}), Law ({domain_distribution['law']}), Education ({domain_distribution['education']}), Fictional/Book Writing ({domain_distribution['fictional']}), Young Entrepreneur/Business Operations ({domain_distribution['entrepreneur']})")
    print(f"   Query types: Co-founders, Role-specific, Person info, Education, Benefits, Products, Character traits, Relationships, Team members, Company info, Funding, Metrics, Contracts")


if __name__ == "__main__":
    main()
