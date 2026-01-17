#!/usr/bin/env python3
"""
Regenerate Dataset with Strict Verbatim Evidence
=================================================
This script regenerates the training dataset ensuring ALL evidence is verbatim from context.
It uses the existing generation logic but enforces verbatim extraction.
"""

import json
import random
import re
import sys
from faker import Faker
from verbatim_evidence_helper import VerbatimEvidenceExtractor, validate_evidence_verbatim

# Import the existing generator
sys.path.insert(0, '.')
from generate_200_real_life_dataset import (
    DOMAINS, RAGChunkGenerator, ExampleGenerator, SYSTEM_PROMPT
)

fake = Faker()
extractor = VerbatimEvidenceExtractor()

# Update system prompt to emphasize verbatim
VERBATIM_SYSTEM_PROMPT = """You are a precise data extraction bot.
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


class VerbatimExampleGenerator(ExampleGenerator):
    """Extended generator that enforces verbatim evidence extraction."""
    
    def generate_cofounders_example(self):
        """Generate co-founders query with STRICT verbatim evidence."""
        # Use parent to generate chunks
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
                # Use cofounder-specific chunk generation
                chunk_parts.append(self.chunk_gen.generate_person_chunk(
                    person, f"Co-Founder and {role}", self.company
                ))
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
        
        # Extract verbatim evidence for each person
        for person in all_people:
            is_cofounder = person in cofounders
            reasoning_lines.append(f"- Item: {person}")
            
            if is_cofounder:
                # Try to find co-founder evidence
                evidence = extractor.extract_cofounder_evidence(person, self.company, full_context)
                if not evidence:
                    # Try person-role with "Co-Founder"
                    role = random.choice(["CEO", "CFO", "COO", "CMO"])
                    evidence = extractor.extract_person_role_evidence(person, f"Co-Founder and {role}", self.company, full_context)
                
                if evidence:
                    reasoning_lines.append(f'  - Evidence: "{evidence}"')
                    reasoning_lines.append("  - Action: [KEEP]")
                    keep_items.append(person)
                else:
                    # Skip if no evidence found (shouldn't happen with proper chunks)
                    continue
            else:
                # Extract evidence for non-cofounder
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
        """Generate role-specific query with verbatim evidence."""
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
        evidence = extractor.extract_person_role_evidence(target_name, target_role, self.company, full_context)
        if evidence:
            reasoning_lines.append(f"- Item: {target_name}")
            reasoning_lines.append(f'  - Evidence: "{evidence}"')
            reasoning_lines.append("  - Action: [KEEP]")
        
        # Extract evidence for others (DISCARD)
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
    
    def _create_example(self, query: str, chunks: list, reasoning: str, final_answer: str) -> dict:
        """Create example with updated system prompt."""
        rag_context = "\n---\n".join(chunks)
        user_content = f"Knowledge context: {rag_context}\n---\nQuestion: {query}"
        
        return {
            "messages": [
                {"role": "system", "content": VERBATIM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"{reasoning}\n\nFINAL ANSWER:\n{final_answer}"}
            ]
        }


def regenerate_dataset(num_examples: int = 200) -> list:
    """Regenerate dataset with verbatim evidence enforcement."""
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
    print("REGENERATING DATASET WITH STRICT VERBATIM EVIDENCE")
    print("=" * 80)
    
    total_generated = 0
    total_validated = 0
    
    for domain_name, count in domain_distribution.items():
        print(f"\nGenerating {count} examples for {domain_name} domain...")
        domain_config = DOMAINS[domain_name]
        generator = VerbatimExampleGenerator(domain_name, domain_config)
        
        for i in range(count):
            # Select query type
            if domain_name == "fictional":
                query_type = random.choice(["character_traits", "relationships"])
            elif domain_name == "entrepreneur":
                query_type = random.choice(["team_members", "company_info", "funding_info", "products_services", "metrics", "contracts"])
            else:
                query_type = random.choice(domain_config["query_types"])
            
            # Generate example
            try:
                if query_type == "cofounders":
                    example = generator.generate_cofounders_example()
                elif query_type == "role_specific":
                    example = generator.generate_role_specific_example()
                else:
                    # Use parent methods for other types (they'll need updating too)
                    # For now, skip and regenerate
                    continue
                
                # Validate verbatim evidence
                is_valid, warnings = validate_evidence_verbatim(example)
                
                if is_valid:
                    examples.append(example)
                    total_validated += 1
                else:
                    # Retry once
                    if query_type == "cofounders":
                        example = generator.generate_cofounders_example()
                        is_valid, warnings = validate_evidence_verbatim(example)
                        if is_valid:
                            examples.append(example)
                            total_validated += 1
                
                total_generated += 1
                
            except Exception as e:
                print(f"  ⚠️  Error generating example {i+1}: {e}")
                continue
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{count} examples (validated: {total_validated})...")
    
    print(f"\n✅ Generated {total_generated} examples, {total_validated} with valid verbatim evidence")
    
    # Shuffle
    random.shuffle(examples)
    
    return examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate dataset with strict verbatim evidence")
    parser.add_argument("--num-examples", type=int, default=200, help="Target number of examples")
    parser.add_argument("--output", type=str, default="rag_cot_training_dataset.json", help="Output filename")
    
    args = parser.parse_args()
    
    print(f"\nRegenerating dataset with STRICT VERBATIM EVIDENCE requirements...")
    examples = regenerate_dataset(args.num_examples)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"\n✅ Saved {len(examples)} examples to {args.output}")
    
    # Final validation
    print(f"\n🔍 Final validation...")
    all_valid = True
    for i, ex in enumerate(examples[:20]):  # Check first 20
        is_valid, warnings = validate_evidence_verbatim(ex)
        if not is_valid:
            all_valid = False
            print(f"  ⚠️  Example {i} has non-verbatim evidence")
    
    if all_valid:
        print(f"  ✅ All checked examples have verbatim evidence!")
    else:
        print(f"  ⚠️  Some examples may need manual review")
    
    print("=" * 80)
