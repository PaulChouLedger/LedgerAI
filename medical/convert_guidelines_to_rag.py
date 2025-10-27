#!/usr/bin/env python3
"""
Convert curated diagnostic guidelines (JSON) to RAG-optimized text format

This creates comprehensive text documents that the LLM can use for
dynamic diagnostic questioning.
"""

import json
from pathlib import Path
from typing import Dict

class GuidelineToRAGConverter:
    """Converts structured diagnostic guidelines to RAG-friendly text"""
    
    def __init__(self, 
                 guidelines_dir: str = None,
                 output_dir: str = None):
        
        if guidelines_dir is None:
            script_dir = Path(__file__).resolve().parent
            repo_root = script_dir.parent
            # Try both possible paths for different systems
            if (repo_root / "llm-medical-container" / "medical" / "guidelines").exists():
                guidelines_dir = repo_root / "llm-medical-container" / "medical" / "guidelines"
            elif (repo_root / "llm-container" / "medical" / "guidelines").exists():
                guidelines_dir = repo_root / "llm-container" / "medical" / "guidelines"
            else:
                # Default to llm-medical-container
                guidelines_dir = repo_root / "llm-medical-container" / "medical" / "guidelines"
        
        if output_dir is None:
            script_dir = Path(__file__).resolve().parent
            repo_root = script_dir.parent
            output_dir = repo_root / "data" / "input"
        
        self.guidelines_dir = Path(guidelines_dir)
        self.output_dir = Path(output_dir)
        
        print(f"[Converter] 📂 Guidelines dir: {self.guidelines_dir}")
        print(f"[Converter] 📂 Output dir: {self.output_dir}")
    
    def convert_to_rag_text(self, guideline: Dict) -> str:
        """
        Convert structured guideline to comprehensive RAG text
        
        Optimized for LLM consumption and dynamic questioning
        """
        sections = []
        
        # === HEADER ===
        sections.append("="*80)
        sections.append(f"DIAGNOSTIC GUIDELINE: {guideline['condition'].upper()}")
        sections.append("="*80)
        sections.append("")
        
        # === METADATA ===
        sections.append(f"Condition: {guideline['condition']}")
        sections.append(f"Category: {guideline.get('category', 'N/A')}")
        sections.append(f"Urgency Level: {guideline.get('urgency', 'N/A')}")
        sections.append(f"Prevalence: {guideline.get('prevalence', 'N/A')}")
        
        if guideline.get('icd10'):
            sections.append(f"ICD-10 Code: {guideline['icd10']}")
        if guideline.get('snomed'):
            sections.append(f"SNOMED Code: {guideline['snomed']}")
        if guideline.get('sex'):
            sections.append(f"Sex: {guideline['sex']}")
        
        sections.append("")
        sections.append("-"*80)
        sections.append("")
        
        # === CHIEF COMPLAINT TRIGGERS ===
        if guideline.get('chief_complaint_triggers'):
            sections.append("CHIEF COMPLAINT TRIGGERS:")
            sections.append("This diagnosis should be considered when patient presents with:")
            for trigger in guideline['chief_complaint_triggers']:
                sections.append(f"  • {trigger}")
            sections.append("")
        
        # === CLASSIC PRESENTATION ===
        if guideline.get('key_features'):
            features = guideline['key_features']
            
            sections.append("CLASSIC PRESENTATION:")
            sections.append(features.get('classic_presentation', 'N/A'))
            sections.append("")
            
            sections.append("TYPICAL PATIENT DEMOGRAPHICS:")
            sections.append(features.get('typical_demographics', 'N/A'))
            sections.append("")
            
            if features.get('pathophysiology_brief'):
                sections.append("PATHOPHYSIOLOGY:")
                sections.append(features['pathophysiology_brief'])
                sections.append("")
            
            # === STRUCTURED OLDCARTS DATA (CRITICAL FOR DYNAMIC QUESTIONING) ===
            if features.get('structured_oldcarts'):
                sections.append("STRUCTURED OLDCARTS ASSESSMENT:")
                sections.append("Use this structured data for dynamic questioning:")
                sections.append("")
                
                structured = features['structured_oldcarts']
                for element, data in structured.items():
                    if isinstance(data, dict):
                        sections.append(f"{element.upper()}:")
                        
                        if data.get('includes'):
                            sections.append(f"  Positive indicators: {', '.join(data['includes'])}")
                        
                        if data.get('excludes'):
                            sections.append(f"  Negative indicators: {', '.join(data['excludes'])}")
                        
                        if data.get('anatomical_type'):
                            sections.append(f"  Anatomical type: {data['anatomical_type']}")
                        
                        sections.append("")
                
                sections.append("-"*40)
                sections.append("")
        
        sections.append("-"*80)
        sections.append("")
        
        # === DIAGNOSTIC QUESTIONS (KEY SECTION FOR LLM) ===
        if guideline.get('diagnostic_questions'):
            sections.append("DIAGNOSTIC QUESTIONING STRATEGY:")
            sections.append("Ask the following questions to establish or rule out this diagnosis:")
            sections.append("")
            
            for i, q in enumerate(guideline['diagnostic_questions'], 1):
                sections.append(f"QUESTION {i}: {q['question_focus'].upper()}")
                sections.append(f"  Diagnostic Value: {q['diagnostic_value']}")
                
                if q.get('expected_positive_responses'):
                    sections.append(f"  Responses suggesting this diagnosis:")
                    for resp in q['expected_positive_responses']:
                        sections.append(f"    ✓ {resp}")
                
                if q.get('negative_responses'):
                    sections.append(f"  Responses arguing against this diagnosis:")
                    for resp in q['negative_responses']:
                        sections.append(f"    ✗ {resp}")
                
                if q.get('context'):
                    sections.append(f"  Clinical Context: {q['context']}")
                
                sections.append("")
        
        sections.append("-"*80)
        sections.append("")
        
        # === RED FLAGS ===
        if guideline.get('red_flags'):
            sections.append("⚠️  EMERGENCY WARNING SIGNS / RED FLAGS:")
            sections.append("If patient reports any of the following, urgent action required:")
            sections.append("")
            
            for flag in guideline['red_flags']:
                if isinstance(flag, dict):
                    sections.append(f"  🚨 {flag.get('finding', flag)}")
                    if flag.get('urgency'):
                        sections.append(f"     Urgency: {flag['urgency']}")
                    if flag.get('action'):
                        sections.append(f"     Action: {flag['action']}")
                else:
                    sections.append(f"  🚨 {flag}")
                sections.append("")
        
        sections.append("-"*80)
        sections.append("")
        
        # === DIFFERENTIAL DIAGNOSES ===
        if guideline.get('differential_diagnoses'):
            sections.append("DIFFERENTIAL DIAGNOSES TO CONSIDER:")
            sections.append("How to distinguish from similar conditions:")
            sections.append("")
            
            for diff in guideline['differential_diagnoses']:
                sections.append(f"  vs. {diff['condition']}:")
                for feature in diff['distinguishing_features']:
                    sections.append(f"    • {feature}")
                sections.append("")
        
        sections.append("-"*80)
        sections.append("")
        
        # === PHYSICAL EXAM ===
        if guideline.get('physical_exam_findings'):
            sections.append("KEY PHYSICAL EXAM FINDINGS:")
            for finding in guideline['physical_exam_findings']:
                sections.append(f"  • {finding['finding']}")
                if finding.get('sensitivity') and finding.get('specificity'):
                    sections.append(f"    Sensitivity: {finding['sensitivity']}, Specificity: {finding['specificity']}")
            sections.append("")
        
        # === DIAGNOSTIC TESTS ===
        if guideline.get('diagnostic_tests'):
            sections.append("DIAGNOSTIC TESTS:")
            for test in guideline['diagnostic_tests']:
                sections.append(f"  • {test['test']}")
                if test.get('indication'):
                    sections.append(f"    Indication: {test['indication']}")
                if test.get('typical_findings'):
                    sections.append(f"    Findings: {test['typical_findings']}")
            sections.append("")
        
        # === TREATMENT ===
        if guideline.get('treatment_summary'):
            sections.append("TREATMENT OVERVIEW:")
            sections.append(guideline['treatment_summary'])
            sections.append("")
        
        # === EVIDENCE SOURCE ===
        if guideline.get('evidence_source'):
            evidence = guideline['evidence_source']
            sections.append("EVIDENCE BASE:")
            if evidence.get('primary_reference'):
                sections.append(f"  Primary Reference: {evidence['primary_reference']}")
            if evidence.get('guideline_source'):
                sections.append(f"  Guideline Source: {evidence['guideline_source']}")
            if evidence.get('last_reviewed'):
                sections.append(f"  Last Reviewed: {evidence['last_reviewed']}")
            sections.append("")
        
        sections.append("="*80)
        sections.append(f"END OF GUIDELINE: {guideline['condition']}")
        sections.append("="*80)
        
        return '\n'.join(sections)
    
    def convert_all_guidelines(self) -> int:
        """Convert all JSON guidelines to RAG text format"""
        # Only convert GI guidelines for now since they have proper structured_oldcarts format
        json_files = list(self.guidelines_dir.glob("GI/*.json"))
        
        if not json_files:
            print("[Converter] ⚠️ No JSON guideline files found")
            return 0
        
        print(f"\n[Converter] 📚 Found {len(json_files)} guideline files to convert")
        
        converted_count = 0
        
        for json_file in json_files:
            try:
                # Load JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    guideline = json.load(f)
                
                # Convert to RAG text
                rag_text = self.convert_to_rag_text(guideline)
                
                # Save to output directory - USE JSON FILENAME for consistency with main.py
                # This ensures GI_Acute_Appendicitis.json → GUIDELINE_GI_Acute_Appendicitis.txt
                base_name = json_file.stem  # Filename without .json extension
                output_filename = f"GUIDELINE_{base_name}.txt"
                output_path = self.output_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(rag_text)
                
                print(f"[Converter] ✅ Converted: {json_file.name} → {output_filename}")
                converted_count += 1
                
            except Exception as e:
                print(f"[Converter] ❌ Error converting {json_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n[Converter] ✅ Converted {converted_count}/{len(json_files)} guidelines")
        print(f"[Converter] 📁 Output: {self.output_dir}")
        
        return converted_count


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  📚 DIAGNOSTIC GUIDELINE TO RAG CONVERTER")
    print("="*80 + "\n")
    
    converter = GuidelineToRAGConverter()
    converted = converter.convert_all_guidelines()
    
    if converted > 0:
        print("\n" + "="*80)
        print("  ✅ CONVERSION COMPLETE!")
        print("="*80)
        print(f"\n  {converted} guidelines ready for RAG ingestion")
        print(f"\n  Next step:")
        print(f"  → Run: python3 medical/ingest_guidelines.py")
        print(f"  → This will build embeddings and make guidelines available")
        print("\n" + "="*80 + "\n")
    else:
        print("\n❌ No guidelines converted\n")


if __name__ == "__main__":
    main()

