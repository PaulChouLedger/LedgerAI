#!/usr/bin/env python3
"""
Medical Guideline Web Scraper

Scrapes authoritative medical guidelines from trusted sources:
- CDC (Centers for Disease Control)
- NIH/MedlinePlus (National Institutes of Health)
- WHO (World Health Organization)
- NHS UK (National Health Service)

Stores guidelines in structured format for RAG ingestion.
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import hashlib
import re

class MedicalGuidelineScraper:
    """
    Scrapes and structures medical guidelines from authoritative sources
    """
    
    def __init__(self, output_dir: str = None):
        # Default to repo root's data/input/medical_guidelines
        if output_dir is None:
            # Get absolute path to repo root
            script_dir = Path(__file__).resolve().parent  # /path/to/LedgerAI/medical
            repo_root = script_dir.parent  # /path/to/LedgerAI
            output_dir = repo_root / "data" / "input" / "medical_guidelines"
        
        self.output_dir = Path(output_dir).resolve()  # Ensure absolute path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Scraper] 📂 Output directory: {self.output_dir}")
        
        # Rate limiting (be respectful to servers)
        self.request_delay = 2.0  # seconds between requests
        self.last_request_time = 0
        
        # User agent (identify ourselves)
        self.headers = {
            'User-Agent': 'LedgerAI-Medical-Assistant/1.0 (Educational/Research; Contact: your-email@example.com)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        print(f"[Scraper] ✅ Initialized - output: {self.output_dir}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL with error handling and rate limiting"""
        self._rate_limit()
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"[Scraper] ❌ Error fetching {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that interfere with RAG
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        return text.strip()
    
    def _generate_guideline_id(self, source: str, title: str) -> str:
        """Generate unique guideline ID"""
        content = f"{source}_{title}_{datetime.now().year}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    # === CDC Scraper ===
    
    def scrape_cdc_condition(self, condition_name: str, url: str) -> Optional[Dict]:
        """
        Scrape CDC guideline for a specific condition
        
        Args:
            condition_name: Name of condition (e.g., "Chest Pain", "Diabetes")
            url: CDC URL for the condition
            
        Returns:
            Structured guideline dictionary
        """
        print(f"[CDC] 🔍 Scraping: {condition_name}")
        
        html = self._fetch_url(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract main content
        # CDC typically uses <div class="syndicate"> or <article>
        content_div = soup.find('div', class_='syndicate') or soup.find('article') or soup.find('main')
        
        if not content_div:
            print(f"[CDC] ⚠️ Could not find main content in {url}")
            return None
        
        # Extract text
        content_text = content_div.get_text(separator='\n', strip=True)
        content_text = self._clean_text(content_text)
        
        # Extract sections (CDC often uses <h2>, <h3> for structure)
        sections = {}
        current_section = "overview"
        sections[current_section] = []
        
        for element in content_div.find_all(['h2', 'h3', 'p', 'ul', 'ol']):
            if element.name in ['h2', 'h3']:
                current_section = element.get_text(strip=True).lower()
                sections[current_section] = []
            else:
                text = element.get_text(strip=True)
                if text:
                    sections[current_section].append(text)
        
        # Build structured guideline
        guideline = {
            "guideline_id": self._generate_guideline_id("CDC", condition_name),
            "source": "CDC",
            "source_url": url,
            "condition": condition_name,
            "title": soup.find('h1').get_text(strip=True) if soup.find('h1') else condition_name,
            "scraped_date": datetime.now().isoformat(),
            "last_verified": datetime.now().isoformat(),
            "category": self._categorize_condition(condition_name),
            "sections": sections,
            "full_content": content_text,
            "metadata": {
                "source_type": "government_health_agency",
                "authority_level": "high",
                "evidence_based": True
            }
        }
        
        # Extract key information
        guideline["symptoms"] = self._extract_symptoms(content_text)
        guideline["red_flags"] = self._extract_red_flags(content_text)
        guideline["questions"] = self._extract_questions(content_text)
        
        print(f"[CDC] ✅ Scraped {condition_name}: {len(content_text)} chars, {len(sections)} sections")
        
        return guideline
    
    def _categorize_condition(self, condition: str) -> str:
        """Categorize condition by medical specialty"""
        condition_lower = condition.lower()
        
        categories = {
            'cardiovascular': ['heart', 'cardiac', 'chest pain', 'hypertension', 'stroke'],
            'respiratory': ['lung', 'asthma', 'copd', 'pneumonia', 'cough', 'dyspnea'],
            'gastrointestinal': ['stomach', 'gastro', 'intestine', 'digestive', 'nausea', 'diarrhea', 'pancreatitis'],
            'neurological': ['brain', 'neuro', 'seizure', 'headache', 'migraine', 'dementia'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'metabolic'],
            'infectious': ['infection', 'covid', 'flu', 'virus', 'bacteria'],
            'musculoskeletal': ['bone', 'joint', 'muscle', 'arthritis', 'fracture'],
            'renal': ['kidney', 'renal', 'urinary', 'bladder'],
            'dermatology': ['skin', 'rash', 'dermatitis', 'eczema']
        }
        
        for category, keywords in categories.items():
            if any(keyword in condition_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptom keywords from guideline text"""
        symptom_patterns = [
            r'symptoms? (?:include|are|may include):?\s*([^.]+)',
            r'signs and symptoms:?\s*([^.]+)',
            r'(?:you|patient) may (?:experience|have|notice):?\s*([^.]+)'
        ]
        
        symptoms = []
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                symptom_text = match.group(1)
                # Split by commas and clean
                items = [s.strip() for s in re.split(r'[,;]', symptom_text)]
                symptoms.extend(items)
        
        return list(set(symptoms))[:20]  # Limit to top 20
    
    def _extract_red_flags(self, text: str) -> List[str]:
        """Extract red flag/emergency indicators"""
        red_flag_patterns = [
            r'seek (?:immediate|emergency) (?:medical )?(?:attention|care|help)(?: if)?:?\s*([^.]+)',
            r'call 911 (?:if|when):?\s*([^.]+)',
            r'warning signs?:?\s*([^.]+)',
            r'emergency symptoms?:?\s*([^.]+)',
            r'(?:go to|visit) (?:the )?(?:emergency room|ER|hospital) (?:if|when):?\s*([^.]+)'
        ]
        
        red_flags = []
        for pattern in red_flag_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                flag_text = match.group(1)
                items = [s.strip() for s in re.split(r'[,;]', flag_text)]
                red_flags.extend(items)
        
        return list(set(red_flags))[:15]  # Limit to top 15
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract medical questions from guideline"""
        # Find sentences ending with ?
        questions = re.findall(r'([A-Z][^.!?]*\?)', text)
        
        # Filter for relevant medical questions
        filtered = []
        for q in questions:
            q_lower = q.lower()
            # Keep questions about symptoms, duration, severity
            if any(keyword in q_lower for keyword in ['pain', 'feel', 'experience', 'symptom', 'how long', 'when', 'where', 'severe']):
                filtered.append(q.strip())
        
        return filtered[:10]  # Limit to 10 questions
    
    # === NIH/MedlinePlus Scraper ===
    
    def scrape_medlineplus_condition(self, condition_name: str, url: str) -> Optional[Dict]:
        """
        Scrape MedlinePlus guideline for a specific condition
        
        MedlinePlus provides excellent patient-friendly medical information
        """
        print(f"[MedlinePlus] 🔍 Scraping: {condition_name}")
        
        html = self._fetch_url(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # MedlinePlus structure: Try multiple selectors in order of preference
        content_sections = []
        
        # Method 1: Find all content sections (MedlinePlus uses specific divs for content)
        section_containers = soup.find_all('div', class_='section')
        if section_containers:
            for section in section_containers:
                section_text = section.get_text(separator='\n', strip=True)
                if len(section_text) > 50:  # Only add substantive sections
                    content_sections.append(section_text)
        
        # Method 2: Look for main content area
        if not content_sections:
            main_content = soup.find('div', id='mplus-content') or soup.find('div', class_='main-content')
            if main_content:
                # Extract all paragraphs and lists
                for elem in main_content.find_all(['p', 'ul', 'ol', 'div']):
                    text = elem.get_text(separator=' ', strip=True)
                    if len(text) > 20:
                        content_sections.append(text)
        
        # Method 3: Fallback to all paragraphs in body
        if not content_sections:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 30:  # Filter out navigation/footer text
                    content_sections.append(text)
        
        if not content_sections:
            print(f"[MedlinePlus] ⚠️ Could not extract content from {url}")
            return None
        
        # Combine all sections
        content_text = '\n\n'.join(content_sections)
        content_text = self._clean_text(content_text)
        
        # Log what we extracted
        print(f"[MedlinePlus] 📄 Extracted {len(content_sections)} sections, {len(content_text)} chars total")
        if len(content_text) < 200:
            print(f"[MedlinePlus] ⚠️ WARNING: Very short content ({len(content_text)} chars)")
            print(f"[MedlinePlus] 📝 Content preview: {content_text[:500]}")
        
        guideline = {
            "guideline_id": self._generate_guideline_id("MedlinePlus", condition_name),
            "source": "NIH MedlinePlus",
            "source_url": url,
            "condition": condition_name,
            "title": soup.find('h1').get_text(strip=True) if soup.find('h1') else condition_name,
            "scraped_date": datetime.now().isoformat(),
            "last_verified": datetime.now().isoformat(),
            "category": self._categorize_condition(condition_name),
            "full_content": content_text,
            "symptoms": self._extract_symptoms(content_text),
            "red_flags": self._extract_red_flags(content_text),
            "questions": self._extract_questions(content_text),
            "metadata": {
                "source_type": "government_health_agency",
                "authority_level": "high",
                "patient_friendly": True,
                "sections_extracted": len(content_sections)
            }
        }
        
        print(f"[MedlinePlus] ✅ Scraped {condition_name}: {len(content_text)} chars, {len(guideline['symptoms'])} symptoms, {len(guideline['red_flags'])} red flags")
        
        return guideline
    
    # === Guideline Management ===
    
    def save_guideline(self, guideline: Dict):
        """Save guideline to JSON file"""
        filename = f"{guideline['source'].replace(' ', '_')}_{guideline['condition'].replace(' ', '_')}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(guideline, f, indent=2, ensure_ascii=False)
            print(f"[Scraper] 💾 Saved: {filepath.name}")
            return filepath
        except Exception as e:
            print(f"[Scraper] ❌ Error saving {filename}: {e}")
            return None
    
    def scrape_and_save(self, source: str, condition_name: str, url: str) -> bool:
        """Scrape from source and save guideline"""
        guideline = None
        
        if source.lower() == 'cdc':
            guideline = self.scrape_cdc_condition(condition_name, url)
        elif source.lower() in ['medlineplus', 'nih']:
            guideline = self.scrape_medlineplus_condition(condition_name, url)
        else:
            print(f"[Scraper] ❌ Unknown source: {source}")
            return False
        
        if guideline:
            self.save_guideline(guideline)
            return True
        return False
    
    def batch_scrape_common_conditions(self):
        """
        Scrape guidelines for most common medical conditions
        
        This provides a good baseline for dynamic medical assessment
        """
        print("\n" + "="*80)
        print("  🏥 SCRAPING COMMON MEDICAL CONDITION GUIDELINES")
        print("="*80 + "\n")
        
        # Common conditions with public URLs
        conditions = [
            # Cardiovascular
            ("MedlinePlus", "Chest Pain", "https://medlineplus.gov/chestpain.html"),
            ("MedlinePlus", "Heart Attack", "https://medlineplus.gov/heartattack.html"),
            ("MedlinePlus", "Hypertension", "https://medlineplus.gov/highbloodpressure.html"),
            
            # Respiratory
            ("MedlinePlus", "Asthma", "https://medlineplus.gov/asthma.html"),
            ("MedlinePlus", "Pneumonia", "https://medlineplus.gov/pneumonia.html"),
            ("MedlinePlus", "COPD", "https://medlineplus.gov/copd.html"),
            
            # Gastrointestinal
            ("MedlinePlus", "Pancreatitis", "https://medlineplus.gov/pancreatitis.html"),
            ("MedlinePlus", "Abdominal Pain", "https://medlineplus.gov/abdominalpain.html"),
            
            # Endocrine
            ("MedlinePlus", "Diabetes", "https://medlineplus.gov/diabetes.html"),
            
            # Neurological
            ("MedlinePlus", "Stroke", "https://medlineplus.gov/stroke.html"),
            ("MedlinePlus", "Headache", "https://medlineplus.gov/headache.html"),
            ("MedlinePlus", "Seizures", "https://medlineplus.gov/seizures.html"),
        ]
        
        success_count = 0
        fail_count = 0
        
        for source, condition, url in conditions:
            print(f"\n[Batch] 📥 Processing: {condition} from {source}")
            
            if self.scrape_and_save(source, condition, url):
                success_count += 1
            else:
                fail_count += 1
            
            # Small delay between conditions
            time.sleep(1.0)
        
        print("\n" + "="*80)
        print(f"  ✅ BATCH SCRAPING COMPLETE")
        print("="*80)
        print(f"  Success: {success_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Output:  {self.output_dir}")
        print("="*80 + "\n")
        
        return success_count
    
    # === Guideline to RAG Format Conversion ===
    
    def convert_guideline_to_rag_text(self, guideline: Dict) -> str:
        """
        Convert structured guideline to RAG-friendly text format
        
        Creates a comprehensive text document that RAG can chunk and embed
        """
        sections = []
        
        # Header
        sections.append(f"MEDICAL GUIDELINE: {guideline['title']}")
        sections.append(f"Source: {guideline['source']}")
        sections.append(f"Condition: {guideline['condition']}")
        sections.append(f"Category: {guideline['category']}")
        sections.append(f"Last Updated: {guideline['scraped_date']}")
        sections.append("")
        
        # Symptoms
        if guideline.get('symptoms'):
            sections.append("COMMON SYMPTOMS:")
            for symptom in guideline['symptoms']:
                sections.append(f"  - {symptom}")
            sections.append("")
        
        # Red Flags / Emergency Signs
        if guideline.get('red_flags'):
            sections.append("EMERGENCY WARNING SIGNS:")
            for flag in guideline['red_flags']:
                sections.append(f"  ⚠️  {flag}")
            sections.append("")
        
        # Diagnostic Questions
        if guideline.get('questions'):
            sections.append("DIAGNOSTIC QUESTIONS TO ASK:")
            for question in guideline['questions']:
                sections.append(f"  - {question}")
            sections.append("")
        
        # Full Content
        sections.append("FULL GUIDELINE CONTENT:")
        sections.append(guideline['full_content'])
        
        return '\n'.join(sections)
    
    def export_all_to_rag_format(self):
        """
        Export all scraped guidelines to RAG-friendly text files
        
        Saves directly to data/input/ for immediate RAG ingestion
        """
        print("\n[Export] 📤 Converting guidelines to RAG format...")
        
        json_files = list(self.output_dir.glob("*.json"))
        
        if not json_files:
            print("[Export] ⚠️ No JSON guidelines found to export")
            return 0
        
        # Export directly to data/input/ (RAG reads from here)
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        rag_input_dir = repo_root / "data" / "input"
        rag_input_dir.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    guideline = json.load(f)
                
                # Convert to RAG text format
                rag_text = self.convert_guideline_to_rag_text(guideline)
                
                # Save directly to data/input/
                txt_filename = json_file.stem + ".txt"
                txt_filepath = rag_input_dir / txt_filename
                
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(rag_text)
                
                print(f"[Export] ✅ Exported: {txt_filename}")
                exported_count += 1
                
            except Exception as e:
                print(f"[Export] ❌ Error exporting {json_file.name}: {e}")
        
        print(f"\n[Export] ✅ Exported {exported_count} guidelines directly to {rag_input_dir}")
        print(f"[Export] 💡 Ready for RAG ingestion (no copying needed)")
        
        return exported_count


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  🏥 MEDICAL GUIDELINE SCRAPER")
    print("="*80 + "\n")
    
    scraper = MedicalGuidelineScraper()
    
    # Scrape common conditions
    scraped = scraper.batch_scrape_common_conditions()
    
    if scraped > 0:
        # Convert to RAG format
        print("\n" + "="*80)
        print("  📤 CONVERTING TO RAG FORMAT")
        print("="*80 + "\n")
        
        exported = scraper.export_all_to_rag_format()
        
        print("\n" + "="*80)
        print("  ✅ SCRAPING COMPLETE!")
        print("="*80)
        print(f"\n  📦 Medical guidelines ready in: data/input/")
        print(f"  📁 JSON backups saved in: data/input/medical_guidelines/")
        print(f"\n  Next step:")
        print(f"  → Run: python3 medical/ingest_guidelines.py")
        print(f"  → This will build embeddings and make guidelines available in RAG")
        print("\n" + "="*80 + "\n")
    else:
        print("\n❌ No guidelines scraped - check network connection\n")


if __name__ == "__main__":
    main()

