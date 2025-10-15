#!/usr/bin/env python3
"""
Test medical guideline scraper with a single condition
"""

from guideline_scraper import MedicalGuidelineScraper

def main():
    print("\n" + "="*80)
    print("  🧪 TESTING MEDICAL GUIDELINE SCRAPER")
    print("="*80 + "\n")
    
    scraper = MedicalGuidelineScraper()
    
    # Test with Abdominal Pain
    print("Testing: Abdominal Pain")
    print("-" * 80)
    
    guideline = scraper.scrape_medlineplus_condition(
        "Abdominal Pain",
        "https://medlineplus.gov/abdominalpain.html"
    )
    
    if guideline:
        print("\n✅ Scraping successful!")
        print(f"\n📊 Stats:")
        print(f"  - Content length: {len(guideline['full_content'])} characters")
        print(f"  - Sections extracted: {guideline['metadata']['sections_extracted']}")
        print(f"  - Symptoms: {len(guideline['symptoms'])}")
        print(f"  - Red flags: {len(guideline['red_flags'])}")
        print(f"  - Questions: {len(guideline['questions'])}")
        
        print(f"\n📄 Content preview (first 500 chars):")
        print("-" * 80)
        print(guideline['full_content'][:500])
        print("...")
        
        if len(guideline['symptoms']) > 0:
            print(f"\n🩺 Symptoms found:")
            for symptom in guideline['symptoms'][:5]:
                print(f"  - {symptom}")
        
        if len(guideline['red_flags']) > 0:
            print(f"\n⚠️  Red flags found:")
            for flag in guideline['red_flags'][:3]:
                print(f"  - {flag}")
        
        # Save it
        print("\n💾 Saving guideline...")
        scraper.save_guideline(guideline)
        
        # Convert to RAG format
        print("\n📤 Converting to RAG format...")
        rag_text = scraper.convert_guideline_to_rag_text(guideline)
        print(f"RAG format length: {len(rag_text)} chars")
        
        print("\n✅ Test complete!")
        
    else:
        print("\n❌ Scraping failed!")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

