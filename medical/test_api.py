#!/usr/bin/env python3
"""
Test MedlinePlus API client with a single condition
"""

from api_guideline_scraper import MedicalAPIClient

def main():
    print("\n" + "="*80)
    print("  🧪 TESTING MEDLINEPLUS API CLIENT")
    print("="*80 + "\n")
    
    client = MedicalAPIClient()
    
    # Test with Abdominal Pain (SNOMED CT: 21522001)
    print("Testing: Abdominal Pain")
    print("-" * 80)
    
    guideline = client.get_condition_info("Abdominal Pain", "21522001")
    
    if guideline:
        print("\n✅ API fetch successful!")
        print(f"\n📊 Stats:")
        print(f"  - Topics: {len(guideline['topics'])}")
        print(f"  - Content length: {len(guideline['full_content'])} characters")
        print(f"  - Category: {guideline['category']}")
        
        print(f"\n📋 Topics found:")
        for i, topic in enumerate(guideline['topics'][:5], 1):
            print(f"  {i}. {topic['title']}")
            if topic['summary']:
                preview = topic['summary'][:100] + "..." if len(topic['summary']) > 100 else topic['summary']
                print(f"     Summary: {preview}")
        
        print(f"\n📄 Full content preview (first 500 chars):")
        print("-" * 80)
        print(guideline['full_content'][:500])
        print("...")
        
        # Save it
        print("\n💾 Saving guideline...")
        client.save_guideline(guideline)
        
        # Convert to RAG format
        print("\n📤 Exporting to RAG format...")
        rag_path = client.export_to_rag_format(guideline)
        
        if rag_path:
            print(f"✅ Exported to: {rag_path}")
            
            # Show RAG format preview
            with open(rag_path, 'r') as f:
                rag_content = f.read()
            
            print(f"\n📄 RAG format preview:")
            print("-" * 80)
            print(rag_content[:500])
            print("...")
        
        print("\n✅ Test complete!")
        
    else:
        print("\n❌ API fetch failed!")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

