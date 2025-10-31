#!/usr/bin/env python3
"""
Location ML Data Extractor
Extracts location data from medical guidelines for ML training
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any
# import pandas as pd  # Optional - can use CSV module instead

class LocationDataExtractor:
    """
    Extract location data from medical guidelines for ML training
    """
    
    def __init__(self, guidelines_dir: str = "/Users/rcabello/Documents/GitHub/LedgerAI/llm-medical-container/medical/guidelines"):
        self.guidelines_dir = Path(guidelines_dir)
        self.location_data = []
        
    def extract_all_guidelines(self) -> List[Dict]:
        """
        Extract location data from all guidelines
        """
        print("🔍 Extracting location data from guidelines...")
        
        # Get all JSON files
        json_files = list(self.guidelines_dir.rglob("*.json"))
        print(f"📁 Found {len(json_files)} guideline files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    guideline = json.load(f)
                
                # Extract location data
                location_info = self._extract_location_info(guideline, json_file)
                if location_info:
                    self.location_data.append(location_info)
                    
            except Exception as e:
                print(f"❌ Error processing {json_file}: {e}")
        
        print(f"✅ Extracted {len(self.location_data)} guidelines with location data")
        return self.location_data
    
    def _extract_location_info(self, guideline: Dict, file_path: Path) -> Dict:
        """
        Extract location information from a single guideline
        """
        condition = guideline.get('condition', '')
        organ_system = file_path.parent.name
        
        # Extract from classic_presentation
        classic_presentation = guideline.get('key_features', {}).get('classic_presentation', '')
        
        # Extract location section - look for LOCATION: until next section
        location_match = re.search(r'LOCATION:\s*([^A-Z]+?)(?=\s+[A-Z]+:|$)', classic_presentation, re.IGNORECASE | re.DOTALL)
        if not location_match:
            # Try alternative - look for LOCATION: until DURATION
            location_match = re.search(r'LOCATION:\s*([^D]+?)(?=\s+DURATION)', classic_presentation, re.IGNORECASE | re.DOTALL)
            if not location_match:
                return None
            
        location_text = location_match.group(1).strip()
        
        # Extract anatomical features
        anatomical_features = self._extract_anatomical_features(location_text)
        
        # Determine anatomical type
        anatomical_type = self._determine_anatomical_type(condition, location_text, anatomical_features)
        
        return {
            'condition': condition,
            'organ_system': organ_system,
            'location_text': location_text,
            'anatomical_features': anatomical_features,
            'anatomical_type': anatomical_type,
            'file_path': str(file_path)
        }
    
    def _extract_anatomical_features(self, location_text: str) -> Dict:
        """
        Extract anatomical features from location text
        """
        text_lower = location_text.lower()
        
        features = {
            'has_right_quadrant': bool(re.search(r'right.*quadrant|ruq|rlq|right.*side', text_lower)),
            'has_left_quadrant': bool(re.search(r'left.*quadrant|luq|llq|left.*side', text_lower)),
            'has_bilateral': bool(re.search(r'bilateral|either side|both sides|unilateral', text_lower)),
            'has_midline': bool(re.search(r'midline|epigastric|periumbilical|central', text_lower)),
            'has_flank': bool(re.search(r'flank|side', text_lower)),
            'has_chest': bool(re.search(r'chest|thoracic', text_lower)),
            'has_back': bool(re.search(r'back|posterior', text_lower)),
            'has_upper': bool(re.search(r'upper|superior', text_lower)),
            'has_lower': bool(re.search(r'lower|inferior', text_lower)),
            'has_anterior': bool(re.search(r'anterior|front', text_lower)),
            'has_posterior': bool(re.search(r'posterior|back', text_lower)),
            'has_radiates': bool(re.search(r'radiates|referred', text_lower)),
            'has_migrates': bool(re.search(r'migrates|moves|travels', text_lower)),
            'has_localizes': bool(re.search(r'localizes|localized|focal', text_lower)),
            'has_diffuse': bool(re.search(r'diffuse|widespread|generalized', text_lower)),
            'spatial_term_count': len(re.findall(r'quadrant|side|flank|epigastric|midline|chest|back', text_lower))
        }
        
        return features
    
    def _determine_anatomical_type(self, condition: str, location_text: str, features: Dict) -> str:
        """
        Determine anatomical type based on condition and features
        """
        # Known bilateral conditions
        bilateral_conditions = [
            'Kidney Stone', 'UTI/Pyelonephritis', 'Acute Gastroenteritis',
            'Severe Constipation', 'IBD Flare', 'IBS', 'Acute Mesenteric Ischemia',
            'Pneumonia', 'Pulmonary Embolism', 'Heart Failure'
        ]
        
        # Known midline conditions  
        midline_conditions = [
            'Peptic Ulcer Disease', 'Acute Gastritis', 'Acute Pancreatitis',
            'Gastric Outlet Obstruction', 'Aortic Dissection', 'Aortic Stenosis'
        ]
        
        # Known right-only conditions
        right_only_conditions = [
            'Acute Appendicitis', 'Acute Cholecystitis', 'Biliary Colic',
            'Acute Cholangitis', 'Acute Hepatitis'
        ]
        
        # Known left-only conditions
        left_only_conditions = [
            'Acute Diverticulitis', 'Sigmoid Volvulus'
        ]
        
        # Check against known conditions
        if condition in bilateral_conditions:
            return 'bilateral'
        elif condition in midline_conditions:
            return 'midline'
        elif condition in right_only_conditions:
            return 'right_only'
        elif condition in left_only_conditions:
            return 'left_only'
        
        # Use features to determine type
        if features['has_bilateral'] or features['has_diffuse']:
            return 'bilateral'
        elif features['has_midline']:
            return 'midline'
        elif features['has_right_quadrant'] and not features['has_left_quadrant']:
            return 'right_only'
        elif features['has_left_quadrant'] and not features['has_right_quadrant']:
            return 'left_only'
        else:
            return 'unknown'
    
    def save_to_csv(self, output_file: str = "location_ml_data.csv"):
        """
        Save extracted data to CSV for ML training
        """
        if not self.location_data:
            print("❌ No data to save. Run extract_all_guidelines() first.")
            return
        
        # Save to CSV using standard library
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if self.location_data:
                fieldnames = self.location_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.location_data)
        
        print(f"💾 Saved {len(self.location_data)} records to {output_file}")
        
        # Print summary
        print("\n📊 LOCATION DATA SUMMARY:")
        print(f"Total guidelines: {len(self.location_data)}")
        
        # Count organ systems
        organ_systems = set(data['organ_system'] for data in self.location_data)
        print(f"Organ systems: {len(organ_systems)}")
        
        # Count anatomical types
        anatomical_types = {}
        for data in self.location_data:
            atype = data['anatomical_type']
            anatomical_types[atype] = anatomical_types.get(atype, 0) + 1
        print(f"Anatomical types: {anatomical_types}")
        
        return self.location_data
    
    def print_sample_data(self, n: int = 5):
        """
        Print sample extracted data
        """
        if not self.location_data:
            print("❌ No data available. Run extract_all_guidelines() first.")
            return
        
        print(f"\n📋 SAMPLE LOCATION DATA ({n} records):")
        for i, data in enumerate(self.location_data[:n]):
            print(f"\n{i+1}. {data['condition']} ({data['organ_system']})")
            print(f"   Type: {data['anatomical_type']}")
            print(f"   Location: {data['location_text']}")
            print(f"   Features: {data['anatomical_features']}")

# Example usage
if __name__ == "__main__":
    extractor = LocationDataExtractor()
    
    # Extract all location data
    data = extractor.extract_all_guidelines()
    
    # Print sample data
    extractor.print_sample_data(10)
    
    # Save to CSV
    df = extractor.save_to_csv("location_ml_data.csv")
    
    print("\n✅ Location data extraction complete!")
