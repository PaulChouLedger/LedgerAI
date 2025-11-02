#!/usr/bin/env python3
"""Verify medical_rules.json against actual guidelines"""
import json
from pathlib import Path

# Load medical_rules.json
with open('config/medical_rules.json', 'r') as f:
    rules = json.load(f)

# Get all guideline files
guidelines_dir = Path('medical/guidelines')
issues = []

for json_file in sorted(guidelines_dir.glob('**/*.json')):
    try:
        with open(json_file, 'r') as f:
            guideline = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        continue
    
    condition_name = guideline.get('condition', '')
    if not condition_name:
        continue
    
    organ_system_dir = json_file.parent.name
    
    # Map directory to organ system
    dir_to_system = {
        'GI': 'GI', 'CARDIO': 'CARDIO', 'PULMONARY': 'PULMONARY',
        'NEURO': 'NEURO', 'MSK': 'MSK', 'RENAL': 'RENAL',
        'GU': 'GU', 'GYN': 'GYN', 'DERM': 'DERM'
    }
    
    organ_system = dir_to_system.get(organ_system_dir)
    if not organ_system or organ_system not in rules:
        continue
    
    # Get anatomical_type from guideline
    anatomical_type = None
    location_data = guideline.get('key_features', {}).get('structured_oldcarts', {}).get('location', {})
    if location_data:
        anatomical_type = location_data.get('anatomical_type')
    
    # Check if condition exists in medical_rules.json
    found = False
    found_in = []
    
    if organ_system in rules and 'anatomical_regions' in rules[organ_system]:
        for region, conditions in rules[organ_system]['anatomical_regions'].items():
            if condition_name in conditions:
                found = True
                found_in.append(region)
    
    # Also check old structure
    for region in ['right_only', 'left_only', 'bilateral', 'midline']:
        if region in rules[organ_system]:
            if condition_name in rules[organ_system][region]:
                found = True
                found_in.append(region)
    
    if not found:
        issues.append({
            'condition': condition_name,
            'organ_system': organ_system,
            'issue': 'MISSING from medical_rules.json',
            'anatomical_type': anatomical_type
        })
    elif anatomical_type:
        # Check if anatomical_type matches
        expected_regions = {
            'right_upper': 'right_upper_quadrant',
            'right_lower': 'right_lower_quadrant',
            'left_upper': 'left_upper_quadrant',
            'left_lower': 'left_lower_quadrant',
            'midline': 'midline'
        }
        expected = expected_regions.get(anatomical_type, anatomical_type)
        if expected not in found_in and anatomical_type not in ['bilateral', 'unilateral']:
            issues.append({
                'condition': condition_name,
                'organ_system': organ_system,
                'issue': f'MISMATCH: guideline has anatomical_type="{anatomical_type}" but found in {found_in}',
                'anatomical_type': anatomical_type,
                'found_in': found_in
            })

# Print issues
print(f"Found {len(issues)} potential issues:\n")
for issue in sorted(issues, key=lambda x: (x['organ_system'], x['condition'])):
    print(f"[{issue['organ_system']}] {issue['condition']}")
    print(f"  Issue: {issue['issue']}")
    if 'anatomical_type' in issue and issue['anatomical_type']:
        print(f"  Guideline anatomical_type: {issue['anatomical_type']}")
    if 'found_in' in issue:
        print(f"  Found in medical_rules.json: {issue['found_in']}")
    print()

