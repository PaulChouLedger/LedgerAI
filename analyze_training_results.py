#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Results Analysis Script
=================================

Analyzes test results to identify failure patterns and suggest improvements.
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Any

def analyze_failures(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze test failures and categorize them."""
    
    failures = {
        "missing_entities": [],
        "wrong_entities": [],
        "incomplete_extraction": [],
        "wrong_synthesis": [],
        "ignored_high_relevance": [],
        "format_issues": []
    }
    
    for result in test_results:
        if not result.get("passed", False):
            issues = result.get("issues", [])
            query = result.get("query", "")
            
            for issue in issues:
                if "Missing expected entity" in issue:
                    failures["missing_entities"].append({
                        "query": query,
                        "issue": issue
                    })
                elif "Should not contain" in issue:
                    failures["wrong_entities"].append({
                        "query": query,
                        "issue": issue
                    })
                elif "incomplete" in issue.lower():
                    failures["incomplete_extraction"].append({
                        "query": query,
                        "issue": issue
                    })
                elif "synthesis" in issue.lower() or "synthesize" in issue.lower():
                    failures["wrong_synthesis"].append({
                        "query": query,
                        "issue": issue
                    })
                elif "HIGH relevance" in issue or "ignored" in issue.lower():
                    failures["ignored_high_relevance"].append({
                        "query": query,
                        "issue": issue
                    })
                else:
                    failures["format_issues"].append({
                        "query": query,
                        "issue": issue
                    })
    
    return failures

def suggest_dataset_improvements(failures: Dict[str, Any]) -> List[str]:
    """Suggest dataset improvements based on failures."""
    suggestions = []
    
    if failures["missing_entities"]:
        count = len(failures["missing_entities"])
        suggestions.append(
            f"❌ Missing Entities ({count} cases):\n"
            f"   - Add 200+ examples where entities are spread across multiple chunks\n"
            f"   - Ensure examples show reading ALL HIGH relevance chunks\n"
            f"   - Add examples with 4-5 chunks, each containing different entities"
        )
    
    if failures["wrong_entities"]:
        count = len(failures["wrong_entities"])
        suggestions.append(
            f"❌ Wrong Entities ({count} cases):\n"
            f"   - Add examples with multiple companies/entities in same chunks\n"
            f"   - Ensure examples show filtering by company name\n"
            f"   - Add negative examples (entities from wrong company)"
        )
    
    if failures["incomplete_extraction"]:
        count = len(failures["incomplete_extraction"])
        suggestions.append(
            f"❌ Incomplete Extraction ({count} cases):\n"
            f"   - Add examples requiring ALL entities (not just first found)\n"
            f"   - Ensure examples show processing ALL chunks completely\n"
            f"   - Add examples with entities in last chunks"
        )
    
    if failures["wrong_synthesis"]:
        count = len(failures["wrong_synthesis"])
        suggestions.append(
            f"❌ Wrong Synthesis ({count} cases):\n"
            f"   - Add more synthesis examples (200+)\n"
            f"   - Ensure examples show combining info from multiple chunks\n"
            f"   - Add examples requiring reasoning across chunks"
        )
    
    if failures["ignored_high_relevance"]:
        count = len(failures["ignored_high_relevance"])
        suggestions.append(
            f"❌ Ignored HIGH Relevance Chunks ({count} cases):\n"
            f"   - Add examples emphasizing score >= 0.70 = HIGH\n"
            f"   - Add examples with mixed relevance (some HIGH, some MEDIUM)\n"
            f"   - Ensure examples show reading ALL HIGH chunks"
        )
    
    if failures["format_issues"]:
        count = len(failures["format_issues"])
        suggestions.append(
            f"❌ Format Issues ({count} cases):\n"
            f"   - Review dataset format consistency\n"
            f"   - Ensure all examples follow same structure\n"
            f"   - Check system prompt clarity"
        )
    
    return suggestions

def generate_improved_dataset_config(failures: Dict[str, Any]) -> Dict[str, Any]:
    """Generate configuration for improved dataset."""
    config = {
        "total_examples": 1000,
        "additional_examples": {},
        "focus_areas": []
    }
    
    # Calculate additional examples needed
    total_failures = sum(len(v) for v in failures.values())
    
    if total_failures > 0:
        # Add 2x examples for each failure type
        for failure_type, cases in failures.items():
            if cases:
                count = len(cases)
                config["additional_examples"][failure_type] = count * 2
                config["focus_areas"].append(failure_type)
        
        config["total_examples"] = 1000 + sum(config["additional_examples"].values())
    
    return config

def print_analysis_report(failures: Dict[str, Any], suggestions: List[str], config: Dict[str, Any]):
    """Print formatted analysis report."""
    print("=" * 80)
    print("Training Results Analysis Report")
    print("=" * 80)
    
    print("\n📊 Failure Summary:")
    print("-" * 80)
    total_failures = sum(len(v) for v in failures.values())
    print(f"Total failures: {total_failures}")
    
    for failure_type, cases in failures.items():
        if cases:
            print(f"  - {failure_type}: {len(cases)} cases")
    
    print("\n💡 Suggested Improvements:")
    print("-" * 80)
    for suggestion in suggestions:
        print(suggestion)
    
    print("\n📝 Recommended Dataset Configuration:")
    print("-" * 80)
    print(f"Total examples: {config['total_examples']}")
    print(f"Focus areas: {', '.join(config['focus_areas'])}")
    print("\nAdditional examples needed:")
    for failure_type, count in config["additional_examples"].items():
        print(f"  - {failure_type}: +{count} examples")
    
    print("\n" + "=" * 80)

# Example usage
if __name__ == "__main__":
    # This would be populated from actual test results
    # For now, showing structure
    
    print("=" * 80)
    print("Training Results Analysis")
    print("=" * 80)
    print("\nTo use this script:")
    print("1. Run your test suite and collect results")
    print("2. Format results as list of dicts with 'passed', 'query', 'issues'")
    print("3. Call analyze_failures(results)")
    print("4. Review suggestions and update dataset accordingly")
    print("\nExample test result format:")
    print(json.dumps({
        "query": "who are the co-founders of TechCorp?",
        "passed": False,
        "issues": [
            "Missing expected entity: Mike Johnson",
            "Missing expected entity: Sarah Williams"
        ]
    }, indent=2))

