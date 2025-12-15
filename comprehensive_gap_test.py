#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Gap Test Suite - 100 Diverse Examples
====================================================

Tests the fine-tuned model for ALL potential training gaps with 100 diverse examples
covering:
- Cross-company filtering
- Role filtering
- Not found handling
- List extraction
- Analytical queries
- Relationship queries
- Comparison queries
- Process queries
- Personal reflection queries
- Business management queries
- Co-founder queries with mixed content

Usage in Colab:
    from comprehensive_gap_test import run_comprehensive_tests
    results = run_comprehensive_tests(model, tokenizer, "unsloth")
"""

import re
from typing import Dict, List, Any

# ============================================================================
# 100 COMPREHENSIVE TEST CASES
# ============================================================================

COMPREHENSIVE_TESTS = [
    # ========================================================================
    # CROSS-COMPANY FILTERING (Tests 1-10)
    # ========================================================================
    {
        "id": 1,
        "category": "cross_company",
        "name": "Cross-Company: Two Companies, Multiple Chunks",
        "query": "who are the co-founders of TechCorp?",
        "chunks": [
            {"text": "John Smith is Co-Founder of TechCorp. Sarah Jones is Co-Founder of DataSystems.", "score": 0.85},
            {"text": "Mike Brown is Co-Founder of TechCorp. Alice Williams is Co-Founder of DataSystems.", "score": 0.82}
        ],
        "expected": ["John Smith", "Mike Brown"],
        "exclude": ["Sarah Jones", "Alice Williams"]
    },
    {
        "id": 2,
        "category": "cross_company",
        "name": "Cross-Company: Same Chunk, Different Companies",
        "query": "who are the co-founders of AlphaCorp?",
        "chunks": [
            {"text": "David Chen is Co-Founder of AlphaCorp. Lisa Wang is Co-Founder of BetaCorp. Robert Kim is Co-Founder of AlphaCorp.", "score": 0.90}
        ],
        "expected": ["David Chen", "Robert Kim"],
        "exclude": ["Lisa Wang"]
    },
    {
        "id": 3,
        "category": "cross_company",
        "name": "Cross-Company: Query Company Name Variation",
        "query": "who are the co-founders of Ledger AI?",
        "chunks": [
            {"text": "Paul Chou is Co-Founder of LedgerAI. Bob Carella is Co-Founder of LedgerAI. Jorge Guinovart is Co-Founder of LedgerAI.", "score": 0.90}
        ],
        "expected": ["Paul Chou", "Bob Carella", "Jorge Guinovart"],
        "exclude": []
    },
    {
        "id": 4,
        "category": "cross_company",
        "name": "Cross-Company: Three Companies Mixed",
        "query": "who are the co-founders of GammaCorp?",
        "chunks": [
            {"text": "Emma White is Co-Founder of GammaCorp. Tom Black is Co-Founder of DeltaCorp. Sue Green is Co-Founder of EpsilonCorp.", "score": 0.88}
        ],
        "expected": ["Emma White"],
        "exclude": ["Tom Black", "Sue Green"]
    },
    {
        "id": 5,
        "category": "cross_company",
        "name": "Cross-Company: Company Name with Spaces",
        "query": "who are the co-founders of Tech Corp?",
        "chunks": [
            {"text": "Alex Brown is Co-Founder of TechCorp. Chris Davis is Co-Founder of TechCorp.", "score": 0.87}
        ],
        "expected": ["Alex Brown", "Chris Davis"],
        "exclude": []
    },
    {
        "id": 6,
        "category": "cross_company",
        "name": "Cross-Company: Inc. vs No Inc.",
        "query": "who are the co-founders of DataSystems?",
        "chunks": [
            {"text": "Maria Garcia is Co-Founder of DataSystems Inc. John Doe is Co-Founder of DataSystems.", "score": 0.89}
        ],
        "expected": ["Maria Garcia", "John Doe"],
        "exclude": []
    },
    {
        "id": 7,
        "category": "cross_company",
        "name": "Cross-Company: Multiple Mentions Same Person",
        "query": "who are the co-founders of StartupXYZ?",
        "chunks": [
            {"text": "James Taylor is Co-Founder of StartupXYZ. James Taylor also leads the engineering team.", "score": 0.85}
        ],
        "expected": ["James Taylor"],
        "exclude": []
    },
    {
        "id": 8,
        "category": "cross_company",
        "name": "Cross-Company: Very Similar Company Names",
        "query": "who are the co-founders of TechCorp?",
        "chunks": [
            {"text": "Frank Miller is Co-Founder of TechCorp. Grace Lee is Co-Founder of TechCorp Inc.", "score": 0.86}
        ],
        "expected": ["Frank Miller", "Grace Lee"],
        "exclude": []
    },
    {
        "id": 9,
        "category": "cross_company",
        "name": "Cross-Company: Long Company Names",
        "query": "who are the co-founders of Advanced Technology Solutions?",
        "chunks": [
            {"text": "Henry Kim is Co-Founder of Advanced Technology Solutions. Irene Park is Co-Founder of SimpleTech.", "score": 0.88}
        ],
        "expected": ["Henry Kim"],
        "exclude": ["Irene Park"]
    },
    {
        "id": 10,
        "category": "cross_company",
        "name": "Cross-Company: All Caps Company Name",
        "query": "who are the co-founders of AI CORP?",
        "chunks": [
            {"text": "Jack Smith is Co-Founder of AI Corp. Jane Doe is Co-Founder of AI Corp.", "score": 0.87}
        ],
        "expected": ["Jack Smith", "Jane Doe"],
        "exclude": []
    },
    
    # ========================================================================
    # ROLE FILTERING (Tests 11-20)
    # ========================================================================
    {
        "id": 11,
        "category": "role_filtering",
        "name": "Role Filtering: CEO Only (Not Co-Founder)",
        "query": "who are the co-founders of GammaCorp?",
        "chunks": [
            {"text": "Alex Brown is CEO of GammaCorp. Emma White is Co-Founder of GammaCorp.", "score": 0.88}
        ],
        "expected": ["Emma White"],
        "exclude": ["Alex Brown"]
    },
    {
        "id": 12,
        "category": "role_filtering",
        "name": "Role Filtering: CTO Only (Not Co-Founder)",
        "query": "who are the co-founders of DeltaCorp?",
        "chunks": [
            {"text": "Chris Davis is CTO of DeltaCorp. Bob Wilson is Co-Founder of DeltaCorp.", "score": 0.87}
        ],
        "expected": ["Bob Wilson"],
        "exclude": ["Chris Davis"]
    },
    {
        "id": 13,
        "category": "role_filtering",
        "name": "Role Filtering: CFO Only (Not Co-Founder)",
        "query": "who are the co-founders of EpsilonCorp?",
        "chunks": [
            {"text": "Diana Prince is CFO of EpsilonCorp. Steve Rogers is Co-Founder of EpsilonCorp.", "score": 0.89}
        ],
        "expected": ["Steve Rogers"],
        "exclude": ["Diana Prince"]
    },
    {
        "id": 14,
        "category": "role_filtering",
        "name": "Role Filtering: CEO and Co-Founder (Both Roles)",
        "query": "who are the co-founders of ZetaCorp?",
        "chunks": [
            {"text": "Tony Stark is CEO and Co-Founder of ZetaCorp. Bruce Banner is Co-Founder of ZetaCorp.", "score": 0.90}
        ],
        "expected": ["Tony Stark", "Bruce Banner"],
        "exclude": []
    },
    {
        "id": 15,
        "category": "role_filtering",
        "name": "Role Filtering: Multiple Non-Founders",
        "query": "who are the co-founders of ThetaCorp?",
        "chunks": [
            {"text": "Peter Parker is CEO of ThetaCorp. Mary Jane is CTO of ThetaCorp. Gwen Stacy is Co-Founder of ThetaCorp.", "score": 0.88}
        ],
        "expected": ["Gwen Stacy"],
        "exclude": ["Peter Parker", "Mary Jane"]
    },
    {
        "id": 16,
        "category": "role_filtering",
        "name": "Role Filtering: President (Not Co-Founder)",
        "query": "who are the co-founders of IotaCorp?",
        "chunks": [
            {"text": "Clark Kent is President of IotaCorp. Lois Lane is Co-Founder of IotaCorp.", "score": 0.87}
        ],
        "expected": ["Lois Lane"],
        "exclude": ["Clark Kent"]
    },
    {
        "id": 17,
        "category": "role_filtering",
        "name": "Role Filtering: VP (Not Co-Founder)",
        "query": "who are the co-founders of KappaCorp?",
        "chunks": [
            {"text": "Barry Allen is VP of Engineering at KappaCorp. Wally West is Co-Founder of KappaCorp.", "score": 0.86}
        ],
        "expected": ["Wally West"],
        "exclude": ["Barry Allen"]
    },
    {
        "id": 18,
        "category": "role_filtering",
        "name": "Role Filtering: Chief Marketing Officer",
        "query": "who are the co-founders of LambdaCorp?",
        "chunks": [
            {"text": "Hal Jordan is CMO of LambdaCorp. John Stewart is Co-Founder of LambdaCorp.", "score": 0.88}
        ],
        "expected": ["John Stewart"],
        "exclude": ["Hal Jordan"]
    },
    {
        "id": 19,
        "category": "role_filtering",
        "name": "Role Filtering: Director (Not Co-Founder)",
        "query": "who are the co-founders of MuCorp?",
        "chunks": [
            {"text": "Arthur Curry is Director of Operations at MuCorp. Mera is Co-Founder of MuCorp.", "score": 0.87}
        ],
        "expected": ["Mera"],
        "exclude": ["Arthur Curry"]
    },
    {
        "id": 20,
        "category": "role_filtering",
        "name": "Role Filtering: All Co-Founders, No Other Roles",
        "query": "who are the co-founders of NuCorp?",
        "chunks": [
            {"text": "Oliver Queen is Co-Founder of NuCorp. Dinah Lance is Co-Founder of NuCorp.", "score": 0.90}
        ],
        "expected": ["Oliver Queen", "Dinah Lance"],
        "exclude": []
    },
    
    # ========================================================================
    # NOT FOUND HANDLING (Tests 21-25)
    # ========================================================================
    {
        "id": 21,
        "category": "not_found",
        "name": "Not Found: Company Mentioned, No Co-Founders",
        "query": "who are the co-founders of UnknownCorp?",
        "chunks": [
            {"text": "UnknownCorp is a technology company. The company has 50 employees.", "score": 0.75}
        ],
        "expected": [],
        "exclude": [],
        "expect_not_found": True
    },
    {
        "id": 22,
        "category": "not_found",
        "name": "Not Found: Company Info But No Leadership",
        "query": "who are the co-founders of MysteryCorp?",
        "chunks": [
            {"text": "MysteryCorp focuses on AI solutions. The company was founded in 2020.", "score": 0.70}
        ],
        "expected": [],
        "exclude": [],
        "expect_not_found": True
    },
    {
        "id": 23,
        "category": "not_found",
        "name": "Not Found: Only Non-Founder Roles",
        "query": "who are the co-founders of NoFounderCorp?",
        "chunks": [
            {"text": "John Manager is CEO of NoFounderCorp. Jane Director is CTO of NoFounderCorp.", "score": 0.80}
        ],
        "expected": [],
        "exclude": ["John Manager", "Jane Director"],
        "expect_not_found": True
    },
    {
        "id": 24,
        "category": "not_found",
        "name": "Not Found: Wrong Company Mentioned",
        "query": "who are the co-founders of TargetCorp?",
        "chunks": [
            {"text": "Alice is Co-Founder of OtherCorp. Bob is Co-Founder of DifferentCorp.", "score": 0.75}
        ],
        "expected": [],
        "exclude": ["Alice", "Bob"],
        "expect_not_found": True
    },
    {
        "id": 25,
        "category": "not_found",
        "name": "Not Found: Empty Chunks",
        "query": "who are the co-founders of EmptyCorp?",
        "chunks": [
            {"text": "Some general information about technology.", "score": 0.50}
        ],
        "expected": [],
        "exclude": [],
        "expect_not_found": True
    },
    
    # ========================================================================
    # LIST EXTRACTION (Tests 26-30)
    # ========================================================================
    {
        "id": 26,
        "category": "list_extraction",
        "name": "List Extraction: Features Query",
        "query": "what are the key features of TechPlatform?",
        "chunks": [
            {"text": "TechPlatform offers real-time analytics, secure data encryption, automated reporting, API integrations, and custom dashboards.", "score": 0.90}
        ],
        "expected_keywords": ["analytics", "encryption", "reporting", "API", "dashboards"],
        "exclude_keywords": ["pricing", "founded"]
    },
    {
        "id": 27,
        "category": "list_extraction",
        "name": "List Extraction: Multiple Items Across Chunks",
        "query": "what are the benefits of ProductX?",
        "chunks": [
            {"text": "ProductX provides cost savings and improved efficiency.", "score": 0.85},
            {"text": "ProductX also offers scalability and reliability.", "score": 0.88}
        ],
        "expected_keywords": ["cost savings", "efficiency", "scalability", "reliability"],
        "exclude_keywords": ["pricing"]
    },
    {
        "id": 28,
        "category": "list_extraction",
        "name": "List Extraction: Items with Irrelevant Info",
        "query": "what are the components of SystemY?",
        "chunks": [
            {"text": "SystemY includes database, API server, and frontend. The system was built in 2021. Pricing starts at $100.", "score": 0.87}
        ],
        "expected_keywords": ["database", "API server", "frontend"],
        "exclude_keywords": ["2021", "pricing", "$100"]
    },
    {
        "id": 29,
        "category": "list_extraction",
        "name": "List Extraction: Mixed Relevance Chunks",
        "query": "what are the advantages of ServiceZ?",
        "chunks": [
            {"text": "ServiceZ offers 24/7 support and global coverage.", "score": 0.90},
            {"text": "The company was founded in 2015. Various other services exist.", "score": 0.55}
        ],
        "expected_keywords": ["24/7 support", "global coverage"],
        "exclude_keywords": ["2015", "founded"]
    },
    {
        "id": 30,
        "category": "list_extraction",
        "name": "List Extraction: Long List",
        "query": "what are the features of PlatformA?",
        "chunks": [
            {"text": "PlatformA features: authentication, authorization, logging, monitoring, backup, restore, scaling, load balancing, and caching.", "score": 0.92}
        ],
        "expected_keywords": ["authentication", "authorization", "logging", "monitoring", "backup", "restore", "scaling", "load balancing", "caching"],
        "exclude_keywords": []
    },
    
    # ========================================================================
    # ANALYTICAL QUERIES (Tests 31-35)
    # ========================================================================
    {
        "id": 31,
        "category": "analytical",
        "name": "Analytical: Why Query with Because",
        "query": "why did the company expand internationally?",
        "chunks": [
            {"text": "The company expanded internationally because of increasing global demand and market opportunities.", "score": 0.88}
        ],
        "expected_keywords": ["because", "demand", "opportunities"],
        "exclude_keywords": ["founded", "location"]
    },
    {
        "id": 32,
        "category": "analytical",
        "name": "Analytical: Why Query with Due To",
        "query": "why was the product launched early?",
        "chunks": [
            {"text": "The product was launched early due to competitive pressures and customer requests.", "score": 0.87}
        ],
        "expected_keywords": ["due to", "competitive", "customer"],
        "exclude_keywords": []
    },
    {
        "id": 33,
        "category": "analytical",
        "name": "Analytical: What Caused Query",
        "query": "what caused the system failure?",
        "chunks": [
            {"text": "The system failure was caused by overloaded servers and insufficient capacity planning.", "score": 0.90}
        ],
        "expected_keywords": ["caused", "overloaded", "capacity"],
        "exclude_keywords": []
    },
    {
        "id": 34,
        "category": "analytical",
        "name": "Analytical: Led To Reasoning",
        "query": "why did sales increase?",
        "chunks": [
            {"text": "Improved marketing strategies led to increased sales and customer engagement.", "score": 0.89}
        ],
        "expected_keywords": ["led to", "marketing", "strategies"],
        "exclude_keywords": []
    },
    {
        "id": 35,
        "category": "analytical",
        "name": "Analytical: Multiple Reasons",
        "query": "why did the merger happen?",
        "chunks": [
            {"text": "The merger happened to achieve market dominance, reduce costs, and expand product offerings.", "score": 0.91}
        ],
        "expected_keywords": ["dominance", "costs", "expand"],
        "exclude_keywords": []
    },
    
    # ========================================================================
    # RELATIONSHIP QUERIES (Tests 36-40)
    # ========================================================================
    {
        "id": 36,
        "category": "relationship",
        "name": "Relationship: Strategic Partners",
        "query": "how are TechCorp and DataSystems related?",
        "chunks": [
            {"text": "TechCorp and DataSystems are strategic partners collaborating on joint product development.", "score": 0.90}
        ],
        "expected_keywords": ["partners", "collaborating", "joint"],
        "exclude_keywords": ["founded", "employees"]
    },
    {
        "id": 37,
        "category": "relationship",
        "name": "Relationship: Parent-Subsidiary",
        "query": "what is the connection between ParentCorp and SubCorp?",
        "chunks": [
            {"text": "ParentCorp owns SubCorp as a subsidiary. They work together on integrated solutions.", "score": 0.88}
        ],
        "expected_keywords": ["owns", "subsidiary", "work together"],
        "exclude_keywords": []
    },
    {
        "id": 38,
        "category": "relationship",
        "name": "Relationship: Alliance",
        "query": "how are AlphaCorp and BetaCorp related?",
        "chunks": [
            {"text": "AlphaCorp and BetaCorp formed an alliance to share technology resources and market access.", "score": 0.89}
        ],
        "expected_keywords": ["alliance", "share", "resources"],
        "exclude_keywords": []
    },
    {
        "id": 39,
        "category": "relationship",
        "name": "Relationship: Connected Through",
        "query": "what is the relationship between CompanyA and CompanyB?",
        "chunks": [
            {"text": "CompanyA and CompanyB are connected through a shared technology platform and mutual customers.", "score": 0.87}
        ],
        "expected_keywords": ["connected", "shared", "platform"],
        "exclude_keywords": []
    },
    {
        "id": 40,
        "category": "relationship",
        "name": "Relationship: Joint Venture",
        "query": "how are XCorp and YCorp related?",
        "chunks": [
            {"text": "XCorp and YCorp established a joint venture to develop new products in emerging markets.", "score": 0.90}
        ],
        "expected_keywords": ["joint venture", "develop", "products"],
        "exclude_keywords": []
    },
    
    # ========================================================================
    # COMPARISON QUERIES (Tests 41-45)
    # ========================================================================
    {
        "id": 41,
        "category": "comparison",
        "name": "Comparison: While Contrast",
        "query": "compare ProductA and ProductB",
        "chunks": [
            {"text": "ProductA focuses on enterprise solutions while ProductB targets small businesses.", "score": 0.90}
        ],
        "expected_keywords": ["while", "enterprise", "small businesses"],
        "exclude_keywords": []
    },
    {
        "id": 42,
        "category": "comparison",
        "name": "Comparison: Whereas Contrast",
        "query": "what is the difference between ServiceX and ServiceY?",
        "chunks": [
            {"text": "ServiceX uses cloud infrastructure whereas ServiceY offers on-premise deployment.", "score": 0.89}
        ],
        "expected_keywords": ["whereas", "cloud", "on-premise"],
        "exclude_keywords": []
    },
    {
        "id": 43,
        "category": "comparison",
        "name": "Comparison: Versus Format",
        "query": "compare Platform1 vs Platform2",
        "chunks": [
            {"text": "Platform1 has extensive customization options versus Platform2 which focuses on simplicity.", "score": 0.88}
        ],
        "expected_keywords": ["versus", "customization", "simplicity"],
        "exclude_keywords": []
    },
    {
        "id": 44,
        "category": "comparison",
        "name": "Comparison: In Contrast",
        "query": "how do SystemA and SystemB differ?",
        "chunks": [
            {"text": "SystemA emphasizes security. In contrast, SystemB prioritizes performance and speed.", "score": 0.91}
        ],
        "expected_keywords": ["contrast", "security", "performance"],
        "exclude_keywords": []
    },
    {
        "id": 45,
        "category": "comparison",
        "name": "Comparison: Multiple Differences",
        "query": "compare Tech1 and Tech2",
        "chunks": [
            {"text": "Tech1 uses Python and has a large community. Tech2 uses Java and focuses on enterprise clients. Tech1 is open source while Tech2 is proprietary.", "score": 0.92}
        ],
        "expected_keywords": ["Python", "Java", "open source", "proprietary"],
        "exclude_keywords": []
    },
    
    # ========================================================================
    # PROCESS QUERIES (Tests 46-50)
    # ========================================================================
    {
        "id": 46,
        "category": "process",
        "name": "Process: How Does Work - Step by Step",
        "query": "how does the authentication system work?",
        "chunks": [
            {"text": "The authentication system works by first verifying user credentials, then generating a token, and finally granting access based on permissions.", "score": 0.90}
        ],
        "expected_keywords": ["first", "then", "finally", "verify", "token", "access"],
        "exclude_keywords": ["founded", "employees"]
    },
    {
        "id": 47,
        "category": "process",
        "name": "Process: How Does Work - Sequential Steps",
        "query": "how does the payment processing work?",
        "chunks": [
            {"text": "Payment processing works by validating the payment method, checking available funds, processing the transaction, and sending confirmation.", "score": 0.89}
        ],
        "expected_keywords": ["validating", "checking", "processing", "sending"],
        "exclude_keywords": []
    },
    {
        "id": 48,
        "category": "process",
        "name": "Process: How Is Processed",
        "query": "how is data processed in the system?",
        "chunks": [
            {"text": "Data is processed by first collecting inputs, then cleaning and validating, next transforming the format, and finally storing in the database.", "score": 0.91}
        ],
        "expected_keywords": ["first", "then", "next", "finally", "collecting", "cleaning", "storing"],
        "exclude_keywords": []
    },
    {
        "id": 49,
        "category": "process",
        "name": "Process: How Do They Work",
        "query": "how do machine learning models work?",
        "chunks": [
            {"text": "Machine learning models work by training on data, learning patterns, making predictions, and improving through feedback loops.", "score": 0.88}
        ],
        "expected_keywords": ["training", "learning", "predictions", "improving"],
        "exclude_keywords": []
    },
    {
        "id": 50,
        "category": "process",
        "name": "Process: Complex Multi-Step",
        "query": "how does the deployment pipeline work?",
        "chunks": [
            {"text": "The deployment pipeline works by building the code, running tests, creating containers, deploying to staging, running integration tests, and finally deploying to production.", "score": 0.92}
        ],
        "expected_keywords": ["building", "running tests", "creating", "deploying", "staging", "production"],
        "exclude_keywords": []
    },
    
    # ========================================================================
    # PERSONAL REFLECTION QUERIES (Tests 51-70)
    # ========================================================================
    {
        "id": 51,
        "category": "personal_reflection",
        "name": "Personal: Goals Extraction",
        "query": "what are my goals?",
        "chunks": [
            {"text": "I have been working on improving my health, learning a new language, and traveling more. These are my main goals for this year.", "score": 0.90}
        ],
        "expected_keywords": ["health", "language", "traveling"],
        "exclude_keywords": ["company", "business"]
    },
    {
        "id": 52,
        "category": "personal_reflection",
        "name": "Personal: Achievements List",
        "query": "what are my achievements?",
        "chunks": [
            {"text": "I completed a marathon last year. I also graduated from university and got promoted at work. These are my key achievements.", "score": 0.88}
        ],
        "expected_keywords": ["marathon", "graduated", "promoted"],
        "exclude_keywords": []
    },
    {
        "id": 53,
        "category": "personal_reflection",
        "name": "Personal: Skills Extraction",
        "query": "what are my skills?",
        "chunks": [
            {"text": "I have developed strong communication skills, leadership abilities, and problem-solving capabilities over the years.", "score": 0.89}
        ],
        "expected_keywords": ["communication", "leadership", "problem-solving"],
        "exclude_keywords": []
    },
    {
        "id": 54,
        "category": "personal_reflection",
        "name": "Personal: Why Did I Decision",
        "query": "why did I decide to change careers?",
        "chunks": [
            {"text": "I decided to change careers because of my passion for technology and the desire for better work-life balance.", "score": 0.87}
        ],
        "expected_keywords": ["passion", "technology", "work-life balance"],
        "exclude_keywords": []
    },
    {
        "id": 55,
        "category": "personal_reflection",
        "name": "Personal: When Did I Event",
        "query": "when did I move to a new city?",
        "chunks": [
            {"text": "I moved to a new city in 2020. This was a major turning point in my life.", "score": 0.85}
        ],
        "expected_keywords": ["2020"],
        "exclude_keywords": []
    },
    {
        "id": 56,
        "category": "personal_reflection",
        "name": "Personal: Interests List",
        "query": "what are my interests?",
        "chunks": [
            {"text": "My interests include photography, cooking, hiking, and reading. I spend most of my free time on these activities.", "score": 0.90}
        ],
        "expected_keywords": ["photography", "cooking", "hiking", "reading"],
        "exclude_keywords": []
    },
    {
        "id": 57,
        "category": "personal_reflection",
        "name": "Personal: Hobbies Extraction",
        "query": "what are my hobbies?",
        "chunks": [
            {"text": "I enjoy playing guitar, painting, and gardening in my spare time. These hobbies help me relax and be creative.", "score": 0.88}
        ],
        "expected_keywords": ["guitar", "painting", "gardening"],
        "exclude_keywords": []
    },
    {
        "id": 58,
        "category": "personal_reflection",
        "name": "Personal: Values List",
        "query": "what are my values?",
        "chunks": [
            {"text": "My core values are integrity, family, growth, and helping others. These guide my decisions in life.", "score": 0.89}
        ],
        "expected_keywords": ["integrity", "family", "growth", "helping others"],
        "exclude_keywords": []
    },
    {
        "id": 59,
        "category": "personal_reflection",
        "name": "Personal: Why Did I Achieve",
        "query": "why did I achieve that goal?",
        "chunks": [
            {"text": "I achieved that goal because of consistent effort, support from family, and clear planning.", "score": 0.87}
        ],
        "expected_keywords": ["effort", "support", "planning"],
        "exclude_keywords": []
    },
    {
        "id": 60,
        "category": "personal_reflection",
        "name": "Personal: Timeline Events",
        "query": "what events happened in my life?",
        "chunks": [
            {"text": "In 2015, I left my corporate job. In 2018, my father passed away. In 2020, I moved to a new city.", "score": 0.90}
        ],
        "expected_keywords": ["2015", "2018", "2020", "left", "passed away", "moved"],
        "exclude_keywords": []
    },
    {
        "id": 61,
        "category": "personal_reflection",
        "name": "Personal: Self-Reflection Why",
        "query": "why did I make that decision?",
        "chunks": [
            {"text": "I made that decision because I was seeking personal fulfillment and wanted to align my career with my values.", "score": 0.88}
        ],
        "expected_keywords": ["fulfillment", "align", "values"],
        "exclude_keywords": []
    },
    {
        "id": 62,
        "category": "personal_reflection",
        "name": "Personal: Lessons Learned",
        "query": "what lessons have I learned?",
        "chunks": [
            {"text": "I learned the importance of patience, resilience, and asking for help when needed. These lessons shaped who I am today.", "score": 0.89}
        ],
        "expected_keywords": ["patience", "resilience", "asking for help"],
        "exclude_keywords": []
    },
    {
        "id": 63,
        "category": "personal_reflection",
        "name": "Personal: Relationships Context",
        "query": "how are my career and personal life related?",
        "chunks": [
            {"text": "My career and personal life are connected through my values of work-life balance and finding meaning in both areas.", "score": 0.87}
        ],
        "expected_keywords": ["connected", "work-life balance", "meaning"],
        "exclude_keywords": []
    },
    {
        "id": 64,
        "category": "personal_reflection",
        "name": "Personal: Compare Past and Present",
        "query": "how have I changed over time?",
        "chunks": [
            {"text": "I have become more confident and self-aware. In contrast, I used to be more hesitant and less reflective.", "score": 0.90}
        ],
        "expected_keywords": ["confident", "self-aware", "contrast", "hesitant"],
        "exclude_keywords": []
    },
    {
        "id": 65,
        "category": "personal_reflection",
        "name": "Personal: Strengths List",
        "query": "what are my strengths?",
        "chunks": [
            {"text": "My strengths include analytical thinking, empathy, and adaptability. These help me in both personal and professional settings.", "score": 0.88}
        ],
        "expected_keywords": ["analytical thinking", "empathy", "adaptability"],
        "exclude_keywords": []
    },
    {
        "id": 66,
        "category": "personal_reflection",
        "name": "Personal: Growth Process",
        "query": "how have I grown as a person?",
        "chunks": [
            {"text": "I have grown by first recognizing my weaknesses, then working on them consistently, and finally developing new skills and perspectives.", "score": 0.91}
        ],
        "expected_keywords": ["first", "then", "finally", "recognizing", "working", "developing"],
        "exclude_keywords": []
    },
    {
        "id": 67,
        "category": "personal_reflection",
        "name": "Personal: Priorities List",
        "query": "what are my priorities?",
        "chunks": [
            {"text": "My priorities are family, health, career growth, and personal development. I focus on these areas daily.", "score": 0.89}
        ],
        "expected_keywords": ["family", "health", "career growth", "personal development"],
        "exclude_keywords": []
    },
    {
        "id": 68,
        "category": "personal_reflection",
        "name": "Personal: Challenges Faced",
        "query": "what challenges have I faced?",
        "chunks": [
            {"text": "I faced financial difficulties, career transitions, and personal loss. These challenges taught me resilience.", "score": 0.87}
        ],
        "expected_keywords": ["financial difficulties", "career transitions", "personal loss"],
        "exclude_keywords": []
    },
    {
        "id": 69,
        "category": "personal_reflection",
        "name": "Personal: Mixed Content Goals",
        "query": "what are my goals?",
        "chunks": [
            {"text": "I have been working on improving my health and learning a new language. The weather was nice today. I also want to travel more this year.", "score": 0.85}
        ],
        "expected_keywords": ["health", "language", "travel"],
        "exclude_keywords": ["weather"]
    },
    {
        "id": 70,
        "category": "personal_reflection",
        "name": "Personal: Not Found Goals",
        "query": "what are my goals?",
        "chunks": [
            {"text": "I enjoy reading books and watching movies. The weekend was relaxing.", "score": 0.70}
        ],
        "expected": [],
        "exclude": [],
        "expect_not_found": True
    },
    
    # ========================================================================
    # BUSINESS MANAGEMENT QUERIES (Tests 71-90)
    # ========================================================================
    {
        "id": 71,
        "category": "business_management",
        "name": "Business: Company Mission",
        "query": "what is TechCorp's mission?",
        "chunks": [
            {"text": "TechCorp's mission is to redefine enterprise intelligence through AI-powered solutions. The company was founded in 2015.", "score": 0.90}
        ],
        "expected_keywords": ["mission", "enterprise intelligence", "AI-powered"],
        "exclude_keywords": ["founded", "2015"]
    },
    {
        "id": 72,
        "category": "business_management",
        "name": "Business: Company Strategy",
        "query": "what is the company's strategy?",
        "chunks": [
            {"text": "The company's strategy focuses on innovation, customer satisfaction, and market expansion. We prioritize these areas.", "score": 0.88}
        ],
        "expected_keywords": ["innovation", "customer satisfaction", "market expansion"],
        "exclude_keywords": []
    },
    {
        "id": 73,
        "category": "business_management",
        "name": "Business: Products List",
        "query": "what products does the company offer?",
        "chunks": [
            {"text": "The company offers cloud storage, data analytics, and AI consulting services. These are our main products.", "score": 0.89}
        ],
        "expected_keywords": ["cloud storage", "data analytics", "AI consulting"],
        "exclude_keywords": []
    },
    {
        "id": 74,
        "category": "business_management",
        "name": "Business: Why Did Company Expand",
        "query": "why did the company expand to new markets?",
        "chunks": [
            {"text": "The company expanded to new markets because of increasing demand and competitive opportunities.", "score": 0.87}
        ],
        "expected_keywords": ["because", "demand", "opportunities"],
        "exclude_keywords": []
    },
    {
        "id": 75,
        "category": "business_management",
        "name": "Business: How Does Company Operate",
        "query": "how does the company operate?",
        "chunks": [
            {"text": "The company operates by first identifying customer needs, then developing solutions, and finally delivering value through partnerships.", "score": 0.91}
        ],
        "expected_keywords": ["first", "then", "finally", "identifying", "developing", "delivering"],
        "exclude_keywords": []
    },
    {
        "id": 76,
        "category": "business_management",
        "name": "Business: Company Values",
        "query": "what are the company's values?",
        "chunks": [
            {"text": "The company's values include integrity, innovation, and customer focus. These guide all our decisions.", "score": 0.88}
        ],
        "expected_keywords": ["integrity", "innovation", "customer focus"],
        "exclude_keywords": []
    },
    {
        "id": 77,
        "category": "business_management",
        "name": "Business: Services List",
        "query": "what services does the company provide?",
        "chunks": [
            {"text": "The company provides consulting, training, and support services. These are our core offerings.", "score": 0.89}
        ],
        "expected_keywords": ["consulting", "training", "support"],
        "exclude_keywords": []
    },
    {
        "id": 78,
        "category": "business_management",
        "name": "Business: Company Culture",
        "query": "what is the company culture like?",
        "chunks": [
            {"text": "The company culture emphasizes collaboration, continuous learning, and work-life balance. Employees value these aspects.", "score": 0.87}
        ],
        "expected_keywords": ["collaboration", "continuous learning", "work-life balance"],
        "exclude_keywords": []
    },
    {
        "id": 79,
        "category": "business_management",
        "name": "Business: Why Did Company Pivot",
        "query": "why did the company pivot its strategy?",
        "chunks": [
            {"text": "The company pivoted its strategy due to market changes and customer feedback indicating new needs.", "score": 0.90}
        ],
        "expected_keywords": ["due to", "market changes", "customer feedback"],
        "exclude_keywords": []
    },
    {
        "id": 80,
        "category": "business_management",
        "name": "Business: Company Goals",
        "query": "what are the company's goals?",
        "chunks": [
            {"text": "The company's goals include reaching 1 million users, expanding to 10 countries, and achieving profitability by 2025.", "score": 0.88}
        ],
        "expected_keywords": ["1 million users", "10 countries", "profitability", "2025"],
        "exclude_keywords": []
    },
    {
        "id": 81,
        "category": "business_management",
        "name": "Business: Compare Business Units",
        "query": "how do the sales and marketing departments differ?",
        "chunks": [
            {"text": "The sales department focuses on closing deals while the marketing department emphasizes brand awareness and lead generation.", "score": 0.89}
        ],
        "expected_keywords": ["while", "closing deals", "brand awareness"],
        "exclude_keywords": []
    },
    {
        "id": 82,
        "category": "business_management",
        "name": "Business: Company Relationships",
        "query": "how is the company related to its partners?",
        "chunks": [
            {"text": "The company works closely with technology partners through strategic alliances and joint development projects.", "score": 0.87}
        ],
        "expected_keywords": ["partners", "strategic alliances", "joint development"],
        "exclude_keywords": []
    },
    {
        "id": 83,
        "category": "business_management",
        "name": "Business: Revenue Sources",
        "query": "what are the company's revenue sources?",
        "chunks": [
            {"text": "The company generates revenue from subscriptions, licensing fees, and professional services.", "score": 0.90}
        ],
        "expected_keywords": ["subscriptions", "licensing fees", "professional services"],
        "exclude_keywords": []
    },
    {
        "id": 84,
        "category": "business_management",
        "name": "Business: Company Challenges",
        "query": "what challenges does the company face?",
        "chunks": [
            {"text": "The company faces challenges including market competition, talent acquisition, and scaling infrastructure.", "score": 0.88}
        ],
        "expected_keywords": ["competition", "talent acquisition", "scaling"],
        "exclude_keywords": []
    },
    {
        "id": 85,
        "category": "business_management",
        "name": "Business: How Does Company Scale",
        "query": "how does the company scale its operations?",
        "chunks": [
            {"text": "The company scales by first automating processes, then hiring strategically, and finally expanding infrastructure.", "score": 0.91}
        ],
        "expected_keywords": ["first", "then", "finally", "automating", "hiring", "expanding"],
        "exclude_keywords": []
    },
    {
        "id": 86,
        "category": "business_management",
        "name": "Business: Company Initiatives",
        "query": "what initiatives is the company working on?",
        "chunks": [
            {"text": "The company is working on sustainability initiatives, diversity programs, and innovation labs.", "score": 0.89}
        ],
        "expected_keywords": ["sustainability", "diversity", "innovation labs"],
        "exclude_keywords": []
    },
    {
        "id": 87,
        "category": "business_management",
        "name": "Business: Mixed Content Mission",
        "query": "what is the company's mission?",
        "chunks": [
            {"text": "The company's mission is to empower businesses through technology. The office is located in downtown. We also focus on innovation.", "score": 0.85}
        ],
        "expected_keywords": ["mission", "empower", "technology", "innovation"],
        "exclude_keywords": ["office", "downtown"]
    },
    {
        "id": 88,
        "category": "business_management",
        "name": "Business: Not Found Revenue",
        "query": "what is the company's revenue?",
        "chunks": [
            {"text": "The company focuses on technology solutions. The office has 50 employees.", "score": 0.70}
        ],
        "expected": [],
        "exclude": [],
        "expect_not_found": True
    },
    {
        "id": 89,
        "category": "business_management",
        "name": "Business: Company Timeline",
        "query": "what are the key milestones in the company's history?",
        "chunks": [
            {"text": "In 2015, the company was founded. In 2018, we reached 100,000 users. In 2020, we expanded internationally.", "score": 0.90}
        ],
        "expected_keywords": ["2015", "founded", "2018", "100,000 users", "2020", "expanded"],
        "exclude_keywords": []
    },
    {
        "id": 90,
        "category": "business_management",
        "name": "Business: Company Advantages",
        "query": "what are the company's competitive advantages?",
        "chunks": [
            {"text": "The company's advantages include proprietary technology, strong brand recognition, and an experienced team.", "score": 0.88}
        ],
        "expected_keywords": ["proprietary technology", "brand recognition", "experienced team"],
        "exclude_keywords": []
    },
    
    # ========================================================================
    # CO-FOUNDER QUERIES WITH MIXED CONTENT (Tests 91-100)
    # ========================================================================
    {
        "id": 91,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: LedgerAI with Irrelevant Info",
        "query": "who are the co-founders of LedgerAI?",
        "chunks": [
            {"text": "Paul Chou is Co-Founder of LedgerAI, serving as CEO. The weather was nice today. Bob Carella is Co-Founder of LedgerAI, serving as CFO. Jorge Guinovart is Co-Founder of LedgerAI.", "score": 0.90},
            {"text": "David Lara is Co-Founder of LedgerAI. The company focuses on AI solutions. Various other companies exist in the market.", "score": 0.85}
        ],
        "expected": ["Paul Chou", "Bob Carella", "Jorge Guinovart", "David Lara"],
        "exclude": [],
        "exclude_keywords": ["weather", "other companies"]
    },
    {
        "id": 92,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Multiple Companies in Chunk",
        "query": "who are the co-founders of TechCorp?",
        "chunks": [
            {"text": "John Smith is Co-Founder of TechCorp. Sarah Jones is Co-Founder of DataSystems. Mike Brown is Co-Founder of TechCorp. The company was founded in 2020.", "score": 0.88}
        ],
        "expected": ["John Smith", "Mike Brown"],
        "exclude": ["Sarah Jones"],
        "exclude_keywords": ["2020", "founded"]
    },
    {
        "id": 93,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Low Relevance Chunks",
        "query": "who are the co-founders of AlphaCorp?",
        "chunks": [
            {"text": "David Chen is Co-Founder of AlphaCorp. Various industry trends are discussed. The market is growing.", "score": 0.90},
            {"text": "General information about technology companies. Some statistics about the industry.", "score": 0.55}
        ],
        "expected": ["David Chen"],
        "exclude": [],
        "exclude_keywords": ["trends", "statistics"]
    },
    {
        "id": 94,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Company Info Mixed",
        "query": "who are the co-founders of BetaCorp?",
        "chunks": [
            {"text": "BetaCorp is a technology company. Lisa Wang is Co-Founder of BetaCorp. The company has 100 employees. Robert Kim is Co-Founder of BetaCorp.", "score": 0.87}
        ],
        "expected": ["Lisa Wang", "Robert Kim"],
        "exclude": [],
        "exclude_keywords": ["100 employees"]
    },
    {
        "id": 95,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Cross-Company with Mixed",
        "query": "who are the co-founders of GammaCorp?",
        "chunks": [
            {"text": "Emma White is Co-Founder of GammaCorp. Tom Black is Co-Founder of DeltaCorp. The weather forecast shows rain. Sue Green is Co-Founder of EpsilonCorp.", "score": 0.89}
        ],
        "expected": ["Emma White"],
        "exclude": ["Tom Black", "Sue Green"],
        "exclude_keywords": ["weather", "forecast", "rain"]
    },
    {
        "id": 96,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Role Filtering with Mixed",
        "query": "who are the co-founders of ZetaCorp?",
        "chunks": [
            {"text": "Tony Stark is CEO and Co-Founder of ZetaCorp. Bruce Banner is Co-Founder of ZetaCorp. The office is located downtown. Various meetings were scheduled.", "score": 0.90}
        ],
        "expected": ["Tony Stark", "Bruce Banner"],
        "exclude": [],
        "exclude_keywords": ["office", "downtown", "meetings"]
    },
    {
        "id": 97,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Name Variation with Mixed",
        "query": "who are the co-founders of Ledger AI?",
        "chunks": [
            {"text": "Paul Chou is Co-Founder of LedgerAI. Bob Carella is Co-Founder of LedgerAI. The quarterly report shows growth. Jorge Guinovart is Co-Founder of LedgerAI.", "score": 0.91}
        ],
        "expected": ["Paul Chou", "Bob Carella", "Jorge Guinovart"],
        "exclude": [],
        "exclude_keywords": ["quarterly report", "growth"]
    },
    {
        "id": 98,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Multiple Chunks Mixed",
        "query": "who are the co-founders of ThetaCorp?",
        "chunks": [
            {"text": "Gwen Stacy is Co-Founder of ThetaCorp. The company focuses on innovation.", "score": 0.88},
            {"text": "Peter Parker is CEO of ThetaCorp. Various other topics are discussed. Mary Jane is CTO of ThetaCorp.", "score": 0.85}
        ],
        "expected": ["Gwen Stacy"],
        "exclude": ["Peter Parker", "Mary Jane"],
        "exclude_keywords": ["other topics"]
    },
    {
        "id": 99,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Long Chunk with Mixed",
        "query": "who are the co-founders of IotaCorp?",
        "chunks": [
            {"text": "Lois Lane is Co-Founder of IotaCorp. Clark Kent is President of IotaCorp. The company was established in 2015. Various industry reports mention growth. The office building has modern facilities. Employees enjoy the work environment.", "score": 0.87}
        ],
        "expected": ["Lois Lane"],
        "exclude": ["Clark Kent"],
        "exclude_keywords": ["2015", "reports", "building", "facilities", "employees"]
    },
    {
        "id": 100,
        "category": "cofounder_mixed",
        "name": "Co-Founder Mixed: Complex Multi-Company Mixed",
        "query": "who are the co-founders of KappaCorp?",
        "chunks": [
            {"text": "Wally West is Co-Founder of KappaCorp. Barry Allen is VP of Engineering at KappaCorp. The weather is sunny. Various market trends are discussed.", "score": 0.89},
            {"text": "Hal Jordan is CMO of LambdaCorp. John Stewart is Co-Founder of LambdaCorp. The quarterly meeting was productive.", "score": 0.85}
        ],
        "expected": ["Wally West"],
        "exclude": ["Barry Allen", "Hal Jordan", "John Stewart"],
        "exclude_keywords": ["weather", "sunny", "trends", "quarterly meeting"]
    }
]

# ============================================================================
# TEST EXECUTION FUNCTIONS
# ============================================================================

def check_response(test: Dict, response: str) -> Dict[str, Any]:
    """Check if response matches expected behavior for a test."""
    results = {
        "passed": True,
        "issues": [],
        "found_items": [],
        "missing_items": [],
        "wrong_items": [],
        "hallucinated": False
    }
    
    # Extract names/entities from response
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', response)
    results["found_items"] = names
    
    # Check based on test type
    if "expected" in test:
        # Entity extraction test
        expected = test["expected"]
        exclude = test.get("exclude", [])
        
        # Check for expected entities
        for exp in expected:
            found = any(exp.lower() in name.lower() or name.lower() in exp.lower() for name in names)
            if not found:
                results["missing_items"].append(exp)
                results["passed"] = False
        
        # Check for excluded entities
        for exc in exclude:
            found = any(exc.lower() in name.lower() or name.lower() in exc.lower() for name in names)
            if found:
                results["wrong_items"].append(exc)
                results["passed"] = False
        
        # Check for not found expectation
        if test.get("expect_not_found", False):
            not_found_phrases = ["don't have", "couldn't find", "not found", "don't have that information"]
            has_not_found = any(phrase in response.lower() for phrase in not_found_phrases)
            if not has_not_found and names:
                results["hallucinated"] = True
                results["passed"] = False
                results["issues"].append("Hallucinated entities when should return 'not found'")
        
        # Check for excluded keywords if present
        if "exclude_keywords" in test:
            response_lower = response.lower()
            for keyword in test["exclude_keywords"]:
                if keyword.lower() in response_lower:
                    results["wrong_items"].append(keyword)
                    results["passed"] = False
    
    elif "expected_keywords" in test:
        # Keyword-based test (list, analytical, relationship, comparison, process)
        expected_keywords = test["expected_keywords"]
        exclude_keywords = test.get("exclude_keywords", [])
        response_lower = response.lower()
        
        # Check for expected keywords
        for keyword in expected_keywords:
            if keyword.lower() not in response_lower:
                results["missing_items"].append(keyword)
                results["passed"] = False
        
        # Check for excluded keywords
        for keyword in exclude_keywords:
            if keyword.lower() in response_lower:
                results["wrong_items"].append(keyword)
                results["passed"] = False
    
    # Generate issue summary
    if results["missing_items"]:
        results["issues"].append(f"Missing: {results['missing_items']}")
    if results["wrong_items"]:
        results["issues"].append(f"Incorrectly included: {results['wrong_items']}")
    if results["hallucinated"]:
        results["issues"].append("Hallucinated information")
    
    return results

def run_comprehensive_tests(model, tokenizer, model_type="unsloth", verbose=True):
    """Run all 100 comprehensive tests with real-time output."""
    import sys
    import time
    
    try:
        from test_rag_analysis_colab import analyze_rag_chunks
    except ImportError:
        print("❌ Error: Could not import analyze_rag_chunks", flush=True)
        print("   Make sure test_rag_analysis_colab.py is available", flush=True)
        return
    
    # Force unbuffered output for real-time display in Colab
    # Use print with flush=True instead of reconfigure (not available in Colab)
    
    print("=" * 80, flush=True)
    print("COMPREHENSIVE GAP TEST SUITE - 100 DIVERSE EXAMPLES", flush=True)
    print("=" * 80, flush=True)
    print(f"Running {len(COMPREHENSIVE_TESTS)} comprehensive tests...\n", flush=True)
    
    results_summary = {
        "total": len(COMPREHENSIVE_TESTS),
        "passed": 0,
        "failed": 0,
        "by_category": {},
        "start_time": time.time()
    }
    
    for idx, test in enumerate(COMPREHENSIVE_TESTS, 1):
        category = test["category"]
        if category not in results_summary["by_category"]:
            results_summary["by_category"][category] = {"total": 0, "passed": 0, "failed": 0}
        
        results_summary["by_category"][category]["total"] += 1
        
        # Show progress immediately
        elapsed = time.time() - results_summary["start_time"]
        print(f"\n{'='*80}", flush=True)
        print(f"TEST {test['id']}/100: {test['name']}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Category: {category}", flush=True)
        print(f"Query: {test['query']}", flush=True)
        print(f"Progress: {idx}/{len(COMPREHENSIVE_TESTS)} ({idx/len(COMPREHENSIVE_TESTS)*100:.1f}%) | Elapsed: {elapsed:.1f}s", flush=True)
        
        test_start = time.time()
        try:
            if verbose:
                print("🤖 Generating response...", flush=True)
            
            response = analyze_rag_chunks(model, tokenizer, test['query'], test['chunks'], model_type)
            
            test_time = time.time() - test_start
            if verbose:
                print(f"⏱️  Response generated in {test_time:.1f}s", flush=True)
                print(f"\n📝 Response: {response[:200]}..." if len(response) > 200 else f"\n📝 Response: {response}", flush=True)
            
            test_results = check_response(test, response)
            
            if test_results["passed"]:
                print(f"✅ PASSED", flush=True)
                results_summary["passed"] += 1
                results_summary["by_category"][category]["passed"] += 1
            else:
                print(f"❌ FAILED", flush=True)
                results_summary["failed"] += 1
                results_summary["by_category"][category]["failed"] += 1
                for issue in test_results["issues"]:
                    print(f"   - {issue}", flush=True)
        
        except Exception as e:
            test_time = time.time() - test_start
            print(f"❌ ERROR after {test_time:.1f}s: {e}", flush=True)
            results_summary["failed"] += 1
            results_summary["by_category"][category]["failed"] += 1
            import traceback
            traceback.print_exc()
        
        # Show running summary every 10 tests
        if idx % 10 == 0:
            elapsed_total = time.time() - results_summary["start_time"]
            avg_time = elapsed_total / idx
            remaining = (len(COMPREHENSIVE_TESTS) - idx) * avg_time
            print(f"\n📊 Progress Update: {idx}/100 tests | Passed: {results_summary['passed']} | Failed: {results_summary['failed']}", flush=True)
            print(f"⏱️  Avg time per test: {avg_time:.1f}s | Est. remaining: {remaining/60:.1f} minutes", flush=True)
    
    # Print summary
    total_time = time.time() - results_summary["start_time"]
    print("\n" + "=" * 80, flush=True)
    print("COMPREHENSIVE TEST SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"\nTotal Tests: {results_summary['total']}", flush=True)
    print(f"✅ Passed: {results_summary['passed']} ({results_summary['passed']/results_summary['total']*100:.1f}%)", flush=True)
    print(f"❌ Failed: {results_summary['failed']} ({results_summary['failed']/results_summary['total']*100:.1f}%)", flush=True)
    print(f"⏱️  Total Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)", flush=True)
    print(f"⏱️  Average Time per Test: {total_time/results_summary['total']:.1f} seconds", flush=True)
    
    print(f"\n📊 Results by Category:", flush=True)
    for category, stats in sorted(results_summary["by_category"].items()):
        pass_rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {category}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)", flush=True)
    
    print(f"\n💡 Recommendations:", flush=True)
    if results_summary["failed"] > 0:
        print(f"  - {results_summary['failed']} tests failed - review failures above", flush=True)
        print(f"  - Update dataset generation to add more examples for failing categories", flush=True)
        print(f"  - Strengthen extraction logic for identified gaps", flush=True)
    else:
        print(f"  - All tests passed! Model is ready for production.", flush=True)
    
    return results_summary

# ============================================================================
# USAGE
# ============================================================================

"""
Usage in Colab:

# IMPORTANT: Force reload to avoid cached 50-test version
import importlib
import comprehensive_gap_test
importlib.reload(comprehensive_gap_test)
from comprehensive_gap_test import run_comprehensive_tests, COMPREHENSIVE_TESTS

# Verify you have 100 tests (not 50)
print(f"Total tests loaded: {len(COMPREHENSIVE_TESTS)}")  # Should show 100

# Run all 100 tests
results = run_comprehensive_tests(model, tokenizer, "unsloth")

# Review results to identify gaps before regenerating dataset
"""
