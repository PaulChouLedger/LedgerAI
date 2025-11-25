#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Intelligent Follow-Up Questions to Medical Dataset
=====================================================
Leverages LLM knowledge to add context-aware, diagnosis-specific follow-up questions
that go beyond the OLD CARTS framework. These questions are based on:
- Chief complaint
- Probable diagnosis
- Medical knowledge (medications, risk factors, associated symptoms, etc.)

Examples:
- Hypertension → Ask about BP medications, family history, lifestyle factors
- Chest Pain → Ask about cardiac risk factors, medications (aspirin, statins)
- GI Bleed → Ask about NSAIDs, anticoagulants, alcohol use
- Diabetes → Ask about medications, blood sugar monitoring, complications
"""

import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# ============================================================================
# Diagnosis-Specific Follow-Up Questions
# ============================================================================

DIAGNOSIS_FOLLOWUP_QUESTIONS = {
    # Cardiovascular
    "Essential Hypertension": {
        "questions": [
            {
                "question": "Are you currently taking any medications for blood pressure?",
                "category": "medications",
                "medical_relevance": "Essential for assessing treatment status and medication adherence",
                "responses": [
                    "Yes, I take lisinopril daily",
                    "Yes, I'm on metoprolol",
                    "Yes, I take amlodipine",
                    "No, I'm not on any blood pressure medications",
                    "I was prescribed something but I don't take it regularly",
                    "I take hydrochlorothiazide"
                ]
            },
            {
                "question": "Do you have a family history of high blood pressure or heart disease?",
                "category": "family_history",
                "medical_relevance": "Family history is a significant risk factor for essential hypertension",
                "responses": [
                    "Yes, my mother has high blood pressure",
                    "Yes, both my parents have hypertension",
                    "Yes, my father had a heart attack",
                    "No, not that I know of",
                    "My grandfather had high blood pressure",
                    "Yes, my siblings also have high blood pressure"
                ]
            },
            {
                "question": "How much salt do you typically consume in your diet?",
                "category": "lifestyle",
                "medical_relevance": "High sodium intake is a modifiable risk factor for hypertension",
                "responses": [
                    "I add salt to most meals",
                    "I try to limit salt but probably still eat too much",
                    "I don't add salt but eat processed foods",
                    "I'm careful about salt intake",
                    "I eat a lot of fast food and processed foods",
                    "I follow a low-sodium diet"
                ]
            },
            {
                "question": "Do you smoke or use tobacco products?",
                "category": "risk_factors",
                "medical_relevance": "Smoking is a major cardiovascular risk factor",
                "responses": [
                    "No, I don't smoke",
                    "Yes, I smoke about a pack a day",
                    "I quit smoking 5 years ago",
                    "I smoke occasionally",
                    "I use chewing tobacco",
                    "I vape"
                ]
            },
            {
                "question": "How often do you exercise?",
                "category": "lifestyle",
                "medical_relevance": "Physical activity helps control blood pressure",
                "responses": [
                    "I exercise 3-4 times a week",
                    "I don't exercise regularly",
                    "I walk daily",
                    "I'm mostly sedentary",
                    "I do intense workouts 5 times a week",
                    "I used to exercise but stopped"
                ]
            }
        ]
    },
    
    "Chest Pain (Cardiac)": {
        "questions": [
            {
                "question": "Do you have any history of heart disease, heart attack, or cardiac procedures?",
                "category": "medical_history",
                "medical_relevance": "Previous cardiac history significantly increases risk of acute coronary syndrome",
                "responses": [
                    "No, no history of heart problems",
                    "Yes, I had a heart attack 2 years ago",
                    "Yes, I have coronary artery disease",
                    "I had a stent placed 5 years ago",
                    "My doctor said I have heart disease",
                    "No cardiac history"
                ]
            },
            {
                "question": "Are you taking any medications like aspirin, clopidogrel, or blood thinners?",
                "category": "medications",
                "medical_relevance": "Antiplatelet and anticoagulant medications are relevant for cardiac risk and bleeding risk",
                "responses": [
                    "Yes, I take daily aspirin",
                    "Yes, I'm on clopidogrel",
                    "Yes, I take warfarin",
                    "No, I'm not on any of those",
                    "I take aspirin occasionally",
                    "I'm on apixaban"
                ]
            },
            {
                "question": "Do you have diabetes, high cholesterol, or high blood pressure?",
                "category": "comorbidities",
                "medical_relevance": "These are major cardiac risk factors",
                "responses": [
                    "Yes, I have diabetes and high blood pressure",
                    "Yes, I have high cholesterol",
                    "I have all three",
                    "No, none of those",
                    "Just high blood pressure",
                    "I have diabetes"
                ]
            },
            {
                "question": "Does the pain occur with exertion or physical activity?",
                "category": "associated_symptoms",
                "medical_relevance": "Exertional chest pain is classic for stable angina",
                "responses": [
                    "Yes, it happens when I walk or climb stairs",
                    "Yes, it comes on with any physical activity",
                    "No, it happens at rest",
                    "Sometimes with exertion, sometimes at rest",
                    "It's worse with activity",
                    "No, it's not related to activity"
                ]
            },
            {
                "question": "Do you have any shortness of breath, nausea, or sweating with the chest pain?",
                "category": "associated_symptoms",
                "medical_relevance": "These are red flag symptoms for acute coronary syndrome",
                "responses": [
                    "Yes, I get short of breath",
                    "Yes, I feel nauseous",
                    "Yes, I break out in a cold sweat",
                    "I have all of those symptoms",
                    "No, just the chest pain",
                    "Sometimes I feel short of breath"
                ]
            }
        ]
    },
    
    "Palpitations": {
        "questions": [
            {
                "question": "Are you taking any medications that could cause palpitations, such as stimulants, decongestants, or asthma medications?",
                "category": "medications",
                "medical_relevance": "Many medications can cause or exacerbate palpitations",
                "responses": [
                    "Yes, I take albuterol for asthma",
                    "Yes, I use nasal decongestants",
                    "I take ADHD medications",
                    "No, I'm not on any of those",
                    "I take pseudoephedrine for colds",
                    "I'm on thyroid medication"
                ]
            },
            {
                "question": "Do you consume caffeine, alcohol, or energy drinks?",
                "category": "lifestyle",
                "medical_relevance": "These substances are common triggers for palpitations",
                "responses": [
                    "Yes, I drink a lot of coffee",
                    "Yes, I drink energy drinks daily",
                    "I drink alcohol regularly",
                    "I consume moderate amounts of caffeine",
                    "No, I avoid caffeine and alcohol",
                    "I drink several cups of coffee a day"
                ]
            },
            {
                "question": "Do you have a history of thyroid problems or hyperthyroidism?",
                "category": "medical_history",
                "medical_relevance": "Hyperthyroidism is a common cause of palpitations",
                "responses": [
                    "Yes, I have an overactive thyroid",
                    "No, no thyroid problems",
                    "I'm not sure, never been checked",
                    "I take thyroid medication",
                    "My doctor mentioned my thyroid might be off",
                    "No thyroid history"
                ]
            },
            {
                "question": "Do the palpitations occur at rest or with activity?",
                "category": "associated_symptoms",
                "medical_relevance": "Helps differentiate benign from concerning causes",
                "responses": [
                    "They happen at rest",
                    "They occur with activity",
                    "Both at rest and with activity",
                    "Mostly when I'm lying down",
                    "They wake me up at night",
                    "Usually when I'm stressed"
                ]
            },
            {
                "question": "Do you have any dizziness, lightheadedness, or fainting spells?",
                "category": "associated_symptoms",
                "medical_relevance": "These symptoms suggest hemodynamic significance",
                "responses": [
                    "Yes, I feel dizzy sometimes",
                    "Yes, I've fainted before",
                    "I get lightheaded with the palpitations",
                    "No, no dizziness or fainting",
                    "I feel faint but haven't passed out",
                    "Sometimes I feel dizzy"
                ]
            }
        ]
    },
    
    # Gastrointestinal
    "Gastroesophageal Reflux Disease": {
        "questions": [
            {
                "question": "Are you taking any medications for acid reflux or heartburn, such as omeprazole, pantoprazole, or ranitidine?",
                "category": "medications",
                "medical_relevance": "Current treatment status and medication effectiveness",
                "responses": [
                    "Yes, I take omeprazole daily",
                    "Yes, I'm on pantoprazole",
                    "I take Tums or antacids occasionally",
                    "No, I'm not on any medications for it",
                    "I was on medication but stopped",
                    "I take famotidine"
                ]
            },
            {
                "question": "What types of foods or drinks trigger your symptoms?",
                "category": "lifestyle",
                "medical_relevance": "Dietary triggers are important for GERD management",
                "responses": [
                    "Spicy foods and coffee",
                    "Tomato-based foods and citrus",
                    "Alcohol and chocolate",
                    "Fried and fatty foods",
                    "Everything seems to trigger it",
                    "I haven't noticed specific triggers"
                ]
            },
            {
                "question": "Do your symptoms worsen when you lie down or bend over?",
                "category": "associated_symptoms",
                "medical_relevance": "Postural worsening is classic for GERD",
                "responses": [
                    "Yes, much worse when I lie down",
                    "Yes, especially after eating",
                    "No, it's not related to position",
                    "It's worse when I bend over",
                    "I wake up with symptoms at night",
                    "Sometimes worse when lying flat"
                ]
            },
            {
                "question": "Do you have any difficulty swallowing or feeling like food gets stuck?",
                "category": "red_flags",
                "medical_relevance": "Dysphagia is a red flag symptom requiring further evaluation",
                "responses": [
                    "No, no swallowing problems",
                    "Yes, sometimes food feels stuck",
                    "I have trouble swallowing solids",
                    "I feel like there's something in my throat",
                    "No difficulty swallowing",
                    "Occasionally I feel like food is stuck"
                ]
            },
            {
                "question": "Have you noticed any weight loss or loss of appetite?",
                "category": "red_flags",
                "medical_relevance": "Unintentional weight loss is a red flag symptom",
                "responses": [
                    "No, my weight is stable",
                    "Yes, I've lost weight without trying",
                    "I've lost my appetite",
                    "No weight loss or appetite changes",
                    "I've lost about 10 pounds",
                    "My appetite is normal"
                ]
            }
        ]
    },
    
    "Acute Cholecystitis": {
        "questions": [
            {
                "question": "Do your symptoms occur after eating fatty or greasy foods?",
                "category": "associated_symptoms",
                "medical_relevance": "Fatty meal intolerance is classic for gallbladder disease",
                "responses": [
                    "Yes, especially after fatty meals",
                    "Yes, fried foods make it worse",
                    "No, it's not related to eating",
                    "It happens regardless of what I eat",
                    "Sometimes after eating",
                    "Yes, greasy foods trigger it"
                ]
            },
            {
                "question": "Do you have any fever or chills?",
                "category": "associated_symptoms",
                "medical_relevance": "Fever suggests infection or inflammation",
                "responses": [
                    "Yes, I have a fever",
                    "Yes, I have chills",
                    "I feel feverish but haven't checked",
                    "No, no fever or chills",
                    "I had a fever earlier",
                    "I feel hot and cold"
                ]
            },
            {
                "question": "Have you noticed any yellowing of your skin or eyes?",
                "category": "red_flags",
                "medical_relevance": "Jaundice suggests biliary obstruction",
                "responses": [
                    "No, no yellowing",
                    "Yes, my eyes look yellow",
                    "My skin looks a bit yellow",
                    "No jaundice",
                    "I'm not sure",
                    "No yellowing noticed"
                ]
            },
            {
                "question": "Do you have a history of gallstones or gallbladder problems?",
                "category": "medical_history",
                "medical_relevance": "Previous history increases likelihood of cholecystitis",
                "responses": [
                    "Yes, I've had gallstones before",
                    "Yes, I had my gallbladder removed",
                    "No, no history of gallbladder problems",
                    "My doctor mentioned gallstones on an ultrasound",
                    "I'm not sure",
                    "No gallbladder history"
                ]
            },
            {
                "question": "Are you taking any medications that could affect the liver or gallbladder?",
                "category": "medications",
                "medical_relevance": "Some medications can cause biliary issues",
                "responses": [
                    "No, I'm not on any such medications",
                    "I take birth control pills",
                    "I'm on hormone replacement therapy",
                    "I'm not sure what medications could affect that",
                    "No relevant medications",
                    "I take various medications but not sure which ones"
                ]
            }
        ]
    },
    
    "Gastroenteritis": {
        "questions": [
            {
                "question": "Have you been in contact with anyone else who has similar symptoms?",
                "category": "epidemiology",
                "medical_relevance": "Suggests infectious cause",
                "responses": [
                    "Yes, my family members are also sick",
                    "Yes, several people at work have it",
                    "No, no one else I know is sick",
                    "My child had similar symptoms",
                    "I'm not sure",
                    "No known contacts"
                ]
            },
            {
                "question": "Have you eaten any unusual foods, traveled recently, or eaten at restaurants?",
                "category": "epidemiology",
                "medical_relevance": "Foodborne illness is common cause of gastroenteritis",
                "responses": [
                    "Yes, I ate at a restaurant yesterday",
                    "Yes, I traveled recently",
                    "I ate some food that tasted off",
                    "No, nothing unusual",
                    "I had takeout food",
                    "I'm not sure"
                ]
            },
            {
                "question": "Are you able to keep fluids down, or are you vomiting everything?",
                "category": "severity_assessment",
                "medical_relevance": "Determines risk of dehydration",
                "responses": [
                    "I can keep some fluids down",
                    "I'm vomiting everything",
                    "I can drink small amounts",
                    "I'm keeping fluids down but not food",
                    "I'm very dehydrated",
                    "I can drink water but it comes back up"
                ]
            },
            {
                "question": "How many times have you had diarrhea or vomited in the last 24 hours?",
                "category": "severity_assessment",
                "medical_relevance": "Helps assess severity and dehydration risk",
                "responses": [
                    "About 5-6 times",
                    "More than 10 times",
                    "Just a few times",
                    "Too many to count",
                    "Maybe 3-4 times",
                    "I've lost track"
                ]
            },
            {
                "question": "Do you have any blood in your stool or vomit?",
                "category": "red_flags",
                "medical_relevance": "Blood suggests more serious pathology",
                "responses": [
                    "No, no blood",
                    "Yes, I see blood in my stool",
                    "Yes, there's blood when I vomit",
                    "No blood noticed",
                    "I'm not sure, it's hard to tell",
                    "No blood"
                ]
            }
        ]
    },
    
    # Add more diagnoses as needed
    "Acute Upper GI Bleed": {
        "questions": [
            {
                "question": "Are you taking any medications like aspirin, ibuprofen, naproxen, or other NSAIDs?",
                "category": "medications",
                "medical_relevance": "NSAIDs are a major cause of upper GI bleeding",
                "responses": [
                    "Yes, I take aspirin daily",
                    "Yes, I take ibuprofen regularly",
                    "Yes, I take naproxen for arthritis",
                    "No, I'm not on any NSAIDs",
                    "I take them occasionally for pain",
                    "I was taking them but stopped"
                ]
            },
            {
                "question": "Are you taking any blood thinners like warfarin, apixaban, or clopidogrel?",
                "category": "medications",
                "medical_relevance": "Anticoagulants increase bleeding risk",
                "responses": [
                    "Yes, I take warfarin",
                    "Yes, I'm on apixaban",
                    "Yes, I take clopidogrel",
                    "No, I'm not on blood thinners",
                    "I take aspirin and clopidogrel",
                    "No anticoagulants"
                ]
            },
            {
                "question": "Do you drink alcohol, and if so, how much?",
                "category": "lifestyle",
                "medical_relevance": "Alcohol is a major risk factor for GI bleeding",
                "responses": [
                    "No, I don't drink",
                    "I drink occasionally",
                    "I drink daily, several drinks",
                    "I'm a heavy drinker",
                    "I drink socially",
                    "I used to drink heavily but stopped"
                ]
            },
            {
                "question": "Do you have a history of ulcers, gastritis, or stomach problems?",
                "category": "medical_history",
                "medical_relevance": "Previous GI history increases risk",
                "responses": [
                    "Yes, I've had ulcers before",
                    "Yes, I have a history of gastritis",
                    "No, no history of stomach problems",
                    "I've had stomach issues in the past",
                    "I'm not sure",
                    "No GI history"
                ]
            },
            {
                "question": "Have you noticed any black, tarry stools?",
                "category": "associated_symptoms",
                "medical_relevance": "Melena suggests upper GI bleeding",
                "responses": [
                    "Yes, my stools are black and tarry",
                    "No, my stools look normal",
                    "I haven't checked",
                    "They're dark but not sure if tarry",
                    "No melena",
                    "I'm not sure what my stools look like"
                ]
            }
        ]
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

def extract_diagnosis_from_conversation(conversation: Dict) -> Optional[str]:
    """Extract the most probable diagnosis from the conversation."""
    messages = conversation.get("messages", [])
    
    # Look for final diagnostic reasoning
    for msg in reversed(messages):
        content = msg.get("content", "")
        if "FINAL DIAGNOSTIC REASONING" in content:
            # Extract diagnosis from final reasoning
            # Pattern: "most probable diagnosis is X"
            match = re.search(r"most probable diagnosis is ([^(]+)", content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            
            # Alternative pattern: "MOST PROBABLE" in ranked diagnosis
            match = re.search(r"1\.\s*([^:]+):\s*\d+%", content)
            if match:
                return match.group(1).strip()
    
    # Look in clinical reasoning for diagnosis mentions
    for msg in messages:
        content = msg.get("content", "")
        if "CLASSIC for" in content:
            match = re.search(r"CLASSIC for ([^.]+)", content)
            if match:
                return match.group(1).strip()
    
    return None

def find_best_matching_diagnosis(diagnosis: str) -> Optional[str]:
    """Find the best matching diagnosis key from DIAGNOSIS_FOLLOWUP_QUESTIONS."""
    diagnosis_lower = diagnosis.lower()
    
    # Direct match
    for key in DIAGNOSIS_FOLLOWUP_QUESTIONS.keys():
        if key.lower() == diagnosis_lower:
            return key
    
    # Partial match
    for key in DIAGNOSIS_FOLLOWUP_QUESTIONS.keys():
        key_lower = key.lower()
        if key_lower in diagnosis_lower or diagnosis_lower in key_lower:
            return key
    
    # Keyword matching (expanded)
    keywords = {
        # Hypertension
        "hypertension": "Essential Hypertension",
        "blood pressure": "Essential Hypertension",
        "high blood pressure": "Essential Hypertension",
        "hypertensive": "Essential Hypertension",
        
        # Cardiac
        "chest pain": "Chest Pain (Cardiac)",
        "cardiac": "Chest Pain (Cardiac)",
        "myocardial infarction": "Chest Pain (Cardiac)",
        "heart attack": "Chest Pain (Cardiac)",
        "angina": "Chest Pain (Cardiac)",
        "unstable angina": "Chest Pain (Cardiac)",
        
        # Palpitations
        "palpitations": "Palpitations",
        "heart racing": "Palpitations",
        "atrial fibrillation": "Palpitations",
        
        # GERD
        "gerd": "Gastroesophageal Reflux Disease",
        "reflux": "Gastroesophageal Reflux Disease",
        "heartburn": "Gastroesophageal Reflux Disease",
        "gastroesophageal": "Gastroesophageal Reflux Disease",
        
        # Cholecystitis
        "cholecystitis": "Acute Cholecystitis",
        "gallbladder": "Acute Cholecystitis",
        
        # Gastroenteritis
        "gastroenteritis": "Gastroenteritis",
        
        # GI Bleed
        "gi bleed": "Acute Upper GI Bleed",
        "upper gi bleed": "Acute Upper GI Bleed",
        "coffee ground": "Acute Upper GI Bleed",
        "lower gi bleed": "Acute Upper GI Bleed",  # Similar questions
    }
    
    for keyword, key in keywords.items():
        if keyword in diagnosis_lower:
            return key
    
    return None

def should_add_followups(conversation: Dict) -> bool:
    """Determine if follow-up questions should be added to this conversation."""
    # Only add to conversations that have completed OLD CARTS assessment
    messages = conversation.get("messages", [])
    
    # Skip if already has follow-ups
    if conversation.get("has_intelligent_followups"):
        return False
    
    has_final_reasoning = any("FINAL DIAGNOSTIC REASONING" in msg.get("content", "") 
                              for msg in messages)
    
    # Should have at least some OLD CARTS questions asked
    has_oldcarts = any("OLD CARTS" in msg.get("content", "") or 
                      "When did" in msg.get("content", "") or
                      "How long" in msg.get("content", "") or
                      "What does" in msg.get("content", "")
                      for msg in messages)
    
    # Need enough conversation to have meaningful context
    return (has_final_reasoning or (has_oldcarts and len(messages) > 8)) and len(messages) > 5

def add_followup_questions(conversation: Dict, num_questions: int = 2) -> Dict:
    """Add intelligent follow-up questions to a conversation."""
    # Extract diagnosis
    diagnosis = extract_diagnosis_from_conversation(conversation)
    if not diagnosis:
        return conversation
    
    # Find matching follow-up questions
    matching_key = find_best_matching_diagnosis(diagnosis)
    if not matching_key:
        return conversation
    
    followup_data = DIAGNOSIS_FOLLOWUP_QUESTIONS.get(matching_key)
    if not followup_data:
        return conversation
    
    # Select random questions (avoid duplicates)
    available_questions = followup_data["questions"]
    selected_questions = random.sample(
        available_questions, 
        min(num_questions, len(available_questions))
    )
    
    # Find insertion point (after final reasoning, before end)
    messages = conversation.get("messages", [])
    insertion_index = len(messages)
    
    for i, msg in enumerate(messages):
        if "FINAL DIAGNOSTIC REASONING" in msg.get("content", ""):
            insertion_index = i + 1
            break
    
    # Add follow-up questions
    new_messages = messages[:insertion_index]
    
    for question_data in selected_questions:
        question = question_data["question"]
        response = random.choice(question_data["responses"])
        category = question_data["category"]
        medical_relevance = question_data["medical_relevance"]
        
        # Add question
        new_messages.append({
            "role": "assistant",
            "content": question
        })
        
        # Add patient response
        new_messages.append({
            "role": "user",
            "content": response
        })
        
        # Add clinical reasoning for follow-up
        reasoning = f"""CLINICAL REASONING: This is a follow-up question ({category}) based on the probable diagnosis ({diagnosis}).

MEDICAL RELEVANCE: {medical_relevance}

PATIENT RESPONSE: The patient reported '{response}'.

CLINICAL SIGNIFICANCE: This information helps:
- Refine the differential diagnosis
- Assess risk factors and comorbidities
- Guide treatment recommendations
- Identify red flag symptoms requiring urgent evaluation

UPDATED ASSESSMENT: Based on this additional information, the diagnosis of {diagnosis} remains the most probable. This follow-up question provides important context for comprehensive clinical assessment."""
        
        new_messages.append({
            "role": "assistant",
            "content": reasoning
        })
    
    # Add remaining messages
    new_messages.extend(messages[insertion_index:])
    
    # Update conversation
    conversation["messages"] = new_messages
    conversation["has_intelligent_followups"] = True
    conversation["followup_diagnosis"] = matching_key
    conversation["num_followups"] = len(selected_questions)
    
    return conversation

# ============================================================================
# Main Processing
# ============================================================================

def process_dataset(input_path: str, output_path: str, num_followups: int = 2):
    """Process dataset and add intelligent follow-up questions."""
    print("=" * 80)
    print("Adding Intelligent Follow-Up Questions to Medical Dataset")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} conversations")
    print()
    
    # Process conversations
    print("Processing conversations...")
    processed = 0
    added_followups = 0
    
    for idx, conversation in enumerate(data):
        if should_add_followups(conversation):
            original_length = len(conversation.get("messages", []))
            conversation = add_followup_questions(conversation, num_followups)
            new_length = len(conversation.get("messages", []))
            
            if new_length > original_length:
                added_followups += 1
            
            processed += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(data)} conversations...")
    
    print(f"✅ Processed {processed} conversations")
    print(f"✅ Added follow-ups to {added_followups} conversations")
    print()
    
    # Save enhanced dataset
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✅ Enhanced dataset saved!")
    print()
    print("=" * 80)
    print("Enhancement Complete!")
    print("=" * 80)
    print(f"Total conversations: {len(data)}")
    print(f"Conversations with follow-ups: {added_followups}")
    print()
    print("Features added:")
    print("  ✅ Intelligent, diagnosis-specific follow-up questions")
    print("  ✅ Questions based on medical knowledge (medications, risk factors, etc.)")
    print("  ✅ Clinical reasoning for follow-up questions")
    print("  ✅ Context-aware question selection")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    input_file = "medical_sft_dataset_enhanced_smart.json"
    output_file = "medical_sft_dataset_enhanced_smart_intelligent.json"
    num_followups = 2  # Number of follow-up questions per conversation
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        num_followups = int(sys.argv[3])
    
    process_dataset(input_file, output_file, num_followups)

