#!/usr/bin/env python3
"""
Generate Complex Medical Dataset
=================================
Focuses on complex presentations requiring:
1. Cross-organ system differentiation (e.g., chest pain - GERD vs cardiac)
2. Clarification questions for ambiguous answers (RUQ vs RLQ)
3. Progressive scoring/ranking with rolling differential diagnosis
4. LLM internal reasoning feeding into ranking system
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ============================================================================
# Complex Case Definitions
# ============================================================================

COMPLEX_CASES = [
    {
        "title": "Chest Pain - GERD vs Cardiac",
        "chief_complaint": "I have chest pain",
        "differential": [
            "Gastroesophageal Reflux Disease (GERD)",
            "Acute Myocardial Infarction (Heart Attack)",
            "Unstable Angina",
            "Pericarditis",
            "Aortic Dissection"
        ],
        "organ_systems": ["gastrointestinal", "cardiovascular"],
        "target_diagnosis": "Gastroesophageal Reflux Disease (GERD)",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": True, "T": True, "S": True
        }
    },
    {
        "title": "Chest Pain - Cardiac",
        "chief_complaint": "I have chest pain",
        "differential": [
            "Acute Myocardial Infarction (Heart Attack)",
            "Unstable Angina",
            "Gastroesophageal Reflux Disease (GERD)",
            "Pericarditis",
            "Aortic Dissection"
        ],
        "organ_systems": ["cardiovascular", "gastrointestinal"],
        "target_diagnosis": "Acute Myocardial Infarction (Heart Attack)",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": True, "T": True, "S": True
        }
    },
    {
        "title": "Abdominal Pain - RUQ vs RLQ Clarification",
        "chief_complaint": "I have abdominal pain",
        "differential": [
            "Acute Cholecystitis",
            "Acute Appendicitis",
            "Acute Gastroenteritis",
            "Hepatitis",
            "Nephrolithiasis (Kidney Stones)"
        ],
        "organ_systems": ["gastrointestinal"],
        "target_diagnosis": "Acute Cholecystitis",
        "clarification_needed": True,  # Location will be ambiguous initially
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": False, "T": True, "S": True
        }
    },
    {
        "title": "Abdominal Pain - RLQ",
        "chief_complaint": "I have abdominal pain",
        "differential": [
            "Acute Appendicitis",
            "Acute Cholecystitis",
            "Acute Gastroenteritis",
            "Nephrolithiasis (Kidney Stones)",
            "Ovarian Torsion"
        ],
        "organ_systems": ["gastrointestinal"],
        "target_diagnosis": "Acute Appendicitis",
        "clarification_needed": True,
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": False, "T": True, "S": True
        }
    },
    {
        "title": "Flank Pain - Renal vs GI",
        "chief_complaint": "I have flank pain",
        "differential": [
            "Nephrolithiasis (Kidney Stones)",
            "Pyelonephritis",
            "Muscle Strain",
            "Herpes Zoster",
            "Aortic Aneurysm"
        ],
        "organ_systems": ["renal", "musculoskeletal"],
        "target_diagnosis": "Nephrolithiasis (Kidney Stones)",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": True, "T": True, "S": True
        }
    },
    {
        "title": "Headache - Migraine vs Tension",
        "chief_complaint": "I have headache",
        "differential": [
            "Migraine",
            "Tension Headache",
            "Sinusitis",
            "Cluster Headache",
            "Medication Overuse Headache"
        ],
        "organ_systems": ["neurological"],
        "target_diagnosis": "Migraine",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": False, "T": True, "S": True
        }
    },
    {
        "title": "Shortness of Breath - Asthma vs Pneumonia",
        "chief_complaint": "I have shortness of breath",
        "differential": [
            "Asthma",
            "Pneumonia",
            "Acute Heart Failure",
            "COPD Exacerbation",
            "Pulmonary Embolism"
        ],
        "organ_systems": ["respiratory", "cardiovascular"],
        "target_diagnosis": "Asthma",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": False, "D": True, "C": False,
            "A_aggravating": True, "A_alleviating": True,
            "R": False, "T": True, "S": True
        }
    },
    {
        "title": "Cough - Pneumonia vs Bronchitis",
        "chief_complaint": "I have cough",
        "differential": [
            "Pneumonia",
            "Acute Bronchitis",
            "Upper Respiratory Infection",
            "Asthma",
            "Post-nasal Drip"
        ],
        "organ_systems": ["respiratory"],
        "target_diagnosis": "Pneumonia",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": False, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": False, "T": True, "S": True
        }
    },
    {
        "title": "Back Pain - Musculoskeletal vs Renal",
        "chief_complaint": "I have back pain",
        "differential": [
            "Muscle Strain",
            "Nephrolithiasis (Kidney Stones)",
            "Pyelonephritis",
            "Herniated Disc",
            "Musculoskeletal Back Pain"
        ],
        "organ_systems": ["musculoskeletal", "renal"],
        "target_diagnosis": "Muscle Strain",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": True, "T": True, "S": True
        }
    },
    {
        "title": "Pelvic Pain - PID vs UTI",
        "chief_complaint": "I have pelvic pain",
        "differential": [
            "Pelvic Inflammatory Disease (PID)",
            "Urinary Tract Infection (UTI)",
            "Ovarian Cyst",
            "Endometriosis",
            "Ovarian Torsion"
        ],
        "organ_systems": ["genitourinary"],
        "target_diagnosis": "Pelvic Inflammatory Disease (PID)",
        "clarification_needed": False,
        "metadata": {
            "O": True, "L": True, "D": True, "C": True,
            "A_aggravating": True, "A_alleviating": True,
            "R": False, "T": True, "S": True
        }
    }
]

# ============================================================================
# Answer Templates by Diagnosis
# ============================================================================

ANSWER_TEMPLATES = {
    "Gastroesophageal Reflux Disease (GERD)": {
        "O": ["started gradually after eating", "came on after dinner", "started after I ate"],
        "L": ["in the center of my chest", "behind my breastbone", "in my chest, more to the center"],
        "D": ["for about an hour", "since after dinner", "for a few hours"],
        "C": ["burning", "burning sensation", "like a burning feeling"],
        "A_aggravating": ["worse when I lie down", "worse after eating", "worse after spicy foods", "worse when lying down after meals"],
        "A_alleviating": ["sitting up helps", "antacids help", "staying upright helps"],
        "R": ["travels up to my throat", "sometimes goes up to my throat", "radiates up after meals"],
        "T": ["comes and goes", "intermittent", "happens after meals"],
        "S": ["about 5 or 6 out of 10", "moderate, maybe 5", "5 to 6"]
    },
    "Acute Myocardial Infarction (Heart Attack)": {
        "O": ["started suddenly", "came on suddenly", "started suddenly about an hour ago"],
        "L": ["in the center of my chest", "behind my sternum", "pressure in my chest"],
        "D": ["for about an hour", "for the past hour", "since this morning"],
        "C": ["pressure and heaviness", "squeezing", "heavy pressure", "crushing"],
        "A_aggravating": ["worse with exertion", "worse with activity", "worse when I move"],
        "A_alleviating": ["rest helps a bit", "nothing really helps", "resting makes it slightly better"],
        "R": ["radiates to my left arm", "goes to my jaw", "down my left arm", "to my shoulder"],
        "T": ["constant", "it's been constant", "continuous"],
        "S": ["8 out of 10", "severe, maybe 8", "7 to 8"]
    },
    "Acute Cholecystitis": {
        "O": ["started after I ate", "came on after eating", "started after dinner"],
        "L": ["on my right side", "in my upper right abdomen", "under my right ribs"],  # Initial ambiguous
        "L_clarified": ["on my right side, under my ribs", "upper right, near my ribs", "right side, upper abdomen"],
        "D": ["for a few hours", "since dinner", "for about 3 hours"],
        "C": ["sharp", "sharp and stabbing", "sharp pain"],
        "A_aggravating": ["worse after eating", "worse with fatty foods", "worse after meals"],
        "A_alleviating": ["nothing really helps", "lying still helps a bit"],
        "T": ["constant", "it's constant", "continuous"],
        "S": ["about 7 out of 10", "7 to 8", "severe, maybe 7"]
    },
    "Acute Appendicitis": {
        "O": ["started gradually", "came on slowly", "started a few hours ago"],
        "L": ["on my right side", "in my abdomen", "lower right"],  # Initial ambiguous
        "L_clarified": ["on my right side, lower down", "lower right, near my hip", "right side, lower abdomen near my groin"],
        "D": ["for a few hours", "since this morning", "for about 6 hours"],
        "C": ["sharp", "sharp and stabbing", "crampy at first, now sharp"],
        "A_aggravating": ["worse with movement", "worse when I walk", "worse with any movement"],
        "A_alleviating": ["lying still helps", "bending over helps", "nothing really helps"],
        "T": ["constant now", "it's constant", "was crampy, now constant"],
        "S": ["about 7 out of 10", "7 to 8", "severe"]
    },
    "Nephrolithiasis (Kidney Stones)": {
        "O": ["started suddenly", "came on suddenly", "started suddenly this morning"],
        "L": ["in my side", "on my right side", "flank area"],
        "D": ["for a few hours", "since this morning", "for about 3 hours"],
        "C": ["severe, sharp", "excruciating", "sharp and stabbing"],
        "A_aggravating": ["nothing makes it worse, it's already terrible", "any movement"],
        "A_alleviating": ["nothing really helps", "maybe some positions"],
        "R": ["radiates to my groin", "goes down to my groin", "radiates down"],
        "T": ["constant, severe", "constant waves", "continuous"],
        "S": ["9 out of 10", "very severe, 9", "almost 10"]
    },
    "Migraine": {
        "O": ["started gradually", "came on slowly", "started a few hours ago"],
        "L": ["on one side of my head", "unilateral", "right side of head"],
        "D": ["for a few hours", "since this morning", "for about 4 hours"],
        "C": ["throbbing", "pulsating", "throbbing pain"],
        "A_aggravating": ["worse with light", "worse with sound", "worse with movement"],
        "A_alleviating": ["lying in dark room helps", "sleep helps", "medication helps"],
        "T": ["intermittent", "comes and goes", "episodic"],
        "S": ["7 out of 10", "severe, 7", "moderate to severe"]
    },
    "Asthma": {
        "O": ["started gradually", "came on slowly", "started after exposure"],
        "D": ["for a few hours", "since this morning", "for about 2 hours"],
        "C": ["wheezing", "tightness in chest", "can't catch my breath"],
        "A_aggravating": ["exercise", "allergens", "cold air"],
        "A_alleviating": ["inhaler helps", "rest helps", "sitting up helps"],
        "T": ["intermittent", "episodic", "comes and goes"],
        "S": ["6 out of 10", "moderate, maybe 6", "moderate"]
    },
    "Pneumonia": {
        "O": ["started gradually", "came on over days", "started a few days ago"],
        "D": ["for a few days", "since a few days ago", "for about 3 days"],
        "C": ["productive cough", "chest pain with breathing", "mucous production"],
        "A_aggravating": ["deep breathing", "lying flat", "coughing"],
        "A_alleviating": ["rest", "sitting up", "medication"],
        "T": ["constant", "persistent", "ongoing"],
        "S": ["7 out of 10", "moderate to severe, 7", "severe"]
    },
    "Muscle Strain": {
        "O": ["started after lifting", "came on after activity", "started after I moved something"],
        "L": ["in my lower back", "lower back area", "mid to lower back"],
        "D": ["for a few hours", "since this morning", "for about 4 hours"],
        "C": ["achy", "dull ache", "tightness"],
        "A_aggravating": ["movement", "bending", "lifting"],
        "A_alleviating": ["rest helps", "heat helps", "lying down"],
        "R": ["sometimes radiates to buttocks", "can go to my leg", "radiates down"],
        "T": ["constant", "worse with movement", "persistent"],
        "S": ["5 out of 10", "moderate, maybe 5", "5 to 6"]
    },
    "Pelvic Inflammatory Disease (PID)": {
        "O": ["started gradually", "came on over days", "started a few days ago"],
        "L": ["in my lower abdomen", "pelvic area", "lower belly"],
        "D": ["for a few days", "since a few days ago", "for about 3 days"],
        "C": ["dull ache", "aching pain", "constant ache"],
        "A_aggravating": ["intercourse", "movement", "pelvic exam"],
        "A_alleviating": ["rest helps slightly", "antibiotics help"],
        "T": ["constant", "persistent", "ongoing"],
        "S": ["6 out of 10", "moderate to severe, 6", "6 to 7"]
    }
}

# ============================================================================
# Scoring Deltas by Answer Pattern
# ============================================================================

# ============================================================================
# Associated Symptoms for Differentiation
# ============================================================================

ASSOCIATED_SYMPTOMS = {
    "Acute Myocardial Infarction (Heart Attack)": {
        "questions": [
            {
                "question": "Are you experiencing any sweating or feeling clammy?",
                "positive_answer": "yes, I'm sweating",
                "negative_answer": "no, no sweating",
                "score_delta": 0.15
            },
            {
                "question": "Do you have any nausea or vomiting?",
                "positive_answer": "yes, I feel nauseous",
                "negative_answer": "no, no nausea",
                "score_delta": 0.10
            },
            {
                "question": "Are you experiencing shortness of breath or difficulty breathing?",
                "positive_answer": "yes, I'm short of breath",
                "negative_answer": "no, breathing is fine",
                "score_delta": 0.12
            }
        ]
    },
    "Gastroesophageal Reflux Disease (GERD)": {
        "questions": [
            {
                "question": "Do you have a sour taste in your mouth or regurgitation?",
                "positive_answer": "yes, sour taste",
                "negative_answer": "no, no sour taste",
                "score_delta": 0.20
            },
            {
                "question": "Do you feel like food or liquid is coming back up into your throat?",
                "positive_answer": "yes, sometimes",
                "negative_answer": "no, nothing comes back up",
                "score_delta": 0.15
            }
        ]
    },
    "Acute Cholecystitis": {
        "questions": [
            {
                "question": "Do you have any nausea or vomiting?",
                "positive_answer": "yes, I've been nauseous and vomiting",
                "negative_answer": "no, no nausea",
                "score_delta": 0.12
            },
            {
                "question": "Do you have a fever or feel feverish?",
                "positive_answer": "yes, I feel feverish",
                "negative_answer": "no, no fever",
                "score_delta": 0.10
            }
        ]
    },
    "Acute Appendicitis": {
        "questions": [
            {
                "question": "Do you have any nausea or vomiting?",
                "positive_answer": "yes, I've been nauseous",
                "negative_answer": "no, no nausea",
                "score_delta": 0.10
            },
            {
                "question": "Have you lost your appetite?",
                "positive_answer": "yes, no appetite at all",
                "negative_answer": "no, appetite is normal",
                "score_delta": 0.08
            }
        ]
    },
    "Nephrolithiasis (Kidney Stones)": {
        "questions": [
            {
                "question": "Have you noticed any blood in your urine?",
                "positive_answer": "yes, I saw blood",
                "negative_answer": "no, no blood",
                "score_delta": 0.15
            },
            {
                "question": "Do you feel like you need to urinate frequently or urgently?",
                "positive_answer": "yes, constantly",
                "negative_answer": "no, normal",
                "score_delta": 0.08
            }
        ]
    },
    "Unstable Angina": {
        "questions": [
            {
                "question": "Are you experiencing any sweating?",
                "positive_answer": "yes, I'm sweating",
                "negative_answer": "no, no sweating",
                "score_delta": 0.10
            },
            {
                "question": "Do you have any shortness of breath?",
                "positive_answer": "yes, short of breath",
                "negative_answer": "no, breathing is fine",
                "score_delta": 0.12
            }
        ]
    },
    "Migraine": {
        "questions": [
            {
                "question": "Do you have any sensitivity to light or sound?",
                "positive_answer": "yes, very sensitive to light and sound",
                "negative_answer": "no, no sensitivity",
                "score_delta": 0.20
            },
            {
                "question": "Do you have any nausea or vomiting?",
                "positive_answer": "yes, I feel nauseous",
                "negative_answer": "no, no nausea",
                "score_delta": 0.12
            }
        ]
    },
    "Asthma": {
        "questions": [
            {
                "question": "Do you have any wheezing?",
                "positive_answer": "yes, I can hear wheezing",
                "negative_answer": "no, no wheezing",
                "score_delta": 0.18
            },
            {
                "question": "Do you have a history of asthma or allergies?",
                "positive_answer": "yes, I have asthma",
                "negative_answer": "no, no history",
                "score_delta": 0.15
            }
        ]
    },
    "Pneumonia": {
        "questions": [
            {
                "question": "Do you have a fever?",
                "positive_answer": "yes, I feel feverish",
                "negative_answer": "no, no fever",
                "score_delta": 0.15
            },
            {
                "question": "Is your cough productive with phlegm or sputum?",
                "positive_answer": "yes, productive with phlegm",
                "negative_answer": "no, dry cough",
                "score_delta": 0.12
            }
        ]
    },
    "Muscle Strain": {
        "questions": [
            {
                "question": "Did this start after a specific activity or movement?",
                "positive_answer": "yes, after lifting something heavy",
                "negative_answer": "no, it just started",
                "score_delta": 0.12
            }
        ]
    },
    "Pelvic Inflammatory Disease (PID)": {
        "questions": [
            {
                "question": "Do you have any abnormal vaginal discharge?",
                "positive_answer": "yes, abnormal discharge",
                "negative_answer": "no, normal discharge",
                "score_delta": 0.15
            },
            {
                "question": "Do you have a fever?",
                "positive_answer": "yes, I feel feverish",
                "negative_answer": "no, no fever",
                "score_delta": 0.10
            },
            {
                "question": "Is the pain worse with intercourse?",
                "positive_answer": "yes, much worse",
                "negative_answer": "no, doesn't change",
                "score_delta": 0.12
            }
        ]
    }
}

# ============================================================================
# Scoring Deltas by Answer Pattern
# ============================================================================

SCORING_DELTAS = {
    "chest_pain_burning": {
        "Gastroesophageal Reflux Disease (GERD)": 0.2,
        "Acute Myocardial Infarction (Heart Attack)": -0.1,
        "Unstable Angina": -0.1,
        "Pericarditis": -0.1,
        "Aortic Dissection": -0.2
    },
    "chest_pain_pressure": {
        "Acute Myocardial Infarction (Heart Attack)": 0.2,
        "Unstable Angina": 0.2,
        "Gastroesophageal Reflux Disease (GERD)": -0.2,
        "Pericarditis": 0.0,
        "Aortic Dissection": 0.0
    },
    "chest_pain_worse_lying_down": {
        "Gastroesophageal Reflux Disease (GERD)": 0.3,
        "Acute Myocardial Infarction (Heart Attack)": -0.2,
        "Unstable Angina": -0.2,
        "Pericarditis": -0.1
    },
    "chest_pain_worse_exertion": {
        "Acute Myocardial Infarction (Heart Attack)": 0.2,
        "Unstable Angina": 0.2,
        "Gastroesophageal Reflux Disease (GERD)": -0.2,
        "Pericarditis": 0.0
    },
    "abdominal_pain_RUQ": {
        "Acute Cholecystitis": 0.3,
        "Hepatitis": 0.2,
        "Acute Appendicitis": -0.3,
        "Acute Gastroenteritis": -0.1
    },
    "abdominal_pain_RLQ": {
        "Acute Appendicitis": 0.3,
        "Ovarian Torsion": 0.2,
        "Acute Cholecystitis": -0.3,
        "Hepatitis": -0.2
    }
}

# ============================================================================
# Conversation Generation
# ============================================================================

def generate_complex_conversation(case_config: Dict, variant: str = "american") -> Dict:
    """Generate a complex conversation with progressive scoring and rankings."""
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": "You are a medical professional conducting a clinical history. Think like a doctor: recognize chief complaints, build differential diagnoses, and rank conditions by probability.\n\nIMPORTANT: Clinical reasoning is provided ONLY for OLD CARTS elements (Onset, Location, Duration, Character, Aggravating, Alleviating, Radiation, Timing, Severity) and associated symptoms. Do NOT provide clinical reasoning for demographic information (age, biological sex) - these are collected for context but do not require differential diagnosis reasoning.\n\nCRITICAL: You must identify which OLD CARTS element the patient's answer corresponds to based on the answer content and conversation context.\n\nSCORING: After each OLD CARTS answer, evaluate how it affects each condition in the differential diagnosis and update their probabilities accordingly."
    })
    
    chief_complaint = case_config["chief_complaint"]
    if variant == "british" and "I have" in chief_complaint:
        chief_complaint = chief_complaint.replace("I have", "I've got")
    
    differential = case_config["differential"]
    target_diagnosis = case_config["target_diagnosis"]
    metadata = case_config["metadata"]
    
    # Initial condition scores (all start at 20% for balanced baseline)
    condition_scores = {cond: 20.0 for cond in differential}
    
    # Chief complaint
    messages.append({
        "role": "user",
        "content": chief_complaint
    })
    
    symptom = extract_symptom(chief_complaint)
    messages.append({
        "role": "assistant",
        "content": f"I understand you're experiencing {symptom}. I'm here to help."
    })
    
    # Chronicity
    messages.append({
        "role": "assistant",
        "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
    })
    
    messages.append({
        "role": "user",
        "content": "It's new"
    })
    
    # Demographics
    age = random.randint(35, 65)
    messages.append({
        "role": "assistant",
        "content": "How old are you?"
    })
    
    messages.append({
        "role": "user",
        "content": f"I'm {age} years old"
    })
    
    messages.append({
        "role": "assistant",
        "content": "What is your biological sex?"
    })
    
    sex = random.choice(["Male", "Female"])
    messages.append({
        "role": "user",
        "content": sex
    })
    
    # OLD CARTS elements with progressive scoring
    element_order = ["O", "L", "D", "C", "A_aggravating", "A_alleviating", "R", "T", "S"]
    pending_location_clarification = None
    
    for element in element_order:
        if not metadata.get(element, False):
            messages.append({
                "role": "assistant",
                "content": f"[SKIP:{element[0]}] This OLD CARTS element is not relevant for this chief complaint and should be skipped.",
                "metadata": {"skip": True, "element": element[0]}
            })
            continue
        
        # Get current top 3 conditions for context-aware question examples
        if condition_scores:
            current_ranked = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
            top_3_conditions = [cond for cond, _ in current_ranked[:3]]
        else:
            # Initial state - use first 3 from differential
            top_3_conditions = differential[:3] if len(differential) >= 3 else differential
        
        # Generate question with examples based on top 3 conditions
        question = generate_question(element, symptom, chief_complaint, top_3_conditions)
        messages.append({
            "role": "assistant",
            "content": question
        })
        
        # Generate answer (with clarification if needed)
        if element == "L" and case_config.get("clarification_needed", False) and pending_location_clarification is None:
            # Initial ambiguous answer
            patient_answer = get_patient_answer(target_diagnosis, element, initial=True)
            messages.append({
                "role": "user",
                "content": patient_answer
            })
            
            pending_location_clarification = patient_answer
            
            # Clarification question
            messages.append({
                "role": "assistant",
                "content": f"Is the pain in your right upper abdomen (near your ribs/liver area) or right lower abdomen (near your hip/appendix area)?"
            })
            
            # Clarified answer
            clarified_answer = get_patient_answer(target_diagnosis, element, clarified=True)
            combined_answer = f"{patient_answer}, {clarified_answer}"
            messages.append({
                "role": "user",
                "content": clarified_answer
            })
            
            patient_answer = combined_answer
        else:
            patient_answer = get_patient_answer(target_diagnosis, element)
            messages.append({
                "role": "user",
                "content": patient_answer
            })
        
        # Calculate score deltas and update rankings
        deltas = calculate_score_deltas(element, patient_answer, chief_complaint, differential, target_diagnosis)
        
        # Update condition scores with more gradual changes
        for condition, delta in deltas.items():
            if condition in condition_scores:
                # Apply delta as percentage points (not raw percentage)
                change = delta * 20  # Scale down deltas to be more gradual
                condition_scores[condition] = max(1.0, min(95.0, condition_scores[condition] + change))
        
        # Normalize scores to sum to 100%, but keep minimum 1% for all
        total = sum(condition_scores.values())
        if total > 0:
            # First normalize
            condition_scores = {k: (v / total) * 100 for k, v in condition_scores.items()}
            # Ensure minimum 1% but cap max at 95%
            condition_scores = {k: max(1.0, min(95.0, v)) for k, v in condition_scores.items()}
            # Re-normalize to sum to 100%
            total_adj = sum(condition_scores.values())
            if total_adj > 0:
                condition_scores = {k: (v / total_adj) * 100 for k, v in condition_scores.items()}
        
        # Sort by score
        ranked_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate clinical reasoning with ranking
        reasoning = generate_clinical_reasoning_with_ranking(
            element, patient_answer, chief_complaint, target_diagnosis,
            deltas, ranked_conditions, condition_scores
        )
        
        messages.append({
            "role": "assistant",
            "content": reasoning
        })
    
    # Associated symptoms section - ask questions based on top 3 conditions
    current_ranked = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
    top_3_conditions = [cond for cond, _ in current_ranked[:3]]
    
    # Generate associated symptom questions to differentiate top conditions
    associated_symptoms = generate_associated_symptom_questions(
        top_3_conditions, chief_complaint, target_diagnosis, condition_scores, differential
    )
    messages.extend(associated_symptoms)
    
    # Final diagnostic reasoning
    final_ranked = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
    messages.append({
        "role": "assistant",
        "content": generate_final_reasoning(final_ranked, target_diagnosis)
    })
    
    return {
        "messages": messages,
        "organ_system": case_config["organ_systems"][0],
        "diagnosis": target_diagnosis,
        "chief_complaint": case_config["chief_complaint"],
        "variant": variant,
        "smart_features": True,
        "relevant_oldcarts": metadata,
        "differential_diagnosis": differential,
        "has_intelligent_followups": False
    }

def extract_symptom(complaint: str) -> str:
    """Extract symptom from chief complaint."""
    import re
    symptom = re.sub(r'^i (have|got|am experiencing|feel|notice)\s+', '', complaint.lower())
    return symptom.strip()

def generate_question(element: str, symptom: str, chief_complaint: str = "", top_conditions: List[str] = None) -> str:
    """Generate OLD CARTS question with example answers based on top 3 ranking conditions."""
    if top_conditions is None:
        top_conditions = []
    
    # Get top 3 conditions
    top_3 = top_conditions[:3] if len(top_conditions) >= 3 else top_conditions
    
    # Generate examples based on top conditions
    examples = _get_example_answers_for_element(element, top_3, chief_complaint)
    
    # Format question with examples
    if examples:
        example_text = ", ".join(examples)
        return _format_question_with_examples(element, symptom, example_text)
    else:
        # Fallback to generic examples
        return _format_question_with_examples(element, symptom, _get_generic_examples(element, chief_complaint))

def _get_example_answers_for_element(element: str, top_conditions: List[str], chief_complaint: str) -> List[str]:
    """Extract 1-2 example answers from top conditions for a given OLD CARTS element."""
    examples = []
    seen = set()
    
    for condition in top_conditions:
        if condition in ANSWER_TEMPLATES:
            templates = ANSWER_TEMPLATES[condition]
            if element in templates:
                # Get first answer from this condition
                answers = templates[element]
                if answers:
                    example = answers[0]
                    # Clean up example for display (remove "I have" prefixes, etc.)
                    example_clean = _clean_answer_for_example(example)
                    if example_clean not in seen and len(examples) < 2:
                        examples.append(example_clean)
                        seen.add(example_clean)
    
    return examples[:2]  # Return max 2 examples

def _clean_answer_for_example(answer: str) -> str:
    """Clean answer text for use as example in question."""
    # Remove common prefixes and make concise
    answer = answer.lower()
    answer = answer.replace("started ", "")
    answer = answer.replace("came on ", "")
    answer = answer.replace("worse when i ", "")
    answer = answer.replace("worse after ", "")
    answer = answer.replace("worse with ", "")
    answer = answer.replace("worse ", "")
    answer = answer.replace("better ", "")
    answer = answer.replace("on my ", "")
    answer = answer.replace("in my ", "")
    answer = answer.replace("the ", "")
    answer = answer.replace("i ", "")
    answer = answer.replace("my ", "")
    # Remove "helps" at end for alleviating examples
    if answer.endswith(" helps"):
        answer = answer[:-6]
    # Capitalize first letter
    if answer:
        answer = answer[0].upper() + answer[1:]
    return answer

def _format_question_with_examples(element: str, symptom: str, examples: str) -> str:
    """Format question with example answers."""
    questions = {
        "O": f"When did {symptom} start? For example, {examples}?",
        "L": f"Where exactly is {symptom} located? For example, {examples}?",
        "D": f"How long has {symptom} been present? For example, {examples}?",
        "C": f"What does {symptom} feel like? For example, {examples}?",
        "A_aggravating": f"What makes {symptom} worse? For example, {examples}?",
        "A_alleviating": f"What makes {symptom} better? For example, {examples}?",
        "R": f"Does {symptom} spread anywhere else? For example, {examples}?",
        "T": f"Is {symptom} constant or does it come and go? For example, {examples}?",
        "S": f"On a scale of 1 to 10, with 10 being the worst imaginable, how severe is {symptom}? For example, {examples}?"
    }
    return questions.get(element, f"Can you tell me more about {symptom}? For example, {examples}?")

def _get_generic_examples(element: str, chief_complaint: str) -> str:
    """Get generic examples if condition-specific ones aren't available."""
    complaint_lower = chief_complaint.lower()
    
    generic = {
        "O": "suddenly, gradually, or after eating",
        "D": "hours, days, or weeks",
        "C": "sharp, dull, burning, or pressure",
        "A_aggravating": "movement, eating, or breathing",
        "A_alleviating": "rest, medication, or position changes",
        "R": "to your arm, jaw, or back",
        "T": "constant or intermittent",
        "S": "mild (1-3), moderate (4-6), or severe (7-10)"
    }
    
    if element == "L":
        if "chest" in complaint_lower:
            return "center of chest, behind breastbone, or left/right side"
        elif "abdominal" in complaint_lower or "belly" in complaint_lower:
            return "upper abdomen, lower abdomen, right side, or left side"
        elif "flank" in complaint_lower:
            return "right side, left side, or wraps around"
        else:
            return "one side, both sides, or specific area"
    
    return generic.get(element, "can you describe it")

def get_patient_answer(diagnosis: str, element: str, initial: bool = False, clarified: bool = False) -> str:
    """Get diagnosis-specific patient answer."""
    templates = ANSWER_TEMPLATES.get(diagnosis, {})
    
    if element == "L" and clarified and f"{element}_clarified" in templates:
        return random.choice(templates[f"{element}_clarified"])
    elif element in templates:
        return random.choice(templates[element])
    
    # Fallback
    fallbacks = {
        "O": ["started suddenly", "came on gradually"],
        "L": ["in my side", "in my abdomen"],
        "D": ["for a few hours", "since this morning"],
        "C": ["sharp", "dull"],
        "A_aggravating": ["movement", "nothing specific"],
        "A_alleviating": ["rest", "nothing"],
        "R": ["no", "yes"],
        "T": ["constant", "comes and goes"],
        "S": ["about 5", "moderate"]
    }
    return random.choice(fallbacks.get(element, ["I'm not sure"]))

def calculate_score_deltas(element: str, answer: str, chief_complaint: str, differential: List[str], target: str) -> Dict[str, float]:
    """Calculate score deltas for each condition based on answer."""
    deltas = {}
    answer_lower = answer.lower()
    complaint_lower = chief_complaint.lower()
    
    # Initialize with small deltas for non-target conditions
    for condition in differential:
        if condition == target:
            deltas[condition] = 0.10  # Boost target
        else:
            deltas[condition] = -0.02  # Slight decrease for others (distributed)
    
    # Pattern-based scoring
    if "chest pain" in complaint_lower:
        if element == "C" and "burning" in answer_lower:
            deltas.update(SCORING_DELTAS.get("chest_pain_burning", {}))
        elif element == "C" and ("pressure" in answer_lower or "heaviness" in answer_lower):
            deltas.update(SCORING_DELTAS.get("chest_pain_pressure", {}))
        elif element == "A_aggravating" and ("lie" in answer_lower or "lay" in answer_lower):
            deltas.update(SCORING_DELTAS.get("chest_pain_worse_lying_down", {}))
        elif element == "A_aggravating" and "exertion" in answer_lower:
            deltas.update(SCORING_DELTAS.get("chest_pain_worse_exertion", {}))
    
    if "abdominal pain" in complaint_lower:
        if element == "L":
            if "upper" in answer_lower or "ribs" in answer_lower:
                deltas.update(SCORING_DELTAS.get("abdominal_pain_RUQ", {}))
            elif "lower" in answer_lower or "hip" in answer_lower or "groin" in answer_lower:
                deltas.update(SCORING_DELTAS.get("abdominal_pain_RLQ", {}))
    
    return deltas

def generate_clinical_reasoning_with_ranking(
    element: str, answer: str, chief_complaint: str, target: str,
    deltas: Dict[str, float], ranked_conditions: List[Tuple[str, float]],
    current_scores: Dict[str, float]
) -> str:
    """Generate clinical reasoning with updated rankings."""
    element_names = {
        "O": "Onset (O)", "L": "Location (L)", "D": "Duration (D)",
        "C": "Character (C)", "A_aggravating": "Aggravating factors (A)",
        "A_alleviating": "Alleviating factors (A)", "R": "Radiation (R)",
        "T": "Timing (T)", "S": "Severity (S)"
    }
    
    element_name = element_names.get(element, element)
    
    # Build ranking text
    ranking_text = "CURRENT DIFFERENTIAL DIAGNOSIS (ranked by probability):\n"
    for i, (condition, score) in enumerate(ranked_conditions[:5], 1):
        change = ""
        if condition in deltas:
            delta_pct = deltas[condition] * 100
            if delta_pct > 0:
                change = f" (+{delta_pct:.1f}%)"
            elif delta_pct < 0:
                change = f" ({delta_pct:.1f}%)"
        ranking_text += f"{i}. {condition}: {score:.1f}% probability{change}\n"
    
    reasoning = f"""CLINICAL REASONING: This is the {element_name} element of OLD CARTS.

ELEMENT IDENTIFICATION: The patient's answer '{answer}' is being evaluated for the {element_name} element.
CRITICAL: This is {element_name}, NOT any other OLD CARTS element.

ANSWER SUFFICIENCY: {'Answer is specific and sufficient' if element != 'L' or 'upper' in answer.lower() or 'lower' in answer.lower() else 'Answer required clarification to be specific enough'}

Patient reported '{answer}' for {element_name.upper()}.

SCORING ANALYSIS:
"""
    
    # Add key changes
    for condition in ranked_conditions[:3]:
        cond_name, score = condition
        if cond_name in deltas:
            delta = deltas[cond_name]
            if abs(delta) > 0.1:
                if delta > 0:
                    reasoning += f"• {cond_name}: {element_name} '{answer}' STRONGLY SUPPORTS this diagnosis. Likelihood increased.\n"
                else:
                    reasoning += f"• {cond_name}: {element_name} '{answer}' is LESS CHARACTERISTIC. Likelihood decreased.\n"
    
    reasoning += f"\n{ranking_text}\n"
    reasoning += f"NEXT STEP: Continue OLD CARTS assessment to further narrow differential. {ranked_conditions[0][0]} currently has highest probability."
    
    return reasoning

def generate_associated_symptom_questions(
    top_conditions: List[str],
    chief_complaint: str,
    target_diagnosis: str,
    condition_scores: Dict[str, float],
    differential: List[str]
) -> List[Dict]:
    """Generate associated symptom questions based on top 3 conditions to help differentiate."""
    messages = []
    
    # Collect unique differentiating questions from top 3 conditions
    questions_to_ask = []
    seen_questions = set()
    
    for condition in top_conditions:
        if condition in ASSOCIATED_SYMPTOMS:
            symptom_data = ASSOCIATED_SYMPTOMS[condition]
            for q_data in symptom_data["questions"]:
                question_text = q_data["question"]
                if question_text not in seen_questions:
                    questions_to_ask.append({
                        "question": question_text,
                        "condition": condition,
                        "positive_answer": q_data["positive_answer"],
                        "negative_answer": q_data["negative_answer"],
                        "score_delta": q_data["score_delta"]
                    })
                    seen_questions.add(question_text)
                    # Limit to 2-3 questions total to keep conversation focused
                    if len(questions_to_ask) >= 3:
                        break
        if len(questions_to_ask) >= 3:
            break
    
    # Ask questions and get answers
    for q_data in questions_to_ask:
        condition = q_data["condition"]
        question = q_data["question"]
        
        # Determine if target diagnosis should have positive answer
        is_target = (condition == target_diagnosis)
        patient_answer = q_data["positive_answer"] if is_target else q_data["negative_answer"]
        
        # Add question
        messages.append({
            "role": "assistant",
            "content": question
        })
        
        # Add patient answer
        messages.append({
            "role": "user",
            "content": patient_answer
        })
        
        # Calculate score deltas
        deltas = {}
        if is_target:
            # Positive answer for target condition - increase its score
            deltas[target_diagnosis] = q_data["score_delta"]
            # Decrease other top conditions slightly
            for other_condition in top_conditions:
                if other_condition != target_diagnosis:
                    deltas[other_condition] = -q_data["score_delta"] * 0.5
        else:
            # Negative answer for non-target condition - decrease its score
            deltas[condition] = -q_data["score_delta"]
            # Slight increase for target if it doesn't have this symptom
            if target_diagnosis in top_conditions:
                deltas[target_diagnosis] = q_data["score_delta"] * 0.3
        
        # Update condition scores
        for cond, delta in deltas.items():
            if cond in condition_scores:
                change = delta * 20  # Scale down deltas
                condition_scores[cond] = max(1.0, min(95.0, condition_scores[cond] + change))
        
        # Normalize scores
        total = sum(condition_scores.values())
        if total > 0:
            condition_scores.update({k: (v / total) * 100 for k, v in condition_scores.items()})
            condition_scores = {k: max(1.0, min(95.0, v)) for k, v in condition_scores.items()}
            total_adj = sum(condition_scores.values())
            if total_adj > 0:
                condition_scores.update({k: (v / total_adj) * 100 for k, v in condition_scores.items()})
        
        # Generate clinical reasoning
        ranked_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
        
        reasoning = f"""CLINICAL REASONING: Associated Symptom Assessment

The patient reported '{patient_answer}' regarding {question.lower()}

This associated symptom pattern helps differentiate between the top differential diagnoses:
"""
        
        # Add scoring changes
        for cond, delta in deltas.items():
            if abs(delta) > 0.05:
                if delta > 0:
                    reasoning += f"• {cond}: Presence of this symptom SUPPORTS the diagnosis. Likelihood increased.\n"
                else:
                    reasoning += f"• {cond}: Absence of this symptom is LESS CHARACTERISTIC. Likelihood decreased.\n"
        
        reasoning += f"\nUPDATED DIFFERENTIAL DIAGNOSIS (ranked by probability):\n"
        for i, (cond, score) in enumerate(ranked_conditions[:5], 1):
            reasoning += f"{i}. {cond}: {score:.1f}% probability\n"
        
        reasoning += f"\nNEXT STEP: Associated symptoms further narrow the differential. {ranked_conditions[0][0]} currently has highest probability."
        
        messages.append({
            "role": "assistant",
            "content": reasoning
        })
    
    return messages

def generate_final_reasoning(ranked_conditions: List[Tuple[str, float]], target: str) -> str:
    """Generate final diagnostic reasoning."""
    reasoning = "FINAL DIAGNOSTIC REASONING:\n\n"
    reasoning += "Based on complete OLD CARTS assessment, the differential diagnosis has been narrowed through progressive scoring:\n\n"
    reasoning += "RANKED DIFFERENTIAL DIAGNOSIS:\n"
    for i, (condition, score) in enumerate(ranked_conditions, 1):
        reasoning += f"{i}. {condition}: {score:.1f}% probability\n"
    
    top_diagnosis = ranked_conditions[0][0]
    reasoning += f"\nCONCLUSION: The clinical presentation most strongly supports {top_diagnosis}.\n"
    reasoning += "This conclusion is based on systematic collection and analysis of OLD CARTS elements, with each answer progressively updating condition probabilities and rankings.\n"
    reasoning += "Each OLD CARTS element provided key information that increased or decreased the likelihood of specific conditions, resulting in the final ranked differential diagnosis."
    
    return reasoning

# ============================================================================
# Main Generation
# ============================================================================

def generate_complex_dataset(output_file: str, conversations_per_case: int = 6):
    """Generate complex dataset with cross-organ differentiation."""
    print("=" * 80)
    print("Generating Complex Medical Dataset")
    print("=" * 80)
    print(f"Conversations per case: {conversations_per_case} (equal American/British variants)")
    print()
    
    all_conversations = []
    
    for case in COMPLEX_CASES:
        print(f"📋 Processing: {case['title']}")
        print(f"   Chief complaint: {case['chief_complaint']}")
        print(f"   Target: {case['target_diagnosis']}")
        print(f"   Differential: {len(case['differential'])} conditions")
        
        # Generate equal number of American and British variants
        conversations_per_variant = conversations_per_case // 2
        
        for i in range(conversations_per_variant):
            # American variant
            conv_american = generate_complex_conversation(case, variant="american")
            all_conversations.append(conv_american)
            
            # British variant (for every American conversation)
            conv_british = generate_complex_conversation(case, variant="british")
            all_conversations.append(conv_british)
        
        print(f"   Generated: {conversations_per_variant} American + {conversations_per_variant} British = {conversations_per_variant * 2} total")
        print()
    
    print(f"✅ Generated {len(all_conversations)} conversations\n")
    
    # Save
    print(f"💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved dataset to {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    conversations_per = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    generate_complex_dataset(
        "medical_sft_dataset_complex.json",
        conversations_per_case=conversations_per
    )

