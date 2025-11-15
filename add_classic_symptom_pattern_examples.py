#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Training Examples That Teach Classic Symptom Patterns

The LLM needs to learn classic medical symptom patterns:
- GERD: Burning chest pain, WORSE when laying down (especially after meals)
- Cardiac: Chest pain worse with exertion, NOT typically worse when laying down
- Pleuritic: Chest pain worse with breathing/coughing
- Biliary: RUQ pain, worse after fatty meals
- Appendicitis: RLQ pain, worse with movement

This script adds examples that explicitly teach these patterns through
conversations where the LLM evaluates conditions based on patient answers.
"""

import json
import os

def create_classic_pattern_examples():
    """Create examples that teach classic symptom patterns."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional conducting a clinical history. "
                        "You have extensive medical knowledge. Your job is to STRUCTURE the conversation "
                        "and REASON SYSTEMATICALLY to arrive at a diagnosis.\n\n"
                        "CRITICAL: Know classic symptom patterns:\n"
                        "- GERD: Burning chest pain, WORSE when laying down (especially after meals), better with antacids/sitting up\n"
                        "- Cardiac: Chest pain worse with exertion, better with rest, NOT typically worse when laying down\n"
                        "- Use your medical knowledge to recognize these patterns and score conditions accordingly."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "45"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "burning"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain worse?"
                },
                {
                    "role": "user",
                    "content": "laying down after meals"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain better?"
                },
                {
                    "role": "user",
                    "content": "sitting up"
                },
                {
                    "role": "assistant",
                    "content": "Based on your symptoms - burning chest pain that's worse when laying down after meals and better when sitting up - this is highly consistent with GERD (Gastroesophageal Reflux Disease). The pattern of symptoms worsening with recumbency, especially after meals, is classic for GERD."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You have medical knowledge. "
                        "Know that cardiac chest pain is typically worse with exertion and better with rest. "
                        "It is NOT typically worse when laying down. "
                        "Use this knowledge to evaluate conditions correctly."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "58"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "pressure, like something heavy on my chest"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain worse?"
                },
                {
                    "role": "user",
                    "content": "walking up stairs or any physical activity"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain better?"
                },
                {
                    "role": "user",
                    "content": "resting"
                },
                {
                    "role": "assistant",
                    "content": "Based on your symptoms - pressure-like chest pain that's worse with exertion and better with rest - this is concerning for cardiac causes like angina or myocardial infarction. The pattern of worsening with physical activity is classic for cardiac chest pain."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that GERD symptoms worsen when laying down, "
                        "especially after meals. This is a classic pattern. "
                        "Cardiac chest pain does NOT typically worsen when laying down."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "35"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "burning sensation"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain worse?"
                },
                {
                    "role": "user",
                    "content": "when I lay down, especially right after eating"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain better?"
                },
                {
                    "role": "user",
                    "content": "sitting up or taking antacids"
                },
                {
                    "role": "assistant",
                    "content": "Your symptoms - burning chest pain that worsens when laying down after meals and improves with sitting up or antacids - are highly characteristic of GERD. This pattern is classic for gastroesophageal reflux."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that right upper quadrant pain "
                        "that worsens after fatty meals is classic for biliary conditions "
                        "(gallbladder, cholecystitis). Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "42"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "right upper side, under my ribs"
                },
                {
                    "role": "assistant",
                    "content": "What makes the abdominal pain worse?"
                },
                {
                    "role": "user",
                    "content": "after eating fatty foods"
                },
                {
                    "role": "assistant",
                    "content": "Based on your symptoms - right upper quadrant pain that worsens after fatty meals - this is highly suggestive of biliary disease, such as cholecystitis or gallstones. The pattern of pain in the RUQ triggered by fatty meals is classic for gallbladder conditions."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that chest pain that worsens "
                        "with breathing or coughing is classic for pleuritic/pulmonary causes. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "28"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "sharp, stabbing"
                },
                {
                    "role": "assistant",
                    "content": "What makes the chest pain worse?"
                },
                {
                    "role": "user",
                    "content": "taking a deep breath or coughing"
                },
                {
                    "role": "assistant",
                    "content": "Based on your symptoms - sharp, stabbing chest pain that worsens with breathing or coughing - this is characteristic of pleuritic pain, which suggests pulmonary causes like pneumonia, pleurisy, or pulmonary embolism."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that right lower quadrant pain "
                        "that worsens with movement is classic for appendicitis. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "25"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "lower right side"
                },
                {
                    "role": "assistant",
                    "content": "What makes the abdominal pain worse?"
                },
                {
                    "role": "user",
                    "content": "moving around or jarring movements"
                },
                {
                    "role": "assistant",
                    "content": "Based on your symptoms - right lower quadrant pain that worsens with movement - this is highly suggestive of appendicitis. The pattern of RLQ pain exacerbated by movement is classic for appendicitis."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that chest pain that radiates "
                        "to the left arm or jaw is classic for cardiac causes, especially MI. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "62"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "heavy pressure"
                },
                {
                    "role": "assistant",
                    "content": "Does the chest pain spread to other areas?"
                },
                {
                    "role": "user",
                    "content": "yes, it goes down my left arm and up to my jaw"
                },
                {
                    "role": "assistant",
                    "content": "Chest pain with radiation to the left arm and jaw is highly concerning for acute myocardial infarction. This pattern of radiation is classic for cardiac ischemia and requires immediate evaluation."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that flank pain that radiates "
                        "to the groin is classic for nephrolithiasis (kidney stones). "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "40"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "in my side, around my back"
                },
                {
                    "role": "assistant",
                    "content": "Does the pain spread to other areas?"
                },
                {
                    "role": "user",
                    "content": "yes, it goes down to my groin"
                },
                {
                    "role": "assistant",
                    "content": "Flank pain that radiates to the groin is highly characteristic of nephrolithiasis (kidney stones). This pattern of pain is classic for ureteral obstruction from a stone."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that epigastric pain with "
                        "nausea and vomiting is classic for peptic ulcer disease or gastritis. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "50"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "upper middle, right in the center below my chest"
                },
                {
                    "role": "assistant",
                    "content": "Are there any other symptoms you're experiencing along with the abdominal pain?"
                },
                {
                    "role": "user",
                    "content": "yes, I've been nauseous and vomiting"
                },
                {
                    "role": "assistant",
                    "content": "Epigastric pain with nausea and vomiting is suggestive of peptic ulcer disease or gastritis. This combination of symptoms in the epigastric region is classic for upper GI conditions."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that lower abdominal pain with "
                        "dysuria (painful urination) and frequency is classic for UTI. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "30"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "lower abdomen"
                },
                {
                    "role": "assistant",
                    "content": "Are there any other symptoms you're experiencing along with the abdominal pain?"
                },
                {
                    "role": "user",
                    "content": "yes, it hurts when I urinate and I have to go frequently"
                },
                {
                    "role": "assistant",
                    "content": "Lower abdominal pain with dysuria and urinary frequency is highly characteristic of a urinary tract infection. This combination of symptoms is classic for UTI."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that productive cough with fever "
                        "and chest pain is classic for pneumonia. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "55"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "sharp, worse when I breathe"
                },
                {
                    "role": "assistant",
                    "content": "Are there any other symptoms you're experiencing along with the chest pain?"
                },
                {
                    "role": "user",
                    "content": "yes, I have a cough with yellow phlegm and I have a fever"
                },
                {
                    "role": "assistant",
                    "content": "Chest pain that worsens with breathing, combined with productive cough and fever, is highly suggestive of pneumonia. This combination of symptoms is classic for pulmonary infection."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that headache with photophobia "
                        "(light sensitivity) and phonophobia (sound sensitivity) is classic for migraine. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have a headache"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing a headache. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "ongoing, I've had these before"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "35"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "What does the headache feel like?"
                },
                {
                    "role": "user",
                    "content": "throbbing, on one side of my head"
                },
                {
                    "role": "assistant",
                    "content": "Are there any other symptoms you're experiencing along with the headache?"
                },
                {
                    "role": "user",
                    "content": "yes, light and sound make it worse"
                },
                {
                    "role": "assistant",
                    "content": "Unilateral throbbing headache with photophobia and phonophobia is highly characteristic of migraine. This combination of symptoms is classic for migraine headaches."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that chest pain with diaphoresis "
                        "(sweating) and nausea is highly concerning for acute myocardial infarction. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "60"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "pressure, like something heavy"
                },
                {
                    "role": "assistant",
                    "content": "Are there any other symptoms you're experiencing along with the chest pain?"
                },
                {
                    "role": "user",
                    "content": "yes, I'm sweating a lot and feeling nauseous"
                },
                {
                    "role": "assistant",
                    "content": "Chest pain with diaphoresis and nausea is highly concerning for acute myocardial infarction. This combination of symptoms, especially in the context of pressure-like chest pain, is classic for cardiac ischemia and requires immediate evaluation."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that lower abdominal pain with "
                        "diarrhea and nausea is classic for gastroenteritis. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "28"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Female"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "lower abdomen, crampy"
                },
                {
                    "role": "assistant",
                    "content": "Are there any other symptoms you're experiencing along with the abdominal pain?"
                },
                {
                    "role": "user",
                    "content": "yes, I have diarrhea and feel nauseous"
                },
                {
                    "role": "assistant",
                    "content": "Lower abdominal crampy pain with diarrhea and nausea is highly characteristic of gastroenteritis. This combination of symptoms is classic for infectious or inflammatory GI conditions."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that shortness of breath with "
                        "wheezing and chest tightness is classic for asthma exacerbation. "
                        "Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have chest pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing chest pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "25"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "What does the chest pain feel like?"
                },
                {
                    "role": "user",
                    "content": "tightness, like I can't get enough air"
                },
                {
                    "role": "assistant",
                    "content": "Are there any other symptoms you're experiencing along with the chest pain?"
                },
                {
                    "role": "user",
                    "content": "yes, I'm short of breath and I can hear wheezing when I breathe"
                },
                {
                    "role": "assistant",
                    "content": "Chest tightness with shortness of breath and wheezing is highly characteristic of asthma exacerbation. This combination of symptoms is classic for reactive airway disease."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Know that right lower quadrant pain "
                        "that starts around the belly button and moves to the right side is "
                        "classic for appendicitis. Use this knowledge to evaluate conditions."
                    )
                },
                {
                    "role": "user",
                    "content": "I have abdominal pain"
                },
                {
                    "role": "assistant",
                    "content": "I understand you're experiencing abdominal pain. I'm here to help."
                },
                {
                    "role": "assistant",
                    "content": "Is this a new issue that just started, or is this an ongoing problem you've had before with a prior diagnosis?"
                },
                {
                    "role": "user",
                    "content": "new"
                },
                {
                    "role": "assistant",
                    "content": "How old are you?"
                },
                {
                    "role": "user",
                    "content": "22"
                },
                {
                    "role": "assistant",
                    "content": "What is your biological sex?"
                },
                {
                    "role": "user",
                    "content": "Male"
                },
                {
                    "role": "assistant",
                    "content": "Where exactly is the abdominal pain located?"
                },
                {
                    "role": "user",
                    "content": "it started around my belly button but now it's in my lower right side"
                },
                {
                    "role": "assistant",
                    "content": "Abdominal pain that starts periumbilically and migrates to the right lower quadrant is highly characteristic of appendicitis. This pattern of pain migration is classic for appendicitis."
                }
            ]
        }
    ]
    
    return examples

def enhance_dataset_with_classic_patterns(input_path, output_path):
    """Add classic symptom pattern examples to the dataset."""
    print(f"📚 Loading dataset: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} conversations")
    
    # Create new examples
    pattern_examples = create_classic_pattern_examples()
    
    # Combine
    enhanced = dataset + pattern_examples
    
    print(f"📝 Added {len(pattern_examples)} classic symptom pattern examples")
    print(f"📊 Total conversations: {len(enhanced)}")
    
    # Save
    print(f"💾 Saving enhanced dataset: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced dataset saved!")
    return enhanced

if __name__ == "__main__":
    # Try enhanced first, then high_quality, then fallback
    input_paths = [
        "medical_sft_dataset_enhanced.json",
        "medical_sft_dataset_high_quality.json",
        "medical_sft_dataset.json"
    ]
    
    output_path = "medical_sft_dataset_enhanced.json"
    
    input_path = None
    for path in input_paths:
        if os.path.exists(path):
            input_path = path
            break
    
    if not input_path:
        print(f"❌ Error: No dataset file found. Tried: {input_paths}")
        exit(1)
    
    print(f"📂 Using input dataset: {input_path}")
    enhance_dataset_with_classic_patterns(input_path, output_path)
    
    print("\n" + "="*80)
    print("✅ Dataset enhancement complete!")
    print("="*80)
    print(f"\n📋 Added examples that teach:")
    print(f"  ✅ GERD: Burning chest pain, WORSE when laying down after meals")
    print(f"  ✅ Cardiac: Chest pain worse with exertion, NOT worse when laying down")
    print(f"  ✅ Pleuritic: Chest pain worse with breathing/coughing")
    print(f"  ✅ Biliary: RUQ pain, worse after fatty meals")
    print(f"  ✅ Appendicitis: RLQ pain, worse with movement")
    print(f"\n💡 These examples teach the LLM to recognize classic symptom patterns")
    print(f"   and score conditions correctly based on these patterns.")
    print(f"\n📝 Next steps:")
    print(f"1. Review {output_path}")
    print(f"2. Retrain model - it should now recognize classic symptom patterns")
    print(f"3. Test with cases like 'burning chest pain worse when laying down after meals'")

