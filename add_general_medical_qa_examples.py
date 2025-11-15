#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Training Examples That Teach General Medical Q&A

The LLM needs to be able to:
1. Answer general medical information questions ("what is diabetes?")
2. Distinguish between informational questions vs symptom complaints
3. Still maintain ability to conduct OLD CARTS assessments when symptoms are mentioned

This script adds examples that show the model can handle both:
- General medical information requests
- Symptom-based assessments (OLD CARTS)
"""

import json
import os

def create_general_qa_examples():
    """Create examples that teach general medical Q&A while maintaining assessment capability."""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional with extensive medical knowledge. "
                        "You can answer general medical questions and provide information. "
                        "You can also conduct clinical assessments when patients mention symptoms.\n\n"
                        "IMPORTANT: Distinguish between:\n"
                        "- Informational questions (e.g., 'what is diabetes?') → Provide information\n"
                        "- Symptom complaints (e.g., 'I have chest pain') → Conduct OLD CARTS assessment"
                    )
                },
                {
                    "role": "user",
                    "content": "what is diabetes?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Diabetes is a chronic medical condition where the body has trouble regulating blood sugar (glucose). "
                        "There are two main types:\n\n"
                        "**Type 1 Diabetes:** The body doesn't produce insulin, a hormone needed to move glucose from the blood into cells. "
                        "It usually develops in children and young adults and requires daily insulin injections.\n\n"
                        "**Type 2 Diabetes:** The body doesn't use insulin properly (insulin resistance) or doesn't make enough. "
                        "It's more common in adults and can often be managed with diet, exercise, and medications.\n\n"
                        "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision. "
                        "If you're experiencing any of these symptoms, I can help you assess them."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can answer medical questions and conduct assessments. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "can you explain what a heart attack is?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "A heart attack, also called a myocardial infarction (MI), occurs when blood flow to part of the heart muscle is blocked, "
                        "usually by a blood clot. This causes damage to the heart muscle.\n\n"
                        "**Common symptoms include:**\n"
                        "- Chest pain or pressure (often described as 'heavy' or 'squeezing')\n"
                        "- Pain that may radiate to the left arm, jaw, or back\n"
                        "- Shortness of breath\n"
                        "- Nausea, sweating, or dizziness\n\n"
                        "**Risk factors:** High blood pressure, high cholesterol, smoking, diabetes, family history, and age.\n\n"
                        "If you're experiencing chest pain or any of these symptoms, I can help assess your situation right away."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer medical questions naturally. "
                        "If the user mentions symptoms, conduct an assessment."
                    )
                },
                {
                    "role": "user",
                    "content": "what causes high blood pressure?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "High blood pressure (hypertension) can have several causes:\n\n"
                        "**Primary (Essential) Hypertension:** Most common type, develops gradually over years. "
                        "Often no single identifiable cause, but risk factors include:\n"
                        "- Age (risk increases with age)\n"
                        "- Family history\n"
                        "- Being overweight or obese\n"
                        "- Lack of physical activity\n"
                        "- High salt intake\n"
                        "- Excessive alcohol consumption\n"
                        "- Stress\n\n"
                        "**Secondary Hypertension:** Caused by an underlying condition like kidney disease, "
                        "thyroid problems, or certain medications.\n\n"
                        "If you're concerned about your blood pressure or experiencing related symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can provide medical information. "
                        "You can also conduct clinical assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "tell me about asthma"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Asthma is a chronic lung condition where the airways become inflamed and narrowed, making breathing difficult.\n\n"
                        "**Common symptoms:**\n"
                        "- Wheezing (whistling sound when breathing)\n"
                        "- Shortness of breath\n"
                        "- Chest tightness\n"
                        "- Coughing (often worse at night or early morning)\n\n"
                        "**Triggers can include:**\n"
                        "- Allergens (pollen, dust, pet dander)\n"
                        "- Exercise\n"
                        "- Cold air\n"
                        "- Respiratory infections\n"
                        "- Stress\n\n"
                        "**Treatment:** Usually involves inhalers (rescue and controller medications) and avoiding triggers.\n\n"
                        "If you're experiencing breathing difficulties or asthma symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "what's the difference between type 1 and type 2 diabetes?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Here are the key differences between Type 1 and Type 2 diabetes:\n\n"
                        "**Type 1 Diabetes:**\n"
                        "- Autoimmune condition where the body attacks insulin-producing cells\n"
                        "- Usually develops in children/young adults\n"
                        "- Always requires insulin therapy\n"
                        "- Onset is usually sudden\n"
                        "- Not preventable\n\n"
                        "**Type 2 Diabetes:**\n"
                        "- Body becomes resistant to insulin or doesn't produce enough\n"
                        "- Usually develops in adults (though increasingly in younger people)\n"
                        "- Can often be managed with diet, exercise, and oral medications\n"
                        "- May eventually need insulin\n"
                        "- Onset is gradual\n"
                        "- Often preventable with lifestyle changes\n\n"
                        "Both types require careful blood sugar management. If you're experiencing symptoms like increased thirst, "
                        "frequent urination, or fatigue, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can answer medical questions. "
                        "You can also conduct assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what is GERD?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "GERD stands for Gastroesophageal Reflux Disease. It's a condition where stomach acid frequently flows back "
                        "into the tube connecting your mouth and stomach (esophagus).\n\n"
                        "**Common symptoms:**\n"
                        "- Heartburn (burning sensation in chest)\n"
                        "- Regurgitation (acid or food coming back up)\n"
                        "- Chest pain\n"
                        "- Difficulty swallowing\n"
                        "- Chronic cough\n\n"
                        "**Common triggers:**\n"
                        "- Lying down after eating\n"
                        "- Large meals\n"
                        "- Fatty or spicy foods\n"
                        "- Certain medications\n\n"
                        "**Treatment:** Lifestyle changes (elevating head of bed, avoiding triggers), antacids, and medications like PPIs.\n\n"
                        "If you're experiencing chest pain or other GERD symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "If symptoms are mentioned, conduct an assessment."
                    )
                },
                {
                    "role": "user",
                    "content": "can you explain what appendicitis is?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Appendicitis is inflammation of the appendix, a small pouch attached to the large intestine.\n\n"
                        "**Classic symptoms:**\n"
                        "- Abdominal pain that starts around the belly button and moves to the right lower quadrant\n"
                        "- Pain that worsens with movement\n"
                        "- Nausea and vomiting\n"
                        "- Loss of appetite\n"
                        "- Fever\n\n"
                        "**Important:** Appendicitis is a medical emergency that usually requires surgical removal of the appendix. "
                        "If left untreated, the appendix can rupture, which is dangerous.\n\n"
                        "If you're experiencing abdominal pain, especially in the right lower quadrant, I can help assess your symptoms right away."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can provide medical information. "
                        "You can also conduct clinical assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what causes kidney stones?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Kidney stones (nephrolithiasis) are hard deposits of minerals and salts that form in the kidneys.\n\n"
                        "**Common causes:**\n"
                        "- Dehydration (not drinking enough water)\n"
                        "- High levels of certain minerals in urine (calcium, oxalate, uric acid)\n"
                        "- Family history\n"
                        "- Certain medical conditions (gout, inflammatory bowel disease)\n"
                        "- Certain medications\n"
                        "- Diet high in salt, protein, or sugar\n\n"
                        "**Common symptoms:**\n"
                        "- Severe pain in the side and back (flank pain)\n"
                        "- Pain that radiates to the lower abdomen and groin\n"
                        "- Pain that comes in waves\n"
                        "- Blood in urine\n"
                        "- Nausea and vomiting\n\n"
                        "If you're experiencing severe flank pain or other symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "what is a UTI?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "UTI stands for Urinary Tract Infection. It's an infection in any part of the urinary system - "
                        "kidneys, bladder, ureters, or urethra. Most UTIs affect the bladder and urethra.\n\n"
                        "**Common symptoms:**\n"
                        "- Painful urination (dysuria)\n"
                        "- Frequent, urgent need to urinate\n"
                        "- Lower abdominal pain\n"
                        "- Cloudy or bloody urine\n"
                        "- Strong-smelling urine\n"
                        "- Pelvic pain (in women)\n\n"
                        "**Common causes:**\n"
                        "- Bacteria entering the urinary tract (most commonly E. coli)\n"
                        "- More common in women due to shorter urethra\n"
                        "- Sexual activity\n"
                        "- Certain birth control methods\n\n"
                        "**Treatment:** Usually antibiotics. Drinking plenty of water helps.\n\n"
                        "If you're experiencing symptoms like painful urination or lower abdominal pain, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can answer medical questions. "
                        "You can also conduct assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "explain what pneumonia is"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Pneumonia is an infection that inflames the air sacs in one or both lungs. "
                        "The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.\n\n"
                        "**Common symptoms:**\n"
                        "- Productive cough (with phlegm/mucus)\n"
                        "- Fever and chills\n"
                        "- Chest pain that worsens with breathing or coughing\n"
                        "- Shortness of breath\n"
                        "- Fatigue\n\n"
                        "**Causes:**\n"
                        "- Bacteria (most common in adults)\n"
                        "- Viruses (including flu and COVID-19)\n"
                        "- Fungi (less common)\n\n"
                        "**Risk factors:** Age (very young or elderly), weakened immune system, chronic lung disease, smoking.\n\n"
                        "**Treatment:** Depends on cause - antibiotics for bacterial, supportive care for viral.\n\n"
                        "If you're experiencing chest pain, productive cough, or fever, I can help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "If symptoms are mentioned, conduct an assessment."
                    )
                },
                {
                    "role": "user",
                    "content": "what is a migraine?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "A migraine is a type of headache disorder characterized by recurrent, moderate to severe headaches, "
                        "often accompanied by other symptoms.\n\n"
                        "**Common symptoms:**\n"
                        "- Throbbing or pulsing pain, usually on one side of the head\n"
                        "- Sensitivity to light (photophobia)\n"
                        "- Sensitivity to sound (phonophobia)\n"
                        "- Nausea and vomiting\n"
                        "- Visual disturbances (aura) in some cases\n\n"
                        "**Triggers can include:**\n"
                        "- Stress\n"
                        "- Hormonal changes\n"
                        "- Certain foods or drinks\n"
                        "- Sleep changes\n"
                        "- Weather changes\n\n"
                        "**Treatment:** Pain relievers, triptans, preventive medications, and lifestyle modifications.\n\n"
                        "If you're experiencing headaches with these characteristics, I can help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can provide medical information. "
                        "You can also conduct clinical assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what is cholecystitis?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Cholecystitis is inflammation of the gallbladder, usually caused by gallstones blocking the bile duct.\n\n"
                        "**Common symptoms:**\n"
                        "- Severe pain in the right upper quadrant of the abdomen\n"
                        "- Pain that may radiate to the right shoulder or back\n"
                        "- Pain that worsens after eating, especially fatty meals\n"
                        "- Nausea and vomiting\n"
                        "- Fever\n"
                        "- Tenderness in the right upper quadrant\n\n"
                        "**Causes:**\n"
                        "- Gallstones blocking the cystic duct (most common)\n"
                        "- Infection\n"
                        "- Tumor (rare)\n\n"
                        "**Treatment:** Usually requires hospitalization, antibiotics, and often surgical removal of the gallbladder (cholecystectomy).\n\n"
                        "If you're experiencing right upper quadrant pain, especially after fatty meals, I can help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "what is the difference between a heart attack and angina?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Both involve chest pain due to reduced blood flow to the heart, but there are important differences:\n\n"
                        "**Angina:**\n"
                        "- Temporary chest pain/discomfort due to reduced blood flow\n"
                        "- Usually triggered by exertion or stress\n"
                        "- Relieved by rest or nitroglycerin\n"
                        "- No permanent heart muscle damage\n"
                        "- Warning sign of coronary artery disease\n\n"
                        "**Heart Attack (Myocardial Infarction):**\n"
                        "- Complete or near-complete blockage of blood flow\n"
                        "- Causes permanent heart muscle damage\n"
                        "- Pain is usually more severe and prolonged\n"
                        "- Not relieved by rest\n"
                        "- Medical emergency requiring immediate treatment\n\n"
                        "**Key difference:** Angina is a warning sign; a heart attack is actual damage to the heart muscle.\n\n"
                        "If you're experiencing chest pain, I can help assess your symptoms right away."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can answer medical questions. "
                        "You can also conduct assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what causes peptic ulcers?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Peptic ulcers are open sores that develop on the inside lining of the stomach (gastric ulcers) "
                        "or the upper portion of the small intestine (duodenal ulcers).\n\n"
                        "**Common causes:**\n"
                        "- H. pylori bacterial infection (most common)\n"
                        "- Long-term use of NSAIDs (aspirin, ibuprofen, naproxen)\n"
                        "- Excessive stomach acid production\n"
                        "- Smoking and alcohol (can worsen ulcers)\n"
                        "- Stress (can worsen but doesn't usually cause)\n\n"
                        "**Common symptoms:**\n"
                        "- Burning stomach pain (often worse when stomach is empty)\n"
                        "- Nausea\n"
                        "- Vomiting\n"
                        "- Bloating\n"
                        "- Feeling of fullness\n\n"
                        "**Treatment:** Antibiotics for H. pylori, acid-reducing medications (PPIs, H2 blockers), and avoiding NSAIDs.\n\n"
                        "If you're experiencing epigastric pain, nausea, or other symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "If symptoms are mentioned, conduct an assessment."
                    )
                },
                {
                    "role": "user",
                    "content": "what is diverticulitis?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Diverticulitis is inflammation or infection of small pouches (diverticula) that can form in the wall of the colon.\n\n"
                        "**Common symptoms:**\n"
                        "- Abdominal pain, usually in the left lower quadrant\n"
                        "- Fever\n"
                        "- Nausea and vomiting\n"
                        "- Constipation or diarrhea\n"
                        "- Abdominal tenderness\n\n"
                        "**Causes:**\n"
                        "- Diverticula (pouches) form when weak spots in the colon wall bulge outward\n"
                        "- When these pouches become inflamed or infected, it's called diverticulitis\n"
                        "- Risk factors include age, low-fiber diet, obesity, and lack of exercise\n\n"
                        "**Treatment:** Mild cases may be treated with antibiotics and diet changes. Severe cases may require hospitalization or surgery.\n\n"
                        "If you're experiencing lower abdominal pain, especially on the left side, I can help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can provide medical information. "
                        "You can also conduct clinical assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what is pericarditis?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Pericarditis is inflammation of the pericardium, the thin sac-like membrane surrounding the heart.\n\n"
                        "**Common symptoms:**\n"
                        "- Sharp chest pain that worsens with breathing or lying down\n"
                        "- Pain that improves when sitting up and leaning forward\n"
                        "- Pain that may radiate to the left shoulder or back\n"
                        "- Fever\n"
                        "- Shortness of breath\n\n"
                        "**Causes:**\n"
                        "- Viral infections (most common)\n"
                        "- Bacterial infections\n"
                        "- Autoimmune conditions\n"
                        "- Heart attack\n"
                        "- Certain medications\n"
                        "- Trauma\n\n"
                        "**Treatment:** Usually involves anti-inflammatory medications (NSAIDs, colchicine) and treating the underlying cause.\n\n"
                        "If you're experiencing chest pain, especially pain that changes with position, I can help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "what is pulmonary embolism?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "A pulmonary embolism (PE) is a blockage in one of the pulmonary arteries in the lungs, "
                        "usually caused by a blood clot that travels from the legs (deep vein thrombosis).\n\n"
                        "**Common symptoms:**\n"
                        "- Sudden shortness of breath\n"
                        "- Chest pain that worsens with breathing (pleuritic)\n"
                        "- Rapid heart rate\n"
                        "- Cough (sometimes with blood)\n"
                        "- Lightheadedness or fainting\n"
                        "- Leg pain or swelling (if DVT is present)\n\n"
                        "**Risk factors:**\n"
                        "- Prolonged immobility\n"
                        "- Surgery\n"
                        "- Certain medical conditions (cancer, heart disease)\n"
                        "- Birth control pills\n"
                        "- Pregnancy\n"
                        "- Family history\n\n"
                        "**Important:** PE is a medical emergency. If you're experiencing sudden shortness of breath or chest pain, "
                        "especially with leg swelling, seek immediate medical attention. I can also help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can answer medical questions. "
                        "You can also conduct assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what is acute kidney injury?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Acute Kidney Injury (AKI), also called acute renal failure, is a sudden episode of kidney failure or damage "
                        "that happens within a few hours or days.\n\n"
                        "**Common symptoms:**\n"
                        "- Decreased urine output\n"
                        "- Fluid retention (swelling in legs, ankles, feet)\n"
                        "- Shortness of breath\n"
                        "- Fatigue\n"
                        "- Confusion\n"
                        "- Nausea\n"
                        "- Chest pain or pressure\n\n"
                        "**Common causes:**\n"
                        "- Reduced blood flow to kidneys (dehydration, blood loss, heart problems)\n"
                        "- Direct damage to kidneys (medications, toxins, infections)\n"
                        "- Blockage of urine flow (kidney stones, enlarged prostate)\n\n"
                        "**Treatment:** Depends on cause - may include fluids, medications, dialysis in severe cases.\n\n"
                        "If you're experiencing decreased urine output, swelling, or other symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "If symptoms are mentioned, conduct an assessment."
                    )
                },
                {
                    "role": "user",
                    "content": "what is glomerulonephritis?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Glomerulonephritis is inflammation of the glomeruli, the tiny filters in the kidneys that remove waste from blood.\n\n"
                        "**Common symptoms:**\n"
                        "- Blood in urine (hematuria)\n"
                        "- Foamy urine (proteinuria)\n"
                        "- High blood pressure\n"
                        "- Swelling (edema) in face, hands, feet, or abdomen\n"
                        "- Decreased urine output\n"
                        "- Fatigue\n\n"
                        "**Causes:**\n"
                        "- Infections (strep throat, endocarditis)\n"
                        "- Autoimmune diseases (lupus, vasculitis)\n"
                        "- Diabetes\n"
                        "- High blood pressure\n"
                        "- Certain medications\n\n"
                        "**Treatment:** Depends on cause - may include antibiotics, immunosuppressants, blood pressure medications, and dialysis in severe cases.\n\n"
                        "If you're experiencing blood in urine, swelling, or other symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can provide medical information. "
                        "You can also conduct clinical assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what is hypothyroidism?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormone.\n\n"
                        "**Common symptoms:**\n"
                        "- Fatigue and weakness\n"
                        "- Weight gain\n"
                        "- Cold intolerance\n"
                        "- Constipation\n"
                        "- Dry skin and hair\n"
                        "- Depression\n"
                        "- Slowed heart rate\n"
                        "- Muscle aches\n\n"
                        "**Common causes:**\n"
                        "- Hashimoto's disease (autoimmune condition)\n"
                        "- Thyroid surgery\n"
                        "- Radiation treatment\n"
                        "- Certain medications\n"
                        "- Iodine deficiency (rare in developed countries)\n\n"
                        "**Treatment:** Usually involves taking synthetic thyroid hormone (levothyroxine) daily.\n\n"
                        "If you're experiencing fatigue, weight gain, or other symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "what is hyperthyroidism?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Hyperthyroidism is a condition where the thyroid gland produces too much thyroid hormone.\n\n"
                        "**Common symptoms:**\n"
                        "- Rapid heart rate or palpitations\n"
                        "- Weight loss despite increased appetite\n"
                        "- Heat intolerance and excessive sweating\n"
                        "- Tremors or nervousness\n"
                        "- Difficulty sleeping\n"
                        "- Frequent bowel movements\n"
                        "- Muscle weakness\n"
                        "- Irregular menstrual periods (in women)\n\n"
                        "**Common causes:**\n"
                        "- Graves' disease (autoimmune condition, most common)\n"
                        "- Thyroid nodules\n"
                        "- Thyroiditis (inflammation)\n"
                        "- Excessive iodine intake\n\n"
                        "**Treatment:** Medications to reduce thyroid hormone, radioactive iodine, or surgery.\n\n"
                        "If you're experiencing rapid heart rate, weight loss, or other symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. You can answer medical questions. "
                        "You can also conduct assessments when symptoms are mentioned."
                    )
                },
                {
                    "role": "user",
                    "content": "what is DKA?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "DKA stands for Diabetic Ketoacidosis. It's a serious complication of diabetes that occurs when the body "
                        "produces high levels of blood acids called ketones.\n\n"
                        "**Common symptoms:**\n"
                        "- Excessive thirst and urination\n"
                        "- High blood sugar\n"
                        "- High ketone levels in urine\n"
                        "- Nausea and vomiting\n"
                        "- Abdominal pain\n"
                        "- Fruity-smelling breath\n"
                        "- Confusion or difficulty concentrating\n"
                        "- Rapid breathing\n"
                        "- Fatigue\n\n"
                        "**Causes:**\n"
                        "- Very high blood sugar and insufficient insulin\n"
                        "- Often triggered by illness, infection, or missed insulin doses\n"
                        "- More common in Type 1 diabetes, but can occur in Type 2\n\n"
                        "**Important:** DKA is a medical emergency requiring immediate treatment with insulin and fluids.\n\n"
                        "If you're experiencing these symptoms, especially if you have diabetes, seek immediate medical attention. "
                        "I can also help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "If symptoms are mentioned, conduct an assessment."
                    )
                },
                {
                    "role": "user",
                    "content": "explain ADHD"
                },
                {
                    "role": "assistant",
                    "content": (
                        "ADHD stands for Attention-Deficit/Hyperactivity Disorder. It's a neurodevelopmental disorder "
                        "characterized by persistent patterns of inattention, hyperactivity, and impulsivity.\n\n"
                        "**Three main types:**\n"
                        "- **Predominantly Inattentive:** Difficulty focusing, organizing tasks, following instructions\n"
                        "- **Predominantly Hyperactive-Impulsive:** Restlessness, fidgeting, interrupting, difficulty waiting\n"
                        "- **Combined:** Features of both types\n\n"
                        "**Common symptoms:**\n"
                        "- Inattention: Difficulty sustaining focus, easily distracted, forgetful\n"
                        "- Hyperactivity: Fidgeting, restlessness, difficulty sitting still\n"
                        "- Impulsivity: Acting without thinking, interrupting others, difficulty waiting turns\n\n"
                        "**Causes:**\n"
                        "- Genetics (runs in families)\n"
                        "- Brain structure and function differences\n"
                        "- Environmental factors (premature birth, exposure to toxins)\n\n"
                        "**Treatment:** Usually involves a combination of medication (stimulants or non-stimulants), "
                        "behavioral therapy, and educational support.\n\n"
                        "If you're concerned about ADHD symptoms in yourself or someone else, I can help assess the situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "If symptoms are mentioned, conduct an assessment."
                    )
                },
                {
                    "role": "user",
                    "content": "what is adrenal crisis?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Adrenal crisis, also called acute adrenal insufficiency, is a life-threatening condition that occurs when "
                        "the adrenal glands don't produce enough cortisol.\n\n"
                        "**Common symptoms:**\n"
                        "- Severe weakness and fatigue\n"
                        "- Low blood pressure\n"
                        "- Dizziness or fainting\n"
                        "- Nausea and vomiting\n"
                        "- Abdominal pain\n"
                        "- Confusion\n"
                        "- Rapid heart rate\n"
                        "- Darkening of the skin (in some cases)\n\n"
                        "**Causes:**\n"
                        "- Sudden stopping of steroid medications\n"
                        "- Severe stress or illness in someone with adrenal insufficiency\n"
                        "- Adrenal gland damage or disease\n"
                        "- Pituitary gland problems\n\n"
                        "**Important:** Adrenal crisis is a medical emergency requiring immediate treatment with hydrocortisone and fluids.\n\n"
                        "If you're experiencing severe weakness, low blood pressure, or other symptoms, especially if you have adrenal insufficiency "
                        "or take steroids, seek immediate medical attention. I can also help assess your symptoms."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "explain what depression is"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Depression is a mental health disorder characterized by persistent feelings of sadness, "
                        "loss of interest, and a range of emotional and physical symptoms.\n\n"
                        "**Common symptoms:**\n"
                        "- Persistent sadness or low mood\n"
                        "- Loss of interest in activities once enjoyed\n"
                        "- Changes in appetite or weight\n"
                        "- Sleep disturbances (insomnia or excessive sleep)\n"
                        "- Fatigue or loss of energy\n"
                        "- Difficulty concentrating\n"
                        "- Feelings of worthlessness or guilt\n"
                        "- Thoughts of death or suicide\n\n"
                        "**Types:**\n"
                        "- Major Depressive Disorder\n"
                        "- Persistent Depressive Disorder (Dysthymia)\n"
                        "- Seasonal Affective Disorder\n"
                        "- Postpartum Depression\n\n"
                        "**Treatment:** Usually involves therapy (CBT, psychotherapy), medications (antidepressants), "
                        "or a combination. Lifestyle changes and support groups can also help.\n\n"
                        "If you're experiencing symptoms of depression, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "what is anxiety?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Anxiety is a normal emotion, but anxiety disorders involve excessive worry or fear that interferes with daily life.\n\n"
                        "**Common types:**\n"
                        "- **Generalized Anxiety Disorder (GAD):** Persistent, excessive worry about various things\n"
                        "- **Panic Disorder:** Recurrent panic attacks with intense fear\n"
                        "- **Social Anxiety Disorder:** Fear of social situations\n"
                        "- **Specific Phobias:** Intense fear of specific objects or situations\n\n"
                        "**Common symptoms:**\n"
                        "- Excessive worry or fear\n"
                        "- Restlessness or feeling on edge\n"
                        "- Rapid heart rate\n"
                        "- Sweating or trembling\n"
                        "- Difficulty concentrating\n"
                        "- Sleep problems\n"
                        "- Avoidance of anxiety-provoking situations\n\n"
                        "**Treatment:** Therapy (CBT, exposure therapy), medications (antidepressants, anti-anxiety medications), "
                        "and lifestyle changes (stress management, exercise).\n\n"
                        "If you're experiencing anxiety symptoms, I can help assess your situation."
                    )
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional. Answer questions naturally. "
                        "Distinguish between information requests and symptom complaints."
                    )
                },
                {
                    "role": "user",
                    "content": "can you explain what bipolar disorder is?"
                },
                {
                    "role": "assistant",
                    "content": (
                        "Bipolar disorder is a mental health condition characterized by extreme mood swings that include "
                        "emotional highs (mania or hypomania) and lows (depression).\n\n"
                        "**Types:**\n"
                        "- **Bipolar I:** At least one manic episode (may or may not include depressive episodes)\n"
                        "- **Bipolar II:** At least one hypomanic episode and one major depressive episode\n"
                        "- **Cyclothymic Disorder:** Periods of hypomanic and depressive symptoms (less severe)\n\n"
                        "**Manic/Hypomanic symptoms:**\n"
                        "- Elevated mood, increased energy\n"
                        "- Reduced need for sleep\n"
                        "- Racing thoughts\n"
                        "- Impulsive behavior\n"
                        "- Grandiose beliefs\n\n"
                        "**Depressive symptoms:**\n"
                        "- Persistent sadness\n"
                        "- Loss of interest\n"
                        "- Fatigue\n"
                        "- Difficulty concentrating\n"
                        "- Thoughts of suicide\n\n"
                        "**Treatment:** Mood stabilizers, antipsychotics, antidepressants, and psychotherapy.\n\n"
                        "If you're experiencing mood swings or other symptoms, I can help assess your situation."
                    )
                }
            ]
        }
    ]
    
    return examples

def enhance_dataset_with_general_qa(input_path, output_path):
    """Add general medical Q&A examples to the dataset."""
    print(f"📚 Loading dataset: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} conversations")
    
    # Create new examples
    qa_examples = create_general_qa_examples()
    
    # Combine
    enhanced = dataset + qa_examples
    
    print(f"📝 Added {len(qa_examples)} general medical Q&A examples")
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
    enhance_dataset_with_general_qa(input_path, output_path)
    
    print("\n" + "="*80)
    print("✅ Dataset enhancement complete!")
    print("="*80)
    print(f"\n📋 Added examples that teach:")
    print(f"  ✅ General medical information questions (e.g., 'what is diabetes?')")
    print(f"  ✅ Ability to provide medical explanations")
    print(f"  ✅ Distinguishing between information requests vs symptom complaints")
    print(f"  ✅ Maintaining conversational ability while being able to conduct assessments")
    print(f"\n💡 These examples ensure the model can:")
    print(f"   - Answer general medical questions naturally")
    print(f"   - Still conduct OLD CARTS assessments when symptoms are mentioned")
    print(f"   - Maintain flexibility between information and assessment modes")
    print(f"\n📝 Next steps:")
    print(f"1. Review {output_path}")
    print(f"2. Retrain model - it should now handle both Q&A and assessments")
    print(f"3. Test with questions like 'what is diabetes?' and 'I have chest pain'")

