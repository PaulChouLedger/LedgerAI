# Intelligent Follow-Up Questions Documentation

## Overview

The enhanced dataset now includes **intelligent, diagnosis-specific follow-up questions** that go beyond the OLD CARTS framework. These questions leverage medical knowledge to ask contextually relevant questions based on:

- **Chief complaint**
- **Probable diagnosis**
- **Medical knowledge** (medications, risk factors, associated symptoms, red flags)

## Key Features

### 1. **Context-Aware Question Selection**

Questions are selected based on the most probable diagnosis identified during OLD CARTS assessment.

**Example:**
- **Diagnosis:** Essential Hypertension
- **Follow-up Questions:**
  - "Are you currently taking any medications for blood pressure?"
  - "Do you have a family history of high blood pressure or heart disease?"
  - "How much salt do you typically consume in your diet?"

### 2. **Medical Knowledge Integration**

Questions are categorized by medical relevance:

- **Medications:** Current medications, medication history
- **Risk Factors:** Smoking, family history, comorbidities
- **Lifestyle:** Diet, exercise, alcohol use
- **Associated Symptoms:** Related symptoms that help refine diagnosis
- **Red Flags:** Warning signs requiring urgent evaluation
- **Medical History:** Previous conditions, procedures
- **Epidemiology:** Exposure history, contacts

### 3. **Clinical Reasoning for Follow-Ups**

Each follow-up question includes clinical reasoning explaining:
- Why the question is medically relevant
- How the answer affects the assessment
- Clinical significance of the information

## Examples by Diagnosis

### Essential Hypertension

**Follow-Up Questions:**
1. **Medications:** "Are you currently taking any medications for blood pressure?"
   - Medical Relevance: Essential for assessing treatment status and medication adherence
   - Example Responses: "Yes, I take lisinopril daily", "No, I'm not on any blood pressure medications"

2. **Family History:** "Do you have a family history of high blood pressure or heart disease?"
   - Medical Relevance: Family history is a significant risk factor for essential hypertension
   - Example Responses: "Yes, my mother has high blood pressure", "No, not that I know of"

3. **Lifestyle:** "How much salt do you typically consume in your diet?"
   - Medical Relevance: High sodium intake is a modifiable risk factor for hypertension
   - Example Responses: "I add salt to most meals", "I try to limit salt but probably still eat too much"

4. **Risk Factors:** "Do you smoke or use tobacco products?"
   - Medical Relevance: Smoking is a major cardiovascular risk factor
   - Example Responses: "No, I don't smoke", "Yes, I smoke about a pack a day"

5. **Lifestyle:** "How often do you exercise?"
   - Medical Relevance: Physical activity helps control blood pressure
   - Example Responses: "I exercise 3-4 times a week", "I don't exercise regularly"

### Chest Pain (Cardiac)

**Follow-Up Questions:**
1. **Medical History:** "Do you have any history of heart disease, heart attack, or cardiac procedures?"
   - Medical Relevance: Previous cardiac history significantly increases risk of acute coronary syndrome
   - Example Responses: "Yes, I had a heart attack 2 years ago", "No, no history of heart problems"

2. **Medications:** "Are you taking any medications like aspirin, clopidogrel, or blood thinners?"
   - Medical Relevance: Antiplatelet and anticoagulant medications are relevant for cardiac risk and bleeding risk
   - Example Responses: "Yes, I take daily aspirin", "No, I'm not on any of those"

3. **Comorbidities:** "Do you have diabetes, high cholesterol, or high blood pressure?"
   - Medical Relevance: These are major cardiac risk factors
   - Example Responses: "Yes, I have diabetes and high blood pressure", "No, none of those"

4. **Associated Symptoms:** "Does the pain occur with exertion or physical activity?"
   - Medical Relevance: Exertional chest pain is classic for stable angina
   - Example Responses: "Yes, it happens when I walk or climb stairs", "No, it happens at rest"

5. **Associated Symptoms:** "Do you have any shortness of breath, nausea, or sweating with the chest pain?"
   - Medical Relevance: These are red flag symptoms for acute coronary syndrome
   - Example Responses: "Yes, I get short of breath", "No, just the chest pain"

### Acute Upper GI Bleed

**Follow-Up Questions:**
1. **Medications:** "Are you taking any medications like aspirin, ibuprofen, naproxen, or other NSAIDs?"
   - Medical Relevance: NSAIDs are a major cause of upper GI bleeding
   - Example Responses: "Yes, I take aspirin daily", "No, I'm not on any NSAIDs"

2. **Medications:** "Are you taking any blood thinners like warfarin, apixaban, or clopidogrel?"
   - Medical Relevance: Anticoagulants increase bleeding risk
   - Example Responses: "Yes, I take warfarin", "No, I'm not on blood thinners"

3. **Lifestyle:** "Do you drink alcohol, and if so, how much?"
   - Medical Relevance: Alcohol is a major risk factor for GI bleeding
   - Example Responses: "No, I don't drink", "I drink daily, several drinks"

4. **Medical History:** "Do you have a history of ulcers, gastritis, or stomach problems?"
   - Medical Relevance: Previous GI history increases risk
   - Example Responses: "Yes, I've had ulcers before", "No, no history of stomach problems"

5. **Associated Symptoms:** "Have you noticed any black, tarry stools?"
   - Medical Relevance: Melena suggests upper GI bleeding
   - Example Responses: "Yes, my stools are black and tarry", "No, my stools look normal"

### Gastroesophageal Reflux Disease (GERD)

**Follow-Up Questions:**
1. **Medications:** "Are you taking any medications for acid reflux or heartburn, such as omeprazole, pantoprazole, or ranitidine?"
   - Medical Relevance: Current treatment status and medication effectiveness
   - Example Responses: "Yes, I take omeprazole daily", "No, I'm not on any medications for it"

2. **Lifestyle:** "What types of foods or drinks trigger your symptoms?"
   - Medical Relevance: Dietary triggers are important for GERD management
   - Example Responses: "Spicy foods and coffee", "I haven't noticed specific triggers"

3. **Associated Symptoms:** "Do your symptoms worsen when you lie down or bend over?"
   - Medical Relevance: Postural worsening is classic for GERD
   - Example Responses: "Yes, much worse when I lie down", "No, it's not related to position"

4. **Red Flags:** "Do you have any difficulty swallowing or feeling like food gets stuck?"
   - Medical Relevance: Dysphagia is a red flag symptom requiring further evaluation
   - Example Responses: "No, no swallowing problems", "Yes, sometimes food feels stuck"

5. **Red Flags:** "Have you noticed any weight loss or loss of appetite?"
   - Medical Relevance: Unintentional weight loss is a red flag symptom
   - Example Responses: "No, my weight is stable", "Yes, I've lost weight without trying"

## Dataset Structure

### Conversation Metadata

Each conversation with intelligent follow-ups includes:

```json
{
  "has_intelligent_followups": true,
  "followup_diagnosis": "Essential Hypertension",
  "num_followups": 2
}
```

### Message Structure

Follow-up questions are inserted after the FINAL DIAGNOSTIC REASONING:

```json
{
  "role": "assistant",
  "content": "FINAL DIAGNOSTIC REASONING: ..."
},
{
  "role": "assistant",
  "content": "Are you currently taking any medications for blood pressure?"
},
{
  "role": "user",
  "content": "Yes, I take lisinopril daily"
},
{
  "role": "assistant",
  "content": "CLINICAL REASONING: This is a follow-up question (medications) based on the probable diagnosis (Essential Hypertension).\n\nMEDICAL RELEVANCE: Essential for assessing treatment status and medication adherence...\n\nCLINICAL SIGNIFICANCE: This information helps:\n- Refine the differential diagnosis\n- Assess risk factors and comorbidities\n- Guide treatment recommendations\n- Identify red flag symptoms requiring urgent evaluation..."
}
```

## Training Benefits

### 1. **Beyond OLD CARTS Framework**

The model learns to ask intelligent questions that go beyond the standard OLD CARTS framework, leveraging medical knowledge to gather relevant information.

### 2. **Context-Aware Questioning**

The model learns to ask different questions based on the probable diagnosis:
- Hypertension → Ask about medications, family history, lifestyle
- Chest Pain → Ask about cardiac risk factors, medications, associated symptoms
- GI Bleed → Ask about NSAIDs, anticoagulants, alcohol use

### 3. **Medical Knowledge Integration**

The model learns to:
- Ask about medications relevant to the condition
- Identify risk factors
- Screen for red flag symptoms
- Gather lifestyle and social history when relevant

### 4. **Clinical Reasoning for Follow-Ups**

The model learns to provide clinical reasoning for follow-up questions, explaining why each question is medically relevant.

## Usage in Training

The intelligent follow-up questions are automatically included in the training dataset. The model learns:

1. **When to ask follow-up questions:** After completing OLD CARTS assessment
2. **What questions to ask:** Based on the probable diagnosis
3. **How to reason about answers:** Clinical significance and impact on assessment

## Expected Model Behavior

### After Training

**Scenario: Patient with Hypertension**

```
Patient: "I have high blood pressure"
[OLD CARTS assessment completed]
Model: "Are you currently taking any medications for blood pressure?" ✅
Patient: "No, I'm not on any medications"
Model: "Do you have a family history of high blood pressure or heart disease?" ✅
Patient: "Yes, my mother has high blood pressure"
Model: [Provides clinical reasoning about family history]
```

**Scenario: Patient with Chest Pain**

```
Patient: "I have chest pain"
[OLD CARTS assessment completed]
Model: "Do you have any history of heart disease, heart attack, or cardiac procedures?" ✅
Patient: "No, no history"
Model: "Are you taking any medications like aspirin, clopidogrel, or blood thinners?" ✅
Patient: "Yes, I take daily aspirin"
Model: [Provides clinical reasoning about medications]
```

## Statistics

- **Total Conversations:** 726
- **Conversations with Intelligent Follow-ups:** 22+ (varies by run)
- **Follow-up Questions per Conversation:** 2 (configurable)
- **Diagnoses Covered:** 
  - Essential Hypertension
  - Chest Pain (Cardiac)
  - Palpitations
  - Gastroesophageal Reflux Disease
  - Acute Cholecystitis
  - Gastroenteritis
  - Acute Upper GI Bleed

## Future Enhancements

1. **Expand Diagnosis Coverage:** Add follow-up questions for more diagnoses
2. **Dynamic Question Generation:** Use LLM to generate questions on-the-fly
3. **Conditional Follow-ups:** Ask different questions based on previous answers
4. **Multi-level Follow-ups:** Chain follow-up questions based on responses

## Summary

The intelligent follow-up questions feature enhances the dataset by:

- ✅ Adding context-aware, diagnosis-specific questions
- ✅ Leveraging medical knowledge (medications, risk factors, etc.)
- ✅ Going beyond the OLD CARTS framework
- ✅ Providing clinical reasoning for follow-up questions
- ✅ Training the model to ask intelligent, relevant questions

This makes the model more clinically useful and better at gathering comprehensive patient information.

