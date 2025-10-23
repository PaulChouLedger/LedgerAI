# System Message Fix - Single Question Enforcement

## 🐛 **Issue Identified:**

The system was asking **multiple questions** in a single response, violating the updated `system_msg` that explicitly states:
- "Output EXACTLY ONE question only"
- "NEVER combine multiple questions"
- "No one prompt should include multiple questions"

### **Example of the Problem:**
```
Patient: "i have abdominal pain"
System Response: "I'm so sorry to hear that you're experiencing abdominal pain. Can I ask, how long have you been feeling this way? How old are you?"
```

**❌ This violates the single question rule by asking TWO questions:**
1. "How long have you been feeling this way?"
2. "How old are you?"

## ✅ **Fixes Applied:**

### **1. Fixed Syntax Error in System Message:**
```python
# OLD (syntax error):
system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Use PLAIN LANGUAGE (no medical jargon). Do not include medical terminology from guidelines. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling'). Do NOT ask about duration/time - that will be covered later." "No one prompt should include multiple questions, in other words do no include muultiple phrases ending with a question mark"

# NEW (fixed):
system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Use PLAIN LANGUAGE (no medical jargon). Do not include medical terminology from guidelines. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling'). Do NOT ask about duration/time - that will be covered later. No one prompt should include multiple questions, in other words do not include multiple phrases ending with a question mark."
```

### **2. Updated All System Messages for Consistency:**

#### **OLDCARTS Question Generation:**
```python
# OLD:
system_msg = "You are a medical assistant. Output ONLY ONE question. Use PLAIN LANGUAGE (no medical jargon). Never combine multiple questions."

# NEW:
system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Use PLAIN LANGUAGE (no medical jargon). Do not include multiple phrases ending with question marks."
```

#### **Chronicity Question Generation:**
```python
# OLD:
system_msg = "You are a medical assistant. Output ONLY the question requested, nothing else. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling')."

# NEW:
system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions. Do NOT ask questions requiring visual inspection (no 'point to', 'show me', 'look at', 'appearance', 'color', 'swelling')."
```

#### **Age Question Generation:**
```python
# OLD:
system_msg = "You are a medical assistant. Output ONLY the question requested, nothing else."

# NEW:
system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions."
```

#### **Sex Question Generation:**
```python
# OLD:
system_msg = "You are a medical assistant. Output ONLY the question requested, nothing else."

# NEW:
system_msg = "You are a medical assistant. CRITICAL: Output EXACTLY ONE question only. NEVER combine multiple questions."
```

## 🎯 **Expected Behavior After Fix:**

### **✅ Correct Single Question Response:**
```
Patient: "i have abdominal pain"
System Response: "I'm sorry to hear you're experiencing abdominal pain. Can you tell me where exactly the pain is located?"
```

### **✅ Follow-up Questions:**
```
Patient: "in my lower left side"
System Response: "How would you describe the pain - is it sharp, dull, or cramping?"
```

## 📊 **Benefits of the Fix:**

### **✅ Consistent Single Question Format:**
- **All system messages** now enforce single questions
- **Clear instructions** to avoid multiple questions
- **Explicit warnings** about question mark usage
- **Consistent behavior** across all question types

### **✅ Better User Experience:**
- **One question at a time** - easier to answer
- **Focused responses** - better data quality
- **Clear progression** through OLDCARTS
- **Reduced confusion** for patients

### **✅ Improved Data Quality:**
- **Single answers** per question
- **Better similarity scoring** with focused responses
- **More accurate ML training** data
- **Cleaner diagnostic flow**

## 🚀 **System Status:**

### **✅ All System Messages Updated:**
- **OLDCARTS questions** - Single question enforcement
- **Age questions** - Single question enforcement
- **Sex questions** - Single question enforcement
- **Chronicity questions** - Single question enforcement
- **Associated symptoms** - Single question enforcement

### **✅ Consistent Behavior:**
- **All question types** now follow single question rule
- **Clear instructions** to avoid multiple questions
- **Explicit warnings** about question mark usage
- **Better user experience** with focused questions

**The system now enforces single questions across all question types, providing a better user experience and improved data quality!** 🏥✅
