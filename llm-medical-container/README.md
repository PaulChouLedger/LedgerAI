# LLM Medical Container - Unified ML System

## 🏥 **Organized Directory Structure**

### **📁 Core Application Files**
- **`adaptive_diagnostic_engine.py`** - Main diagnostic engine with unified ML system
- **`clinician_mode.py`** - Clinician interface and advanced features
- **`container_rest.py`** - REST API endpoints
- **`rag_client.py`** - RAG (Retrieval-Augmented Generation) client
- **`thinking_fillers.py`** - Audio filler system
- **ML System** - Handles all validation and similarity scoring

### **📁 ML System (`ml/`)**
- **`medical_rule_engine.py`** - Medical rule engine with ML predictions
- **`continuous_learning.py`** - Background learning system
- **`learning_data_collector.py`** - Learning data collection
- **`learning_tracker.py`** - Learning progress tracking
- **`performance_monitor.py`** - Performance monitoring
- **`performance_dashboard.py`** - Performance visualization
- **`user_feedback_interface.py`** - User feedback collection
- **`location_ml_trainer.py`** - ML model training
- **`location_ml_data_extractor.py`** - Data extraction for ML
- **`location_ml_data.csv`** - ML training data
- **`location_ml_model.pkl`** - Trained ML model

### **📁 Configuration (`config/`)**
- **`medical_rules.json`** - Medical rules for anatomical types
- **`medical_term_mappings.json`** - OLDCARTS term mappings
- **`config.env.example`** - Environment configuration template

### **📁 Medical Guidelines (`medical/`)**
- **`guidelines/`** - Organized by organ system:
  - **`CARDIO/`** - Cardiovascular conditions
  - **`DERM/`** - Dermatology conditions
  - **`GI/`** - Gastrointestinal conditions
  - **`GU/`** - Genitourinary conditions
  - **`GYN/`** - Gynecological conditions
  - **`MSK/`** - Musculoskeletal conditions
  - **`NEURO/`** - Neurological conditions
  - **`PULMONARY/`** - Pulmonary conditions
  - **`RENAL/`** - Renal conditions
- **`synonyms/`** - Medical term synonyms by organ system

### **📁 Tests (`tests/`)**
- **`debug_ml_system.py`** - ML system debugging
- **`test_monitoring.py`** - Monitoring system testing

### **📁 Scripts (`scripts/`)**
- **`ehr_toggle.sh`** - EHR integration toggle
- **`update_acronyms.py`** - Medical acronym updates

### **📁 Documentation (`docs/`)**
- **`UNIFIED_ML_SYSTEM.md`** - Unified ML system documentation
- **`ML_PROGRESS_TRACKING.md`** - ML progress tracking guide
- **`MONITORING_GUIDE.md`** - System monitoring guide
- **`QUICK_START.md`** - Quick start guide
- **`README_EHR_INTEGRATION.md`** - EHR integration guide

### **📁 Data (`data/`)**
- **`learning/`** - Learning system data
  - **`feedback.json`** - User feedback data
  - **`learning_export.json`** - Learning data export
  - **`performance.json`** - Performance metrics
  - **`predictions.json`** - ML predictions
  - **`user_feedback_export.json`** - User feedback export
- **`models/`** - ML model storage

## 🧠 **Unified ML System Features**

### **✅ All OLDCARTS Components Use ML:**
- **Location (L)** - ML-powered anatomical similarity
- **Onset (O)** - ML-powered temporal similarity
- **Duration (D)** - ML-powered time similarity
- **Character (C)** - ML-powered pain descriptor similarity
- **Aggravating (A)** - ML-powered trigger similarity
- **Relieving (R)** - ML-powered relief similarity
- **Timing (T)** - ML-powered temporal pattern similarity
- **Severity (S)** - ML-powered intensity similarity

### **✅ Comprehensive Learning System:**
- **Learning Data Collection** - From all OLDCARTS interactions
- **Performance Monitoring** - Across all components
- **Continuous Learning** - Background model updates
- **User Feedback** - Rating and improvement system

### **✅ Smart Medical Rules:**
- **Anatomical Type Classification** - Bilateral, midline, unilateral
- **Hardcoded Medical Rules** - Critical anatomical relationships
- **ML Predictions** - Learned patterns and relationships
- **Dynamic Thresholds** - Adaptive rule-out thresholds

## 🚀 **Quick Start**

### **1. Environment Setup:**
```bash
# Copy configuration template
cp config/config.env.example .env

# Install dependencies
pip install -r requirements.txt
```

### **2. Run Diagnostic System:**
```bash
# Start the diagnostic engine
python adaptive_diagnostic_engine.py

# Start with clinician mode
python clinician_mode.py
```

### **3. Monitor ML System:**
```bash
# Test ML system
python tests/debug_ml_system.py

# Monitor learning progress
python tests/test_monitoring.py
```

## 📊 **System Architecture**

### **Core Components:**
1. **Diagnostic Engine** - Main diagnostic logic
2. **ML System** - Unified ML across all OLDCARTS
3. **Learning System** - Continuous improvement
4. **Performance Monitoring** - System optimization
5. **Medical Guidelines** - Comprehensive condition database

### **Data Flow:**
1. **Patient Input** → **Text Normalization** → **ML Processing** → **Similarity Scoring** → **Condition Ranking** → **Diagnosis**

### **Learning Flow:**
1. **Patient Interactions** → **Learning Data Collection** → **Performance Monitoring** → **Continuous Learning** → **Model Updates**

## 🎯 **Key Benefits**

### **✅ Unified ML System:**
- **Consistent ML processing** across all OLDCARTS components
- **Intelligent similarity scoring** for all elements
- **Comprehensive learning** from all interactions
- **Adaptive performance** with continuous improvement

### **✅ Organized Structure:**
- **Clean separation** of concerns
- **Easy maintenance** and updates
- **Scalable architecture** for growth
- **Clear documentation** and guides

### **✅ Production Ready:**
- **Comprehensive testing** and debugging
- **Performance monitoring** and optimization
- **User feedback** and improvement system
- **Continuous learning** and adaptation

## 🏥 **Medical System Status**

- **144 Guidelines** - Comprehensive condition coverage
- **8 Organ Systems** - Complete medical coverage
- **Unified ML System** - All OLDCARTS components
- **Learning System** - Continuous improvement
- **Performance Monitoring** - System optimization
- **User Feedback** - Quality assurance

**The LLM Medical Container is now a clean, organized, unified ML-powered diagnostic system!** 🏥✅
