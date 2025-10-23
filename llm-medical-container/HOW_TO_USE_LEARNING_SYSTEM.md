# 🎯 **HOW TO USE THE LEARNING SYSTEM**

## 📊 **TRACK LEARNING DATA**

### **1. Check Learning Status**
```python
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine

# Initialize engine
engine = AdaptiveDiagnosticEngine()

# Get learning status
status = engine.get_learning_status()
print(f"Learning Status: {status}")
```

### **2. Use Learning Tracker**
```python
from learning_tracker import LearningTracker

# Initialize tracker
tracker = LearningTracker()

# Get comprehensive learning summary
summary = tracker.get_learning_summary()
print(f"Learning Summary: {summary}")

# Get recent activity (last 24 hours)
activity = tracker.get_recent_activity(24)
print(f"Recent Activity: {activity}")

# Export all learning data
export_path = tracker.export_learning_data()
print(f"Data exported to: {export_path}")
```

---

## 📈 **MONITOR PERFORMANCE**

### **1. Use Performance Dashboard**
```python
from performance_dashboard import PerformanceDashboard

# Initialize dashboard
dashboard = PerformanceDashboard()

# Get performance overview
overview = dashboard.get_performance_overview()
print(f"Performance Overview: {overview}")

# Print formatted dashboard
dashboard.print_dashboard()
```

### **2. Check Performance Metrics**
```python
# Get learning status from engine
status = engine.get_learning_status()

# Check performance monitor
if status['performance_monitor']:
    performance_summary = status['performance_summary']
    print(f"Performance Summary: {performance_summary}")
```

---

## 💬 **PROVIDE FEEDBACK**

### **1. Use Feedback Guide**
```python
from feedback_guide import FeedbackGuide

# Initialize feedback guide
guide = FeedbackGuide()

# Initialize engine
guide.initialize_engine()

# Provide prediction feedback
guide.provide_prediction_feedback(
    patient_text="right lower quadrant pain",
    guideline_text="right lower quadrant pain",
    condition_name="Acute Appendicitis",
    user_rating=5,  # 1-5 stars
    user_comment="Excellent prediction - very accurate"
)

# Provide accuracy feedback
guide.provide_accuracy_feedback(
    condition_name="Acute Appendicitis",
    predicted_accuracy=0.8,
    actual_accuracy=0.85,
    user_comment="Close prediction, very good"
)
```

### **2. Direct Engine Feedback**
```python
# Collect user feedback on prediction
engine.collect_user_feedback(
    prediction_id="pred_123",
    prediction={'similarity': 0.8, 'method': 'ml_prediction'},
    user_rating=4,  # 1-5 stars
    user_comment="The prediction was accurate",
    condition_name="Acute Appendicitis"
)

# Collect accuracy feedback
engine.collect_accuracy_feedback(
    prediction_id="pred_123",
    predicted_accuracy=0.8,
    actual_accuracy=0.85,
    user_comment="Close prediction",
    condition_name="Acute Appendicitis"
)
```

---

## 🔄 **CONTINUOUS LEARNING**

### **1. Automatic Learning**
The system automatically:
- **Collects learning data** from every prediction
- **Monitors performance** in real-time
- **Retrains models** when enough new data is available
- **Tracks user feedback** and ratings
- **Generates performance reports** every hour

### **2. Manual Learning Status Check**
```python
# Check if continuous learning is active
status = engine.get_learning_status()
if status['continuous_learning']:
    learning_status = status['continuous_learning_status']
    print(f"Continuous Learning: {learning_status}")
```

---

## 📁 **DATA FILES**

### **Learning Data Location: `./data/learning/`**
- **`feedback.json`** - ML prediction feedback
- **`predictions.json`** - All ML predictions
- **`performance.json`** - Performance metrics
- **`user_feedback.json`** - User ratings and comments

### **Model Data Location: `./data/models/`**
- **`location_ml_model.pkl`** - Current ML model
- **`model_metadata.json`** - Model versioning info

---

## 🚀 **QUICK START EXAMPLES**

### **Example 1: Complete Learning Workflow**
```python
from adaptive_diagnostic_engine import AdaptiveDiagnosticEngine
from learning_tracker import LearningTracker
from performance_dashboard import PerformanceDashboard
from feedback_guide import FeedbackGuide

# 1. Initialize everything
engine = AdaptiveDiagnosticEngine()
tracker = LearningTracker()
dashboard = PerformanceDashboard()
guide = FeedbackGuide()

# 2. Check system status
status = engine.get_learning_status()
print(f"System Status: {status}")

# 3. Provide feedback
guide.initialize_engine()
guide.provide_prediction_feedback(
    patient_text="right lower quadrant pain",
    guideline_text="right lower quadrant pain",
    condition_name="Acute Appendicitis",
    user_rating=5,
    user_comment="Perfect prediction!"
)

# 4. Monitor performance
dashboard.print_dashboard()

# 5. Check learning data
summary = tracker.get_learning_summary()
print(f"Learning Summary: {summary}")
```

### **Example 2: Daily Performance Check**
```python
from performance_dashboard import PerformanceDashboard

# Check daily performance
dashboard = PerformanceDashboard()
dashboard.print_dashboard()

# Get recent activity
activity = dashboard._get_recent_activity(24)
print(f"Last 24 hours: {activity}")
```

### **Example 3: Export Learning Data**
```python
from learning_tracker import LearningTracker

# Export all learning data
tracker = LearningTracker()
export_path = tracker.export_learning_data("daily_export.json")
print(f"Data exported to: {export_path}")
```

---

## 📊 **MONITORING COMMANDS**

### **Check Learning Status**
```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI/llm-medical-container
python learning_tracker.py
```

### **View Performance Dashboard**
```bash
python performance_dashboard.py
```

### **Test Feedback System**
```bash
python feedback_guide.py
```

---

## 🎯 **KEY BENEFITS**

### **For Learning:**
- **Real-time tracking** - See learning progress immediately
- **Performance monitoring** - Track accuracy over time
- **User feedback integration** - Learn from user input
- **Automated reports** - Get performance insights

### **For Medical Accuracy:**
- **Continuous improvement** - Model gets better with use
- **User expertise** - Leverage clinician feedback
- **Performance metrics** - Quantify improvement
- **Quality assurance** - Track prediction accuracy

### **For System Health:**
- **Data freshness** - Monitor data recency
- **Learning activity** - Track system usage
- **Performance indicators** - Identify issues
- **Overall health** - System status assessment

---

## 🚀 **READY TO USE!**

Your learning system is now fully operational and will:

1. **Automatically collect** learning data from every interaction
2. **Monitor performance** in real-time
3. **Retrain models** when enough new data is available
4. **Track user feedback** and ratings
5. **Generate reports** and insights

**Start using the system and watch it learn and improve!** 🎉
