# 📊 Monitoring Scripts Guide

## 🎯 **QUICK ANSWER**

**You can run monitoring scripts from BOTH host and container!**

- **Host**: Direct access to data files
- **Container**: Full system integration with learning components

---

## 🚀 **OPTION 1: FROM HOST (Recommended for Monitoring)**

### **Prerequisites**
```bash
# Install Python dependencies on host
pip install scikit-learn joblib numpy pandas
```

### **Run Monitoring Scripts**
```bash
# Navigate to container directory
cd /Users/rcabello/Documents/GitHub/LedgerAI/llm-medical-container

# Run monitoring scripts directly
python learning_tracker.py
python performance_dashboard.py
python feedback_guide.py
```

### **Access Data Files**
```bash
# Check if data directory exists
ls -la ./data/learning/

# If not, create it
mkdir -p ./data/learning
```

---

## 🐳 **OPTION 2: FROM CONTAINER (Full Integration)**

### **Build Updated Container**
```bash
# Build container with learning components
docker build -t llm-medical-container:latest .
```

### **Run Container with Volume Mounts**
```bash
# Run container with data volume mounted
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/shared:/shared \
  -p 11434:11434 \
  llm-medical-container:latest
```

### **Execute Monitoring Scripts Inside Container**
```bash
# Inside container
cd /app
python learning_tracker.py
python performance_dashboard.py
python feedback_guide.py
```

---

## 📋 **MONITORING SCRIPT USAGE**

### **1. Learning Tracker**
```bash
# Check learning progress
python learning_tracker.py

# Output example:
# 📊 Learning System Status
# ├── Total Predictions: 150
# ├── Learning Data Points: 45
# ├── Feedback Entries: 12
# └── Performance Metrics: 8
```

### **2. Performance Dashboard**
```bash
# View performance metrics
python performance_dashboard.py

# Output example:
# 📈 Performance Dashboard
# ├── Accuracy: 85.2%
# ├── Precision: 82.1%
# ├── Recall: 88.3%
# └── F1-Score: 85.1%
```

### **3. Feedback Guide**
```bash
# Test feedback system
python feedback_guide.py

# Output example:
# 💬 Feedback System Test
# ├── Prediction Rating: ✅ Collected
# ├── Accuracy Feedback: ✅ Collected
# └── General Feedback: ✅ Collected
```

---

## 🔧 **DATA DIRECTORY STRUCTURE**

```
./data/learning/
├── predictions.json      # ML prediction data
├── feedback.json         # User feedback data
├── performance.json      # Performance metrics
└── performance_metrics.json  # Detailed metrics
```

---

## 🚀 **RECOMMENDED WORKFLOW**

### **Daily Monitoring (Host)**
```bash
# 1. Check learning progress
python learning_tracker.py

# 2. Review performance
python performance_dashboard.py

# 3. Test feedback system
python feedback_guide.py
```

### **Full System Testing (Container)**
```bash
# 1. Build updated container
docker build -t llm-medical-container:latest .

# 2. Run with volume mounts
docker run -it -v $(pwd)/data:/app/data -p 11434:11434 llm-medical-container:latest

# 3. Test inside container
python learning_tracker.py
python performance_dashboard.py
python feedback_guide.py
```

---

## 🎯 **KEY DIFFERENCES**

| Aspect | Host | Container |
|--------|------|-----------|
| **Data Access** | Direct file access | Volume mounted |
| **Dependencies** | Manual install | Pre-installed |
| **Integration** | Limited | Full system |
| **Performance** | Faster startup | Slower startup |
| **Debugging** | Easier | More complex |

---

## 💡 **RECOMMENDATIONS**

### **For Development & Monitoring**
- **Use Host**: Faster, easier debugging
- **Direct file access**: No volume mounting needed
- **Quick iteration**: No container rebuilds

### **For Production & Full Testing**
- **Use Container**: Complete system integration
- **Volume mounts**: Persistent data storage
- **Full isolation**: Production-like environment

---

## 🔧 **TROUBLESHOOTING**

### **Host Issues**
```bash
# Missing dependencies
pip install scikit-learn joblib numpy pandas

# Missing data directory
mkdir -p ./data/learning
```

### **Container Issues**
```bash
# Rebuild container
docker build -t llm-medical-container:latest .

# Check volume mounts
docker run -it -v $(pwd)/data:/app/data llm-medical-container:latest ls -la /app/data
```

---

## 📊 **MONITORING COMMANDS**

### **Quick Status Check**
```bash
# Check if data exists
ls -la ./data/learning/

# Check learning progress
python learning_tracker.py

# Check performance
python performance_dashboard.py
```

### **Full System Test**
```bash
# Build and run container
docker build -t llm-medical-container:latest .
docker run -it -v $(pwd)/data:/app/data -p 11434:11434 llm-medical-container:latest

# Test inside container
python learning_tracker.py
python performance_dashboard.py
python feedback_guide.py
```

---

## 🎯 **SUMMARY**

**You can run monitoring scripts from both host and container!**

- **Host**: Faster, easier for development
- **Container**: Full integration, production-like
- **Data**: Shared via volume mounts
- **Dependencies**: Pre-installed in container

**Choose based on your needs:**
- **Development**: Use host
- **Production**: Use container
- **Testing**: Use both!
