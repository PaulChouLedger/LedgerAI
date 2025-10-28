# Background Learning System Architecture

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Main Process (Fast)                        │
│  Main Diagnostic Engine                                      │
│  - Processes patient answers                              │
│  - Async recording of issues                              │
│  - Never blocks on learning                               │
└──────────────────┬────────────────────────────────────────┘
                   │
                   ▼ async writes
┌─────────────────────────────────────────────────────────────┐
│            Background Learning Process                      │
│  - Reads learning data (corrections, synonyms, patterns)    │
│  - Analyzes patterns                                        │
│  - Generates updates                                        │
│  - Runs periodically (nightly/weekly)                       │
└──────────────────┬────────────────────────────────────────┘
                   │
                   ▼ updates
┌─────────────────────────────────────────────────────────────┐
│                 Updated Files                               │
│  - structured_oldcarts (includes/excludes refined)          │
│  - synonym files (new patient terms added)                  │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 How It Works

### **Phase 1: Real-Time Recording (Async, Non-Blocking)**

During normal operation, system **asynchronously** records learning data:

```python
# In adaptive_diagnostic_engine.py
async def _process_clinical_answer(self, answer: str):
    # Normal processing (fast, synchronous)
    result = self._compute_similarity(answer, guideline_text)
    
    # Background recording (async, doesn't block)
    if self._should_record_learning(context):
        await self.learner.record_correction_async(
            condition=top_condition,
            user_answer=answer,
            result=result
        )
    
    return result
```

**Key Points:**
- ✅ **Zero performance impact** - async writes don't block
- ✅ **Fire-and-forget** - don't wait for completion
- ✅ **Buffered writes** - batch writes every N seconds

### **Phase 2: Background Analysis (Separate Process)**

Separate background worker analyzes data:

```python
# ml/background_learning_worker.py
import schedule
import time

def analyze_and_update():
    """Runs nightly to analyze learning data"""
    print("[Background] Starting nightly analysis...")
    
    # 1. Load learning data
    learner = InteractionLearning()
    
    # 2. Analyze patterns
    suggestions = learner.analyze_corrections(min_occurrences=5)
    
    # 3. Generate updates
    updates = learner.generate_updates(llm_fn=your_llm)
    
    # 4. Apply updates
    apply_learning_updates(updates, dry_run=False)
    
    print("[Background] ✅ Updates applied")

# Schedule to run nightly at 2 AM
schedule.every().day.at("02:00").do(analyze_and_update)

# Run scheduler in background thread
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)
```

### **Phase 3: Manual Triggers (Optional)**

For testing or immediate updates:

```bash
# Run analysis now
python ml/background_learning_worker.py --run-now

# Check what would be updated (dry run)
python ml/background_learning_worker.py --dry-run

# Manual update of specific condition
python ml/update_guideline_from_learning.py --condition "Acute Appendicitis"
```

## 📊 Implementation Details

### **1. Async Recording**

```python
# ml/async_learning_recorder.py
import asyncio
from collections import deque
import threading

class AsyncLearningRecorder:
    def __init__(self):
        self.buffer = deque()
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        
        # Start background flush thread
        threading.Thread(target=self._flush_loop, daemon=True).start()
    
    async def record_correction_async(self, **kwargs):
        """Async record - fire and forget"""
        self.buffer.append(kwargs)
        
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_loop(self):
        """Background thread flushes buffer periodically"""
        while True:
            time.sleep(self.flush_interval)
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered records to disk"""
        while self.buffer:
            record = self.buffer.popleft()
            # Write to file (non-blocking)
            self._append_to_file(record)
```

### **2. Background Worker**

```python
# ml/background_learning_worker.py
import schedule
import time
from datetime import datetime
import logging

class BackgroundLearningWorker:
    def __init__(self):
        self.learner = InteractionLearning()
        self.llm_fn = self._init_llm()
    
    def run_analysis(self):
        """Main analysis loop"""
        try:
            print(f"[{datetime.now()}] Starting analysis...")
            
            # Step 1: Analyze corrections
            suggestions = self.learner.analyze_corrections(min_occurrences=3)
            print(f"Found {len(suggestions)} conditions needing updates")
            
            # Step 2: Generate updates
            updates = self.learner.generate_updates(llm_fn=self.llm_fn)
            
            # Step 3: Apply updates
            apply_learning_updates(updates, dry_run=False)
            
            print(f"[{datetime.now()}] ✅ Analysis complete")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Analysis failed: {e}")
    
    def start(self):
        """Start background worker"""
        # Schedule nightly at 2 AM
        schedule.every().day.at("02:00").do(self.run_analysis)
        
        print("Background learning worker started")
        print("Scheduled: Nightly at 2:00 AM")
        
        # Run forever
        while True:
            schedule.run_pending()
            time.sleep(60)
```

### **3. Docker Integration**

```dockerfile
# In Dockerfile
# Install requirements
RUN pip install schedule

# Add background worker script
COPY ml/background_learning_worker.py /app/ml/

# Start background worker alongside main app
CMD ["python", "-c", "from multiprocessing import Process; \
      Process(target=lambda: __import__('ml.background_learning_worker').start()).start(); \
      __import__('container_rest')"]
```

## 🎯 Usage Patterns

### **Development/Testing**
```bash
# Run one-time analysis
python ml/background_learning_worker.py --run-now

# Dry run - see what would change
python ml/background_learning_worker.py --dry-run

# Check learning data stats
python ml/check_learning_stats.py
```

### **Production**
```bash
# Start background worker as daemon
python ml/background_learning_worker.py --daemon &

# Main app continues normally
python container_rest.py
```

### **Monitoring**
```bash
# Check if worker is running
ps aux | grep background_learning_worker

# View recent updates
tail -f ml/learning_data/updates.log

# Check learning data size
du -sh ml/learning_data/
```

## ⚙️ Configuration

```json
{
  "learning": {
    "enabled": true,
    "async_recording": true,
    "flush_interval_seconds": 60,
    "buffer_size": 100,
    "min_occurrences": 5,
    "analysis_schedule": "0 2 * * *",
    "auto_apply": true,
    "confidence_threshold": 0.7
  }
}
```

## 🚨 Important Considerations

1. **Performance**: Async recording ensures zero impact on response time
2. **Disk I/O**: Buffered writes minimize disk operations
3. **Resource Usage**: Background worker only runs when needed
4. **Error Handling**: Failures don't affect main system
5. **Rollback**: Keep backups before applying updates

## 📈 Benefits

- ✅ **Always improving** - System learns 24/7
- ✅ **No downtime** - Updates applied during maintenance window
- ✅ **Safe updates** - Dry-run and confidence checks
- ✅ **Transparent** - Logs show what changed and why
- ✅ **Scalable** - Handles thousands of interactions

