# Simple RAG Toggle - CPU vs GPU

## 🎯 **Simple Toggle Design:**

### **Two Distinct Pathways:**

#### **CPU Mode (Default):**
- **Local CPU FAISS** in LLM container
- **No network calls** 
- **Direct processing** within container
- **Faster for small batches**

#### **GPU Mode:**
- **GPU FAISS** in RAG container
- **RAG API calls** to external container
- **Network overhead** but GPU acceleration
- **Faster for large batches**

## ⚙️ **Configuration:**

### **Environment Variable:**
```bash
# CPU Mode (Default)
RAG_MODE=CPU

# GPU Mode
RAG_MODE=GPU
```

### **Docker Compose:**
```yaml
services:
  llm-medical:
    environment:
      - RAG_MODE=CPU    # or GPU
    # For GPU mode, also include:
    depends_on:
      - rag-container

  rag-container:  # Only needed for GPU mode
    image: rag-gpu:latest
    ports:
      - "11435:11435"
```

## 🔄 **How It Works:**

### **CPU Mode (RAG_MODE=CPU):**
```python
# Local CPU FAISS processing
if self.rag_mode == 'CPU':
    # Use local embedding model
    embeddings = self.embedding_model.encode(all_texts)
    # Direct CPU processing
    similarity = np.dot(query_embedding, trigger_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(trigger_embedding)
    )
```

### **GPU Mode (RAG_MODE=GPU):**
```python
# GPU RAG API processing
if self.rag_mode == 'GPU':
    # Use RAG API for GPU-accelerated embeddings
    embeddings = self.rag_api.encode(all_texts)
    # GPU processing via API
    similarity = np.dot(query_embedding, trigger_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(trigger_embedding)
    )
```

## 📊 **Performance Comparison:**

### **CPU Mode:**
- **Small batches (1-10 triggers):** ~50ms
- **Medium batches (10-50 triggers):** ~100-300ms
- **Large batches (50+ triggers):** ~500-2000ms
- **Network calls:** 0
- **Resource usage:** CPU only

### **GPU Mode:**
- **Small batches (1-10 triggers):** ~100-200ms (network overhead)
- **Medium batches (10-50 triggers):** ~100-200ms
- **Large batches (50+ triggers):** ~100-300ms
- **Network calls:** 1 per batch
- **Resource usage:** GPU + network

## 🎯 **When to Use Each Mode:**

### **Use CPU Mode When:**
- **Small to medium batches** (<50 triggers)
- **No GPU available**
- **Network latency is high**
- **Simple deployment** (single container)

### **Use GPU Mode When:**
- **Large batches** (50+ triggers)
- **GPU available** and accessible
- **High-volume processing**
- **Distributed architecture**

## 🚀 **Quick Start:**

### **CPU Mode (Default):**
```bash
# No configuration needed
# System uses local CPU FAISS automatically
```

### **GPU Mode:**
```bash
# Set environment variable
export RAG_MODE=GPU

# Ensure RAG container is running
docker run -p 11435:11435 rag-gpu:latest
```

## 📈 **Expected Performance:**

### **Small Batches (1-10 triggers):**
- **CPU:** 50ms (no network)
- **GPU:** 100-200ms (network overhead)
- **Winner:** CPU

### **Large Batches (50+ triggers):**
- **CPU:** 500-2000ms (CPU bottleneck)
- **GPU:** 100-300ms (GPU parallelization)
- **Winner:** GPU

### **Very Large Batches (200+ triggers):**
- **CPU:** 2-10 seconds (CPU bottleneck)
- **GPU:** 200-500ms (GPU parallelization)
- **Winner:** GPU (10-20x faster)

## 🔧 **Implementation:**

### **Simple Toggle Logic:**
```python
# Initialize mode
self.rag_mode = os.environ.get('RAG_MODE', 'CPU').upper()

# Use mode in processing
if self.rag_mode == 'GPU':
    # GPU pathway: RAG API calls
    self._perform_gpu_semantic_search(...)
else:
    # CPU pathway: Local FAISS
    self._perform_cpu_semantic_search(...)
```

### **Debug Output:**
```
[Engine] 🎯 RAG Mode: CPU
[Engine] 🧠 CPU FAISS semantic search (local processing)...

# OR

[Engine] 🎯 RAG Mode: GPU
[Engine] 🚀 GPU RAG API semantic search (GPU-accelerated)...
```

## ✅ **Benefits:**

### **CPU Mode Benefits:**
- **No network calls** for local operations
- **Faster for small batches**
- **Simpler architecture**
- **Lower resource usage**

### **GPU Mode Benefits:**
- **Faster for large batches**
- **Better parallelization**
- **Scalable for high-volume**
- **Advanced GPU features**

**Simple toggle between two distinct data analysis pathways - CPU (local) vs GPU (RAG API)!** 🏥⚡
