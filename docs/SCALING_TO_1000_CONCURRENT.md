# Scaling to 1000 Concurrent Conversations - On-Premise Only

## What is k3s and Why Do You Need It?

### Current Setup: Docker Compose

Your current system uses **Docker Compose** to run containers on a single machine (Jetson Orin NX). This works well for:
- Single server deployments
- Development and testing
- Small-scale production (3-5 concurrent users)

**Limitations for 1000 concurrent conversations:**
- ❌ Runs on **one machine only** - can't distribute across multiple servers
- ❌ **No automatic scaling** - must manually add/remove containers
- ❌ **No load balancing** - all traffic goes to one server
- ❌ **No high availability** - if server fails, everything goes down
- ❌ **Manual management** - must SSH to each server to manage containers

### What is Kubernetes (K8s)?

**Kubernetes** is a container orchestration platform that manages containers across **multiple servers** (called a "cluster"). Think of it as:
- **Docker Compose** = Managing containers on **one computer**
- **Kubernetes** = Managing containers across **many computers** (cluster)

**What Kubernetes does:**
1. **Distributes containers** across multiple servers automatically
2. **Scales up/down** based on load (auto-scaling)
3. **Load balances** traffic across all containers
4. **Restarts failed containers** automatically (self-healing)
5. **Rolls out updates** without downtime (rolling updates)
6. **Manages resources** (CPU, memory, GPU) across the cluster

### What is k3s?

**k3s** is a **lightweight version of Kubernetes** designed for:
- **Edge computing** and IoT devices
- **Resource-constrained** environments
- **Simpler deployments** (easier to set up than full Kubernetes)
- **On-premise** infrastructure

**Key Differences: k3s vs Full Kubernetes:**

| Feature | Full Kubernetes (kubeadm) | k3s |
|---------|---------------------------|-----|
| **Installation** | Complex, multiple steps | Single binary, one command |
| **Resource Usage** | ~512MB-1GB RAM per node | ~50-100MB RAM per node |
| **Dependencies** | Requires etcd, container runtime | Self-contained, includes everything |
| **Configuration** | Many config files | Minimal configuration |
| **Learning Curve** | Steep | Easier |
| **Production Ready** | ✅ Yes | ✅ Yes (CNCF certified) |
| **Features** | Full feature set | Most features, optimized subset |

**k3s is perfect for on-premise because:**
- ✅ **Easier to install** - One command: `curl -sfL https://get.k3s.io | sh -`
- ✅ **Lower overhead** - Uses less RAM/CPU on each server
- ✅ **Same Kubernetes API** - All your Kubernetes knowledge applies
- ✅ **Production ready** - Used by major companies
- ✅ **GPU support** - Works with NVIDIA GPUs via device plugin

### What k3s Does for Your 1000 Concurrent Conversations

**Without k3s (current Docker Compose):**
```
Internet → Single Server (Jetson Orin NX)
           ├─ Container 1 (LLM)
           ├─ Container 2 (LLM)
           └─ Container 3 (LLM)
           ❌ Limited to one server's capacity
           ❌ No automatic distribution
           ❌ Manual scaling
```

**With k3s (scaled deployment):**
```
Internet → Load Balancer (Nginx)
           ↓
           Kubernetes Cluster (k3s)
           ├─ Server 1 (GPU) → 5 LLM containers
           ├─ Server 2 (GPU) → 5 LLM containers
           ├─ Server 3 (GPU) → 5 LLM containers
           ├─ ... (10-15 GPU servers total)
           └─ Server 15 (GPU) → 5 LLM containers
           ✅ Automatically distributes load
           ✅ Scales containers across servers
           ✅ High availability (if one server fails, others continue)
```

### Real-World Example

**Scenario**: You have 1000 users trying to chat simultaneously.

**With Docker Compose (current):**
- All 1000 requests hit **one server**
- Server gets overwhelmed
- Users experience slow responses or timeouts
- ❌ **Can't handle 1000 concurrent**

**With k3s:**
- 1000 requests hit **load balancer**
- k3s **distributes** requests across 10-15 GPU servers
- Each server handles 70-100 conversations
- ✅ **Handles 1000 concurrent easily**

### How k3s Works

1. **Master Node**: Controls the cluster (schedules containers, manages resources)
2. **Worker Nodes**: Run your containers (GPU servers with LLM containers)
3. **kubectl**: Command-line tool to manage the cluster
4. **Pods**: Your containers running in the cluster
5. **Services**: Network endpoints that load balance between pods
6. **Deployments**: Define how many containers to run and where

**Example k3s command:**
```bash
# Install k3s on master node (one command!)
curl -sfL https://get.k3s.io | sh -

# Join worker nodes to cluster
curl -sfL https://get.k3s.io | K3S_URL=https://master-ip:6443 K3S_TOKEN=xxx sh -

# Deploy your LLM containers
kubectl apply -f llm-medical-deployment.yaml

# Scale to 50 containers across all servers
kubectl scale deployment llm-medical --replicas=50
```

### Do You Need k3s?

**You need k3s if:**
- ✅ You want to run containers on **multiple servers** (not just one)
- ✅ You need **automatic scaling** (add/remove containers based on load)
- ✅ You want **high availability** (survive server failures)
- ✅ You need **load balancing** across multiple servers
- ✅ You're scaling to **1000+ concurrent conversations**

**You DON'T need k3s if:**
- ❌ You're staying on **one server** (Jetson Orin NX only)
- ❌ You only need **3-5 concurrent conversations**
- ❌ You're okay with **manual management**
- ❌ You don't need **high availability**

### Migration Path: Docker Compose → k3s

**Current (Docker Compose):**
```yaml
# docker-compose.yml
services:
  llm-medical:
    build: ../llm-medical-container
    ports:
      - "11434:11434"
```

**With k3s (Kubernetes):**
```yaml
# llm-medical-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-medical
spec:
  replicas: 50  # Run 50 containers across all servers
  template:
    spec:
      containers:
      - name: llm-medical
        image: your-registry/llm-medical:latest
        ports:
        - containerPort: 11434
```

**Benefits:**
- Same Docker containers, but now distributed across multiple servers
- Automatic load balancing
- Easy scaling: `kubectl scale deployment llm-medical --replicas=100`

## Current Architecture Limitations

Your current system has these bottlenecks for scaling:

1. **Single LLM Instance with Lock**: `threading.Lock()` serializes all inference (line 96 in `container_rest.py`)
2. **Single Flask Instance**: No horizontal scaling capability
3. **In-Memory Session Storage**: Sessions stored in Python dicts (not persistent/distributed)
4. **Edge Device Focus**: Designed for Jetson Orin NX (single device)

## Required Architectural Changes

### 1. Remove Thread Lock (Enable True Concurrency)
- Replace single LLM instance with **per-request or pooled LLM instances**
- Use async/await with proper resource management
- Implement connection pooling for llama.cpp

### 2. Horizontal Scaling
- **Load Balancer**: Nginx or HAProxy (self-hosted)
- **Container Orchestration**: Kubernetes or Docker Swarm
- **Stateless Services**: Move session state to Redis/database

### 3. Session Management
- **Redis Cluster**: Self-hosted Redis for distributed session storage
- **PostgreSQL**: Self-hosted database for persistent conversation history
- **Session Affinity**: Optional sticky sessions for better caching

### 4. Request Queue System
- **Message Queue**: RabbitMQ or Redis Queue (self-hosted)
- **Worker Pool**: Separate workers for LLM inference
- **Priority Queue**: Handle urgent medical queries first

## On-Premise Infrastructure Architecture

### Complete System Architecture:
```
Internet → Nginx Load Balancer → Kubernetes Cluster (GPU Workers) → Redis Cluster + PostgreSQL
                                    ↓
                            [10-15 GPU Servers]
                                    ↓
                            [Docker Containers]
```

## Hardware Specifications

### Option 1: Enterprise GPU Servers (Recommended)

#### GPU Worker Nodes (10-15 servers needed)

**Per Server Specifications:**
- **GPU**: 1x NVIDIA A100 (40GB) or 1x NVIDIA A10 (24GB)
- **CPU**: AMD EPYC 7543 (32 cores) or Intel Xeon Gold 6338 (32 cores)
- **RAM**: 128GB DDR4 ECC
- **Storage**: 2TB NVMe SSD (for models and data)
- **Network**: 10GbE network card
- **Power**: 1000W+ PSU

**Per Server Cost**: $15,000-25,000
**Total for 12 servers**: $180,000-300,000

**Capacity**: 100-150 concurrent conversations per server
**Total Capacity**: 1,200-1,800 concurrent (with headroom)

---

### Option 2: Budget GPU Servers (Consumer GPUs)

#### GPU Worker Nodes (15-20 servers needed)

**Per Server Specifications:**
- **GPU**: 1x NVIDIA RTX 4090 (24GB) - Consumer GPU
- **CPU**: AMD Ryzen 9 7950X (16 cores) or Intel i9-13900K (24 cores)
- **RAM**: 64GB DDR5
- **Storage**: 2TB NVMe SSD
- **Network**: 1GbE (upgrade to 10GbE recommended)
- **Power**: 850W+ PSU

**Per Server Cost**: $3,500-5,000
**Total for 18 servers**: $63,000-90,000

**Capacity**: 50-70 concurrent conversations per server
**Total Capacity**: 900-1,260 concurrent (with headroom)

**Note**: Consumer GPUs (RTX 4090) work well but:
- No ECC memory
- May need driver workarounds for multi-GPU
- Better for budget deployments

---

### Option 3: Hybrid Approach (Mix of Enterprise + Consumer)

- **3-5 Enterprise servers** (A100/A10) for high-priority traffic
- **10-12 Budget servers** (RTX 4090) for general traffic
- **Total Cost**: $100,000-150,000

---

## Supporting Infrastructure

### 1. Load Balancer Server (1-2 servers for HA)

**Specifications:**
- **CPU**: 8+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 32GB
- **Network**: 10GbE (dual for redundancy)
- **Storage**: 500GB SSD
- **Software**: Nginx or HAProxy
- **Cost**: $2,000-3,000 per server

### 2. Redis Cluster (3-6 nodes for redundancy)

**Per Node Specifications:**
- **CPU**: 4-8 cores
- **RAM**: 32-64GB (Redis is memory-intensive)
- **Storage**: 500GB SSD (for persistence)
- **Network**: 10GbE
- **Cost**: $1,500-2,500 per node

**Total Redis Cluster**: $4,500-15,000

### 3. PostgreSQL Database (1-2 servers for HA)

**Per Server Specifications:**
- **CPU**: 8-16 cores
- **RAM**: 64GB
- **Storage**: 2TB NVMe SSD (RAID 10 recommended)
- **Network**: 10GbE
- **Cost**: $3,000-5,000 per server

**Total Database**: $3,000-10,000 (with HA)

### 4. Control Plane / Kubernetes Master Nodes (3 nodes for HA)

**Per Node Specifications:**
- **CPU**: 4-8 cores
- **RAM**: 16-32GB
- **Storage**: 500GB SSD
- **Network**: 10GbE
- **Cost**: $1,500-2,500 per node

**Total Control Plane**: $4,500-7,500

### 5. Network Infrastructure

- **10GbE Switch**: 24-48 port managed switch
- **Cost**: $2,000-5,000
- **Cables, Racks, PDU**: $1,000-2,000

### 6. Storage Server (Optional - for backups/models)

- **NAS or SAN**: 10TB+ storage
- **Cost**: $3,000-8,000

---

## Total Hardware Cost Summary

| Component | Budget Option | Enterprise Option |
|-----------|--------------|-------------------|
| **GPU Workers** | $63,000-90,000 (18x RTX 4090) | $180,000-300,000 (12x A100/A10) |
| **Load Balancer** | $2,000-3,000 | $4,000-6,000 (HA) |
| **Redis Cluster** | $4,500-7,500 (3 nodes) | $9,000-15,000 (6 nodes) |
| **PostgreSQL** | $3,000-5,000 | $6,000-10,000 (HA) |
| **K8s Control Plane** | $4,500-7,500 | $4,500-7,500 |
| **Network** | $3,000-7,000 | $3,000-7,000 |
| **Storage/Backup** | $3,000-8,000 | $5,000-10,000 |
| **TOTAL** | **$83,000-128,000** | **$211,500-365,500** |

---

## Power & Cooling Requirements

### Power Consumption:
- **Per GPU Server**: 600-1000W (under load)
- **18 Budget Servers**: ~15-18kW
- **12 Enterprise Servers**: ~12-15kW
- **Supporting Infrastructure**: ~3-5kW
- **Total**: 18-23kW

### Cooling:
- **Air Conditioning**: 60,000-80,000 BTU (5-7 tons)
- **Server Room**: Minimum 200 sq ft, preferably 400+ sq ft
- **Raised Floor**: Recommended for cable management

### Power Cost (at $0.12/kWh):
- **Monthly**: $1,600-2,000
- **Annual**: $19,200-24,000

---

## Software Stack (All Self-Hosted)

### Container Orchestration:
- **Kubernetes**: k3s (lightweight) or full K8s
- **Alternative**: Docker Swarm (simpler, less features)

### Load Balancing:
- **Nginx**: Open-source, high performance
- **HAProxy**: Alternative, excellent for TCP/HTTP

### Session Storage:
- **Redis**: Self-hosted Redis Cluster
- **Setup**: 3-6 node cluster with sentinel for HA

### Database:
- **PostgreSQL**: Self-hosted with streaming replication
- **Backup**: pg_dump + WAL archiving

### Message Queue:
- **RabbitMQ**: Self-hosted cluster
- **Alternative**: Redis Queue (simpler, less features)

### Monitoring:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **AlertManager**: Alerting

### Container Registry:
- **Harbor**: Self-hosted Docker registry
- **Alternative**: Docker Registry (basic)

---

## Deployment Architecture

### Network Layout:
```
Internet
  ↓
[Firewall/Router]
  ↓
[Nginx Load Balancer] (10.0.1.10-11)
  ↓
[Kubernetes Cluster]
  ├─ Master Nodes (10.0.1.20-22)
  ├─ GPU Worker Nodes (10.0.1.30-47)
  └─ Infrastructure Nodes
      ├─ Redis Cluster (10.0.1.50-55)
      ├─ PostgreSQL (10.0.1.60-61)
      └─ Monitoring (10.0.1.70)
```

### Kubernetes Node Labels:
- **GPU Workers**: `node-type=gpu-worker`
- **Infrastructure**: `node-type=infra`
- **Masters**: `node-role=master`

---

## Advantages of On-Premise

1. **Full Control**: Complete control over hardware and software
2. **No Recurring Cloud Costs**: One-time hardware investment
3. **Data Privacy**: All data stays on-premise
4. **Customization**: Can optimize for specific workloads
5. **No Vendor Lock-in**: Use any hardware/software combination
6. **Predictable Costs**: Fixed hardware cost vs. variable cloud costs
7. **Consumer GPUs**: Can use RTX 4090 for significant cost savings

---

## Disadvantages & Considerations

1. **High Initial Investment**: $80,000-365,000 upfront
2. **Maintenance**: Requires IT staff for hardware/software maintenance
3. **Scaling**: Manual scaling (add more servers) vs. auto-scaling
4. **Redundancy**: Must plan for hardware failures
5. **Power & Cooling**: Requires proper data center facilities
6. **Updates**: Manual security updates and patches
7. **Backup & DR**: Must implement own backup/disaster recovery

---

## ROI Analysis

### Cloud Alternative Cost (for comparison):
- **AWS/GCP/Azure**: $8,000-12,000/month
- **Annual**: $96,000-144,000/year

### On-Premise Break-Even:
- **Budget Option** ($83,000-128,000): **7-10 months**
- **Enterprise Option** ($211,500-365,500): **18-30 months**

### 3-Year Total Cost:
- **Cloud**: $288,000-432,000
- **On-Premise Budget**: $83,000 + ($2,000/month × 36) = **$155,000**
- **On-Premise Enterprise**: $211,500 + ($2,000/month × 36) = **$283,500**

**Savings**: $133,000-149,000 (budget) or $4,500-148,500 (enterprise) over 3 years

---

## Code Changes Required

### 1. Remove Thread Lock & Enable Concurrency

**File**: `llm-medical-container/container_rest.py`

```python
# OLD (Current):
llm_lock = threading.Lock()

def llm_chat_simple(messages, ...):
    with llm_lock:  # ❌ Serializes all requests
        response = llm_simple.create_chat_completion(...)
```

**NEW (Concurrent)**:
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio
from queue import Queue

# LLM connection pool
llm_pool = Queue(maxsize=5)  # Pool of 5 LLM instances
llm_pool_lock = threading.Lock()

def init_llm_pool(pool_size=5):
    """Initialize pool of LLM instances"""
    for _ in range(pool_size):
        llm = Llama(
            model_path=SIMPLE_MODEL_PATH,
            n_ctx=SIMPLE_N_CTX,
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            n_gpu_layers=-1,
            # ... other params
        )
        llm_pool.put(llm)

def llm_chat_simple(messages, ...):
    """Get LLM from pool, use, return to pool"""
    llm = llm_pool.get(timeout=30)  # Wait max 30s for available LLM
    try:
        response = llm.create_chat_completion(...)
        return response
    finally:
        llm_pool.put(llm)  # Return to pool
```

### 2. Move to Async Framework

**Replace Flask with FastAPI** for better async support:

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import aioredis
import os

app = FastAPI()

# Redis connection pool (self-hosted Redis cluster)
redis_pool = None
REDIS_HOST = os.getenv("REDIS_HOST", "redis-cluster.local")  # Your Redis cluster endpoint
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

@app.on_event("startup")
async def startup():
    global redis_pool
    # Connect to self-hosted Redis cluster
    redis_pool = await aioredis.from_url(
        f"redis://{REDIS_HOST}:{REDIS_PORT}",
        encoding="utf-8",
        decode_responses=True,
        max_connections=100  # Connection pool for high concurrency
    )

@app.post("/chat-tg")
async def chat_tg(request: Request):
    data = await request.json()
    session_id = data.get("chat_id", "default")
    
    # Get session from self-hosted Redis
    session_data = await redis_pool.get(f"session:{session_id}")
    if session_data:
        session_data = json.loads(session_data)
    else:
        session_data = {"active": True, "history": []}
    
    # Process with LLM (async)
    response = await process_with_llm_async(data, session_data)
    
    # Save session to Redis
    await redis_pool.setex(
        f"session:{session_id}",
        3600,  # 1 hour TTL
        json.dumps(session_data)
    )
    
    return {"response": response}
```

### 3. Session Management with Self-Hosted Redis

**File**: `llm-medical-container/session_manager.py` (NEW)

```python
import aioredis
import json
import os
from typing import Optional, Dict

class SessionManager:
    def __init__(self, redis_host: str = None, redis_port: int = None):
        self.redis = None
        # Use environment variables or defaults for self-hosted Redis
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "redis-cluster.local")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_url = f"redis://{self.redis_host}:{self.redis_port}"
    
    async def connect(self):
        """Connect to self-hosted Redis cluster"""
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=100,  # Connection pool for high concurrency
            retry_on_timeout=True,
            health_check_interval=30
        )
    
    async def get_session(self, session_id: str) -> Dict:
        """Get session data from Redis"""
        try:
            data = await self.redis.get(f"session:{session_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"[SessionManager] Error getting session: {e}")
        return {"active": True, "history": []}
    
    async def save_session(self, session_id: str, session_data: Dict, ttl: int = 3600):
        """Save session to Redis with TTL"""
        try:
            await self.redis.setex(
                f"session:{session_id}",
                ttl,
                json.dumps(session_data)
            )
        except Exception as e:
            print(f"[SessionManager] Error saving session: {e}")
    
    async def delete_session(self, session_id: str):
        """Delete session from Redis"""
        try:
            await self.redis.delete(f"session:{session_id}")
        except Exception as e:
            print(f"[SessionManager] Error deleting session: {e}")
```

### 4. Add Request Queue (Self-Hosted RabbitMQ or Redis Queue)

**Option A: Redis Queue (Simpler, uses existing Redis)**

**File**: `llm-medical-container/queue_worker.py` (NEW)

```python
import redis
from rq import Queue, Worker
import os

# Redis connection for queue (same self-hosted Redis cluster)
redis_conn = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis-cluster.local"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=1,  # Use different DB for queue vs sessions
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

# Create queue
task_queue = Queue('llm_inference', connection=redis_conn)

def process_llm_request(session_id: str, prompt: str):
    """Worker function for LLM processing"""
    # This runs in separate worker processes
    navigator = get_medical_navigator()
    result = navigator.process_message(session_id, prompt, stream=False)
    return result

# In container_rest.py:
@app.post("/chat-tg")
async def chat_tg(request: Request):
    data = await request.json()
    session_id = data.get("chat_id")
    prompt = data.get("prompt")
    
    # Enqueue job to self-hosted Redis queue
    job = task_queue.enqueue(
        process_llm_request,
        session_id,
        prompt,
        job_timeout=30,
        result_ttl=3600  # Keep result for 1 hour
    )
    
    # Wait for result (or return job ID for async polling)
    result = job.result(timeout=30)
    return {"response": result.get("response")}
```

**Option B: RabbitMQ (More features, separate service)**

```python
import pika
import json
import os

# RabbitMQ connection (self-hosted)
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq.local")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
)
channel = connection.channel()

# Declare queue
channel.queue_declare(queue='llm_inference', durable=True)

def enqueue_request(session_id: str, prompt: str):
    """Enqueue LLM request"""
    message = json.dumps({"session_id": session_id, "prompt": prompt})
    channel.basic_publish(
        exchange='',
        routing_key='llm_inference',
        body=message,
        properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
    )
```

### 5. Kubernetes Deployment (Self-Hosted K8s)

**File**: `k8s/llm-medical-deployment.yaml` (NEW)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-medical
  namespace: aura
spec:
  replicas: 10  # Start with 10 pods, manually scale to 20 as needed
  selector:
    matchLabels:
      app: llm-medical
  template:
    metadata:
      labels:
        app: llm-medical
    spec:
      nodeSelector:
        node-type: gpu-worker  # Schedule on GPU worker nodes
      containers:
      - name: llm-medical
        image: harbor.local/aura/llm-medical:latest  # Self-hosted registry
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: REDIS_HOST
          value: "redis-cluster.aura.svc.cluster.local"  # K8s service DNS
        - name: REDIS_PORT
          value: "6379"
        - name: POSTGRES_HOST
          value: "postgres.aura.svc.cluster.local"
        - name: POSTGRES_PORT
          value: "5432"
        - name: POSTGRES_DB
          value: "aura_medical"
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
---
apiVersion: v1
kind: Service
metadata:
  name: llm-medical-service
  namespace: aura
spec:
  selector:
    app: llm-medical
  ports:
  - port: 11434
    targetPort: 11434
  type: ClusterIP  # Use NodePort or LoadBalancer if needed
---
# Manual scaling (on-premise doesn't auto-scale, but you can use HPA)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-medical-hpa
  namespace: aura
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-medical
  minReplicas: 10
  maxReplicas: 20  # Based on available GPU nodes
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**File**: `k8s/redis-cluster.yaml` (NEW - Self-hosted Redis)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: aura
data:
  redis.conf: |
    cluster-enabled yes
    cluster-config-file nodes.conf
    cluster-node-timeout 5000
    appendonly yes
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: aura
spec:
  serviceName: redis-cluster
  replicas: 6  # 6 Redis nodes for cluster
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      nodeSelector:
        node-type: infra  # Run on infrastructure nodes
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
        - name: redis-config
          mountPath: /etc/redis
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 50Gi
```

**File**: `k8s/postgres.yaml` (NEW - Self-hosted PostgreSQL)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: aura
spec:
  serviceName: postgres
  replicas: 1  # Add replica for HA if needed
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      nodeSelector:
        node-type: infra
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: aura_medical
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 500Gi
```

## Performance Estimates

### Per Instance Capacity (After Optimizations):

| Configuration | Concurrent Conversations | Response Time |
|--------------|-------------------------|---------------|
| **Single GPU (T4/A10)** | 50-100 | 2-4 seconds |
| **Single GPU (A100)** | 100-200 | 1-3 seconds |
| **With Request Queue** | 200-300 | 3-6 seconds (queued) |

### For 1000 Concurrent:

- **Minimum**: 10-15 GPU instances (T4/A10)
- **Recommended**: 15-20 GPU instances (with 20% headroom)
- **Optimal**: 10-12 A100 instances (faster, fewer instances)

## Migration Path

### Phase 1: Remove Lock & Add Redis (Week 1-2)
- Remove `threading.Lock()`
- Add Redis for session storage
- Test with 10-20 concurrent users

### Phase 2: Move to FastAPI (Week 3-4)
- Convert Flask to FastAPI
- Add async/await support
- Test with 50-100 concurrent users

### Phase 3: Add Load Balancer (Week 5-6)
- Deploy 2-3 instances behind load balancer
- Test with 200-300 concurrent users

### Phase 4: Full Kubernetes Deployment (Week 7-8)
- Deploy to Kubernetes
- Configure auto-scaling
- Test with 1000 concurrent users

## On-Premise Deployment Steps

### Phase 1: Infrastructure Setup (Week 1-2)
1. **Procure Hardware**: Order GPU servers, networking, storage
2. **Data Center Prep**: Power, cooling, rack space
3. **Network Setup**: Configure switches, VLANs, firewall rules
4. **Initial Server Setup**: Install OS (Ubuntu Server 22.04 LTS recommended)

### Phase 2: Kubernetes Cluster Setup (Week 3-4)
1. **Install Kubernetes**: k3s (lightweight) or full K8s
2. **Configure GPU Support**: NVIDIA Device Plugin for K8s
3. **Set Up Container Registry**: Harbor or Docker Registry
4. **Configure Networking**: Calico or Flannel CNI

### Phase 3: Supporting Services (Week 5-6)
1. **Deploy Redis Cluster**: 3-6 node Redis cluster
2. **Deploy PostgreSQL**: With streaming replication for HA
3. **Set Up Load Balancer**: Nginx or HAProxy
4. **Configure Monitoring**: Prometheus + Grafana

### Phase 4: Application Deployment (Week 7-8)
1. **Build Container Images**: Push to self-hosted registry
2. **Deploy LLM Containers**: Start with 5-10 replicas
3. **Configure Session Management**: Connect to Redis
4. **Load Testing**: Gradually increase to 1000 concurrent

### Phase 5: Production Hardening (Week 9-10)
1. **Backup Strategy**: Database backups, container image backups
2. **Security Hardening**: Firewall rules, SSL/TLS, secrets management
3. **Monitoring & Alerting**: Set up alerts for failures
4. **Documentation**: Runbooks, disaster recovery procedures

## Recommendations

1. **Start with Budget Option (RTX 4090)** - Lower initial investment, test at scale
2. **Use k3s for Kubernetes** - Lighter weight, easier to manage
3. **Implement Redis Cluster** - Critical for distributed sessions
4. **Use Nginx for Load Balancing** - Simple, reliable, high performance
5. **Deploy Prometheus + Grafana** - Essential for monitoring on-premise
6. **Plan for Redundancy** - N+1 for critical components
7. **Start Small, Scale Up** - Begin with 5-10 GPU servers, add more as needed
8. **Consider Colocation** - If you don't have data center space

## Next Steps

1. **Hardware Procurement**: Order GPU servers and supporting infrastructure
2. **Data Center Preparation**: Ensure power, cooling, and network capacity
3. **Create Redis session manager module**: For distributed session storage
4. **Remove thread lock from container_rest.py**: Enable true concurrency
5. **Convert Flask to FastAPI**: Better async support
6. **Set up self-hosted container registry**: Harbor or Docker Registry
7. **Create Kubernetes manifests**: For deployment automation
8. **Deploy supporting services**: Redis cluster, PostgreSQL
9. **Configure load balancer**: Nginx or HAProxy
10. **Set up monitoring**: Prometheus + Grafana
11. **Load test gradually**: Start with 10 users, scale to 1000

## Additional Resources

### Self-Hosted Alternatives to Cloud Services:

| Cloud Service | On-Premise Alternative |
|--------------|----------------------|
| AWS RDS | PostgreSQL (self-hosted) |
| AWS ElastiCache | Redis Cluster (self-hosted) |
| AWS ECR | Harbor or Docker Registry |
| AWS ALB | Nginx or HAProxy |
| AWS CloudWatch | Prometheus + Grafana |
| AWS SQS | RabbitMQ or Redis Queue |
| AWS Secrets Manager | HashiCorp Vault or Kubernetes Secrets |

### Useful Open-Source Tools:

- **Kubernetes**: k3s (lightweight) or kubeadm (full)
- **Container Registry**: Harbor (enterprise) or Docker Registry (basic)
- **Load Balancer**: Nginx, HAProxy, or Traefik
- **Monitoring**: Prometheus + Grafana + AlertManager
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana) or Loki
- **Secrets Management**: HashiCorp Vault
- **CI/CD**: GitLab CI, Jenkins, or GitHub Actions (self-hosted runner)

## Maintenance Considerations

### Regular Tasks:
- **Weekly**: Review monitoring dashboards, check disk space
- **Monthly**: Security updates, backup verification
- **Quarterly**: Capacity planning, hardware health checks
- **Annually**: Hardware refresh planning, disaster recovery drills

### Staff Requirements:
- **DevOps Engineer**: Kubernetes, container orchestration
- **Systems Administrator**: Hardware, networking, OS
- **Database Administrator**: PostgreSQL optimization, backups
- **On-call Rotation**: 24/7 support for production issues

### Backup Strategy:
- **Database**: Daily automated backups, weekly full backups
- **Redis**: Periodic snapshots (Redis persistence)
- **Container Images**: Regular exports to external storage
- **Configuration**: Version control (Git) + regular backups
- **Disaster Recovery**: Test restore procedures quarterly

