# Open Voice OS (OVOS) Integration Analysis

## Executive Summary

**Question:** Is installing Open Voice OS on Jetson to function as Aura assistant bot reasonable? Can it be integrated into the current structure?

**Short Answer:** 
- ⚠️ **Full OVOS replacement is NOT recommended** - would lose critical custom medical features
- ✅ **Selective integration is possible** but complex
- ✅ **Better alternative:** Use lightweight wake word detection (Porcupine) instead of full OVOS

---

## Current Aura Architecture

### Components
1. **Whisper Container** - Custom faster-whisper STT (port 5000)
2. **LLM Containers** - Medical (11434) + Generic (11436) with custom RAG
3. **RAG Container** - GPU/CPU FAISS for medical knowledge retrieval
4. **Aura Control** - Main orchestrator:
   - `listener.py` - VAD + audio capture (Silero VAD, no wake word)
   - `speaker.py` - TTS via ElevenLabs API
   - `main.py` - Docker orchestration + GUI
   - Custom PyQt5 GUI

### Current Pipeline
```
Microphone (ReSpeaker XVF3800)
    ↓
Hardware DSP (Beamforming + AGC)
    ↓
Silero VAD (always listening, no wake word)
    ↓
Advanced Multi-Feature Filter
    ↓
Whisper STT (Docker container)
    ↓
LLM Processing (Medical/Generic with RAG)
    ↓
ElevenLabs TTS
```

### Key Features
- ✅ Custom medical RAG system
- ✅ Dual LLM modes (medical/generic)
- ✅ Advanced audio filtering
- ✅ Custom GUI
- ✅ Docker-based microservices
- ❌ **No wake word detection** (always listening)

---

## Open Voice OS (OVOS) Overview

### What OVOS Provides
1. **Wake Word Detection** - Precise, Porcupine, or custom
2. **STT Plugins** - Vosk (offline), Google, DeepSpeech, etc.
3. **TTS Plugins** - Mimic3 (offline), Google, Amazon Polly, etc.
4. **Skills Framework** - Modular skill system
5. **NLP/Intent Handling** - Built-in intent classification
6. **Message Bus** - Inter-service communication
7. **GUI Framework** - Optional GUI components

### OVOS Architecture
```
OVOS Core
    ├─ Wake Word Engine
    ├─ STT Plugin (Vosk/Google/etc.)
    ├─ Intent Parser
    ├─ Skills Manager
    ├─ TTS Plugin (Mimic3/Google/etc.)
    └─ Message Bus (MQTT/WebSocket)
```

---

## Integration Analysis

### Option 1: Full OVOS Replacement ❌ NOT RECOMMENDED

**What it means:**
- Replace entire Aura system with OVOS
- Use OVOS skills instead of custom LLM containers
- Use OVOS STT/TTS instead of Whisper/ElevenLabs

**Pros:**
- ✅ Full wake word support
- ✅ Skills framework
- ✅ Community support
- ✅ Offline capabilities (Vosk + Mimic3)

**Cons:**
- ❌ **Loses custom medical RAG system** (critical feature)
- ❌ **Loses dual LLM modes** (medical/generic switching)
- ❌ **Loses custom medical knowledge base**
- ❌ **Loses ElevenLabs TTS** (high-quality voice)
- ❌ **Loses custom GUI** (PyQt5 interface)
- ❌ **Complex migration** (weeks of work)
- ❌ **OVOS skills don't match medical use case** (general-purpose, not medical)

**Verdict:** ❌ **Not reasonable** - Would destroy core medical functionality

---

### Option 2: Hybrid Integration (OVOS as Wake Word + STT) ⚠️ COMPLEX

**What it means:**
- Use OVOS for wake word detection and STT only
- Keep custom LLM containers, RAG, TTS, GUI
- Bridge OVOS message bus to Aura REST APIs

**Architecture:**
```
OVOS (Wake Word + STT)
    ↓
Custom Bridge Service
    ↓
Aura LLM Containers (existing)
    ↓
Aura TTS (ElevenLabs, existing)
```

**Pros:**
- ✅ Adds wake word detection
- ✅ Keeps medical features
- ✅ Can use offline STT (Vosk) if desired

**Cons:**
- ❌ **Complex integration** - OVOS message bus → REST API bridge needed
- ❌ **Redundant STT** - Already have Whisper (better quality)
- ❌ **Heavyweight** - OVOS includes many unused components
- ❌ **Maintenance burden** - Two systems to maintain
- ❌ **Resource overhead** - OVOS + Aura running simultaneously

**Implementation Complexity:**
- Need to create bridge service between OVOS message bus and Aura REST APIs
- OVOS expects skills, but Aura uses custom LLM
- Would need to disable OVOS TTS, intent parser, skills manager
- Essentially using 10% of OVOS, maintaining 100% of complexity

**Verdict:** ⚠️ **Possible but not recommended** - Too complex for limited benefit

---

### Option 3: OVOS Wake Word Only ⚠️ OVERKILL

**What it means:**
- Install OVOS just for wake word detection
- Disable all other OVOS features
- Use existing Aura pipeline after wake word

**Pros:**
- ✅ Adds wake word detection

**Cons:**
- ❌ **Massive overkill** - Installing full platform for one feature
- ❌ **Resource waste** - OVOS includes many unused services
- ❌ **Complexity** - Full OVOS installation just for wake word

**Verdict:** ⚠️ **Not recommended** - Use dedicated wake word library instead

---

## Recommended Alternative: Lightweight Wake Word Detection ✅

### Use Porcupine (Already Documented)

**Why Porcupine over OVOS:**
- ✅ **Lightweight** - Single library, not full platform
- ✅ **Jetson optimized** - ARM64 support
- ✅ **Low CPU** - ~5-10% vs OVOS's ~20-30%
- ✅ **Simple integration** - Direct Python library
- ✅ **Already documented** - See `docs/WAKE_WORD_DETECTION_GUIDE.md`
- ✅ **No redundant features** - Just wake word, nothing else

**Integration:**
```python
# Simple addition to listener.py
import pvporcupine

wake_word = pvporcupine.create(keywords=['hey aura'])
# Add to existing listener loop
```

**Architecture:**
```
Microphone
    ↓
Hardware DSP
    ↓
Porcupine Wake Word (NEW) ← Simple addition
    ↓
Silero VAD (existing)
    ↓
Whisper STT (existing)
    ↓
LLM + RAG (existing)
    ↓
ElevenLabs TTS (existing)
```

**Effort:** ~2-4 hours vs weeks for OVOS integration

---

## Comparison Table

| Feature | Current Aura | OVOS Full | OVOS Hybrid | Porcupine Only |
|---------|-------------|-----------|-------------|----------------|
| **Wake Word** | ❌ None | ✅ Yes | ✅ Yes | ✅ Yes |
| **STT** | ✅ Whisper | Vosk/Google | ✅ Whisper (keep) | ✅ Whisper (keep) |
| **TTS** | ✅ ElevenLabs | Mimic3/Google | ✅ ElevenLabs (keep) | ✅ ElevenLabs (keep) |
| **Medical RAG** | ✅ Custom | ❌ Lost | ✅ Keep | ✅ Keep |
| **Dual LLM Modes** | ✅ Yes | ❌ Lost | ✅ Keep | ✅ Keep |
| **Custom GUI** | ✅ PyQt5 | OVOS GUI | ✅ Keep | ✅ Keep |
| **CPU Usage** | ~30-40% | ~40-50% | ~50-60% | ~35-45% |
| **Memory** | ~2GB | ~2.5GB | ~3GB | ~2.1GB |
| **Integration Effort** | - | Weeks | Days | Hours |
| **Maintenance** | Low | Medium | High | Low |

---

## Jetson Compatibility

### OVOS on Jetson
- ✅ **Supported** - OVOS runs on ARM64 (Raspberry Pi, Jetson)
- ⚠️ **Performance** - May need optimization for Jetson Orin
- ⚠️ **Docker** - OVOS can run in Docker, but adds complexity
- ⚠️ **Dependencies** - Many Python packages, may conflict with existing

### Current Aura on Jetson
- ✅ **Already working** - Docker containers optimized for Jetson
- ✅ **GPU support** - CUDA for RAG and LLM
- ✅ **Tested** - Current architecture is Jetson-ready

---

## Recommendations

### ✅ **Recommended: Add Porcupine Wake Word**

**Steps:**
1. Install Porcupine: `pip install pvporcupine`
2. Train/download "hey aura" model from Picovoice Console
3. Integrate into `listener.py` (see `docs/WAKE_WORD_DETECTION_GUIDE.md`)
4. Add GUI feedback for wake word activation
5. Test and tune sensitivity

**Benefits:**
- ✅ Minimal code changes (~100 lines)
- ✅ Keeps all existing features
- ✅ Low resource overhead
- ✅ Simple maintenance

---

### ⚠️ **Alternative: OVOS Hybrid (Only if you need OVOS skills)**

**When to consider:**
- You want to add general-purpose skills (weather, calendar, etc.)
- You need OVOS's skills marketplace
- You want to experiment with OVOS ecosystem

**Implementation:**
1. Install OVOS alongside Aura
2. Configure OVOS to use custom STT/TTS (bridge to Aura)
3. Create custom OVOS skill that calls Aura LLM API
4. Handle message bus routing

**Complexity:** High (days/weeks of work)

---

### ❌ **Not Recommended: Full OVOS Replacement**

**Why:**
- Loses critical medical features
- Massive migration effort
- No clear benefit over current architecture
- OVOS is general-purpose, Aura is medical-specific

---

## Conclusion

**Is OVOS reasonable for Aura?**
- **Full replacement:** ❌ No - Would destroy medical features
- **Hybrid integration:** ⚠️ Possible but complex - Not worth the effort
- **Wake word only:** ⚠️ Overkill - Use Porcupine instead

**Recommended Path:**
1. ✅ **Add Porcupine wake word detection** (simple, lightweight)
2. ✅ **Keep existing architecture** (medical RAG, dual LLM, custom GUI)
3. ✅ **Consider OVOS later** if you need general-purpose skills

**Integration Effort:**
- Porcupine: **2-4 hours** ✅
- OVOS Hybrid: **Days/weeks** ⚠️
- OVOS Full: **Weeks/months** ❌

---

## References

- **OVOS Documentation:** https://openvoiceos.github.io/
- **Porcupine (Recommended):** https://github.com/Picovoice/porcupine
- **Aura Wake Word Guide:** `docs/WAKE_WORD_DETECTION_GUIDE.md`
- **Current Architecture:** `docs/CURRENT_VS_FHIR_DATA_FLOW.md`

---

## Questions to Consider

1. **Do you need wake word detection?** 
   - If yes → Use Porcupine (simple)
   - If no → Keep current VAD-only system

2. **Do you need general-purpose skills?**
   - If yes → Consider OVOS hybrid (complex)
   - If no → Keep custom medical-focused system

3. **Do you want offline capabilities?**
   - Current: Whisper (online), ElevenLabs (online)
   - OVOS: Vosk (offline), Mimic3 (offline)
   - Trade-off: Quality vs. offline capability

4. **Resource constraints?**
   - Jetson Orin: Can handle OVOS but adds overhead
   - Current system: Already optimized for Jetson

---

**Final Recommendation:** ✅ **Use Porcupine for wake word detection, keep existing Aura architecture**

