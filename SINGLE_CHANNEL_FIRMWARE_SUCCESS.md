# Single-Channel Firmware Flash Success

## 🎉 **SUCCESS: Firmware Flash Completed!**

The ReSpeaker 4 Mic Array is now running **single-channel firmware**.

```
Before: Available input channels: 6
After:  Available input channels: 1  ✅
```

---

## 📊 **What Changed**

### **Hardware**
- ✅ Flashed from 6-channel to 1-channel firmware
- ✅ Cleaner signal path (no multi-channel interference)
- ✅ Lower latency (less data to process)
- ✅ Simplified audio pipeline

### **Listener**
- ✅ Auto-detects firmware type (1-ch or 6-ch)
- ✅ Handles both firmware versions automatically
- ✅ Filters low-RMS noise (< 0.010 RMS)
- ✅ No permission spam (runtime AGC reset disabled)

---

## 🔧 **Current Configuration**

### **Hardware AGC** (ReSpeaker DSP)
```
Target RMS:  0.03
Max Gain:    20x (26 dB)
Strategy:    Gentle to prevent clipping
Result:      Clean ~0.03 RMS output
```

### **Software AGC** (Listener)
```
Target RMS:  0.20
Max Gain:    10x
Strategy:    Boost hardware output to Whisper optimal
Result:      Consistent 0.20 RMS for Whisper
```

### **Noise Filtering**
```
MIN_SPEECH_RMS:          0.010
VAD_START_THRESHOLD:     0.30
VAD_SILENCE_THRESHOLD:   0.05
SILENCE_TIMEOUT:         0.40s
```

---

## 📋 **How Firmware Flash Happened**

### **Initial Attempts (Failed)**
1. `sudo python dfu.py --download 1_channel_firmware.bin`
   - Got stuck at "entering dfu mode"
   - USB permissions interfered
   - Device didn't complete flash

### **Final Success**
- Device eventually entered DFU mode properly
- Flash completed successfully
- Device rebooted with 1-channel firmware
- Listener detected the change automatically

---

## ✅ **Benefits of Single-Channel Firmware**

### **1. Cleaner Signal Path**
- **Before:** 6 mics → beamforming → phase alignment issues
- **After:** 1 mic → direct signal → no interference ✅

### **2. Lower Latency**
- **Before:** 6 channels × 16kHz = 96k samples/sec
- **After:** 1 channel × 16kHz = 16k samples/sec ✅

### **3. Simpler Processing**
- **Before:** Extract channel 0 from 6-channel array
- **After:** Use mono audio directly ✅

### **4. Better Far-Field**
- **Before:** Phase issues at distance caused destructive interference
- **After:** Single mic, no phase issues ✅

---

## 🚀 **Usage**

### **Start Listener:**
```bash
python3 aura-control/main.py
```

### **Expected Output:**
```
[Aura/listener] 🎙️  Available input channels: 1
[Audio] ✅ Single-Channel Processing (1-ch firmware detected)
[Audio] 🎉 Using single-channel firmware (cleaner signal!)
[Audio] 🔧 Hardware: Gentle AGC → ~0.03 RMS (prevents clipping & drift)
[Audio] 🔧 Software: Boost to 0.2 RMS (max 10.0x, does main work)
[Audio] 🔧 Low-RMS Filter: Skips audio below 0.01 RMS (filters noise)
[Audio] 💡 Hardware tuned by systemd service (boot-time configuration)
```

---

## 🛠️ **Troubleshooting**

### **If VAD Still Freezes After Idle**

The hardware AGC may drift. **Quick fix:**
```bash
# Restart listener (Ctrl+C)
python3 aura-control/main.py
```

Or **reboot system** to reset hardware:
```bash
sudo reboot
```

The systemd service will reconfigure hardware on boot.

### **If AGC Permission Errors Appear**

Runtime AGC reset is **disabled** by default (requires sudo). Configuration happens at:
- **Boot time:** via systemd service
- **Startup:** when listener starts

If you need runtime reset:
1. Set `AGC_ENABLE_RUNTIME_RESET = True` in listener.py
2. Run listener with sudo: `sudo python3 aura-control/main.py` (not recommended)

**Better:** Just restart the listener when drift occurs.

### **If Audio Too Quiet (RMS < 0.01)**

The listener will skip and show:
```
[Audio] ⚠️  RMS too low (0.003100 < 0.010), skipping (likely noise/drift)
[Audio] 💡 AGC may have drifted - restart listener or speak louder
```

**Solutions:**
1. Restart listener
2. Speak louder/closer
3. Reboot system (resets hardware AGC)

---

## 📈 **Performance Expectations**

### **Near Field (1-3 feet):**
```
Hardware RMS:  0.03-0.05
Software Boost: 4-7x
Final RMS:     0.20
Transcription: ✅ Excellent
```

### **Mid Field (4-8 feet):**
```
Hardware RMS:  0.01-0.03
Software Boost: 7-10x
Final RMS:     0.15-0.20
Transcription: ✅ Good
```

### **Far Field (8-16 feet):**
```
Hardware RMS:  0.005-0.01
Software Boost: 10x (max)
Final RMS:     0.05-0.10
Transcription: ⚠️  May struggle (acoustic limits)
```

---

## 🔄 **Reverting to 6-Channel Firmware**

If you want to go back:

```bash
cd ~/usb_4_mic_array
sudo python dfu.py --download 6_channels_firmware.bin
```

The listener will auto-detect and work with either firmware! firmware!

---

## 📚 **Related Files**

### **Modified:**
- `aura-control/listener.py` - Firmware-agnostic processing
- `scripts/tune_respeaker.py` - Updated AGC settings
- `scripts/install_auto_tune.sh` - Systemd service installer
- `scripts/respeaker-tuning.service` - Boot-time configuration

### **Created:**
- `scripts/flash_firmware.sh` - Firmware flash helper (not needed anymore)
- `scripts/rebuild_rag.sh` - Quick RAG rebuild
- `scripts/rebuild_whisper.sh` - Quick Whisper rebuild

---

## 🎯 **Next Steps**

1. ✅ **Firmware flashed** (complete)
2. ⏳ **Test far-field** (8-16 feet)
3. ⏳ **Rebuild RAG container** (for phonetic matching fix)
4. ⏳ **Rebuild Whisper container** (for name guidance)

### **Rebuild Commands:**
```bash
# RAG (fixes phonetic matching bugs)
bash scripts/rebuild_rag.sh

# Whisper (adds name guidance support)
bash scripts/rebuild_whisper.sh
```

---

**Last Updated:** October 9, 2025

**Status:** ✅ Single-channel firmware operational, listener fully compatible


