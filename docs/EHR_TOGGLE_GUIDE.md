# EHR Integration Toggle Guide

**Quick control over FHIR/SystmOne integration** - Turn it on when ready, keep it off while developing.

---

## 🎛️ The Toggle System

Your Aura system now has a **simple on/off switch** for EHR integration:

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   EHR TOGGLE: OFF ←────────────→ ON                        │
│                                                              │
│   Normal Development         Ready to Test with FHIR        │
│   • No FHIR calls            • Connects to SystmOne         │
│   • Local data only          • Saves to EHR                 │
│   • Fast iteration           • Full integration             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Use the Toggle Script (Easiest)

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI/llm-medical-container

# Turn EHR integration ON
./ehr_toggle.sh on

# Turn EHR integration OFF
./ehr_toggle.sh off

# Check current status
./ehr_toggle.sh status
```

### Option 2: Interactive Menu

```bash
./ehr_toggle.sh

# You'll see:
========================================
   EHR INTEGRATION TOGGLE
========================================

Current Status:
  ○ EHR Integration: DISABLED

What would you like to do?
  1) Enable EHR integration
  2) Disable EHR integration
  3) Show status only
  4) Exit

Enter choice [1-4]:
```

### Option 3: Manual (Edit .env file)

```bash
# Edit the config file
nano llm-medical-container/.env

# Add or change this line:
EHR_INTEGRATION_ENABLED=true   # ON
# or
EHR_INTEGRATION_ENABLED=false  # OFF
```

---

## 📖 Detailed Usage

### Scenario 1: Normal Development (EHR OFF)

**When:** You're developing Aura, testing features, debugging

**Command:**
```bash
cd llm-medical-container
./ehr_toggle.sh off
```

**What happens:**
```
✅ EHR Integration DISABLED

Normal Aura mode:
  • No FHIR calls made
  • Data saved locally only
  • No connection to SystmOne
```

**Your workflow:**
```
1. Run main.py as usual
2. Test patient assessments
3. Data saved to: /app/data/sessions/{session_id}.json
4. No external API calls
5. Fast and simple!
```

---

### Scenario 2: Testing FHIR Integration (EHR ON)

**When:** Ready to test EHR integration with test server

**Command:**
```bash
cd llm-medical-container
./ehr_toggle.sh on
```

**What happens:**
```
✅ EHR Integration ENABLED

Configuration:
  • FHIR calls will be made to SystmOne
  • Data will be saved to EHR
  • Local JSON files still saved (backup)

Server: https://hapi.fhir.org/baseR4
```

**Your workflow:**
```
1. Run main.py as usual
2. Test patient assessments
3. Data saved to:
   - Local: /app/data/sessions/{session_id}.json
   - EHR: SystmOne FHIR API (test server)
4. Can verify in FHIR test server
```

---

### Scenario 3: Check Current Status

**Command:**
```bash
./ehr_toggle.sh status
```

**Output when OFF:**
```
Current Status:
  ○ EHR Integration: DISABLED

  Data flow:
    Patient → Aura Assessment → Local JSON only
```

**Output when ON:**
```
Current Status:
  ● EHR Integration: ENABLED

  FHIR Server: https://hapi.fhir.org/baseR4

  Data flow:
    Patient → Aura Assessment → Local JSON + SystmOne EHR
```

---

## 🔧 How It Works

### The Configuration File

Location: `llm-medical-container/.env`

```bash
# EHR Integration Toggle
EHR_INTEGRATION_ENABLED=false    # ← This line controls everything

# FHIR Server URL (when enabled)
SYSTMONE_FHIR_URL=https://hapi.fhir.org/baseR4

# Production settings (commented out by default)
# SYSTMONE_FHIR_URL=https://api.systmone.nhs.uk/fhir
# NHS_CLIENT_ID=your_client_id
# NHS_CLIENT_SECRET=your_client_secret
```

### What Happens in Code

**When EHR is OFF (`false`):**
```python
# In clinician_mode.py

if EHR_INTEGRATION_ENABLED:  # False, so this block skips
    # All FHIR code is skipped
    pass

# Normal Aura code runs
save_to_local_json()  # ✅ Still happens
# No FHIR calls made
```

**When EHR is ON (`true`):**
```python
# In clinician_mode.py

if EHR_INTEGRATION_ENABLED:  # True, so this runs
    ehr_client.search_patient(nhs_number)      # ✅ Runs
    ehr_client.create_encounter(patient_id)    # ✅ Runs
    ehr_client.create_observation(...)         # ✅ Runs

# Normal Aura code also runs
save_to_local_json()  # ✅ Still happens
```

---

## 🔄 Switching Between Modes

### Development → Testing Workflow

```bash
# 1. Start in development mode (OFF)
cd llm-medical-container
./ehr_toggle.sh off

# 2. Develop and test Aura normally
cd ../aura-control
python main.py
# ... test features, iterate, debug ...

# 3. When ready to test EHR integration
cd ../llm-medical-container
./ehr_toggle.sh on

# 4. Restart Docker container
cd ..
docker-compose restart llm

# 5. Test with EHR
cd aura-control
python main.py
# ... now FHIR calls are made ...

# 6. Back to development
cd ../llm-medical-container
./ehr_toggle.sh off
docker-compose restart llm
```

---

## 📊 Comparison: OFF vs ON

| Feature | EHR OFF (Development) | EHR ON (Testing) |
|---------|----------------------|------------------|
| **FHIR API calls** | ❌ None | ✅ Made to test server |
| **Local JSON files** | ✅ Saved | ✅ Saved (backup) |
| **SystmOne data** | ❌ Not saved | ✅ Saved to EHR |
| **Speed** | ⚡ Fast (no network) | 🐌 Slower (API calls) |
| **Internet required** | ❌ No | ✅ Yes |
| **NHS credentials** | ❌ Not needed | ⚠️ Needed for production |
| **Use case** | Development, debugging | Testing, demo, production |

---

## 🧪 Testing the Toggle

### Test 1: Verify Toggle Works

```bash
# Check initial status
cd llm-medical-container
./ehr_toggle.sh status

# Toggle ON
./ehr_toggle.sh on
./ehr_toggle.sh status   # Should show ENABLED

# Toggle OFF
./ehr_toggle.sh off
./ehr_toggle.sh status   # Should show DISABLED
```

### Test 2: Verify Aura Still Works (OFF)

```bash
# Make sure EHR is OFF
./ehr_toggle.sh off

# Restart container
cd ..
docker-compose restart llm

# Run Aura
cd aura-control
python main.py

# Test a conversation
# Patient: "I have a headache"
# Aura should respond normally, no errors
```

### Test 3: Verify FHIR Integration (ON)

```bash
# Turn EHR ON
cd llm-medical-container
./ehr_toggle.sh on

# Install FHIR dependencies (first time only)
pip install -r requirements_ehr.txt

# Restart container
cd ..
docker-compose restart llm

# Run Aura
cd aura-control
python main.py

# Watch for EHR messages in console:
# [EHR] 🏥 Integration enabled: https://hapi.fhir.org/baseR4
# [EHR] ✅ Found patient: Smith
# [EHR] ✅ Started encounter: 12345
```

---

## 🎯 Best Practices

### 1. Keep EHR OFF During Development

```bash
# At start of work session
./ehr_toggle.sh off

# Develop freely
# - Fast iteration
# - No network dependencies
# - No accidental API calls
```

### 2. Turn ON Only for EHR Testing

```bash
# When specifically testing EHR
./ehr_toggle.sh on

# Test EHR features
# - Verify FHIR calls work
# - Check data in test server
# - Validate integration

# Turn back OFF when done
./ehr_toggle.sh off
```

### 3. Always Check Status Before Testing

```bash
# Before important testing
./ehr_toggle.sh status

# Make sure it's in the mode you expect
```

### 4. Document Your Test Mode

```bash
# Good practice: Add to your test notes
echo "# Test performed with EHR: $(./ehr_toggle.sh status | grep ENABLED)"
```

---

## 🚨 Important Notes

### 1. Restart Required

**After toggling, restart the Docker container:**

```bash
docker-compose restart llm
```

Why? The container reads `.env` file at startup.

### 2. Local Data Always Saved

**EHR ON or OFF, your local data is always saved:**

```
/app/data/sessions/{session_id}.json
```

This is your **backup** and **development data**.

### 3. Test Server vs Production

**Toggle script uses TEST server by default:**
```
https://hapi.fhir.org/baseR4  ← Safe, public, free
```

**For production, manually edit `.env`:**
```bash
SYSTMONE_FHIR_URL=https://api.systmone.nhs.uk/fhir  ← Real NHS
NHS_CLIENT_ID=your_credentials
NHS_CLIENT_SECRET=your_credentials
```

### 4. No Data Loss

Toggling EHR integration **does not delete any data**:
- ✅ Local JSON files remain
- ✅ EHR data stays in SystmOne
- ✅ Safe to switch modes anytime

---

## 🔍 Troubleshooting

### Problem: Toggle doesn't work

**Solution:**
```bash
# Make sure script is executable
chmod +x llm-medical-container/ehr_toggle.sh

# Check .env file exists
ls -la llm-medical-container/.env
```

### Problem: EHR calls still happening after turning OFF

**Solution:**
```bash
# Verify it's really OFF
./ehr_toggle.sh status

# Restart container (required!)
docker-compose restart llm

# Check logs
docker logs aura-llm | grep EHR
# Should NOT see: [EHR] 🏥 Integration enabled
```

### Problem: EHR calls not happening after turning ON

**Solution:**
```bash
# 1. Verify it's ON
./ehr_toggle.sh status

# 2. Check dependencies installed
cd llm-medical-container
pip list | grep fhir

# 3. Restart container
cd ..
docker-compose restart llm

# 4. Check logs for errors
docker logs aura-llm | grep EHR
```

### Problem: Can't connect to FHIR server

**Solution:**
```bash
# Test connection manually
curl https://hapi.fhir.org/baseR4/metadata

# If it fails, check internet connection
ping hapi.fhir.org

# Or use a different test server
# Edit .env:
# SYSTMONE_FHIR_URL=https://server.fire.ly/r4
```

---

## 📝 Quick Reference

### Common Commands

```bash
# Enable EHR
./ehr_toggle.sh on

# Disable EHR
./ehr_toggle.sh off

# Check status
./ehr_toggle.sh status

# Interactive menu
./ehr_toggle.sh

# Apply changes
docker-compose restart llm
```

### File Locations

```
Configuration:     llm-medical-container/.env
Toggle script:     llm-medical-container/ehr_toggle.sh
Local data:        data/sessions/{session_id}.json
FHIR client code:  llm-medical-container/ehr_integration_example.py
```

### Status Indicators

```
OFF:  ○ EHR Integration: DISABLED    (red)
ON:   ● EHR Integration: ENABLED     (green)
```

---

## 🎓 Example Session

```bash
# Morning: Start development work
$ cd llm-medical-container
$ ./ehr_toggle.sh off
✅ EHR Integration DISABLED

$ cd ../aura-control
$ python main.py
# Work on features...

# Afternoon: Test EHR integration
$ cd ../llm-medical-container
$ ./ehr_toggle.sh on
✅ EHR Integration ENABLED

$ cd ..
$ docker-compose restart llm
$ cd aura-control
$ python main.py
# Test with FHIR...

# Evening: Back to development
$ cd ../llm-medical-container
$ ./ehr_toggle.sh off
✅ EHR Integration DISABLED

$ docker-compose restart llm
```

---

## 🌟 Summary

**You now have complete control:**

- 🔴 **OFF** = Normal Aura development (fast, local, no EHR)
- 🟢 **ON** = EHR integration testing (FHIR calls, SystmOne)

**Switch anytime with one command:**
```bash
./ehr_toggle.sh on   # or off
```

**Your development workflow stays smooth, but EHR is ready when you need it!** 🚀

---

**Questions? Issues?**
- Check status: `./ehr_toggle.sh status`
- View logs: `docker logs aura-llm | grep EHR`
- Test connection: `curl https://hapi.fhir.org/baseR4/metadata`

