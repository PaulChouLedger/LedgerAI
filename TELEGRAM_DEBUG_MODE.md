# Telegram Debug Mode

## Overview

The Telegram bot now includes **comprehensive debug information** showing the internal reasoning of the Adaptive Diagnostic Engine. This helps verify the system is working correctly and troubleshoot issues.

## Enabling Debug Mode

Set in `.env`:
```bash
TELEGRAM_DEBUG=true   # Show debug info
# OR
TELEGRAM_DEBUG=false  # Hide debug info (production mode)
```

## Debug Information Displayed

After each response, the bot sends a **second message** with internal reasoning:

### Example Debug Output

```
🔍 INTERNAL REASONING
========================================
👤 Patient: 35 y/o female
📝 Question: #3
📋 OLDCARTS: OL______ (2/8)
🔁 Clarifications: L:1
💬 Answer: 'lower abdomen'

Pool: Active=5, Reserve=16, Ruled out=7

📊 TOP DIFFERENTIALS:
  1. Acute Gastroenteritis (65%) 📋
  2. Peptic Ulcer Disease (65%) 📋
  3. Severe Constipation (64%) 📋
  4. Irritable Bowel Syndrome (IBS) (64%) 📋
  5. Pelvic Inflammatory Disease (PID) (64%) ⚠️
```

## Debug Fields Explained

| Field | Meaning | Example |
|-------|---------|---------|
| **👤 Patient** | Demographics collected | `35 y/o female` |
| **📝 Question** | Question number in assessment | `#3` |
| **📋 OLDCARTS** | Coverage (O=Onset, L=Location, etc.) | `OL______ (2/8)` = Only Onset & Location covered |
| **🔁 Clarifications** | Number of clarification attempts | `L:1` = 1 location clarification asked |
| **💬 Answer** | Last answer provided | `'lower abdomen'` |
| **Pool** | Active, reserve, ruled out counts | `Active=5, Reserve=16, Ruled out=7` |
| **📊 TOP DIFFERENTIALS** | Top 5 ranked diagnoses with scores | See below |

### Differential Rankings

Each differential shows:
- **Rank**: Position in sorted list
- **Name**: Condition name
- **Score**: Confidence percentage (0-100%)
- **Urgency Emoji**:
  - 🚨 = Emergent (life-threatening)
  - ⚠️ = Urgent (needs prompt attention)
  - 📋 = Routine (non-urgent)

### OLDCARTS Coverage String

```
OLDCARTS = "OL______ (2/8)"
           ↓↓
           ||
           |└─ L = Location (covered)
           └── O = Onset (covered)
           
Missing: D,C,A,R,T,S (6 remaining)
```

## How to Use Debug Mode

### During Development/Testing

```bash
# Enable debug
TELEGRAM_DEBUG=true

# Start bot
cd ~/LedgerAI/aura-control/server
python3 telegram_bot.py
```

**Each message shows:**
1. The bot's question/response
2. *(Separate message)* Internal reasoning with scores and rankings

### In Production

```bash
# Disable debug (cleaner user experience)
TELEGRAM_DEBUG=false
```

**Only the bot's question/response is shown** (no debug info).

## Example Conversation with Debug

### Message 1
**User:** i have abdominal pain

**Bot:** I understand. How old are you?

**Debug:**
```
🔍 INTERNAL REASONING
========================================
📝 Question: #0
📋 OLDCARTS: ________ (0/8)

Pool: Active=5, Reserve=25, Ruled out=0

📊 TOP DIFFERENTIALS:
  1. Acute Cholangitis (50%) 🚨
  2. Acute Appendicitis (60%) ⚠️
  3. Acute Cholecystitis (60%) ⚠️
  4. Acute Pancreatitis (60%) ⚠️
  5. Kidney Stone (60%) ⚠️
```

### Message 2
**User:** 35

**Bot:** Are you male or female?

**Debug:**
```
🔍 INTERNAL REASONING
========================================
👤 Patient: 35 y/o ?
📝 Question: #0
📋 OLDCARTS: ________ (0/8)

Pool: Active=5, Reserve=25, Ruled out=0

📊 TOP DIFFERENTIALS:
  1. Acute Cholangitis (50%) 🚨
  2. Acute Appendicitis (60%) ⚠️
  3. Acute Cholecystitis (60%) ⚠️
  4. Acute Pancreatitis (60%) ⚠️
  5. Kidney Stone (60%) ⚠️
```

### Message 3
**User:** female

**Bot:** When did the pain start?

**Debug:**
```
🔍 INTERNAL REASONING
========================================
👤 Patient: 35 y/o female
📝 Question: #1
📋 OLDCARTS: O_______ (1/8)

Pool: Active=5, Reserve=20, Ruled out=5

📊 TOP DIFFERENTIALS:
  1. Acute Cholangitis (50%) 🚨
  2. Acute Appendicitis (60%) ⚠️
  3. Acute Cholecystitis (60%) ⚠️
  4. Acute Pancreatitis (60%) ⚠️
  5. Kidney Stone (60%) ⚠️
```
*(Note: Female-specific conditions ruled out)*

### Message 4
**User:** 2 hours ago

**Bot:** Where exactly does it hurt?

**Debug:**
```
🔍 INTERNAL REASONING
========================================
👤 Patient: 35 y/o female
📝 Question: #2
📋 OLDCARTS: OL______ (2/8)
💬 Answer: '2 hours ago'

Pool: Active=5, Reserve=20, Ruled out=5

📊 TOP DIFFERENTIALS:
  1. Acute Pancreatitis (62%) ⚠️
  2. Acute Gastroenteritis (61%) 📋
  3. Urinary Tract Infection (60%) ⚠️
  4. Peptic Ulcer Disease (60%) 📋
  5. Kidney Stone (59%) ⚠️
```

### Message 5
**User:** left side

**Bot:** Is it in the upper part or lower part?

**Debug:**
```
🔍 INTERNAL REASONING
========================================
👤 Patient: 35 y/o female
📝 Question: #3
📋 OLDCARTS: OL______ (2/8)
🔁 Clarifications: L:1
💬 Answer: 'left side'

Pool: Active=5, Reserve=15, Ruled out=10

📊 TOP DIFFERENTIALS:
  1. Acute Diverticulitis (65%) ⚠️
  2. Sigmoid Volvulus (63%) 🚨
  3. IBS (62%) 📋
  4. Acute Pancreatitis (60%) ⚠️
  5. Peptic Ulcer (58%) 📋
```
*(Note: Right-sided conditions ruled out)*

### Message 6
**User:** lower

**Bot:** How long does the pain last?

**Debug:**
```
🔍 INTERNAL REASONING
========================================
👤 Patient: 35 y/o female
📝 Question: #4
📋 OLDCARTS: OLD_____ (3/8)
🔁 Clarifications: L:2
💬 Answer: 'lower'

Pool: Active=5, Reserve=15, Ruled out=10

📊 TOP DIFFERENTIALS:
  1. Acute Diverticulitis (68%) ⚠️
  2. Sigmoid Volvulus (65%) 🚨
  3. IBS (64%) 📋
  4. Severe Constipation (63%) 📋
  5. Ruptured Ovarian Cyst (62%) ⚠️
```
*(Note: Moved to DURATION after L hit max clarifications)*

## Benefits of Debug Mode

✅ **Verify scoring**: See if answers are increasing/decreasing the right conditions
✅ **Track clarifications**: Know when hitting the 2-attempt limit
✅ **Monitor OLDCARTS**: Ensure all 8 elements are being covered
✅ **See ruled out**: Understand which conditions were eliminated and why
✅ **Watch rankings change**: See differential evolve with each answer
✅ **Catch infinite loops**: Clarification counter prevents endless location questions

## Toggling Debug Mode

You can toggle debug mode **without restarting the bot**:

```bash
# In .env
TELEGRAM_DEBUG=true

# Restart just the bot (not containers)
pkill -f telegram_bot.py
python3 aura-control/server/telegram_bot.py &
```

## Production Use

For actual users, set `TELEGRAM_DEBUG=false` to hide internal reasoning and provide a cleaner experience.

Debug info is **always** logged to stdout for server-side monitoring, regardless of the Telegram setting.

