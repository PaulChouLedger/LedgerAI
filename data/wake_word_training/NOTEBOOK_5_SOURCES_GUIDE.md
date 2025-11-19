# Handling "5 Sources" Requirement in OpenWakeWord Notebook

The notebook is asking for "5 sources" - this typically means it wants **5 different audio sources or variations** of your wake phrase.

## What "5 Sources" Means

The notebook wants diversity in your training data. This could mean:
- **5 different speakers** saying "hey aura" (ideal)
- **5 different recording sessions** (different times/environments)
- **5 different variations** of how "hey aura" is spoken
- **5 groups** of samples organized by source

## Solution Options

### Option 1: Use Your Existing Samples (Recommended)

If you have 20+ positive samples, you can:

1. **Organize samples into 5 groups:**
   - Group 1: Samples 1-4
   - Group 2: Samples 5-8
   - Group 3: Samples 9-12
   - Group 4: Samples 13-16
   - Group 5: Samples 17-20

2. **Or select 5 representative samples:**
   - Pick 5 diverse samples (different tones, speeds, volumes)
   - Use these as your "sources"
   - The notebook will augment them to create more training data

### Option 2: Create 5 Source Folders

Reorganize your data into 5 source folders:

```bash
# In Colab, after uploading your data:
!mkdir -p /content/formatted/positive/source1
!mkdir -p /content/formatted/positive/source2
!mkdir -p /content/formatted/positive/source3
!mkdir -p /content/formatted/positive/source4
!mkdir -p /content/formatted/positive/source5

# Distribute your samples across the 5 sources
# (You can split your 20 samples: 4 samples per source)
```

Then point the notebook to:
```python
positive_dir = "/content/formatted/positive"
# It will find the 5 source subdirectories
```

### Option 3: Use All Samples as One Source

If the notebook allows it, you can:

1. **Put all samples in one folder:**
   ```python
   positive_dir = "/content/formatted/positive"
   # All 20 samples in one folder
   ```

2. **Tell the notebook you have 1 source with 20 samples:**
   - Some notebooks allow this
   - They'll use data augmentation to create diversity

### Option 4: Record More Samples (If Needed)

If you need actual 5 different sources:

1. **Record from 5 different people** (if possible)
2. **Or record 5 different sessions** yourself:
   - Session 1: Morning, quiet room
   - Session 2: Afternoon, normal room
   - Session 3: Evening, different distance
   - Session 4: Different speaking style
   - Session 5: Different volume/tone

## What the Notebook Actually Needs

Check what the notebook cell is asking for:

1. **If it asks for "5 source folders":**
   - Create 5 subdirectories in your positive folder
   - Distribute samples across them

2. **If it asks for "5 source files":**
   - Select 5 representative WAV files
   - The notebook will augment them

3. **If it asks for "5 speakers":**
   - You can use the same speaker but different recordings
   - Or organize by recording session

## Quick Fix: Use Your Existing Data

**Simplest approach:** Your 20 samples should work fine. Try:

```python
# In the notebook cell, point to your data:
positive_dir = "/content/formatted/positive"
negative_dir = "/content/formatted/negative"

# If it still asks for 5 sources, create them:
import os
import shutil

# Create 5 source folders
for i in range(1, 6):
    os.makedirs(f"/content/formatted/positive/source{i}", exist_ok=True)

# Distribute your 20 samples (4 per source)
import glob
files = sorted(glob.glob("/content/formatted/positive/*.wav"))
for i, file in enumerate(files):
    source_num = (i % 5) + 1
    shutil.move(file, f"/content/formatted/positive/source{source_num}/")
```

## Check the Notebook Instructions

Look at the notebook cell that's asking for "5 sources" - it should have instructions or comments explaining what it needs. The exact requirement may vary depending on which notebook you're using.

## Your Current Data

You have:
- ✅ 20 positive samples (should be enough)
- ✅ 50 negative samples (plenty)
- ✅ TTS echo samples (critical for echo rejection)

This should be sufficient for training. The "5 sources" requirement is likely just for organization/augmentation purposes.

