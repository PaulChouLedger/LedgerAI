# OpenWakeWord Colab Training Guide

## Quick Start

Your training data is ready! Follow these steps to train your "hey aura" model in Google Colab.

## Step 1: Open the Training Notebook

The correct notebook names in the OpenWakeWord repository are:

**Option A (Recommended):** Automatic Model Training Notebook
- https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

**Option B:** Training Models Notebook (Alternative)
- https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb

**Option C:** Check the Repository Directly
- Visit: https://github.com/dscripka/openWakeWord
- Navigate to the `notebooks/` folder
- Look for `.ipynb` files with "train" or "training" in the name
- Open the notebook in Colab by clicking the Colab badge or using the Colab link

## Step 2: Prepare Your Training Data

Your training data is located at:
```
data/wake_word_training/formatted/
├── positive/  (20+ WAV files)
└── negative/  (50+ WAV files)
```

### Option A: Upload via Colab UI
1. In Colab, click the folder icon (📁) in the left sidebar
2. Click "Upload" button
3. Upload the entire `formatted` folder, OR
4. Upload `positive` and `negative` folders separately
5. Wait for upload to complete

### Option B: Upload via Code (Recommended)
In a Colab cell, run:
```python
from google.colab import files
import zipfile
import os

# Create a zip file of your training data
# (Do this on your local machine first, then upload the zip)
# Or use the Colab file browser to upload the formatted folder
```

### Option C: Use Google Drive
1. Upload `formatted` folder to Google Drive
2. In Colab, mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Copy data to Colab workspace:
```python
!cp -r /content/drive/MyDrive/path/to/formatted /content/
```

## Step 3: Configure Training

In the Colab notebook, you'll need to:

1. **Set your wake phrase:**
   ```python
   wake_phrase = "hey aura"
   ```

2. **Point to your data:**
   ```python
   positive_dir = "/content/formatted/positive"
   negative_dir = "/content/formatted/negative"
   ```

3. **Handle "5 Sources" Requirement (if asked):**
   
   The notebook may ask for "5 sources" - this means it wants 5 different audio sources or variations. You have options:
   
   **Option A:** Create 5 source subdirectories:
   ```python
   import os
   import shutil
   import glob
   
   # Create 5 source folders
   for i in range(1, 6):
       os.makedirs(f"/content/formatted/positive/source{i}", exist_ok=True)
   
   # Distribute your samples (4 samples per source from your 20 total)
   files = sorted(glob.glob("/content/formatted/positive/*.wav"))
   for i, file in enumerate(files):
       source_num = (i % 5) + 1
       shutil.move(file, f"/content/formatted/positive/source{source_num}/")
   ```
   
   **Option B:** If the notebook allows, use all samples as one source - your 20 samples should work fine.
   
   See `NOTEBOOK_5_SOURCES_GUIDE.md` for detailed instructions.

4. **Configure training parameters** (optional):
   - Number of epochs (default: 50-100)
   - Learning rate (default: usually auto)
   - Batch size (default: usually auto)

## Step 4: Train the Model

1. Run all cells in the notebook
2. Training typically takes 10-30 minutes
3. The notebook will:
   - Load and augment your data
   - Train the model
   - Evaluate performance
   - Export the model as `.onnx` file

## Step 5: Download the Trained Model

1. After training completes, download the `.onnx` file:
   ```python
   from google.colab import files
   files.download('hey_aura_v0.1.onnx')
   ```

2. Or use the Colab file browser:
   - Right-click the `.onnx` file
   - Select "Download"

## Step 6: Install the Model

1. Copy the downloaded model to your system:
   ```bash
   # On your Jetson/Ubuntu system
   cp ~/Downloads/hey_aura_v0.1.onnx ~/LedgerAI/data/models/wake_words/
   ```

2. Verify it's in place:
   ```bash
   ls -lh ~/LedgerAI/data/models/wake_words/
   ```

## Step 7: Update Configuration

Edit `aura-control/core/openwakeword_wake_word.py`:

```python
# Change this line:
DEFAULT_MODEL = "hey_aura_v0.1"  # or use full path
```

Or the system will auto-detect it if placed in:
```
data/models/wake_words/hey_aura_v0.1.onnx
```

## Step 8: Test the Model

1. Start the listener:
   ```bash
   cd ~/LedgerAI/aura-control/core
   python3 listener.py
   ```

2. Test wake word detection:
   - Say "hey aura" - should trigger ✅
   - Play TTS saying "hey aura" - should NOT trigger ❌
   - Say similar phrases - should NOT trigger ❌

3. Adjust threshold if needed (in Settings or `state.py`)

## Troubleshooting

### Training fails in Colab
- Check that all WAV files are valid (16kHz, mono)
- Ensure you have enough samples (20+ positive, 50+ negative)
- Try reducing batch size or number of epochs

### Model doesn't work after installation
- Verify model file is `.onnx` format
- Check file permissions: `chmod 644 hey_aura_v0.1.onnx`
- Check model path in `openwakeword_wake_word.py`
- Review logs for loading errors

### Model triggers on TTS
- You may need more TTS negative samples
- Retrain with additional TTS echo samples
- Increase threshold in settings

## Training Data Summary

- **Positive samples:** Human speech saying "hey aura"
- **Negative samples:** Other phrases + TTS echo samples
- **TTS echo samples:** Recorded TTS playback (critical for echo rejection)

## Additional Resources

- OpenWakeWord GitHub: https://github.com/dscripka/openWakeWord
- Training Documentation: https://github.com/dscripka/openWakeWord#training-custom-models
- Issues/Support: https://github.com/dscripka/openWakeWord/issues

