# Wake Word Training Data

This directory contains training data for the custom "hey aura" OpenWakeWord model.

## Directory Structure

```
wake_word_training/
├── positive/              # Human "hey aura" recordings (20+ files)
├── negative/              # Other phrases, noise (30+ files)
├── negative_tts/          # TTS-generated "hey aura" (20+ files)
├── formatted/             # Formatted for Colab training
│   ├── positive/         # Formatted positive samples
│   └── negative/         # Formatted negative samples
├── device_config.json    # Saved microphone device preference
├── dataset_manifest.json # Training data manifest
├── COLAB_TRAINING_GUIDE.md  # Step-by-step Colab training guide
└── README.md             # This file
```

## Quick Start

1. **Training data is ready!** The `formatted/` directory contains your data ready for Colab.

2. **Train in Colab:**
   - Open: https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/train_custom_model.ipynb
   - Upload the `formatted/` folder
   - Follow the notebook instructions

3. **See `COLAB_TRAINING_GUIDE.md` for detailed instructions**

## Data Collection

To collect more training data, run:
```bash
python3 train_openwakeword_hey_aura.py --mode collect
```

## Model Installation

After training in Colab, download the `.onnx` model and place it in:
```
../models/wake_words/hey_aura_v0.1.onnx
```

Then update `aura-control/core/openwakeword_wake_word.py`:
```python
DEFAULT_MODEL = "hey_aura_v0.1"
```

