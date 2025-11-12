# LLM Fine-Tuning Directory

This directory contains all scripts and resources needed for fine-tuning the medical chatbot LLM using Unsloth on Jetson devices.

## Directory Structure

```
LLM_tuning/
├── finetune_unsloth.py    # Main fine-tuning script
├── setup_unsloth.sh       # Setup script for dependencies
└── README.md              # This file
```

## Quick Start

1. **Setup dependencies:**
   ```bash
   cd LLM_tuning
   ./setup_unsloth.sh
   ```

2. **Run fine-tuning:**
   ```bash
   cd LLM_tuning
   python3 finetune_unsloth.py
   ```
   
   **Note:** The model will be automatically downloaded from Hugging Face on first use (~1-2GB) and cached in `~/.cache/huggingface/hub/`.

## Files Location

- **Dataset**: `medical_sft_dataset.json` (in this directory)
- **Output models**: `../models/finetuned_medical` (in project root)
- **Documentation**: `FINETUNE_UNSLOTH_GUIDE.md` (in this directory)

## Usage

All paths in the scripts are relative to this directory. The dataset is in the same directory, and output models are saved to `../models/finetuned_medical` in the project root.

For detailed instructions, see: [FINETUNE_UNSLOTH_GUIDE.md](./FINETUNE_UNSLOTH_GUIDE.md)

