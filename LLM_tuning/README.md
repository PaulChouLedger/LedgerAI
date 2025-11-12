# LLM Fine-Tuning Directory

This directory contains all scripts and resources needed for fine-tuning the medical chatbot LLM using Unsloth on Jetson devices.

## Directory Structure

```
LLM_tuning/
├── finetune_unsloth.py      # Main fine-tuning script
├── setup_clean_env.sh        # Setup script for clean virtual environment (RECOMMENDED)
├── setup_unsloth.sh          # Setup script for system-wide installation
├── diagnose_imports.py       # Diagnostic script for troubleshooting
├── medical_sft_dataset.json  # Fine-tuning dataset
└── README.md                 # This file
```

## Quick Start

### Recommended: Clean Virtual Environment (Easiest)

1. **Setup clean virtual environment:**
   ```bash
   cd LLM_tuning
   bash setup_clean_env.sh
   ```

2. **Activate environment and run fine-tuning:**
   ```bash
   source unsloth-env/bin/activate
   python3 finetune_unsloth.py --dataset_path ./medical_sft_dataset.json
   ```

### Alternative: System-wide Installation

1. **Setup dependencies:**
   ```bash
   cd LLM_tuning
   bash setup_unsloth.sh
   ```

2. **Run fine-tuning:**
   ```bash
   python3 finetune_unsloth.py --dataset_path ./medical_sft_dataset.json
   ```

**Note:** The model will be automatically downloaded from Hugging Face on first use (~1-2GB) and cached in `~/.cache/huggingface/hub/`.

## Files Location

- **Dataset**: `medical_sft_dataset.json` (in this directory)
- **Output models**: `../models/finetuned_medical` (in project root)
- **Documentation**: `FINETUNE_UNSLOTH_GUIDE.md` (in this directory)

## Usage

All paths in the scripts are relative to this directory. The dataset is in the same directory, and output models are saved to `../models/finetuned_medical` in the project root.

For detailed instructions, see: [FINETUNE_UNSLOTH_GUIDE.md](./FINETUNE_UNSLOTH_GUIDE.md)

