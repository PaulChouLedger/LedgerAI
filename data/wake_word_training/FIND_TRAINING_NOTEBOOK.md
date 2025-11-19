# Finding the OpenWakeWord Training Notebook

The exact notebook path may vary. Here's how to find it:

## Method 1: Check GitHub Repository

1. Visit: https://github.com/dscripka/openWakeWord
2. Click on the `notebooks/` folder (if it exists)
3. Look for training-related `.ipynb` files
4. Or check the README.md for training instructions

## Method 2: Search the Repository

1. On the GitHub repository page, use the search box
2. Search for: "train" or "training" or ".ipynb"
3. This will show all training-related files

## Method 3: Check Documentation

1. Visit: https://github.com/dscripka/openWakeWord#training-custom-models
2. The README should have a link to training resources
3. Follow the provided links

## Method 4: Clone and Explore

```bash
git clone https://github.com/dscripka/openWakeWord.git
cd openWakeWord
find . -name "*.ipynb" | grep -i train
ls -la notebooks/  # if notebooks folder exists
```

## Method 5: Use Training Script (Alternative)

OpenWakeWord may provide a Python training script instead of a notebook:

```bash
git clone https://github.com/dscripka/openWakeWord.git
cd openWakeWord
python3 train.py --help  # or similar script name
```

## Your Training Data Location

Your formatted training data is ready at:
```
data/wake_word_training/formatted/
├── positive/  (your positive samples)
└── negative/  (your negative samples)
```

Once you find the training notebook/script, upload or point to this `formatted/` directory.

## Alternative: Manual Training Setup

If no notebook is available, you can set up training manually:

1. Install openwakeword training dependencies
2. Use the training API directly (check openWakeWord docs)
3. Or use their training utilities if available

Check the openWakeWord repository for the most current training method.

