# Wake Word Model Location Guide

## Where Models Are Stored

### Custom Models (Your Trained Models)

**Location:** `~/LedgerAI/data/models/wake_words/`

This is where you place your trained `.onnx` files.

**Example:**
```
~/LedgerAI/data/models/wake_words/
├── hey_aura_v0.1.onnx  ← Your custom model
└── hey_aura_v0.2.onnx  ← Updated version
```

### Built-in Models (Default Models)

**Location:** openWakeWord's cache directory (managed automatically)

Built-in models like `"hey_mycroft_v0.1"` are downloaded and stored by openWakeWord in its own cache directory (typically `~/.local/share/openwakeword/` or similar). You don't need to manage these.

## How Model Loading Works

The system checks locations in this order:

1. **Custom Directory** (`~/LedgerAI/data/models/wake_words/`) ✅ **CHECKED FIRST**
   - Looks for: `{model_name}.onnx`
   - Example: If `DEFAULT_MODEL = "hey_aura_v0.1"`, it looks for `hey_aura_v0.1.onnx`

2. **Built-in Models** (openWakeWord's cache)
   - Only used if custom model not found
   - Models like `"hey_mycroft_v0.1"` are built-in

## Configuration

### Setting the Model Name

In `aura-control/core/openwakeword_wake_word.py`:

```python
DEFAULT_MODEL = "hey_aura_v0.1"  # Without .onnx extension
```

**Important:**
- ✅ Use model name **without** `.onnx` extension: `"hey_aura_v0.1"`
- ✅ File should be named **with** `.onnx` extension: `hey_aura_v0.1.onnx`
- ✅ The code automatically adds `.onnx` when searching

### File Naming

**Correct:**
- Model name: `"hey_aura_v0.1"`
- File name: `hey_aura_v0.1.onnx`
- Location: `~/LedgerAI/data/models/wake_words/hey_aura_v0.1.onnx`

**Incorrect:**
- Model name: `"hey_aura_v0.1.onnx"` ❌ (don't include extension)
- File name: `hey_aura_v0.1` ❌ (must have .onnx extension)

## Verification

When you start the listener, you should see:

```
[OpenWakeWord] 📁 Custom models directory: .../data/models/wake_words
[OpenWakeWord]    Available models: hey_aura_v0.1.onnx
[OpenWakeWord] 📁 Found custom model: .../hey_aura_v0.1.onnx
[OpenWakeWord] ✅ Custom model loaded: .../hey_aura_v0.1.onnx
```

If you see:
```
[OpenWakeWord] ⚠️  Custom model not found, will try built-in models
```

This means:
- The model file is not in `~/LedgerAI/data/models/wake_words/`
- Or the filename doesn't match (e.g., missing `.onnx` extension)

## Quick Setup

1. **Place your model:**
   ```bash
   mkdir -p ~/LedgerAI/data/models/wake_words
   cp ~/Downloads/hey_aura_v0.1.onnx ~/LedgerAI/data/models/wake_words/
   ```

2. **Update configuration:**
   ```python
   # In openwakeword_wake_word.py
   DEFAULT_MODEL = "hey_aura_v0.1"  # Without .onnx
   ```

3. **Verify:**
   ```bash
   ls -lh ~/LedgerAI/data/models/wake_words/
   # Should show: hey_aura_v0.1.onnx
   ```

## Summary

- ✅ **Custom models go in:** `~/LedgerAI/data/models/wake_words/`
- ✅ **File must have:** `.onnx` extension
- ✅ **Model name in code:** Without `.onnx` extension
- ✅ **System checks custom directory FIRST** before built-in models
- ✅ **Default model** (`hey_mycroft_v0.1`) is built-in, not in custom directory

