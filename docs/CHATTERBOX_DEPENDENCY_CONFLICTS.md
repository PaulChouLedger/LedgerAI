# ChatterboxTTS Dependency Conflicts

## Overview

ChatterboxTTS 0.1.4 has strict version requirements that conflict with your existing system packages. This document explains the conflicts and provides solutions.

## Detected Conflicts

Based on your installation attempt, the following conflicts exist:

### 1. PyTorch Versions
- **System has**: `torch==2.8.0`, `torchaudio==2.8.0`
- **ChatterboxTTS requires**: `torch==2.6.0`, `torchaudio==2.6.0`
- **Impact**: ⚠️ **HIGH** - Downgrading PyTorch may break other components

### 2. NumPy Version
- **System has**: `numpy==1.26.0`
- **ChatterboxTTS requires**: `numpy>=1.24.0,<1.26.0` (i.e., 1.24.x or 1.25.x)
- **Impact**: ⚠️ **MEDIUM** - Minor version downgrade

### 3. Diffusers Version
- **System has**: `diffusers==0.35.2`
- **ChatterboxTTS requires**: `diffusers==0.29.0`
- **Impact**: ⚠️ **MEDIUM** - Significant version downgrade

### 4. Safetensors Version
- **System has**: `safetensors==0.6.2`
- **ChatterboxTTS requires**: `safetensors==0.5.3`
- **Impact**: ⚠️ **LOW** - Minor version downgrade

### 5. Transformers Version (CRITICAL CONFLICT)
- **System has**: `transformers==4.45.2`
- **ChatterboxTTS requires**: `transformers==4.46.3`
- **unsloth-zoo requires**: `transformers>=4.51.3,<=4.57.2`
- **Impact**: 🔴 **CRITICAL** - **Incompatible requirements!**

## Solutions

### Option 1: Use Virtual Environment (Recommended)

Create an isolated environment for ChatterboxTTS to avoid conflicts:

```bash
# Create virtual environment
python3 -m venv ~/chatterbox-env

# Activate it
source ~/chatterbox-env/bin/activate

# Install ChatterboxTTS with exact versions
bash setup/scripts/install_chatterbox_without_pkuseg.sh

# When using ChatterboxTTS, activate the environment first
source ~/chatterbox-env/bin/activate
python3 aura-control/core/main.py
```

**Pros:**
- ✅ No conflicts with system packages
- ✅ Other components continue working
- ✅ Easy to remove if needed

**Cons:**
- ⚠️ Need to activate environment before use
- ⚠️ Slightly more disk space

### Option 2: Downgrade System Packages (Risky)

Downgrade packages to match ChatterboxTTS requirements:

```bash
# WARNING: This may break other components!
pip3 install torch==2.6.0 torchaudio==2.6.0
pip3 install "numpy>=1.24.0,<1.26.0"
pip3 install diffusers==0.29.0 safetensors==0.5.3 transformers==4.46.3
```

**Pros:**
- ✅ No environment switching needed
- ✅ Simpler workflow

**Cons:**
- 🔴 **May break unsloth-zoo** (requires transformers>=4.51.3)
- 🔴 May break other ML components expecting newer PyTorch
- 🔴 May require reinstalling other packages

### Option 3: Remove unsloth-zoo (RECOMMENDED - Not Needed for Runtime)

**`unsloth-zoo` is NOT used in the running application** - it's only used for training models in Colab scripts. You can safely remove it:

```bash
# Remove unsloth-zoo (not needed for runtime)
pip3 uninstall unsloth-zoo

# Then install ChatterboxTTS
bash setup/scripts/install_chatterbox_without_pkuseg.sh
```

**Pros:**
- ✅ Resolves transformers conflict
- ✅ No environment needed
- ✅ Safe to remove (not used in runtime)
- ✅ Frees up disk space

**Cons:**
- ⚠️ Still need to downgrade PyTorch (may break other things)
- ⚠️ Can't train models locally (but you can use Colab for that)

**See:** `docs/UNSLOTH_ZOO_NOT_NEEDED.md` for details

### Option 4: Wait for ChatterboxTTS Update

ChatterboxTTS may release a version compatible with newer dependencies. Check:

```bash
pip3 index versions chatterbox-tts
```

**Pros:**
- ✅ No conflicts
- ✅ Uses latest packages

**Cons:**
- ⏳ May take time
- ⏳ May never happen

## Recommended Approach

**For Production Use:**
1. Use **Option 1 (Virtual Environment)** - safest and most reliable
2. Modify `aura-control/core/speaker.py` to activate the environment when using ChatterboxTTS

**For Development/Testing:**
1. **Use Option 3** (remove unsloth-zoo) - it's not needed for runtime
2. If you need unsloth for training, use **Option 1** (virtual environment) for training only

## Checking What Uses Which Packages

To see what packages depend on conflicting versions:

```bash
# Check what requires transformers
pip3 show transformers

# Check what requires torch
pip3 show torch

# List all installed packages
pip3 list | grep -E "(torch|transformers|diffusers|numpy|safetensors)"
```

## Testing After Installation

After resolving conflicts, test ChatterboxTTS:

```bash
python3 setup/scripts/test_chatterbox_install.py
```

If successful, test voice cloning:

```bash
python3 setup/scripts/generate_chatterbox_voice_sample.py
```

## Summary

| Solution | Risk Level | Complexity | Recommended For |
|----------|-----------|------------|-----------------|
| Remove unsloth-zoo + Install | 🟢 Low | Low | **RECOMMENDED** - Runtime use |
| Virtual Environment | 🟢 Low | Medium | Production (if keeping unsloth) |
| Downgrade Packages | 🔴 High | Low | Not recommended |
| Wait for Update | 🟢 Low | None | Future |

**Bottom Line:** Use a virtual environment to avoid breaking your existing system.

