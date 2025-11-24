# ChatterboxTTS Installation Steps

## Quick Start (After Removing unsloth-zoo)

Since you've already removed `unsloth-zoo`, you can proceed with installation:

### Option 1: System-Wide Installation (Simpler)

```bash
# Install ChatterboxTTS with required versions
bash setup/scripts/install_chatterbox_without_pkuseg.sh
```

**Note:** This will downgrade PyTorch and other packages. If you have other components that need newer versions, use Option 2.

### Option 2: Virtual Environment (Recommended for Production)

```bash
# Create isolated environment
python3 -m venv ~/chatterbox-env

# Activate it
source ~/chatterbox-env/bin/activate

# Install ChatterboxTTS
bash setup/scripts/install_chatterbox_without_pkuseg.sh

# When using AuraVision, activate the environment first
source ~/chatterbox-env/bin/activate
cd ~/LedgerAI/aura-control
python3 core/main.py
```

## What Gets Installed

The script installs these exact versions required by ChatterboxTTS 0.1.4:

- `torch==2.6.0`
- `torchaudio==2.6.0`
- `numpy>=1.24.0,<1.26.0` (will install 1.25.x)
- `librosa==0.11.0`
- `transformers==4.46.3`
- `diffusers==0.29.0`
- `safetensors==0.5.3`
- `conformer==0.3.2`
- `resemble-perth==1.0.1`
- `s3tokenizer`
- `pykakasi==2.3.0`
- `jaconv`
- `gradio==5.44.1`
- `chatterbox-tts==0.1.4` (without pkuseg)

## After Installation

1. **Test the installation:**
   ```bash
   python3 setup/scripts/test_chatterbox_install.py
   ```

2. **Generate a voice sample (if you want voice cloning):**
   ```bash
   python3 setup/scripts/generate_chatterbox_voice_sample.py
   ```

3. **Enable ChatterboxTTS in the app:**
   - Open Settings dialog
   - Toggle "TTS Engine" to ON (switches to ChatterboxTTS)
   - Optionally enable "Voice Cloning" if you have a voice sample

## Troubleshooting

### If installation fails:

1. **Check for remaining conflicts:**
   ```bash
   pip3 check
   ```

2. **Verify unsloth is removed:**
   ```bash
   pip3 show unsloth-zoo
   # Should show: WARNING: Package(s) not found
   ```

3. **Try manual installation:**
   ```bash
   pip3 install --upgrade setuptools
   pip3 install torch==2.6.0 torchaudio==2.6.0
   pip3 install "numpy>=1.24.0,<1.26.0"
   pip3 install librosa==0.11.0 transformers==4.46.3 diffusers==0.29.0 safetensors==0.5.3
   pip3 install conformer==0.3.2 resemble-perth==1.0.1 s3tokenizer pykakasi==2.3.0 jaconv gradio==5.44.1
   pip3 install chatterbox-tts --no-deps
   ```

### If import fails after installation:

```bash
# Test import
python3 -c "from chatterbox import ChatterboxTTS; print('✅ OK')"

# If it fails, check what's missing
python3 -c "from chatterbox import ChatterboxTTS" 2>&1 | grep -i "missing\|no module"
```

## Next Steps

Once installed and tested:
- See `docs/CHATTERBOX_VOICE_CLONING.md` for voice cloning setup
- See `docs/TTS_LATENCY_COMPARISON.md` for performance details
- See `docs/USING_ELEVENLABS_SAMPLES.md` for using existing voice samples

