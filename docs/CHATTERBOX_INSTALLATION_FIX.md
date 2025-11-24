# Fixing ChatterboxTTS Installation Error

## Important: Installation Location

**Install ChatterboxTTS on the Jetson device, not on your local Mac!**

- ✅ **Jetson**: Python 3.8-3.10 (compatible)
- ❌ **Mac with Python 3.13**: Not compatible (pkuseg build fails)

The installation errors you see on Mac are expected - install directly on the Jetson where it will be used.

## Error Explanation

The error occurs when installing `chatterbox-tts` because one of its dependencies (`pkuseg`) is incompatible with newer Python versions:

**Python 3.13+ errors:**
- `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`
- `fatal error: 'longintrepr.h' file not found`

**Python 3.10-3.12 errors:**
- `ModuleNotFoundError: No module named 'distutils.msvccompiler'`

This happens because `pkuseg` is an older package that hasn't been updated for newer Python versions.

## Python Version Compatibility

**ChatterboxTTS compatibility:**
- ✅ **Python 3.8-3.9**: Full support (recommended)
- ⚠️ **Python 3.10-3.11**: May work with workarounds
- ❌ **Python 3.12+**: Not compatible (pkuseg build fails)

**Note:** Jetson devices typically use Python 3.8-3.10, which should work fine.

## Solutions

### Solution 1: Install setuptools (Recommended)

`setuptools` provides compatibility for `distutils`:

```bash
pip install setuptools
pip install chatterbox-tts
```

### Solution 2: Install pkuseg separately with workaround

If Solution 1 doesn't work:

```bash
# Install setuptools first
pip install setuptools

# Try installing pkuseg with legacy build
pip install pkuseg --no-build-isolation

# Then install chatterbox-tts
pip install chatterbox-tts
```

### Solution 3: Use Python 3.8-3.9 (Recommended for Local Testing)

If you're testing locally (not on Jetson), use Python 3.8 or 3.9:

```bash
# Check available Python versions
python3.9 --version  # or python3.8

# Create virtual environment with compatible Python
python3.9 -m venv aura-env
source aura-env/bin/activate
pip install chatterbox-tts
```

**Note:** On Jetson, the system Python (usually 3.8-3.10) should work fine.

### Solution 4: Skip pkuseg if not needed

If `pkuseg` is only needed for Chinese text processing and you don't need that feature:

```bash
# Install chatterbox-tts without pkuseg
pip install chatterbox-tts --no-deps
pip install torch torchaudio  # Install core dependencies manually
```

Then test if voice cloning works without pkuseg.

## Installation on Jetson (Recommended)

**On your Jetson device (Python 3.10):**

**DO NOT update Ubuntu or Python** - this could break other functionality!

### Fix 1: Install setuptools (Try This First)

```bash
# SSH into Jetson
ssh aura@jetson-ip

# Navigate to project
cd ~/LedgerAI/aura-control

# Install setuptools (provides distutils compatibility)
pip3 install --upgrade setuptools

# Try installing chatterbox-tts
pip3 install chatterbox-tts
```

### Fix 2: Install pkuseg with workaround

If Fix 1 doesn't work:

```bash
# Install setuptools
pip3 install --upgrade setuptools

# Install pkuseg with no build isolation (bypasses distutils check)
pip3 install pkuseg --no-build-isolation

# Then install chatterbox-tts
pip3 install chatterbox-tts
```

### Fix 3: Skip pkuseg (if English-only TTS)

If you only need English TTS and don't need Chinese text processing:

```bash
# Install core dependencies
pip3 install torch torchaudio

# Install chatterbox-tts without pkuseg dependency
pip3 install chatterbox-tts --no-deps

# Test if it works
python3 -c "from chatterbox import ChatterboxTTS; print('✅ ChatterboxTTS installed')"
```

**Note:** This may work if pkuseg is only needed for Chinese text processing.

## Local Mac Testing (Not Recommended)

If you need to test locally on Mac:

1. **Use Python 3.9 (if available):**
   ```bash
   python3.9 -m venv test-env
   source test-env/bin/activate
   pip install chatterbox-tts
   ```

2. **Or skip local testing:**
   - Install directly on Jetson
   - Test the integration there
   - Mac Python 3.13 is not compatible

## Alternative: Use Pre-built Wheel

If building from source fails, check if there's a pre-built wheel:

```bash
pip install --only-binary :all: chatterbox-tts
```

## If All Else Fails

If `pkuseg` continues to cause issues:

1. **Install on Jetson (recommended):**
   - Jetson typically uses Python 3.8-3.10 which is compatible
   - Install directly on the Jetson device where it will be used
   - Local Mac testing may not be necessary

2. **Make chatterbox-tts optional:**
   - The system will fall back to ElevenLabs if ChatterboxTTS isn't available
   - You can still use ElevenLabs as your TTS engine
   - Test the integration on Jetson where Python version is compatible

3. **Use Docker:**
   - Install in a Docker container with Python 3.9
   - Or use a pre-built image with dependencies

4. **Wait for package update:**
   - The `pkuseg` maintainers may release an update
   - Or `chatterbox-tts` may remove the dependency

## Verification

After successful installation, test:

```python
from chatterbox import ChatterboxTTS
model = ChatterboxTTS.from_pretrained()
print("✅ ChatterboxTTS working!")
```

## Important: DO NOT Update System

**⚠️ DO NOT update Ubuntu or Python to fix this!**

**Why not:**
- Updating Ubuntu/Python could break other system functionality
- Your Jetson setup is likely tuned for the current Python version
- Other containers/services may depend on current Python version
- System updates can cause dependency conflicts

**Instead:**
- Use the workarounds above (setuptools, --no-build-isolation)
- Or skip pkuseg if you only need English TTS
- The error is a build-time issue, not a system issue

## Notes

- `pkuseg` is used for Chinese word segmentation
- If you're only using English TTS, this dependency may not be critical
- The error is a build-time issue, not a runtime issue
- **Python 3.10 on Jetson** should work with setuptools workaround
- **DO NOT update system Python** - use workarounds instead
- **Local Mac testing** with Python 3.13 will fail - test on Jetson instead

