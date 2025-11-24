# Fixing ChatterboxTTS Installation Error

## Error Explanation

The error occurs when installing `chatterbox-tts` because one of its dependencies (`pkuseg`) tries to use `distutils.msvccompiler`, which was removed in Python 3.12+ and deprecated in Python 3.10+.

**Error:**
```
ModuleNotFoundError: No module named 'distutils.msvccompiler'
```

This happens because `pkuseg` is an older package that hasn't been updated for newer Python versions.

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

### Solution 3: Use Python 3.9 (if available)

If you have Python 3.9 available, it still has full `distutils` support:

```bash
# Create virtual environment with Python 3.9
python3.9 -m venv aura-env
source aura-env/bin/activate
pip install chatterbox-tts
```

### Solution 4: Skip pkuseg if not needed

If `pkuseg` is only needed for Chinese text processing and you don't need that feature:

```bash
# Install chatterbox-tts without pkuseg
pip install chatterbox-tts --no-deps
pip install torch torchaudio  # Install core dependencies manually
```

Then test if voice cloning works without pkuseg.

## Step-by-Step Fix (Recommended)

1. **Install setuptools:**
   ```bash
   pip install --upgrade setuptools
   ```

2. **Try installing chatterbox-tts:**
   ```bash
   pip install chatterbox-tts
   ```

3. **If it still fails, try with legacy build:**
   ```bash
   pip install pkuseg --no-build-isolation
   pip install chatterbox-tts
   ```

4. **Verify installation:**
   ```bash
   python3 -c "from chatterbox import ChatterboxTTS; print('✅ ChatterboxTTS installed successfully')"
   ```

## Alternative: Use Pre-built Wheel

If building from source fails, check if there's a pre-built wheel:

```bash
pip install --only-binary :all: chatterbox-tts
```

## If All Else Fails

If `pkuseg` continues to cause issues and you don't need Chinese text processing:

1. **Make chatterbox-tts optional:**
   - The system will fall back to ElevenLabs if ChatterboxTTS isn't available
   - You can still use ElevenLabs as your TTS engine

2. **Wait for package update:**
   - The `pkuseg` maintainers may release an update
   - Or `chatterbox-tts` may remove the dependency

3. **Use Docker:**
   - Install in a Docker container with Python 3.9
   - Or use a pre-built image with dependencies

## Verification

After successful installation, test:

```python
from chatterbox import ChatterboxTTS
model = ChatterboxTTS.from_pretrained()
print("✅ ChatterboxTTS working!")
```

## Notes

- `pkuseg` is used for Chinese word segmentation
- If you're only using English TTS, this dependency may not be critical
- The error is a build-time issue, not a runtime issue
- Once installed, it should work fine

