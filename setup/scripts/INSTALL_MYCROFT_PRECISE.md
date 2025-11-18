# Install Mycroft Precise (Standalone)

Quick installation commands for Mycroft Precise wake word detection without running the full installation script.

## Option 1: Run the Standalone Script (Recommended)

```bash
cd ~/LedgerAI/setup/scripts
bash install_mycroft_precise.sh
```

## Option 2: Manual Installation Commands

If you prefer to run commands manually:

### 1. Activate Virtual Environment

```bash
source ~/aura-env/bin/activate
```

### 2. Install Python Packages

```bash
pip install precise-runner precise-engine
```

### 3. Download precise-engine Binary

```bash
# Create directory
mkdir -p ~/.mycroft/precise/precise-engine

# Download and extract
cd /tmp
wget https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz
tar xzf precise-all_0.3.0_aarch64.tar.gz

# Copy to installation directory
cp -r precise/* ~/.mycroft/precise/precise-engine/
chmod +x ~/.mycroft/precise/precise-engine/precise-engine

# Cleanup
rm -rf precise precise-all_0.3.0_aarch64.tar.gz
```

### 4. Download Wake Word Model

```bash
# Create directory
mkdir -p ~/precise-models

# Download model
wget -O ~/precise-models/hey-mycroft.pb https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb

# Create symlink for compatibility
ln -s ~/precise-models/hey-mycroft.pb ~/hey-mycroft.pb
```

### 5. Verify Installation

```bash
# Test Python import
python3 -c "from precise_runner import PreciseEngine, PreciseRunner; print('✅ Python package OK')"

# Check binary
test -x ~/.mycroft/precise/precise-engine/precise-engine && echo "✅ Binary OK" || echo "❌ Binary missing"

# Check model
test -f ~/precise-models/hey-mycroft.pb && echo "✅ Model OK" || echo "❌ Model missing"
```

## Installation Locations

After installation, Mycroft Precise will be located at:

- **Python package**: Installed in virtual environment (`~/aura-env/`)
- **Binary**: `~/.mycroft/precise/precise-engine/precise-engine`
- **Model**: `~/precise-models/hey-mycroft.pb` (with symlink at `~/hey-mycroft.pb`)

## Enable Wake Word Detection

After installation, enable wake word detection in Aura:

1. Open Settings dialog
2. Go to **AI Model Settings**
3. Toggle **Wake Word Detection** to **ON**
4. Adjust sensitivity if needed

## Troubleshooting

### Python package not importable
```bash
# Reinstall in virtual environment
source ~/aura-env/bin/activate
pip install --upgrade precise-runner precise-engine
```

### Binary not found
```bash
# Re-download binary
cd /tmp
wget https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-all_0.3.0_aarch64.tar.gz
tar xzf precise-all_0.3.0_aarch64.tar.gz
mkdir -p ~/.mycroft/precise/precise-engine
cp -r precise/* ~/.mycroft/precise/precise-engine/
chmod +x ~/.mycroft/precise/precise-engine/precise-engine
rm -rf precise precise-all_0.3.0_aarch64.tar.gz
```

### Model not found
```bash
# Re-download model
mkdir -p ~/precise-models
wget -O ~/precise-models/hey-mycroft.pb https://github.com/MycroftAI/precise-data/raw/models/hey-mycroft.pb
ln -sf ~/precise-models/hey-mycroft.pb ~/hey-mycroft.pb
```

