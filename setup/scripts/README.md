# Setup Scripts

## Scripts Overview

### Microphone Configuration

- **`tune_xvf3800.py`** - Configure XVF3800 USB 4 Mic Array DSP
  - Supports multiple presets for different use cases
  - Can be run manually or via systemd service
  - See [HARDWARE_CONFIGURATION_GUIDE.md](../HARDWARE_CONFIGURATION_GUIDE.md) for full documentation

### Tuning and Testing Scripts

All tuning scripts are configured for the **XVF3800 4-Mic Array** (2 channels):

- **`find_optimal_rms.py`** - Find optimal RMS level for Whisper transcription
- **`record_frequencies_simple.py`** - Simple frequency analyzer (terminal output only)
- **`transcription_tuner.py`** - Real-time transcription tuning with VAD
- **`transcription_tuner_chunks.py`** - Generator-based transcription tuner
- **`test_transcription.py`** - Comprehensive transcription testing
- **`test_jetson_fan_noise.py`** - Measure fan noise impact on microphone

### Systemd Services

- **`xvf3800-tuning.service`** - Auto-configure XVF3800 on boot
  - Installs to `/etc/systemd/system/`
  - Runs `tune_xvf3800.py` automatically on system startup
  - User aura (no sudo required)

### Installation

To set up auto-tuning on boot:

```bash
# Copy service file
sudo cp setup/scripts/xvf3800-tuning.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable xvf3800-tuning.service
sudo systemctl start xvf3800-tuning.service

# Verify it's working
systemctl status xvf3800-tuning.service
```

### Manual Usage

To configure microphone manually:

```bash
python3 setup/scripts/tune_xvf3800.py balanced_beam
```

Available presets: `balanced_beam`, `ultra_sensitive`, `far_field`, `near_field`, `reset`, `show`

