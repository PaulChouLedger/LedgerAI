# Setup Scripts

## Scripts Overview

### Complete Installation

- **`install_aura_bootable.sh`** - Complete bootable installation script
  - Sets up Python virtual environment with all requirements
  - Installs jetson-containers
  - Installs and builds XVF3800 USB 4-Mic Array support
  - Configures systemd services for boot startup
  - Sets up Docker and audio permissions
  - Makes Aura start automatically on boot
  
  **Usage:**
  ```bash
  cd /path/to/LedgerAI/setup/scripts
  bash install_aura_bootable.sh
  ```
  
  **What it does:**
  1. Updates system packages and installs dependencies
  2. Creates Python 3.10 virtual environment at `~/aura-env`
  3. Installs all Python requirements from `aura-control/requirements/`
  4. Clones and installs jetson-containers
  5. Clones and builds XVF3800 mic array support (xvf_host)
  6. Configures display settings (disables lock screen)
  7. Installs XVF3800 tuning service (runs on boot)
  8. Creates Aura systemd service (starts on boot)
  9. Configures Docker and audio permissions
  10. Creates necessary data directories

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

- **`disable-keyboard-monitor.service`** - Disable Ubuntu keyboard while Aura runs
  - Monitors `aura.service` and disables Ubuntu on-screen keyboard (onboard, caribou, matchbox-keyboard) while Aura is running
  - Automatically starts/stops with `aura.service`
  - Runs as user aura (no sudo required)
  - Checks every 2 seconds if Aura is running and kills keyboard processes

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

Available presets: `balanced_beam`, `ultra_sensitive`, `far_field`, `near_field`, `hpf_only`, `reset`, `show`

