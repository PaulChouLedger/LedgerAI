#!/bin/bash
# Aura Bootable Installation Script
# Installs Aura with virtual environment, jetson-containers, and XVF3800 mic support
# Usage: bash install_aura_bootable.sh

set -e  # Exit on error

echo "=========================================="
echo "  Aura Bootable Installation Script"
echo "=========================================="
echo ""

# Configuration
AURA_USER="${SUDO_USER:-$USER}"
AURA_HOME="/home/$AURA_USER"
LEDGERAI_DIR="$AURA_HOME/LedgerAI"
VENV_DIR="$AURA_HOME/aura-env"
JETSON_CONTAINERS_DIR="$AURA_HOME/jetson-containers"
XVF3800_REPO_DIR="$AURA_HOME/reSpeaker_XVF3800_USB_4MIC_ARRAY"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then 
    print_error "Please run this script as a regular user (not root). It will prompt for sudo when needed."
    exit 1
fi

# Check if LedgerAI directory exists
if [ ! -d "$LEDGERAI_DIR" ]; then
    print_error "LedgerAI directory not found at $LEDGERAI_DIR"
    print_info "Please clone or copy LedgerAI to $LEDGERAI_DIR first"
    exit 1
fi

print_step "Starting Aura installation..."
print_info "User: $AURA_USER"
print_info "LedgerAI: $LEDGERAI_DIR"
print_info "Virtual Env: $VENV_DIR"
echo ""

# ============================================================================
# Step 1: Update system packages
# ============================================================================
print_step "1. Updating system packages..."

# Detect Python version
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    PYTHON_VENV_PKG="python3.10-venv"
    print_info "Using Python 3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    PYTHON_VENV_PKG="python3.9-venv"
    print_info "Using Python 3.9"
elif command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
    PYTHON_VENV_PKG="python3.8-venv"
    print_info "Using Python 3.8"
else
    PYTHON_CMD="python3"
    PYTHON_VENV_PKG="python3-venv"
    print_info "Using default Python 3"
fi

sudo apt update

# Install essential packages (required)
print_info "Installing essential packages..."
sudo apt install -y \
    nano \
    git \
    "$PYTHON_CMD" \
    "$PYTHON_VENV_PKG" \
    python3-pip \
    build-essential \
    cmake \
    pkg-config \
    libusb-1.0-0-dev \
    unclutter \
    xdotool \
    wmctrl \
    x11-xserver-utils \
    alsa-utils \
    libasound2-dev \
    pulseaudio \
    qtbase5-dev \
    qttools5-dev \
    qttools5-dev-tools \
    python3-pyqt5 \
    python3-sip-dev

# Install audio packages (try multiple variants for different systems)
print_info "Installing audio packages..."
AUDIO_PACKAGES=""
PORT_AUDIO_AVAILABLE=false

# Check for available portaudio packages
if apt-cache show portaudio19-dev > /dev/null 2>&1; then
    AUDIO_PACKAGES="$AUDIO_PACKAGES portaudio19-dev"
    PORT_AUDIO_AVAILABLE=true
fi
if apt-cache show libportaudio2 > /dev/null 2>&1; then
    AUDIO_PACKAGES="$AUDIO_PACKAGES libportaudio2"
    PORT_AUDIO_AVAILABLE=true
fi
if apt-cache show libportaudio-dev > /dev/null 2>&1; then
    AUDIO_PACKAGES="$AUDIO_PACKAGES libportaudio-dev"
    PORT_AUDIO_AVAILABLE=true
fi

if [ -n "$AUDIO_PACKAGES" ]; then
    if sudo apt install -y $AUDIO_PACKAGES; then
        print_info "PortAudio packages installed successfully"
    else
        print_info "⚠️  Some PortAudio packages failed to install, continuing anyway..."
        print_info "   sounddevice may work with ALSA/PulseAudio if PortAudio isn't available"
        PORT_AUDIO_AVAILABLE=false
    fi
else
    print_info "⚠️  PortAudio packages not found in repositories"
    print_info "   sounddevice will use ALSA/PulseAudio instead (should work on Jetson)"
    PORT_AUDIO_AVAILABLE=false
fi

echo ""

# ============================================================================
# Step 2: Create and configure virtual environment
# ============================================================================
print_step "2. Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment already exists at $VENV_DIR"
    print_info "Removing old virtual environment..."
    rm -rf "$VENV_DIR"
fi

"$PYTHON_CMD" -m venv "$VENV_DIR"
print_info "Virtual environment created at $VENV_DIR using $PYTHON_CMD"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Auto-activate on login
if ! grep -q "source $VENV_DIR/bin/activate" "$AURA_HOME/.bashrc"; then
    echo "" >> "$AURA_HOME/.bashrc"
    echo "# Auto-activate Aura virtual environment" >> "$AURA_HOME/.bashrc"
    echo "source \"$VENV_DIR/bin/activate\"" >> "$AURA_HOME/.bashrc"
    print_info "Added auto-activation to .bashrc"
else
    print_info "Virtual environment auto-activation already configured"
fi

echo ""

# ============================================================================
# Step 3: Install Python requirements
# ============================================================================
print_step "3. Installing Python requirements..."

# Ensure we're in the virtual environment
source "$VENV_DIR/bin/activate"

# Install core requirements
if [ -f "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" ]; then
    print_info "Installing core requirements..."
    
    # Note: sounddevice uses portaudio via ctypes, so it needs the shared library
    # On Jetson systems, ALSA/PulseAudio may be sufficient
    if [ "$PORT_AUDIO_AVAILABLE" = false ]; then
        print_info "Note: PortAudio system packages not available"
        print_info "      sounddevice will attempt to use ALSA/PulseAudio"
        print_info "      If audio doesn't work, you may need to build portaudio from source"
    fi
    
    # Install PyQt5 separately with better error handling
    # First, try to install all requirements
    print_info "Installing all requirements..."
    if pip install -r "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" 2>&1 | tee /tmp/pip_install.log; then
        print_info "✅ All requirements installed successfully"
    else
        PIP_ERROR=$(cat /tmp/pip_install.log)
        if echo "$PIP_ERROR" | grep -q "PyQt5\|pyqt5"; then
            print_info "⚠️  PyQt5 installation failed, trying alternative approach..."
            
            # Check if system PyQt5 is available and can be used
            if python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import PyQt5" 2>/dev/null; then
                print_info "System PyQt5 is available, will use it"
                # Create requirements without PyQt5
                TEMP_REQUIREMENTS="/tmp/requirements_no_pyqt5.txt"
                grep -v "^PyQt5" "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" > "$TEMP_REQUIREMENTS" || true
                pip install -r "$TEMP_REQUIREMENTS"
                rm -f "$TEMP_REQUIREMENTS"
                
                # Make system PyQt5 available in venv by creating a symlink or using --system-site-packages
                print_info "Note: Using system PyQt5. If import fails, you may need to:"
                print_info "      Recreate venv with: python3 -m venv --system-site-packages $VENV_DIR"
            else
                print_info "Attempting to install PyQt5 with Qt5 dev tools..."
                # Check if qmake is available
                if command -v qmake > /dev/null 2>&1; then
                    print_info "qmake found: $(which qmake)"
                    QMAKE_PATH=$(which qmake)
                else
                    print_info "qmake not in PATH, searching..."
                    if [ -f "/usr/bin/qmake" ]; then
                        QMAKE_PATH="/usr/bin/qmake"
                        export PATH="/usr/bin:$PATH"
                    elif [ -f "/usr/lib/qt5/bin/qmake" ]; then
                        QMAKE_PATH="/usr/lib/qt5/bin/qmake"
                        export PATH="/usr/lib/qt5/bin:$PATH"
                    fi
                fi
                
                # Try installing PyQt5 with --no-build-isolation (often works better)
                if pip install PyQt5 --no-build-isolation; then
                    print_info "✅ PyQt5 installed successfully with --no-build-isolation"
                elif [ -n "$QMAKE_PATH" ]; then
                    print_info "Trying PyQt5 install with explicit qmake path..."
                    export QMAKE="$QMAKE_PATH"
                    pip install PyQt5 --no-build-isolation || {
                        print_error "PyQt5 installation failed"
                        print_info "Trying to install remaining packages without PyQt5..."
                        TEMP_REQUIREMENTS="/tmp/requirements_no_pyqt5.txt"
                        grep -v "^PyQt5" "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" > "$TEMP_REQUIREMENTS" || true
                        pip install -r "$TEMP_REQUIREMENTS"
                        rm -f "$TEMP_REQUIREMENTS"
                        print_info "⚠️  PyQt5 not installed. GUI may not work."
                    }
                else
                    print_error "qmake not found. PyQt5 cannot be built."
                    print_info "Trying to install remaining packages without PyQt5..."
                    TEMP_REQUIREMENTS="/tmp/requirements_no_pyqt5.txt"
                    grep -v "^PyQt5" "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" > "$TEMP_REQUIREMENTS" || true
                    pip install -r "$TEMP_REQUIREMENTS"
                    rm -f "$TEMP_REQUIREMENTS"
                    print_info "⚠️  PyQt5 not installed. GUI may not work."
                    print_info "   Install Qt5 dev tools: sudo apt install qtbase5-dev qttools5-dev"
                fi
            fi
        else
            print_error "Installation failed for other reasons. Check logs above."
        fi
        rm -f /tmp/pip_install.log
    fi
else
    print_error "Core requirements file not found!"
    exit 1
fi

# Install upload requirements (optional)
if [ -f "$LEDGERAI_DIR/aura-control/requirements/requirements_upload.txt" ]; then
    print_info "Installing upload requirements..."
    pip install -r "$LEDGERAI_DIR/aura-control/requirements/requirements_upload.txt"
fi

# Install Google Drive requirements (optional)
if [ -f "$LEDGERAI_DIR/aura-control/requirements/requirements_gdrive.txt" ]; then
    print_info "Installing Google Drive requirements..."
    pip install -r "$LEDGERAI_DIR/aura-control/requirements/requirements_gdrive.txt"
fi

echo ""

# ============================================================================
# Step 4: Install jetson-containers
# ============================================================================
print_step "4. Installing jetson-containers..."

if [ -d "$JETSON_CONTAINERS_DIR" ]; then
    print_info "jetson-containers already exists, updating..."
    cd "$JETSON_CONTAINERS_DIR"
    git pull
else
    print_info "Cloning jetson-containers repository..."
    cd "$AURA_HOME"
    git clone https://github.com/dusty-nv/jetson-containers.git
    cd "$JETSON_CONTAINERS_DIR"
fi

# Install jetson-containers tools
print_info "Installing jetson-containers tools..."
bash "$JETSON_CONTAINERS_DIR/install.sh"

# Add to PATH if not already there
if ! grep -q "jetson-containers" "$AURA_HOME/.bashrc"; then
    echo "" >> "$AURA_HOME/.bashrc"
    echo "# jetson-containers" >> "$AURA_HOME/.bashrc"
    echo "export PATH=\"\$PATH:$JETSON_CONTAINERS_DIR\"" >> "$AURA_HOME/.bashrc"
    print_info "Added jetson-containers to PATH"
fi

echo ""

# ============================================================================
# Step 5: Install XVF3800 USB 4-Mic Array support
# ============================================================================
print_step "5. Installing XVF3800 USB 4-Mic Array support..."

if [ -d "$XVF3800_REPO_DIR" ]; then
    print_info "XVF3800 repository already exists, updating..."
    cd "$XVF3800_REPO_DIR"
    git pull || print_info "Git pull failed, continuing with existing code..."
else
    print_info "Cloning XVF3800 repository..."
    cd "$AURA_HOME"
    git clone https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY.git
    cd "$XVF3800_REPO_DIR"
fi

# Build xvf_host for Jetson
print_info "Building xvf_host for Jetson..."
if [ -d "$XVF3800_REPO_DIR/host_control/jetson" ]; then
    cd "$XVF3800_REPO_DIR/host_control/jetson"
    
    # Check if already built
    if [ -f "xvf_host" ]; then
        print_info "xvf_host already built, skipping..."
    else
        print_info "Compiling xvf_host..."
        make clean || true
        
        # Try to build
        if make; then
            if [ -f "xvf_host" ]; then
                print_info "xvf_host built successfully"
                # Make it executable
                chmod +x xvf_host
            else
                print_error "Build completed but xvf_host not found"
            fi
        else
            print_error "Failed to build xvf_host"
            print_info "You may need to install additional build dependencies:"
            print_info "  sudo apt install libusb-1.0-0-dev libasound2-dev"
        fi
    fi
else
    print_error "Jetson build directory not found!"
    print_info "The repository structure may have changed"
    print_info "Expected path: $XVF3800_REPO_DIR/host_control/jetson"
fi

echo ""

# ============================================================================
# Step 6: Configure display settings
# ============================================================================
print_step "6. Configuring display settings..."

# Disable password prompt when waking from sleep
sudo -u "$AURA_USER" gsettings set org.gnome.desktop.screensaver lock-enabled false || true

# Disable lock screen after suspend
sudo -u "$AURA_USER" gsettings set org.gnome.desktop.lockdown disable-lock-screen true || true

print_info "Display settings configured"

echo ""

# ============================================================================
# Step 7: Install XVF3800 tuning service
# ============================================================================
print_step "7. Installing XVF3800 tuning service..."

SERVICE_FILE="$LEDGERAI_DIR/setup/scripts/xvf3800-tuning.service"
SYSTEMD_SERVICE="/etc/systemd/system/xvf3800-tuning.service"

if [ -f "$SERVICE_FILE" ]; then
    # Update service file with correct paths
    sudo cp "$SERVICE_FILE" "$SYSTEMD_SERVICE"
    
    # Update paths in service file if needed
    sudo sed -i "s|/home/aura|$AURA_HOME|g" "$SYSTEMD_SERVICE"
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable and start service
    sudo systemctl enable xvf3800-tuning.service
    sudo systemctl start xvf3800-tuning.service
    
    print_info "XVF3800 tuning service installed and enabled"
    print_info "Service will configure microphone on boot"
else
    print_error "XVF3800 service file not found at $SERVICE_FILE"
fi

echo ""

# ============================================================================
# Step 8: Create Aura systemd service for boot
# ============================================================================
print_step "8. Creating Aura systemd service for boot startup..."

AURA_SERVICE_FILE="/tmp/aura.service"
AURA_UID=$(id -u "$AURA_USER")
cat > "$AURA_SERVICE_FILE" << EOF
[Unit]
Description=Aura Voice Assistant
After=network.target docker.service xvf3800-tuning.service
Wants=docker.service xvf3800-tuning.service
Requires=docker.service

[Service]
Type=simple
User=$AURA_USER
Group=$AURA_USER
WorkingDirectory=$LEDGERAI_DIR/aura-control/core
Environment="DISPLAY=:0"
Environment="HOME=$AURA_HOME"
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="XDG_RUNTIME_DIR=/run/user/$AURA_UID"
ExecStartPre=/bin/sleep 10
ExecStart=$VENV_DIR/bin/python3 $LEDGERAI_DIR/aura-control/core/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOF

# Copy service file to systemd
sudo cp "$AURA_SERVICE_FILE" /etc/systemd/system/aura.service
sudo systemctl daemon-reload

# Enable service (but don't start yet - user may want to test first)
sudo systemctl enable aura.service

print_info "Aura systemd service created and enabled"
print_info "Service will start Aura automatically on boot"
print_info "To start now: sudo systemctl start aura.service"
print_info "To stop: sudo systemctl stop aura.service"
print_info "To check status: sudo systemctl status aura.service"
print_info "To view logs: journalctl -u aura.service -f"

echo ""

# ============================================================================
# Step 9: Install keyboard monitor service
# ============================================================================
print_step "9. Installing keyboard monitor service..."

KEYBOARD_SERVICE_FILE="$LEDGERAI_DIR/setup/scripts/disable-keyboard-monitor.service"
SYSTEMD_KEYBOARD_SERVICE="/etc/systemd/system/disable-keyboard-monitor.service"

if [ -f "$KEYBOARD_SERVICE_FILE" ]; then
    # Update service file with correct user
    sudo cp "$KEYBOARD_SERVICE_FILE" "$SYSTEMD_KEYBOARD_SERVICE"
    
    # Update user in service file if needed
    sudo sed -i "s|User=aura|User=$AURA_USER|g" "$SYSTEMD_KEYBOARD_SERVICE"
    sudo sed -i "s|Group=aura|Group=$AURA_USER|g" "$SYSTEMD_KEYBOARD_SERVICE"
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable service (it will start automatically with aura.service)
    sudo systemctl enable disable-keyboard-monitor.service
    
    print_info "Keyboard monitor service installed and enabled"
    print_info "Service will disable Ubuntu keyboard while Aura is running"
    print_info "To check status: sudo systemctl status disable-keyboard-monitor.service"
    print_info "To view logs: journalctl -u disable-keyboard-monitor.service -f"
else
    print_error "Keyboard monitor service file not found at $KEYBOARD_SERVICE_FILE"
fi

echo ""

# ============================================================================
# Step 10: Set up Docker (if not already configured)
# ============================================================================
print_step "10. Configuring Docker..."

# Check if user is in docker group
if ! groups "$AURA_USER" | grep -q docker; then
    print_info "Adding $AURA_USER to docker group..."
    sudo usermod -aG docker "$AURA_USER"
    print_info "User added to docker group (logout/login required for changes)"
else
    print_info "User already in docker group"
fi

# Ensure Docker is running
if ! systemctl is-active --quiet docker; then
    print_info "Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
fi

print_info "Docker configured"

echo ""

# ============================================================================
# Step 11: Set permissions for audio devices
# ============================================================================
print_step "11. Configuring audio device permissions..."

# Add user to audio group
if ! groups "$AURA_USER" | grep -q audio; then
    sudo usermod -aG audio "$AURA_USER"
    print_info "User added to audio group"
else
    print_info "User already in audio group"
fi

# Create udev rules for XVF3800 (if needed)
UDEV_RULES="/etc/udev/rules.d/99-xvf3800.rules"
if [ ! -f "$UDEV_RULES" ]; then
    print_info "Creating udev rules for XVF3800..."
    sudo tee "$UDEV_RULES" > /dev/null << 'EOF'
# XVF3800 USB 4-Mic Array
SUBSYSTEM=="usb", ATTRS{idVendor}=="20b1", ATTRS{idProduct}=="0011", MODE="0666", GROUP="audio"
EOF
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    print_info "Udev rules created"
fi

echo ""

# ============================================================================
# Step 12: Create data directories if needed
# ============================================================================
print_step "12. Creating data directories..."

mkdir -p "$LEDGERAI_DIR/data/input"
mkdir -p "$LEDGERAI_DIR/data/parsed"
mkdir -p "$LEDGERAI_DIR/data/embeddings"
mkdir -p "$LEDGERAI_DIR/shared/input_audio"
mkdir -p "$LEDGERAI_DIR/shared/output_audio"

print_info "Data directories created"

echo ""

# ============================================================================
# Installation Summary
# ============================================================================
print_step "Installation Complete!"
echo ""
echo "=========================================="
echo "  Installation Summary"
echo "=========================================="
echo ""
echo "✅ Virtual environment: $VENV_DIR"
echo "✅ Python requirements: Installed"
echo "✅ jetson-containers: $JETSON_CONTAINERS_DIR"
echo "✅ XVF3800 support: $XVF3800_REPO_DIR"
echo "✅ XVF3800 tuning service: Enabled (runs on boot)"
echo "✅ Aura service: Enabled (will start on boot)"
echo "✅ Keyboard monitor service: Enabled (disables Ubuntu keyboard while Aura runs)"
echo "✅ Docker: Configured"
echo "✅ Display settings: Configured"
echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "1. Logout and login again (or reboot) to apply group changes"
echo "2. Ensure Docker containers are built:"
echo "   cd $LEDGERAI_DIR/setup"
echo "   docker compose build"
echo ""
echo "3. Test Aura manually first:"
echo "   cd $LEDGERAI_DIR/aura-control/core"
echo "   source $VENV_DIR/bin/activate"
echo "   python3 main.py"
echo ""
echo "4. If everything works, reboot to test auto-start:"
echo "   sudo reboot"
echo ""
echo "5. Check service status after boot:"
echo "   sudo systemctl status aura.service"
echo "   sudo systemctl status xvf3800-tuning.service"
echo "   sudo systemctl status disable-keyboard-monitor.service"
echo ""
echo "6. View logs:"
echo "   journalctl -u aura.service -f"
echo "   journalctl -u xvf3800-tuning.service -n 50"
echo "   journalctl -u disable-keyboard-monitor.service -f"
echo ""
echo "7. If audio doesn't work (PortAudio not available):"
echo "   sounddevice may need PortAudio library. Try:"
echo "   - Install from source: https://www.portaudio.com/download.html"
echo "   - Or use ALSA directly (may require code changes)"
echo ""
echo "=========================================="
echo "  Service Management"
echo "=========================================="
echo ""
echo "Start Aura service:     sudo systemctl start aura.service"
echo "Stop Aura service:      sudo systemctl stop aura.service"
echo "Restart Aura service:   sudo systemctl restart aura.service"
echo "Disable auto-start:     sudo systemctl disable aura.service"
echo "Enable auto-start:      sudo systemctl enable aura.service"
echo ""
echo "=========================================="

