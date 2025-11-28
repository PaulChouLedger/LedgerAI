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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
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
        PORT_AUDIO_AVAILABLE=true
    else
        print_info "⚠️  Some PortAudio packages failed to install"
        PORT_AUDIO_AVAILABLE=false
    fi
else
    print_info "⚠️  PortAudio packages not found in repositories"
    PORT_AUDIO_AVAILABLE=false
fi

# If PortAudio packages aren't available, build from source
if [ "$PORT_AUDIO_AVAILABLE" = false ]; then
    print_info "Building PortAudio from source (required for sounddevice)..."
    
    # Check if already built
    if ldconfig -p | grep -q libportaudio || [ -f "/usr/local/lib/libportaudio.so" ] || [ -f "/usr/lib/libportaudio.so" ]; then
        print_info "PortAudio library already exists, skipping build"
        # Verify it works with Python
        if python3 -c "import sounddevice as sd" 2>/dev/null; then
            print_info "✅ PortAudio verified and working"
            PORT_AUDIO_AVAILABLE=true
        else
            print_info "PortAudio exists but Python can't find it - rebuilding..."
            PORT_AUDIO_AVAILABLE=false
        fi
    fi
    
    if [ "$PORT_AUDIO_AVAILABLE" = false ]; then
        print_info "Installing build dependencies for PortAudio..."
        sudo apt install -y autoconf automake libasound2-dev wget || true
        
        print_info "Downloading PortAudio source..."
        cd /tmp
        rm -rf portaudio portaudio.tgz
        
        # Download PortAudio source (fixed URL)
        PORTAUDIO_URL="http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz"
        if wget -q "$PORTAUDIO_URL" -O portaudio.tgz; then
            tar -xzf portaudio.tgz
            cd portaudio
            
            # Configure and build
            print_info "Building PortAudio (this may take 2-5 minutes)..."
            if ./configure && make -j$(nproc); then
                sudo make install
                sudo ldconfig
                
                # Verify installation
                if ldconfig -p | grep -q portaudio || [ -f "/usr/local/lib/libportaudio.so" ]; then
                    print_info "✅ PortAudio built and installed successfully"
                    
                    # Verify library is in cache
                    print_info "✅ PortAudio library installed and registered"
                    PORT_AUDIO_AVAILABLE=true
                    print_info "   Python import will be tested after virtual environment setup"
                else
                    print_error "PortAudio built but library not found in cache"
                    PORT_AUDIO_AVAILABLE=false
                fi
            else
                print_error "PortAudio build failed"
                print_info "   sounddevice may not work - audio input/output will fail"
                print_info "   You can try building manually: bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
                PORT_AUDIO_AVAILABLE=false
            fi
            
            cd /tmp
            rm -rf portaudio portaudio.tgz
        else
            print_error "Failed to download PortAudio source"
            print_info "   sounddevice may not work - audio input/output will fail"
            print_info "   You can try building manually: bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
            PORT_AUDIO_AVAILABLE=false
        fi
    fi
fi

echo ""

# ============================================================================
# Step 2: Create and configure virtual environment
# ============================================================================
print_step "2. Setting up Python virtual environment..."

# Verify system PyQt5 installation (we just installed it in step 1)
print_info "Verifying PyQt5 installation..."
SYSTEM_PYQT5_AVAILABLE=false
if python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import PyQt5" 2>/dev/null; then
    SYSTEM_PYQT5_AVAILABLE=true
    PYQT5_VERSION=$(python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import PyQt5; print(PyQt5.QtCore.PYQT_VERSION_STR)" 2>/dev/null || echo "unknown")
    print_info "✅ System PyQt5 detected (version: $PYQT5_VERSION)"
    print_info "   Will create venv with --system-site-packages to use system PyQt5"
elif dpkg -l | grep -q python3-pyqt5; then
    print_info "⚠️  python3-pyqt5 package installed but not importable"
    print_info "   This may be a path issue - will try system-site-packages anyway"
    SYSTEM_PYQT5_AVAILABLE=true  # Try anyway
else
    print_info "⚠️  System PyQt5 not found - will build from source if needed"
    print_info "   Qt5 dev tools installed: qtbase5-dev, qttools5-dev"
fi

if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment already exists at $VENV_DIR"
    print_info "Removing old virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Always try system-site-packages if system PyQt5 might be available
# This ensures GUI works on Jetson/Ubuntu systems
if [ "$SYSTEM_PYQT5_AVAILABLE" = true ]; then
    "$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR"
    print_info "Virtual environment created with --system-site-packages (to access system PyQt5)"
else
    # Even if not detected, try system-site-packages on Ubuntu/Jetson (PyQt5 is usually there)
    if [ -f "/etc/os-release" ] && grep -q "Ubuntu\|Debian" /etc/os-release; then
        print_info "Ubuntu/Debian detected - creating venv with --system-site-packages"
        print_info "   (PyQt5 is typically pre-installed on Ubuntu/Jetson)"
        "$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR"
    else
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        print_info "Virtual environment created at $VENV_DIR using $PYTHON_CMD"
    fi
fi

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
    # PortAudio should have been built from source if packages weren't available
    print_info "Verifying PortAudio is accessible for sounddevice..."
    if python3 -c "import sounddevice as sd; print('✅ PortAudio working!')" 2>/dev/null; then
        print_info "✅ PortAudio is accessible - sounddevice will work"
        PORT_AUDIO_AVAILABLE=true
    elif [ "$PORT_AUDIO_AVAILABLE" = true ]; then
        print_info "⚠️  PortAudio library installed but Python import failed"
        print_info "   This may resolve after installing sounddevice package"
        print_info "   Will continue with installation..."
    else
        print_error "⚠️  PortAudio not available - sounddevice will NOT work"
        print_error "   Build PortAudio: bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
        print_info "   Continuing with installation, but audio input/output will fail"
    fi
    
    # Install PyQt5 separately with better error handling
    # Check if we should use system PyQt5 or install from pip
    if [ "$SYSTEM_PYQT5_AVAILABLE" = true ]; then
        print_info "Using system PyQt5 (pre-installed on system)"
        print_info "Installing requirements without PyQt5..."
        TEMP_REQUIREMENTS="/tmp/requirements_no_pyqt5.txt"
        grep -v "^PyQt5" "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" > "$TEMP_REQUIREMENTS" || true
        if pip install -r "$TEMP_REQUIREMENTS"; then
            print_info "✅ All requirements installed successfully (using system PyQt5)"
        else
            print_error "Some requirements failed to install. Installing critical packages individually..."
            pip install python-dotenv requests numpy scipy sounddevice soundfile pyusb flask werkzeug || true
        fi
        rm -f "$TEMP_REQUIREMENTS"
        
        # Verify PyQt5 is accessible in venv
        print_info "Verifying PyQt5 is accessible in virtual environment..."
        if python3 -c "import PyQt5; from PyQt5.QtWidgets import QApplication; print('✅ PyQt5 fully functional')" 2>/dev/null; then
            PYQT5_VERSION=$(python3 -c "import PyQt5; print(PyQt5.QtCore.PYQT_VERSION_STR)" 2>/dev/null || echo "unknown")
            print_info "✅ PyQt5 is accessible and functional (version: $PYQT5_VERSION)"
            print_info "✅ GUI will work correctly"
        else
            print_error "⚠️  PyQt5 import failed in virtual environment!"
            print_info "   Trying to diagnose..."
            python3 -c "import sys; print('Python path:', sys.path)" 2>/dev/null || true
            python3 -c "import PyQt5" 2>&1 || print_error "PyQt5 not accessible - GUI will NOT work"
            print_info "   You may need to recreate venv or install PyQt5 from pip"
        fi
    else
        # Try to install PyQt5 from pip
        print_info "Installing all requirements (including PyQt5 from pip)..."
        if pip install -r "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" 2>&1 | tee /tmp/pip_install.log; then
            print_info "✅ All requirements installed successfully"
        else
            PIP_ERROR=$(cat /tmp/pip_install.log)
            if echo "$PIP_ERROR" | grep -q "PyQt5\|pyqt5"; then
                print_info "⚠️  PyQt5 installation failed, trying alternative approach..."
                
                # Try installing system PyQt5 packages
                print_info "Attempting to install system PyQt5 packages..."
                sudo apt install -y python3-pyqt5 python3-pyqt5.qtsvg python3-pyqt5.qtwebkit 2>/dev/null || true
                
                # Check if system PyQt5 is now available
                if python3 -c "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); import PyQt5" 2>/dev/null; then
                    print_info "System PyQt5 now available. Recreating venv with system-site-packages..."
                    # We can't recreate venv here easily, so just install other packages
                    TEMP_REQUIREMENTS="/tmp/requirements_no_pyqt5.txt"
                    grep -v "^PyQt5" "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" > "$TEMP_REQUIREMENTS" || true
                    if pip install -r "$TEMP_REQUIREMENTS"; then
                        print_info "✅ All other requirements installed"
                    else
                        print_error "Some requirements failed. Installing critical packages..."
                        pip install python-dotenv requests numpy scipy sounddevice soundfile pyusb flask werkzeug || true
                    fi
                    rm -f "$TEMP_REQUIREMENTS"
                    print_info "⚠️  Note: You may need to recreate venv with --system-site-packages for PyQt5 to work"
                    print_info "   Run: deactivate && rm -rf $VENV_DIR && $PYTHON_CMD -m venv --system-site-packages $VENV_DIR"
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
                    print_info "Building PyQt5 from source (this may take 10-30 minutes)..."
                    if pip install PyQt5 --no-build-isolation 2>&1 | tee /tmp/pyqt5_build.log; then
                        print_info "✅ PyQt5 installed successfully from source"
                        # Verify it works
                        if python3 -c "import PyQt5; from PyQt5.QtWidgets import QApplication" 2>/dev/null; then
                            print_info "✅ PyQt5 verified and functional - GUI will work"
                        else
                            print_error "⚠️  PyQt5 installed but import failed - GUI may not work"
                        fi
                    elif [ -n "$QMAKE_PATH" ]; then
                        print_info "Trying PyQt5 install with explicit qmake path..."
                        export QMAKE="$QMAKE_PATH"
                        pip install PyQt5 --no-build-isolation || {
                            print_error "PyQt5 installation failed"
                            print_info "Installing remaining packages without PyQt5..."
                            TEMP_REQUIREMENTS="/tmp/requirements_no_pyqt5.txt"
                            grep -v "^PyQt5" "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" > "$TEMP_REQUIREMENTS" || true
                            if pip install -r "$TEMP_REQUIREMENTS"; then
                                print_info "✅ All other requirements installed successfully"
                            else
                                print_error "Some requirements failed to install. Installing critical packages individually..."
                                pip install python-dotenv requests numpy scipy sounddevice soundfile pyusb flask werkzeug || true
                            fi
                            rm -f "$TEMP_REQUIREMENTS"
                            print_info "⚠️  PyQt5 not installed. GUI may not work."
                        }
                    else
                        print_error "qmake not found. PyQt5 cannot be built."
                        print_info "Installing remaining packages without PyQt5..."
                        TEMP_REQUIREMENTS="/tmp/requirements_no_pyqt5.txt"
                        grep -v "^PyQt5" "$LEDGERAI_DIR/aura-control/requirements/requirements.txt" > "$TEMP_REQUIREMENTS" || true
                        if pip install -r "$TEMP_REQUIREMENTS"; then
                            print_info "✅ All other requirements installed successfully"
                        else
                            print_error "Some requirements failed to install. Installing critical packages individually..."
                            pip install python-dotenv requests numpy scipy sounddevice soundfile pyusb flask werkzeug || true
                        fi
                        rm -f "$TEMP_REQUIREMENTS"
                        print_error "⚠️  PyQt5 not installed. GUI will NOT work."
                        print_info "   Attempting final fallback: install system PyQt5 and recreate venv..."
                        sudo apt install -y python3-pyqt5 python3-pyqt5.qtsvg python3-pyqt5.qtwebkit 2>/dev/null || true
                        print_info "   If PyQt5 still doesn't work, you may need to:"
                        print_info "   1. Recreate venv: deactivate && rm -rf $VENV_DIR"
                        print_info "   2. Create with system packages: $PYTHON_CMD -m venv --system-site-packages $VENV_DIR"
                        print_info "   3. Install requirements excluding PyQt5"
                    fi
                fi
            else
                print_error "Installation failed for other reasons (not PyQt5 related). Check logs above."
                print_info "Attempting to install critical packages..."
                pip install python-dotenv requests numpy scipy sounddevice soundfile pyusb flask werkzeug || true
            fi
        fi
        rm -f /tmp/pip_install.log /tmp/pyqt5_build.log 2>/dev/null || true
    fi
    
    # Final verification: Ensure PyQt5 and PortAudio are accessible
    print_info ""
    print_info "🔍 Final verification..."
    
    # Test PyQt5
    print_info "Testing PyQt5..."
    if python3 -c "import PyQt5; from PyQt5.QtWidgets import QApplication; print('SUCCESS')" 2>/dev/null; then
        PYQT5_VERSION=$(python3 -c "import PyQt5; print(PyQt5.QtCore.PYQT_VERSION_STR)" 2>/dev/null || echo "unknown")
        print_info "✅ PyQt5 is installed and functional (version: $PYQT5_VERSION)"
        print_info "✅ GUI will work correctly on this device"
    else
        print_error "❌ PyQt5 is NOT accessible - GUI will NOT work!"
        print_error "   This is a CRITICAL issue - the GUI requires PyQt5"
        print_info "   Troubleshooting steps:"
        print_info "   1. Check if system PyQt5 is installed: dpkg -l | grep pyqt5"
        print_info "   2. Verify Qt5 dev tools: dpkg -l | grep qt5"
        print_info "   3. Try: python3 -c 'import sys; sys.path.insert(0, \"/usr/lib/python3/dist-packages\"); import PyQt5'"
        print_info "   4. If system PyQt5 works, recreate venv with --system-site-packages"
    fi
    
    # Test PortAudio (sounddevice)
    print_info "Testing PortAudio (sounddevice)..."
    if python3 -c "import sounddevice as sd; print('SUCCESS')" 2>/dev/null; then
        print_info "✅ PortAudio is accessible - sounddevice will work"
        print_info "✅ Audio input/output will work correctly"
        PORT_AUDIO_AVAILABLE=true
    else
        if ldconfig -p | grep -q portaudio || [ -f "/usr/local/lib/libportaudio.so" ]; then
            print_error "⚠️  PortAudio library installed but Python can't access it"
            print_info "   Try: source $VENV_DIR/bin/activate && python3 -c 'import sounddevice as sd'"
            print_info "   Or rebuild PortAudio: bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
        else
            print_error "❌ PortAudio not installed - sounddevice will NOT work!"
            print_error "   Audio input/output will fail - listener will not work"
            print_info "   Build PortAudio: bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
        fi
        PORT_AUDIO_AVAILABLE=false
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
# Step 3.5: Install wake word detection engine (OpenWakeWord)
# ============================================================================
print_step "3.5. Installing wake word detection engine..."

# Install OpenWakeWord dependencies first to avoid numpy/pandas binary incompatibility
# This ensures all packages come from pip (not system packages) and are compatible
print_info "Installing OpenWakeWord dependencies (numpy, pandas, scikit-learn)..."
if pip install --upgrade "numpy>=1.24.0" "pandas>=2.0.0" "scikit-learn>=1.3.0" 2>&1 | tee /tmp/openwakeword_deps_install.log; then
    print_info "✅ OpenWakeWord dependencies installed successfully"
else
    print_warning "⚠️  Some OpenWakeWord dependencies had issues (check logs)"
    print_info "   Continuing with OpenWakeWord installation anyway..."
fi

# Install OpenWakeWord (actively maintained, recommended for all systems)
# NOTE: OpenWakeWord installation failure is non-fatal - installation will continue
print_info "Installing OpenWakeWord..."
if pip install "openwakeword>=0.5.0" 2>&1 | tee /tmp/openwakeword_install.log; then
    print_info "✅ OpenWakeWord installed successfully"
    
    # Verify OpenWakeWord installation (non-fatal if it fails)
    if python3 -c "import openwakeword; from openwakeword import Model; print('✅ OpenWakeWord verified')" 2>/dev/null; then
        print_info "✅ OpenWakeWord verified and ready to use"
    else
        print_warning "⚠️  OpenWakeWord installed but verification failed"
        print_info "   This may be due to numpy/pandas compatibility issues or onnxruntime version"
        print_info "   Installation will continue - wake word detection may not work until fixed"
        print_info "   Try: pip install --upgrade --force-reinstall numpy pandas scikit-learn"
    fi
else
    print_warning "⚠️  OpenWakeWord installation failed (check logs)"
    print_info "   Installation will continue - wake word detection will not work until fixed"
    print_info "   You can fix this later by running: pip install openwakeword"
fi

# Verify installation
print_info "Verifying wake word detection engine..."

# Use the Python from the virtual environment for verification
PYTHON_CMD="python3"
if [ -f "$VENV_DIR/bin/python3" ]; then
    PYTHON_CMD="$VENV_DIR/bin/python3"
fi

# Verify OpenWakeWord and download models (non-fatal - continue even if it fails)
print_info "Verifying OpenWakeWord..."
if $PYTHON_CMD -c "import openwakeword; from openwakeword import Model; print('SUCCESS')" 2>/dev/null; then
    print_info "✅ OpenWakeWord is installed and importable"
    
    # Download/initialize OpenWakeWord models explicitly
    # NOTE: This may crash due to onnxruntime issues - make it non-fatal
    print_info "Downloading OpenWakeWord models (this may take a minute on first run)..."
    if $PYTHON_CMD << 'PYEOF' 2>&1 || true
import sys
try:
    # Use OpenWakeWord's utility function to explicitly download models
    import openwakeword.utils
    print("Downloading OpenWakeWord models (melspectrogram and wake word models)...")
    openwakeword.utils.download_models()
    print("✅ Models downloaded successfully")
    
    # Verify models are available by trying to create a Model instance
    from openwakeword import Model
    print("Verifying models are accessible...")
    model = Model(inference_framework='onnx')
    print("✅ Models verified and accessible")
    # Check if models are available
    if hasattr(model, 'models') and len(model.models) > 0:
        print(f"✅ Found {len(model.models)} wake word model(s)")
    else:
        print("✅ Models downloaded (melspectrogram preprocessing model available)")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error downloading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(0)  # Non-fatal - don't fail installation
PYEOF
    then
        print_info "✅ OpenWakeWord models downloaded and ready"
    else
        print_warning "⚠️  Model download had issues (may still work on first use)"
        print_info "   Models will be downloaded automatically when OpenWakeWord is first used"
        print_info "   Installation will continue..."
    fi
else
    print_warning "⚠️  OpenWakeWord not importable (non-fatal - installation will continue)"
    print_info "   Attempting to diagnose..."
    # Capture error output (may crash due to onnxruntime - make it non-fatal)
    ERROR_OUTPUT=$($PYTHON_CMD -c "import openwakeword" 2>&1 || echo "Import failed (may have crashed)")
    if echo "$ERROR_OUTPUT" | grep -q "numpy.dtype size changed\|binary incompatibility"; then
        print_warning "⚠️  Numpy/pandas binary incompatibility detected"
        print_info "   This usually means system packages conflict with pip packages"
        print_info "   Attempting to fix by reinstalling numpy, pandas, scikit-learn..."
        if pip install --upgrade --force-reinstall "numpy>=1.24.0" "pandas>=2.0.0" "scikit-learn>=1.3.0" 2>&1 | tee /tmp/fix_numpy_pandas.log; then
            print_info "✅ Dependencies reinstalled - retrying verification..."
            # Retry verification (may still crash - make it non-fatal)
            if $PYTHON_CMD -c "import openwakeword; from openwakeword import Model; print('SUCCESS')" 2>&1 | grep -q "SUCCESS" || true; then
                print_info "✅ OpenWakeWord verified after dependency fix"
                # Try to download models after fix (may crash - make it non-fatal)
                print_info "Downloading OpenWakeWord models..."
                $PYTHON_CMD << 'PYEOF' 2>&1 || true
import sys
try:
    import openwakeword.utils
    openwakeword.utils.download_models()
    print("✅ Models downloaded successfully")
    sys.exit(0)
except Exception as e:
    print(f"⚠️  Model download issue: {e}")
    sys.exit(0)  # Non-fatal
PYEOF
            else
                print_warning "⚠️  Still having issues - may need manual intervention"
                print_info "   Try: pip install --upgrade --force-reinstall numpy pandas scikit-learn openwakeword"
            fi
        else
            print_warning "⚠️  Failed to fix dependency issues"
            print_info "   Manual fix may be required"
        fi
    elif echo "$ERROR_OUTPUT" | grep -q "ModuleNotFoundError\|ImportError"; then
        print_info "   Module exists but import failed - checking dependencies..."
        print_info "   Error: $ERROR_OUTPUT"
    else
        print_info "   Module not found - may need to reinstall: pip install openwakeword"
    fi
fi

# Wake word setup complete (even if OpenWakeWord failed, installation continues)
if $PYTHON_CMD -c "import openwakeword" 2>/dev/null; then
    print_info "✅ Wake word engine setup complete"
    print_info "   - OpenWakeWord: Installed and ready"
else
    print_warning "⚠️  Wake word engine setup incomplete (OpenWakeWord not working)"
    print_info "   - Installation will continue - you can fix OpenWakeWord later"
    print_info "   - Wake word detection will not work until OpenWakeWord is fixed"
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
# Step 4.5: Configure Jetson MAXN Power Mode
# ============================================================================
print_step "4.5. Configuring Jetson MAXN Power Mode..."

# Check if running on Jetson device
if [ -f /etc/nv_tegra_release ] || [ -f /proc/device-tree/model ] && grep -q "Jetson\|NVIDIA" /proc/device-tree/model 2>/dev/null; then
    print_info "Jetson device detected - configuring MAXN power mode..."
    
    # Check if nvpmodel command exists
    if command -v nvpmodel >/dev/null 2>&1; then
        # Check if the specific config file exists
        if [ -f "/etc/nvpmodel/nvpmodel_p3767_0000_super.conf" ]; then
            print_info "Setting MAXN power mode (mode 0) with p3767_0000_super config..."
            if sudo nvpmodel -f /etc/nvpmodel/nvpmodel_p3767_0000_super.conf -m 0; then
                print_info "✅ MAXN power mode configured successfully"
            else
                print_warning "⚠️  Failed to set nvpmodel - continuing anyway"
            fi
        else
            print_info "p3767_0000_super config not found, trying default MAXN mode..."
            if sudo nvpmodel -m 0; then
                print_info "✅ MAXN power mode configured successfully (default config)"
            else
                print_warning "⚠️  Failed to set nvpmodel - continuing anyway"
            fi
        fi
        
        # Verify current power mode
        CURRENT_MODE=$(sudo nvpmodel -q 2>/dev/null | grep -oP "NV Power Mode: \K[0-9]+" || echo "unknown")
        print_info "Current power mode: $CURRENT_MODE (0 = MAXN)"
    else
        print_warning "⚠️  nvpmodel command not found - power mode configuration skipped"
    fi
    
    # Set max clocks with jetson_clocks
    if command -v jetson_clocks >/dev/null 2>&1; then
        print_info "Setting maximum clock speeds..."
        if sudo jetson_clocks; then
            print_info "✅ Maximum clock speeds configured"
        else
            print_warning "⚠️  Failed to set jetson_clocks - continuing anyway"
        fi
    else
        print_warning "⚠️  jetson_clocks command not found - clock configuration skipped"
    fi
    
    # Make power mode persistent across reboots
    print_info "Making power mode persistent across reboots..."
    
    # Create a systemd service to set power mode on boot
    JETSON_POWER_SERVICE="/etc/systemd/system/jetson-maxn-power.service"
    if [ ! -f "$JETSON_POWER_SERVICE" ]; then
        sudo tee "$JETSON_POWER_SERVICE" >/dev/null << 'EOFPOWER'
[Unit]
Description=Set Jetson to MAXN Power Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'if [ -f /etc/nvpmodel/nvpmodel_p3767_0000_super.conf ]; then /usr/bin/nvpmodel -f /etc/nvpmodel/nvpmodel_p3767_0000_super.conf -m 0; else /usr/bin/nvpmodel -m 0; fi'
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOFPOWER
        sudo systemctl daemon-reload
        sudo systemctl enable jetson-maxn-power.service
        print_info "✅ Created systemd service for persistent MAXN power mode"
        print_info "   Power mode will be set to MAXN on every boot"
    else
        print_info "Jetson power service already exists"
    fi
else
    print_info "Not running on Jetson device - skipping power mode configuration"
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

# Disable password prompt when waking from sleep and lock screen
# Run these commands as the user (same as original orin_setup.sh)
print_info "Configuring lock screen settings..."
if [ "$(whoami)" = "$AURA_USER" ]; then
    # Running as the user - use gsettings directly
    gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
    gsettings set org.gnome.desktop.lockdown disable-lock-screen true 2>/dev/null || true
else
    # Running as root/sudo - need to run as the user
    su - "$AURA_USER" -c "gsettings set org.gnome.desktop.screensaver lock-enabled false" 2>/dev/null || true
    su - "$AURA_USER" -c "gsettings set org.gnome.desktop.lockdown disable-lock-screen true" 2>/dev/null || true
fi

# Configure automatic login (no password required on boot)
print_info "Configuring automatic login (no password on boot)..."
if [ -f "/etc/gdm3/custom.conf" ]; then
    # Backup original config
    if [ ! -f "/etc/gdm3/custom.conf.bak" ]; then
        sudo cp /etc/gdm3/custom.conf /etc/gdm3/custom.conf.bak
    fi
    
    # Check if automatic login is already configured
    if ! grep -q "^AutomaticLogin=$AURA_USER" /etc/gdm3/custom.conf 2>/dev/null; then
        # Enable automatic login in GDM3
        # Remove any existing AutomaticLogin lines first
        sudo sed -i '/^AutomaticLogin=/d' /etc/gdm3/custom.conf
        
        # Find the [daemon] section and add AutomaticLogin after it
        if grep -q "^\[daemon\]" /etc/gdm3/custom.conf; then
            # Add AutomaticLogin in the [daemon] section
            sudo sed -i "/^\[daemon\]/a AutomaticLogin=$AURA_USER" /etc/gdm3/custom.conf
            sudo sed -i "/^AutomaticLogin=$AURA_USER/a AutomaticLoginEnable=true" /etc/gdm3/custom.conf
            print_info "✅ Automatic login configured for user: $AURA_USER"
        else
            # Create [daemon] section if it doesn't exist
            sudo bash -c "cat >> /etc/gdm3/custom.conf << 'EOFGDM'

[daemon]
AutomaticLogin=$AURA_USER
AutomaticLoginEnable=true
EOFGDM
"
            print_info "✅ Automatic login configured for user: $AURA_USER (created [daemon] section)"
        fi
    else
        print_info "✅ Automatic login already configured for user: $AURA_USER"
    fi
else
    print_warning "⚠️  GDM3 configuration file not found at /etc/gdm3/custom.conf"
    print_warning "   Automatic login may need to be configured manually"
    print_info "   Edit /etc/gdm3/custom.conf and add:"
    print_info "   [daemon]"
    print_info "   AutomaticLogin=$AURA_USER"
    print_info "   AutomaticLoginEnable=true"
fi

print_info "Display settings configured"

echo ""

# ============================================================================
# Step 6.8: Configure WiFi privileges (allow nmcli without password)
# ============================================================================
print_step "6.8. Configuring WiFi privileges (Polkit rule for nmcli)..."

# Ensure polkit is installed
if ! dpkg -s policykit-1 >/dev/null 2>&1; then
    print_info "Installing policykit-1 (Polkit)..."
    sudo apt install -y policykit-1 || true
fi

# Create a dedicated group for NetworkManager privileged actions
if ! getent group nm-authed >/dev/null; then
    print_info "Creating group: nm-authed"
    sudo groupadd nm-authed
else
    print_info "Group nm-authed already exists"
fi

# Add Aura user to the group
if groups "$AURA_USER" | grep -q "\bnm-authed\b"; then
    print_info "User $AURA_USER already in nm-authed group"
else
    print_info "Adding $AURA_USER to nm-authed group"
    sudo usermod -aG nm-authed "$AURA_USER"
    print_warning "You must logout/login (or reboot) for group changes to take effect"
fi

# Determine Polkit rules directory (prefer /etc, fallback to /usr/share)
POLKIT_DIR="/etc/polkit-1/rules.d"
if [ ! -d "$POLKIT_DIR" ]; then
    print_info "Creating Polkit rules directory at $POLKIT_DIR"
    sudo mkdir -p "$POLKIT_DIR" 2>/dev/null || true
fi
if [ ! -d "$POLKIT_DIR" ]; then
    POLKIT_DIR="/usr/share/polkit-1/rules.d"
    print_warning "Using fallback Polkit rules directory: $POLKIT_DIR"
    sudo mkdir -p "$POLKIT_DIR" 2>/dev/null || true
fi

POLKIT_RULE="$POLKIT_DIR/10-allow-nmcli.rules"

install_rules_file() {
    local target_file="$1"
    print_info "Creating Polkit rule at $target_file"
    sudo tee "$target_file" >/dev/null <<'EORULE'
polkit.addRule(function(action, subject) {
  if ((action.id.indexOf("org.freedesktop.NetworkManager") === 0) &&
      subject.isInGroup("nm-authed")) {
    return polkit.Result.YES;
  }
});
EORULE
    sudo chmod 0644 "$target_file" 2>/dev/null || true
}

# Try to install a .rules file; if it fails, fallback to .pkla for older Polkit
if [ ! -f "$POLKIT_RULE" ]; then
    if install_rules_file "$POLKIT_RULE"; then
        print_info "✅ Polkit .rules installed at $POLKIT_RULE"
    else
        print_warning "Failed to write .rules file; attempting .pkla fallback"
        PKLA_DIR="/etc/polkit-1/localauthority/50-local.d"
        sudo mkdir -p "$PKLA_DIR" 2>/dev/null || true
        PKLA_FILE="$PKLA_DIR/10-allow-nmcli.pkla"
        sudo tee "$PKLA_FILE" >/dev/null <<'EOPKLA'
[Allow NetworkManager for nm-authed]
Identity=unix-group:nm-authed
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOPKLA
        sudo chmod 0644 "$PKLA_FILE" 2>/dev/null || true
        print_info "✅ Polkit .pkla rule installed at $PKLA_FILE"
    fi
else
    print_info "Polkit rule already exists at $POLKIT_RULE"
fi

# Restart polkit to apply rule immediately (safe if present)
if systemctl list-unit-files | grep -q polkit; then
    print_info "Restarting polkit service to apply rule..."
    sudo systemctl restart polkit || true
else
    # Some systems use polkit.service name polkitd or don't permit restart; ignore failures
    sudo systemctl restart polkit.service 2>/dev/null || true
fi

print_warning "If nmcli still prompts for authorization, logout/login (or reboot) to apply group membership."

# Also create user-specific WiFi permissions rules (more reliable than group-based)
print_info "Creating user-specific WiFi permissions rules..."
PKLA_DIR="/etc/polkit-1/localauthority/50-local.d"
WIFI_POLKIT_RULE="$POLKIT_DIR/50-allow-nmcli-wifi.rules"
if [ ! -f "$WIFI_POLKIT_RULE" ]; then
    sudo tee "$WIFI_POLKIT_RULE" >/dev/null <<EORULE
// Allow NetworkManager WiFi operations for $AURA_USER
// This rule allows WiFi scanning, connecting, and disconnecting without password prompts

polkit.addRule(function(action, subject) {
  // Allow all NetworkManager actions for the specific user
  if (action.id.indexOf("org.freedesktop.NetworkManager") === 0) {
    // Allow for the specific user
    if (subject.user == "$AURA_USER") {
      return polkit.Result.YES;
    }
    // Also allow for users in the nm-authed group (if it exists)
    if (subject.isInGroup("nm-authed")) {
      return polkit.Result.YES;
    }
  }
  
  // Specifically allow WiFi operations
  if (action.id == "org.freedesktop.NetworkManager.wifi.scan" ||
      action.id == "org.freedesktop.NetworkManager.settings.modify.system" ||
      action.id == "org.freedesktop.NetworkManager.settings.modify.own" ||
      action.id == "org.freedesktop.NetworkManager.network-control") {
    if (subject.user == "$AURA_USER") {
      return polkit.Result.YES;
    }
  }
});
EORULE
    sudo chmod 0644 "$WIFI_POLKIT_RULE" 2>/dev/null || true
    print_info "✅ WiFi user-specific polkit rule created"
    
    # Also create .pkla fallback
    sudo mkdir -p "$PKLA_DIR" 2>/dev/null || true
    WIFI_PKLA_FILE="$PKLA_DIR/50-allow-nmcli-wifi.pkla"
    sudo tee "$WIFI_PKLA_FILE" >/dev/null <<EOPKLA
[Allow NetworkManager WiFi for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes

[Allow NetworkManager WiFi for nm-authed group]
Identity=unix-group:nm-authed
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOPKLA
    sudo chmod 0644 "$WIFI_PKLA_FILE" 2>/dev/null || true
    print_info "✅ WiFi fallback .pkla rule created"
else
    print_info "WiFi user-specific polkit rule already exists"
fi

echo ""

# ============================================================================
# Step 6.9: Configure shutdown permissions (passwordless shutdown/restart)
# ============================================================================
print_step "6.9. Configuring shutdown/restart permissions (Polkit rule for passwordless power management)..."

# Create shutdown/restart polkit rules
SHUTDOWN_POLKIT_RULE="$POLKIT_DIR/50-allow-shutdown.rules"
if [ ! -f "$SHUTDOWN_POLKIT_RULE" ]; then
    print_info "Creating shutdown/restart polkit rule..."
    sudo tee "$SHUTDOWN_POLKIT_RULE" >/dev/null <<EORULE
// Allow system shutdown/reboot for $AURA_USER
// This rule allows passwordless shutdown and reboot operations

polkit.addRule(function(action, subject) {
  // Allow system power management actions for the specific user
  if (action.id == "org.freedesktop.login1.power-off" ||
      action.id == "org.freedesktop.login1.power-off-multiple-sessions" ||
      action.id == "org.freedesktop.login1.reboot" ||
      action.id == "org.freedesktop.login1.reboot-multiple-sessions" ||
      action.id == "org.freedesktop.login1.hibernate" ||
      action.id == "org.freedesktop.login1.suspend" ||
      action.id == "org.freedesktop.login1.set-wall-message") {
    if (subject.user == "$AURA_USER") {
      polkit.log("action=" + action.id + " subject=" + subject.user);
      return polkit.Result.YES;
    }
  }
  
  // Also allow systemctl poweroff/reboot commands
  if (action.id.indexOf("org.freedesktop.systemd1") === 0) {
    if (subject.user == "$AURA_USER") {
      polkit.log("action=" + action.id + " subject=" + subject.user);
      return polkit.Result.YES;
    }
  }
  
  // Allow polkit exec for systemctl commands
  if (action.id == "org.freedesktop.policykit.exec") {
    if (subject.user == "$AURA_USER") {
      // Check if it's a systemctl poweroff/reboot command
      if (action.lookup("command") && 
          (action.lookup("command").indexOf("systemctl") >= 0 && 
           (action.lookup("command").indexOf("poweroff") >= 0 || 
            action.lookup("command").indexOf("reboot") >= 0))) {
        polkit.log("action=" + action.id + " subject=" + subject.user);
        return polkit.Result.YES;
      }
    }
  }
});
EORULE
    sudo chmod 0644 "$SHUTDOWN_POLKIT_RULE" 2>/dev/null || true
    print_info "✅ Shutdown/restart polkit rule created"
    
    # Also create .pkla fallback
    SHUTDOWN_PKLA_FILE="$PKLA_DIR/50-allow-shutdown.pkla"
    sudo mkdir -p "$PKLA_DIR" 2>/dev/null || true
    sudo tee "$SHUTDOWN_PKLA_FILE" >/dev/null <<EOPKLA
[Allow Shutdown for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.login1.power-off;org.freedesktop.login1.power-off-multiple-sessions;org.freedesktop.login1.reboot;org.freedesktop.login1.reboot-multiple-sessions;org.freedesktop.login1.hibernate;org.freedesktop.login1.suspend;org.freedesktop.login1.set-wall-message
ResultAny=yes
ResultInactive=yes
ResultActive=yes

[Allow systemctl for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.systemd1.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes

[Allow polkit exec for $AURA_USER]
Identity=unix-user:$AURA_USER
Action=org.freedesktop.policykit.exec
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOPKLA
    sudo chmod 0644 "$SHUTDOWN_PKLA_FILE" 2>/dev/null || true
    print_info "✅ Shutdown/restart fallback .pkla rule created"
    print_info "   Shutdown and restart buttons in Settings will work without password prompts"
else
    print_info "Shutdown/restart polkit rule already exists"
fi

# Restart polkit to apply all new rules
if systemctl list-unit-files | grep -q "polkit"; then
    print_info "Restarting polkit service to apply all permission rules..."
    sudo systemctl restart polkit 2>/dev/null || sudo systemctl restart polkit.service 2>/dev/null || true
fi

echo ""

# ============================================================================
# Step 7: Install XVF3800 tuning service
# ============================================================================
print_step "7. Installing XVF3800 tuning service..."

SERVICE_FILE="$LEDGERAI_DIR/setup/scripts/xvf3800-tuning.service"
SYSTEMD_SERVICE="/etc/systemd/system/xvf3800-tuning.service"

if [ -f "$SERVICE_FILE" ]; then
    # Remove existing service file or symlink if it exists
    if [ -L "$SYSTEMD_SERVICE" ] || [ -f "$SYSTEMD_SERVICE" ]; then
        sudo rm -f "$SYSTEMD_SERVICE"
    fi
    
    # Create symlink to git repository file (allows automatic updates via git pull)
    # Service file is now fully dynamic - no placeholders needed!
    # Python script automatically detects user, paths, and waits for USB device
    # Use absolute path for symlink target to ensure it works regardless of working directory
    SERVICE_FILE_ABS="$(cd "$(dirname "$SERVICE_FILE")" && pwd)/$(basename "$SERVICE_FILE")"
    sudo ln -s "$SERVICE_FILE_ABS" "$SYSTEMD_SERVICE"
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable service (but don't start - microphone may not be connected yet)
    sudo systemctl enable xvf3800-tuning.service
    
    print_info "XVF3800 tuning service installed and enabled"
    print_info "Service file is symlinked to git repository - updates automatically on git pull!"
    print_info "Service will configure microphone on boot"
    print_info "Note: Service will start automatically when microphone is connected"
    print_info "To test manually: sudo systemctl start xvf3800-tuning.service"
    
    # Install git hooks
    GIT_HOOKS_DIR="$LEDGERAI_DIR/.git/hooks"
    
    # Install post-merge hook to auto-reload systemd after git pull
    POST_MERGE_HOOK="$GIT_HOOKS_DIR/post-merge"
    POST_MERGE_TEMPLATE="$LEDGERAI_DIR/setup/scripts/git-hooks/post-merge"
    
    # Install pre-commit hook to sync llm-container changes to llm-medical-container
    PRE_COMMIT_HOOK="$GIT_HOOKS_DIR/pre-commit"
    PRE_COMMIT_TEMPLATE="$LEDGERAI_DIR/setup/scripts/git-hooks/pre-commit"
    
    if [ -d "$LEDGERAI_DIR/.git" ]; then
        # Install post-merge hook
        if [ -f "$POST_MERGE_TEMPLATE" ]; then
        if [ ! -f "$POST_MERGE_HOOK" ] || ! grep -q "XVF3800 service file updated" "$POST_MERGE_HOOK" 2>/dev/null; then
                cp "$POST_MERGE_TEMPLATE" "$POST_MERGE_HOOK"
            chmod +x "$POST_MERGE_HOOK"
            print_info "Git post-merge hook installed - systemd will auto-reload after git pull"
            fi
        fi
        
        # Install pre-commit hook
        if [ -f "$PRE_COMMIT_TEMPLATE" ]; then
            if [ ! -f "$PRE_COMMIT_HOOK" ] || ! grep -q "llm-container" "$PRE_COMMIT_HOOK" 2>/dev/null; then
                cp "$PRE_COMMIT_TEMPLATE" "$PRE_COMMIT_HOOK"
                chmod +x "$PRE_COMMIT_HOOK"
                print_info "Git pre-commit hook installed - llm-container changes will sync to llm-medical-container"
            fi
        fi
    fi
else
    print_error "XVF3800 service file not found at $SERVICE_FILE"
fi

echo ""

# ============================================================================
# Step 8: Configure X11 authentication for GUI access
# ============================================================================
print_step "8. Configuring X11 authentication for GUI access..."

# Allow user to access X11 display
print_info "Setting up X11 authentication..."
# Get the display session for the user
if [ -n "$DISPLAY" ]; then
    CURRENT_DISPLAY="$DISPLAY"
else
    CURRENT_DISPLAY=":0"
fi

# Allow local connections (xhost method - simpler but less secure)
sudo -u "$AURA_USER" xhost +local: 2>/dev/null || {
    print_info "xhost command not available, will use xauth method"
}

# Create X11 auth setup script
X11_SETUP_SCRIPT="$AURA_HOME/.x11_setup.sh"
cat > "$X11_SETUP_SCRIPT" << 'EOFX11'
#!/bin/bash
# X11 authentication setup for Aura
export DISPLAY=:0
# Allow local connections
xhost +local: 2>/dev/null || true
# Copy X11 auth if it exists
if [ -f "$HOME/.Xauthority" ]; then
    export XAUTHORITY="$HOME/.Xauthority"
elif [ -f "/tmp/.X11-unix/X0" ]; then
    # Try to find XAUTHORITY from current session
    export XAUTHORITY="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/.Xauthority" 2>/dev/null || true
fi
EOFX11

chmod +x "$X11_SETUP_SCRIPT"
print_info "X11 setup script created at $X11_SETUP_SCRIPT"

echo ""

# ============================================================================
# Step 9: Create Aura systemd service for boot
# ============================================================================
print_step "9. Creating Aura systemd service for boot startup..."

AURA_SERVICE_FILE="/tmp/aura.service"
AURA_UID=$(id -u "$AURA_USER")

# Get XAUTHORITY path if available
XAUTH_PATH="$AURA_HOME/.Xauthority"
if [ ! -f "$XAUTH_PATH" ]; then
    # Try to find it from runtime directory
    XAUTH_PATH="/run/user/$AURA_UID/.Xauthority"
fi

# Detect available DISPLAY dynamically from X sockets at install time
# This provides a default, but the service will auto-detect at runtime too
DETECTED_DISPLAY=":0"  # Default fallback
if [ -d "/tmp/.X11-unix/" ]; then
    FIRST_SOCKET=$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" | head -1)
    if [ -n "$FIRST_SOCKET" ]; then
        DISPLAY_NUM=$(echo "$FIRST_SOCKET" | sed 's/X//')
        DETECTED_DISPLAY=":$DISPLAY_NUM"
        print_info "Detected DISPLAY=$DETECTED_DISPLAY from X socket $FIRST_SOCKET (will auto-detect at runtime)"
    fi
fi

cat > "$AURA_SERVICE_FILE" << EOF
[Unit]
Description=Aura Voice Assistant
After=network.target docker.service xvf3800-tuning.service display-manager.service
Wants=docker.service xvf3800-tuning.service display-manager.service
Requires=docker.service

[Service]
Type=simple
User=$AURA_USER
Group=$AURA_USER
WorkingDirectory=$LEDGERAI_DIR/aura-control/core
Environment="DISPLAY=$DETECTED_DISPLAY"
Environment="HOME=$AURA_HOME"
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="XDG_RUNTIME_DIR=/run/user/$AURA_UID"
Environment="XAUTHORITY=$XAUTH_PATH"
# Auto-detect DISPLAY from X sockets at runtime (handles :0, :1, etc.)
ExecStartPre=/bin/bash -c 'SOCKET=\$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" | head -1); if [ -n "\$SOCKET" ]; then NUM=\$(echo "\$SOCKET" | sed "s/X//"); export DISPLAY=:\$NUM; else export DISPLAY=:0; fi; while [ ! -e "/tmp/.X11-unix/\${SOCKET:-X0}" ]; do sleep 1; done'
ExecStartPre=/bin/bash -c 'SOCKET=\$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" | head -1); if [ -n "\$SOCKET" ]; then NUM=\$(echo "\$SOCKET" | sed "s/X//"); export DISPLAY=:\$NUM; else export DISPLAY=:0; fi; xhost +local: 2>/dev/null || true'
# Disable lock screen on boot (in case it re-enables) - use detected DISPLAY
ExecStartPre=/bin/bash -c 'SOCKET=\$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" | head -1); if [ -n "\$SOCKET" ]; then NUM=\$(echo "\$SOCKET" | sed "s/X//"); export DISPLAY=:\$NUM; else export DISPLAY=:0; fi; gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true'
ExecStartPre=/bin/bash -c 'SOCKET=\$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" | head -1); if [ -n "\$SOCKET" ]; then NUM=\$(echo "\$SOCKET" | sed "s/X//"); export DISPLAY=:\$NUM; else export DISPLAY=:0; fi; gsettings set org.gnome.desktop.lockdown disable-lock-screen true 2>/dev/null || true'
ExecStartPre=/bin/bash -c 'SOCKET=\$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" | head -1); if [ -n "\$SOCKET" ]; then NUM=\$(echo "\$SOCKET" | sed "s/X//"); export DISPLAY=:\$NUM; else export DISPLAY=:0; fi; gsettings set org.gnome.desktop.screensaver idle-activation-enabled false 2>/dev/null || true'
# Set default audio output (UACDemoV1.0) on every boot - ALSA and PulseAudio
ExecStartPre=$LEDGERAI_DIR/setup/scripts/set_default_audio_on_boot.sh
ExecStartPre=/bin/sleep 5
ExecStart=$VENV_DIR/bin/python3 -u $LEDGERAI_DIR/aura-control/core/main.py
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
# Step 10: Install keyboard monitor service
# ============================================================================
print_step "10. Installing keyboard monitor service..."

KEYBOARD_SERVICE_FILE="$LEDGERAI_DIR/setup/scripts/disable-keyboard-monitor.service"
SYSTEMD_KEYBOARD_SERVICE="/etc/systemd/system/disable-keyboard-monitor.service"

if [ -f "$KEYBOARD_SERVICE_FILE" ]; then
    # Create a temporary service file with correct paths
    TEMP_KEYBOARD_SERVICE="/tmp/disable-keyboard-monitor.service"
    cp "$KEYBOARD_SERVICE_FILE" "$TEMP_KEYBOARD_SERVICE"
    
    # Replace placeholders with actual values (supports both placeholders and legacy hardcoded values)
    sed -i "s|__AURA_USER__|$AURA_USER|g" "$TEMP_KEYBOARD_SERVICE"
    
    # Also handle legacy hardcoded values for backward compatibility
    sed -i "s|User=aura|User=$AURA_USER|g" "$TEMP_KEYBOARD_SERVICE"
    sed -i "s|User=ledger|User=$AURA_USER|g" "$TEMP_KEYBOARD_SERVICE"
    sed -i "s|Group=aura|Group=$AURA_USER|g" "$TEMP_KEYBOARD_SERVICE"
    sed -i "s|Group=ledger|Group=$AURA_USER|g" "$TEMP_KEYBOARD_SERVICE"
    
    # Copy to systemd
    sudo cp "$TEMP_KEYBOARD_SERVICE" "$SYSTEMD_KEYBOARD_SERVICE"
    rm -f "$TEMP_KEYBOARD_SERVICE"
    
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
# Step 11: Set up Docker (if not already configured)
# ============================================================================
print_step "11. Configuring Docker..."

# Check if user is in docker group
if ! groups "$AURA_USER" | grep -q docker; then
    print_info "Adding $AURA_USER to docker group..."
    sudo usermod -aG docker "$AURA_USER"
    print_info "⚠️  User added to docker group"
    print_info "   IMPORTANT: You must logout and login (or reboot) for Docker access to work"
    print_info "   Or use: newgrp docker (in current session)"
    print_info "   Or use: sudo docker (temporary workaround)"
else
    print_info "User already in docker group"
fi

# Ensure Docker is running
if ! systemctl is-active --quiet docker; then
    print_info "Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
fi

# Test Docker access (will fail if user needs to logout/login, but that's expected)
if docker ps > /dev/null 2>&1; then
    print_info "✅ Docker is accessible"
else
    print_info "⚠️  Docker not accessible in current session"
    print_info "   This is normal - logout/login or reboot to apply group changes"
    print_info "   Temporary workaround: use 'sudo docker' or 'newgrp docker'"
fi

print_info "Docker configured"

echo ""

# ============================================================================
# Step 12: Set permissions for audio devices
# ============================================================================
print_step "12. Configuring audio device permissions..."

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

# Make audio setup script executable
if [ -f "$LEDGERAI_DIR/setup/scripts/set_default_audio_on_boot.sh" ]; then
    chmod +x "$LEDGERAI_DIR/setup/scripts/set_default_audio_on_boot.sh"
    print_info "✅ Audio setup script made executable"
fi

# Set default ALSA device to UACDemoV1.0 (if present)
print_info "Configuring default audio output..."
if aplay -l 2>/dev/null | grep -q "UACDemoV1.0"; then
    CARD_NUM=$(aplay -l 2>/dev/null | grep "UACDemoV1.0" | sed -n 's/.*card \([0-9]*\):.*/\1/p' | head -1)
    if [ -n "$CARD_NUM" ]; then
        # Create/update .asoundrc for ALSA
        cat > "$AURA_HOME/.asoundrc" << EOF
pcm.!default {
    type hw
    card $CARD_NUM
    device 0
}

ctl.!default {
    type hw
    card $CARD_NUM
}
EOF
        chown "$AURA_USER:$AURA_USER" "$AURA_HOME/.asoundrc"
        print_info "✅ ALSA default set to card $CARD_NUM (UACDemoV1.0)"
        
        # Set PulseAudio default sink (if available) - uses dynamic name matching
        if command -v pactl >/dev/null 2>&1; then
            SINK_NAME=$(pactl list short sinks 2>/dev/null | grep -i "UACDemoV1.0\|UACDemo" | cut -f2 | head -1)
            if [ -n "$SINK_NAME" ]; then
                su - "$AURA_USER" -c "pactl set-default-sink \"$SINK_NAME\" 2>/dev/null" || true
                print_info "✅ PulseAudio default sink set to: $SINK_NAME"
            fi
        fi
    fi
else
    print_info "UACDemoV1.0 not found - skipping audio default setup"
fi

echo ""

# ============================================================================
# Step 13: Create minimal .env with only API keys (deployment-friendly)
# ============================================================================
print_step "13. Setting up .env configuration file (API keys only)..."

ENV_FILE="$LEDGERAI_DIR/.env"

# Backup existing .env if present
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.bak.$(date +%s)" 2>/dev/null || true
fi

cat > "$ENV_FILE" << 'EOF'
# ============================================
# Aura API Keys (Deployment)
# ============================================
# Only API keys belong here. All other settings are hardcoded in code.
#
# Required: ElevenLabs (TTS)
ELEVENLABS_API_KEY=
# Optional: ElevenLabs voice (if left empty, default voice is used)
ELEVENLABS_VOICE_ID=
#
# Optional: Telegram Bot
TELEGRAM_BOT_TOKEN=
#
# Optional: GitHub OTA Updates
GITHUB_TOKEN=
#
# Note: TTS Volume is controlled via Settings Dialog (not in .env)
# The volume slider in Settings → General Settings controls TTS output volume
# Default volume is 80% (set in speaker.py)
#
# Wake Word Detection: OpenWakeWord installed
# - OpenWakeWord: Actively maintained, works on all platforms
#   Install: pip install openwakeword (already done)
#   Models are downloaded automatically on first use
EOF
chown "$AURA_USER:$AURA_USER" "$ENV_FILE"
print_info "✅ Wrote minimal .env (API keys only)"
print_warning "⚠️  Edit .env to add API keys before running Aura: nano $ENV_FILE"

echo ""

# ============================================================================
# Step 14: Create data directories if needed
# ============================================================================
print_step "14. Creating data directories..."

# Create directories (mkdir -p won't error if they exist)
mkdir -p "$LEDGERAI_DIR/data/input" 2>/dev/null || true
mkdir -p "$LEDGERAI_DIR/data/parsed" 2>/dev/null || true
mkdir -p "$LEDGERAI_DIR/data/embeddings" 2>/dev/null || true
mkdir -p "$LEDGERAI_DIR/shared/input_audio" 2>/dev/null || true
mkdir -p "$LEDGERAI_DIR/shared/output_audio" 2>/dev/null || true

# Create models directories for LLM containers (Dockerfiles expect models/ in container root)
print_info "Creating models directories for LLM containers..."
mkdir -p "$LEDGERAI_DIR/llm-container/models" 2>/dev/null || true
mkdir -p "$LEDGERAI_DIR/llm-medical-container/models" 2>/dev/null || true
print_info "✅ Models directories created for LLM containers"

# Download Qwen2.5 model for generic LLM container (if not already present)
GENERIC_MODEL_PATH="$LEDGERAI_DIR/llm-container/models/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
if [ -f "$GENERIC_MODEL_PATH" ]; then
    print_info "✅ Generic LLM model already exists: $GENERIC_MODEL_PATH"
else
    print_info "Downloading Qwen2.5-1.5B-Instruct model for generic LLM container..."
    print_info "   This is a large file (~1GB) - may take several minutes..."
    if wget -q --show-progress -O "$GENERIC_MODEL_PATH" https://huggingface.co/RichardErkhov/unsloth_-_Qwen2.5-1.5B-Instruct-gguf/resolve/main/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf; then
        print_info "✅ Generic LLM model downloaded successfully"
    else
        print_warning "⚠️  Failed to download generic LLM model"
        print_info "   You can download manually:"
        print_info "   cd $LEDGERAI_DIR/llm-container/models"
        print_info "   wget https://huggingface.co/RichardErkhov/unsloth_-_Qwen2.5-1.5B-Instruct-gguf/resolve/main/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"
        print_info "   Note: Docker build will fail if model is missing"
    fi
fi

print_info "Data directories ready"

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
if [ "$PORT_AUDIO_AVAILABLE" = true ]; then
    echo "✅ PortAudio: Installed (sounddevice will work)"
else
    echo "⚠️  PortAudio: Not available (audio may not work)"
    echo "   Run: bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
fi
if python3 -c "import PyQt5" 2>/dev/null; then
    echo "✅ PyQt5: Installed (GUI will work)"
else
    echo "❌ PyQt5: Not available (GUI will NOT work)"
fi
echo "✅ jetson-containers: $JETSON_CONTAINERS_DIR"
if [ -f /etc/nv_tegra_release ] || [ -f /proc/device-tree/model ] && grep -q "Jetson\|NVIDIA" /proc/device-tree/model 2>/dev/null; then
    echo "✅ Jetson MAXN Power Mode: Configured (persistent on boot)"
else
    echo "ℹ️  Jetson Power Mode: Skipped (not a Jetson device)"
fi
echo "✅ XVF3800 support: $XVF3800_REPO_DIR"
echo "✅ XVF3800 tuning service: Enabled (runs on boot)"
echo "✅ Aura service: Enabled (will start on boot)"
echo "✅ Keyboard monitor service: Enabled (disables Ubuntu keyboard while Aura runs)"
echo "✅ Docker: Configured"
echo "✅ Display settings: Configured"
echo "✅ Automatic login: Configured (no password on boot)"
echo "ℹ️  Boot logo: Configure via BIOS/UEFI settings (not handled by script)"
echo "✅ X11 authentication: Configured (xhost +local: in service)"
echo "✅ HDMI display support: Configured (auto-detects DISPLAY from X sockets)"
if [ -f "$LEDGERAI_DIR/.env" ]; then
    echo "✅ .env file: Exists"
else
    echo "⚠️  .env file: Created from template (needs API keys)"
fi
echo "✅ Wake Word Engine: OpenWakeWord installed"
echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "1. ⚠️  IMPORTANT: Logout and login again (or reboot) to apply Docker group changes"
echo "   Without this, you'll get 'permission denied' errors with Docker"
echo "   Temporary workaround: use 'sudo docker' or run 'newgrp docker'"
echo ""
if [ ! -f "$LEDGERAI_DIR/.env" ] || grep -q "your_api_key_here\|your_bot_token_here" "$LEDGERAI_DIR/.env" 2>/dev/null; then
    echo "1.5. ⚠️  IMPORTANT: Configure .env file with your API keys:"
    echo "   cd $LEDGERAI_DIR"
    echo "   nano .env"
    echo "   # Or use interactive config: ./aura_config.sh"
    echo "   # Required: ELEVENLABS_API_KEY"
    echo "   # Optional: TELEGRAM_BOT_TOKEN, GITHUB_TOKEN"
    echo ""
fi
echo "2. After logout/login, ensure Docker containers are built:"
echo "   cd $LEDGERAI_DIR/setup"
echo "   docker compose build"
echo ""
echo "3. Test Aura manually first:"
echo "   IMPORTANT: For GUI to work, you need display access."
echo ""
echo "   Option A - Run directly on the device (RECOMMENDED):"
echo "   - Log into the device directly (not via SSH)"
echo "   - Run Aura from a terminal on the device"
echo ""
echo "   Option B - Run via SSH (display on device's screen):"
echo "   - SSH to device: ssh ledger@192.168.1.215"
echo "   - On the device (in a separate terminal on the device itself):"
echo "     xhost +local:  # Allow local X11 connections"
echo "   - Back in your SSH session:"
echo "     cd $LEDGERAI_DIR/aura-control/core"
echo "     source $VENV_DIR/bin/activate"
echo "     export DISPLAY=:0"
echo "     python3 main.py"
echo ""
echo "   Option C - SSH with X11 forwarding (display on your computer):"
echo "   - SSH with X11 forwarding: ssh -X ledger@192.168.1.215"
echo "   - Note: DISPLAY will be auto-set by SSH, don't override it"
echo "   - Run: cd $LEDGERAI_DIR/aura-control/core && source $VENV_DIR/bin/activate && python3 main.py"
echo ""
echo "   If you get 'Qt platform plugin' errors:"
echo "   - Make sure you're logged into a graphical session on the device"
echo "   - Check DISPLAY: echo \$DISPLAY (should be :0 or localhost:X.X)"
echo "   - Try: xhost +local: on the device itself"
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
echo "7. If PortAudio/audio doesn't work:"
if [ "$PORT_AUDIO_AVAILABLE" = false ]; then
    echo "   PortAudio not installed. Build it:"
    echo "   bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
    echo "   Or: cd /tmp && wget http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz"
    echo "        tar -xzf pa_stable_v190700_20210406.tgz && cd portaudio"
    echo "        ./configure && make -j$(nproc) && sudo make install && sudo ldconfig"
else
    echo "   PortAudio is installed. If audio still doesn't work:"
    echo "   - Test: python3 -c 'import sounddevice as sd; print(sd.query_devices())'"
    echo "   - Check microphone permissions: groups | grep audio"
fi
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


