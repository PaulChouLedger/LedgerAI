#!/bin/bash
# Test RMS levels in a completely fresh virtual environment
# This helps isolate microphone issues from Aura installation dependencies

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RMS Test in Clean Virtual Environment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Detect Python version
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
else
    PYTHON_CMD="python3"
fi

echo -e "${YELLOW}[INFO]${NC} Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEDGERAI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_rms_fresh_install.py"

# Check if PortAudio is installed system-wide (required for sounddevice)
echo -e "${YELLOW}[STEP]${NC} Checking PortAudio installation..."
PORTAUDIO_AVAILABLE=false

# Check if PortAudio library exists
if ldconfig -p 2>/dev/null | grep -q libportaudio || [ -f "/usr/local/lib/libportaudio.so" ] || [ -f "/usr/lib/libportaudio.so" ]; then
    echo -e "${GREEN}✅${NC} PortAudio library found system-wide"
    PORTAUDIO_AVAILABLE=true
else
    echo -e "${RED}❌${NC} PortAudio library not found"
    echo -e "${YELLOW}[INFO]${NC} PortAudio must be installed system-wide for sounddevice to work"
    echo -e "${YELLOW}[INFO]${NC} Attempting to install PortAudio..."
    
    # Try to install via apt first
    if sudo apt install -y libportaudio2 libportaudio-dev 2>/dev/null; then
        sudo ldconfig
        if ldconfig -p 2>/dev/null | grep -q libportaudio; then
            echo -e "${GREEN}✅${NC} PortAudio installed via apt"
            PORTAUDIO_AVAILABLE=true
        fi
    fi
    
    # If apt install failed, try building from source
    if [ "$PORTAUDIO_AVAILABLE" = false ]; then
        echo -e "${YELLOW}[INFO]${NC} Building PortAudio from source..."
        cd /tmp
        rm -rf portaudio portaudio.tgz
        
        if wget -q "http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz" -O portaudio.tgz; then
            tar -xzf portaudio.tgz
            cd portaudio
            
            if ./configure && make -j$(nproc); then
                sudo make install
                sudo ldconfig
                
                if ldconfig -p 2>/dev/null | grep -q libportaudio; then
                    echo -e "${GREEN}✅${NC} PortAudio built and installed from source"
                    PORTAUDIO_AVAILABLE=true
                fi
            fi
            
            cd /tmp
            rm -rf portaudio portaudio.tgz
        fi
    fi
fi

if [ "$PORTAUDIO_AVAILABLE" = false ]; then
    echo -e "${RED}❌ ERROR:${NC} PortAudio is required but could not be installed"
    echo -e "${YELLOW}[INFO]${NC} Please install PortAudio manually:"
    echo -e "${YELLOW}[INFO]${NC}   sudo apt install -y libportaudio2 libportaudio-dev"
    echo -e "${YELLOW}[INFO]${NC}   Or build from source: bash $LEDGERAI_DIR/setup/scripts/build_portaudio.sh"
    exit 1
fi

echo ""

# Create persistent virtual environment (survives reboot)
VENV_DIR="$HOME/rms_test_venv"
echo -e "${YELLOW}[STEP]${NC} Setting up virtual environment..."
echo "  Location: $VENV_DIR"
echo ""

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}[INFO]${NC} Virtual environment already exists"
    echo -e "${YELLOW}[INFO]${NC} Reusing existing environment (will reinstall packages if needed)"
    read -p "Recreate from scratch? (yes/no): " RECREATE
    if [ "$RECREATE" = "yes" ]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}✅${NC} Virtual environment recreated"
    else
        echo -e "${GREEN}✅${NC} Using existing virtual environment"
    fi
else
    # Create venv
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✅${NC} Virtual environment created"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[STEP]${NC} Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✅${NC} Virtual environment activated"
echo ""

# Upgrade pip
echo -e "${YELLOW}[STEP]${NC} Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✅${NC} pip upgraded"
echo ""

# Install minimal requirements
echo -e "${YELLOW}[STEP]${NC} Installing minimal requirements..."
echo "  Installing: sounddevice, numpy"
pip install sounddevice numpy --quiet
echo -e "${GREEN}✅${NC} Requirements installed"
echo ""

# Verify installations
echo -e "${YELLOW}[STEP]${NC} Verifying installations..."
python3 -c "import sounddevice; print(f'✅ sounddevice: {sounddevice.__version__}')" || {
    echo -e "${RED}❌${NC} sounddevice import failed"
    exit 1
}
python3 -c "import numpy; print(f'✅ numpy: {numpy.__version__}')" || {
    echo -e "${RED}❌${NC} numpy import failed"
    exit 1
}
echo ""

# Check if test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo -e "${RED}❌ ERROR:${NC} Test script not found: $TEST_SCRIPT"
    exit 1
fi

# Run the test
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Running RMS Level Test${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}[INFO]${NC} This will test microphone RMS levels in a clean environment"
echo -e "${YELLOW}[INFO]${NC} No Aura dependencies are installed - just sounddevice and numpy"
echo ""

python3 "$TEST_SCRIPT"
TEST_EXIT_CODE=$?

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Test Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Note: Virtual environment is preserved for reboot testing
echo -e "${YELLOW}[INFO]${NC} Virtual environment preserved at: $VENV_DIR"
echo -e "${YELLOW}[INFO]${NC} To use after reboot:"
echo -e "${YELLOW}[INFO]${NC}   source $VENV_DIR/bin/activate"
echo -e "${YELLOW}[INFO]${NC}   python3 $TEST_SCRIPT"
echo ""
deactivate 2>/dev/null || true
echo ""

# Exit with test result
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Test passed! Microphone is working in clean environment.${NC}"
    exit 0
else
    echo -e "${RED}❌ Test failed. Check microphone connection and permissions.${NC}"
    exit $TEST_EXIT_CODE
fi

