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

# Create temporary virtual environment
VENV_DIR="/tmp/rms_test_venv_$$"
echo -e "${YELLOW}[STEP]${NC} Creating fresh virtual environment..."
echo "  Location: $VENV_DIR"
echo ""

# Remove if exists
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi

# Create venv
$PYTHON_CMD -m venv "$VENV_DIR"
echo -e "${GREEN}✅${NC} Virtual environment created"
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

# Cleanup
echo -e "${YELLOW}[STEP]${NC} Cleaning up virtual environment..."
deactivate 2>/dev/null || true
rm -rf "$VENV_DIR"
echo -e "${GREEN}✅${NC} Cleanup complete"
echo ""

# Exit with test result
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Test passed! Microphone is working in clean environment.${NC}"
    exit 0
else
    echo -e "${RED}❌ Test failed. Check microphone connection and permissions.${NC}"
    exit $TEST_EXIT_CODE
fi

