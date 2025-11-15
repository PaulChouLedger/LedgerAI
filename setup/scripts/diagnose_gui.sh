#!/bin/bash
#
# GUI Diagnostic Script for Aura
#
# This script checks for common issues that prevent the GUI from working
# on a new clean install, even when the code is the same.
#
# Usage:
#   bash setup/scripts/diagnose_gui.sh
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}[CHECK] $1${NC}"
    echo -e "${BLUE}==========================================${NC}"
}

print_step "1. Checking X Server Status"

# Check if X server is running (try multiple process names)
X_SERVER_FOUND=false
X_PID=""
X_PROCESS=""

# Try different X server process names
for xproc in Xorg Xwayland X Xtigervnc Xvnc; do
    if pgrep -x "$xproc" > /dev/null 2>&1; then
        X_SERVER_FOUND=true
        X_PID=$(pgrep -x "$xproc" | head -1)
        X_PROCESS="$xproc"
        break
    fi
done

# Also check for processes containing X in their name (for Xorg variants)
if [ "$X_SERVER_FOUND" = false ]; then
    X_PID=$(pgrep -f "^.*/X.*" | head -1)
    if [ -n "$X_PID" ]; then
        X_SERVER_FOUND=true
        X_PROCESS=$(ps -p "$X_PID" -o comm= 2>/dev/null || echo "unknown")
    fi
fi

# Check display manager processes (indicates GUI session)
DISPLAY_MANAGER=""
if pgrep -x "gdm3\|gdm\|lightdm\|sddm" > /dev/null 2>&1; then
    DISPLAY_MANAGER=$(pgrep -xl "gdm3\|gdm\|lightdm\|sddm" | head -1 | awk '{print $2}')
fi

# Check if DISPLAY is set and accessible (more reliable than process check)
DISPLAY_VAL="${DISPLAY:-NOT SET}"
X_ACCESSIBLE=false
if [ "$DISPLAY_VAL" != "NOT SET" ]; then
    # Try to test X11 connection
    if command -v xset >/dev/null 2>&1; then
        if xset q >/dev/null 2>&1; then
            X_ACCESSIBLE=true
        fi
    fi
fi

# Report findings
if [ "$X_SERVER_FOUND" = true ] && [ -n "$X_PID" ]; then
    print_info "✅ X server process found: $X_PROCESS (PID: $X_PID)"
elif [ "$X_ACCESSIBLE" = true ]; then
    print_info "✅ X server accessible (DISPLAY=$DISPLAY_VAL) - process detection may have failed"
elif [ -n "$DISPLAY_MANAGER" ]; then
    print_warning "⚠️  Display manager running ($DISPLAY_MANAGER) but X server process not detected"
    print_info "   This might be normal if using Wayland or remote X11"
elif [ -d "/tmp/.X11-unix/" ]; then
    print_warning "⚠️  X socket directory exists (/tmp/.X11-unix/) but process not detected"
    print_info "   X server may be running under a different name or via remote connection"
else
    print_error "❌ X server appears to be NOT running"
    print_warning "   Solution: Make sure you're logged into a graphical session"
    print_warning "   Note: If GUI works, this check may be too strict - check DISPLAY and X sockets below"
fi

# Check X server sockets (most reliable indicator)
print_info "Checking X server sockets..."
if [ -d "/tmp/.X11-unix/" ]; then
    sockets=$(ls /tmp/.X11-unix/ 2>/dev/null | grep "^X" || true)
    if [ -n "$sockets" ]; then
        print_info "✅ Found X sockets: $(echo $sockets | tr '\n' ' ')"
        print_info "   This indicates X server IS running, even if process not detected"
        
        # Test if sockets are accessible
        for socket in $sockets; do
            socket_path="/tmp/.X11-unix/$socket"
            if [ -S "$socket_path" ]; then
                print_info "   Socket $socket exists and is accessible"
            fi
        done
    else
        print_warning "⚠️  No X sockets found in /tmp/.X11-unix/ (but directory exists)"
    fi
else
    print_warning "⚠️  /tmp/.X11-unix/ directory doesn't exist"
    print_info "   This might be normal if using Wayland or remote X11 connection"
fi

print_step "2. Checking DISPLAY Environment"

DISPLAY_VAL="${DISPLAY:-NOT SET}"
print_info "Current DISPLAY: $DISPLAY_VAL"

if [ "$DISPLAY_VAL" = "NOT SET" ]; then
    print_warning "⚠️  DISPLAY not set"
    # Try to detect available display
    if [ -d "/tmp/.X11-unix/" ]; then
        socket=$(ls /tmp/.X11-unix/ | grep "^X" | head -1)
        if [ -n "$socket" ]; then
            display_num=$(echo $socket | sed 's/X//')
            export DISPLAY=":${display_num}"
            print_info "✅ Set DISPLAY to: $DISPLAY"
        fi
    fi
fi

# Test X11 connection
print_info "Testing X11 connection..."
if command -v xset >/dev/null 2>&1; then
    if xset q >/dev/null 2>&1; then
        print_info "✅ X11 connection successful"
    else
        print_error "❌ X11 connection failed"
        print_warning "   Run: xhost +local:"
    fi
else
    print_warning "⚠️  xset not found (cannot test X11 connection)"
fi

print_step "3. Checking XAUTHORITY"

XAUTH_VAL="${XAUTHORITY:-NOT SET}"
if [ "$XAUTH_VAL" = "NOT SET" ]; then
    # Try default location
    home_dir="${HOME:-$(eval echo ~$USER)}"
    default_xauth="$home_dir/.Xauthority"
    if [ -f "$default_xauth" ]; then
        export XAUTHORITY="$default_xauth"
        print_info "✅ Set XAUTHORITY to: $XAUTHORITY"
    else
        print_warning "⚠️  .Xauthority not found at $default_xauth"
    fi
else
    if [ -f "$XAUTH_VAL" ]; then
        print_info "✅ XAUTHORITY set and file exists: $XAUTH_VAL"
    else
        print_error "❌ XAUTHORITY points to non-existent file: $XAUTH_VAL"
    fi
fi

print_step "4. Checking X11 Permissions"

if command -v xhost >/dev/null 2>&1; then
    print_info "Current xhost access:"
    xhost 2>&1 | grep -v "^Access control" || true
    
    # Try to enable local access
    print_info "Enabling local X11 access..."
    if xhost +local: >/dev/null 2>&1; then
        print_info "✅ Enabled xhost +local:"
    else
        print_warning "⚠️  Failed to enable xhost +local: (may need to run on device directly)"
    fi
else
    print_warning "⚠️  xhost not found"
fi

print_step "5. Checking Qt/PyQt5 Installation"

# Check if PyQt5 is installed
if python3 -c "import PyQt5" 2>/dev/null; then
    print_info "✅ PyQt5 is installed"
    
    # Check Qt version
    QT_VERSION=$(python3 -c "from PyQt5.QtCore import QT_VERSION_STR; print(QT_VERSION_STR)" 2>/dev/null)
    if [ -n "$QT_VERSION" ]; then
        print_info "   Qt version: $QT_VERSION"
    fi
    
    # Check available Qt platform plugins
    print_info "Checking Qt platform plugins..."
    python3 -c "
from PyQt5.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
plugins = app.platformName()
print('   Available platforms:', ', '.join(app.availablePlatformNames()))
print('   Current platform:', plugins)
" 2>&1 | grep -E "(Available|Current)" || print_warning "⚠️  Could not query Qt platforms"
else
    print_error "❌ PyQt5 is NOT installed"
    print_warning "   Install: sudo apt-get install python3-pyqt5"
fi

print_step "6. Checking Display Manager (GDM/LightDM)"

if systemctl is-active --quiet gdm3; then
    print_info "✅ GDM3 display manager is running"
elif systemctl is-active --quiet lightdm; then
    print_info "✅ LightDM display manager is running"
elif systemctl is-active --quiet sddm; then
    print_info "✅ SDDM display manager is running"
else
    print_warning "⚠️  No active display manager detected"
    print_info "   Check: systemctl status gdm3"
fi

print_step "7. Checking Window Manager"

WM=$(pgrep -l "gnome-shell\|kwin\|xfwm\|openbox\|i3" | head -1 | awk '{print $2}')
if [ -n "$WM" ]; then
    print_info "✅ Window manager detected: $WM"
else
    print_warning "⚠️  No window manager detected"
fi

print_step "8. Checking Aura GUI Scripts"

WORKSPACE_DIR="${1:-$(pwd)}"
if [ ! -d "$WORKSPACE_DIR" ]; then
    WORKSPACE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
fi

print_info "Workspace directory: $WORKSPACE_DIR"

GUI_SCRIPT="$WORKSPACE_DIR/aura-control/gui/aura_gui.py"
if [ -f "$GUI_SCRIPT" ]; then
    print_info "✅ GUI script exists: $GUI_SCRIPT"
    
    # Check if script is executable
    if [ -x "$GUI_SCRIPT" ]; then
        print_info "✅ GUI script is executable"
    else
        print_warning "⚠️  GUI script is not executable (not required, but checked)"
    fi
    
    # Try to compile/validate
    if python3 -m py_compile "$GUI_SCRIPT" 2>/dev/null; then
        print_info "✅ GUI script compiles without errors"
    else
        print_error "❌ GUI script has syntax errors"
    fi
else
    print_error "❌ GUI script not found: $GUI_SCRIPT"
fi

print_step "9. Testing Simple Qt Application"

print_info "Testing minimal Qt window creation..."
python3 << 'EOF'
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

try:
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Aura Test Window")
    window.setGeometry(100, 100, 400, 300)
    
    label = QLabel("If you can see this, Qt is working!", window)
    label.setGeometry(50, 100, 300, 50)
    label.setStyleSheet("font-size: 18px; color: white; background: black;")
    
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    
    print("✅ Test window created successfully")
    print("   Window should be visible on screen")
    print("   Press Ctrl+C to close")
    
    # Keep window open for 3 seconds
    import time
    time.sleep(3)
    
except Exception as e:
    print(f"❌ Failed to create test window: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_info "✅ Qt test window was created successfully"
else
    print_error "❌ Qt test window creation failed"
    print_warning "   This indicates a Qt/X11 display issue"
fi

print_step "10. Systemd Service Check (if running as service)"

if systemctl is-active --quiet aura.service; then
    print_info "✅ Aura service is running"
    
    # Check service environment
    print_info "Service DISPLAY:"
    systemctl show aura.service | grep -i display || print_warning "   DISPLAY not set in service"
    
    print_info "Service environment:"
    systemctl show aura.service | grep -i "environment\|user\|group" | head -5
else
    print_info "ℹ️  Aura service is not running (this is OK if running manually)"
fi

echo ""
print_step "SUMMARY"
echo ""
print_info "Common fixes if GUI is not working:"
echo "  1. Ensure you're logged into a graphical session"
echo "  2. Set DISPLAY: export DISPLAY=:0"
echo "  3. Allow X11 access: xhost +local:"
echo "  4. Check XAUTHORITY: ls -la ~/.Xauthority"
echo "  5. Verify PyQt5: python3 -c 'import PyQt5'"
echo "  6. Test Qt: python3 setup/scripts/diagnose_gui.sh (run step 9 test)"
echo ""
print_info "If test window (step 9) doesn't appear, the issue is with X11/Qt setup, not the Aura code."
echo ""

