#!/bin/bash
# Check reSpeaker XVF3800 repository for initialization scripts and documentation

XVF3800_REPO_DIR="${HOME}/reSpeaker_XVF3800_USB_4MIC_ARRAY"
JETSON_DIR="$XVF3800_REPO_DIR/host_control/jetson"

echo "=========================================="
echo "  Checking XVF3800 Repository"
echo "=========================================="
echo ""

if [ ! -d "$XVF3800_REPO_DIR" ]; then
    echo "❌ Repository not found at: $XVF3800_REPO_DIR"
    echo "   Run: git clone https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY.git"
    exit 1
fi

echo "✅ Repository found: $XVF3800_REPO_DIR"
echo ""

# Check jetson directory
if [ -d "$JETSON_DIR" ]; then
    echo "📁 Files in host_control/jetson/:"
    ls -lah "$JETSON_DIR" 2>/dev/null | grep -v "^total" || echo "  (empty or inaccessible)"
    echo ""
    
    # Check for README
    if [ -f "$JETSON_DIR/README.md" ] || [ -f "$JETSON_DIR/README" ] || [ -f "$JETSON_DIR/readme.txt" ]; then
        echo "📄 README found:"
        find "$JETSON_DIR" -maxdepth 1 -iname "readme*" -type f
        echo ""
        echo "Content:"
        find "$JETSON_DIR" -maxdepth 1 -iname "readme*" -type f -exec head -50 {} \;
        echo ""
    fi
    
    # Check for Makefile (might have comments)
    if [ -f "$JETSON_DIR/Makefile" ]; then
        echo "📄 Makefile found - checking for comments/documentation:"
        grep -E "^#|^##" "$JETSON_DIR/Makefile" | head -20 || echo "  (no comments found)"
        echo ""
    fi
    
    # Check for any shell scripts
    echo "📜 Shell scripts found:"
    find "$JETSON_DIR" -maxdepth 1 -type f \( -name "*.sh" -o -name "*.bash" \) 2>/dev/null
    echo ""
    
    # Check for any initialization scripts
    echo "🔧 Initialization scripts (checking for init, setup, configure):"
    find "$JETSON_DIR" -maxdepth 1 -type f -iname "*init*" -o -iname "*setup*" -o -iname "*config*" 2>/dev/null
    echo ""
fi

# Check root of repository for README
echo "📁 Checking repository root for documentation:"
if [ -f "$XVF3800_REPO_DIR/README.md" ]; then
    echo "✅ README.md found"
    echo ""
    echo "First 100 lines:"
    head -100 "$XVF3800_REPO_DIR/README.md"
    echo ""
elif [ -f "$XVF3800_REPO_DIR/README" ]; then
    echo "✅ README found"
    head -100 "$XVF3800_REPO_DIR/README"
    echo ""
fi

# Check host_control directory
if [ -d "$XVF3800_REPO_DIR/host_control" ]; then
    echo "📁 Files in host_control/:"
    ls -lah "$XVF3800_REPO_DIR/host_control" 2>/dev/null | grep -v "^total" || echo "  (empty)"
    echo ""
    
    # Check for any README in host_control
    if [ -f "$XVF3800_REPO_DIR/host_control/README.md" ] || [ -f "$XVF3800_REPO_DIR/host_control/README" ]; then
        echo "📄 host_control README:"
        find "$XVF3800_REPO_DIR/host_control" -maxdepth 1 -iname "readme*" -type f -exec head -50 {} \;
        echo ""
    fi
fi

# Check for udev rules or initialization scripts
echo "🔍 Checking for udev rules or system initialization files:"
find "$XVF3800_REPO_DIR" -type f \( -name "*udev*" -o -name "*rules*" -o -name "*init*" -o -name "*systemd*" \) 2>/dev/null | head -10
echo ""

# Check xvf_host binary info
if [ -f "$JETSON_DIR/xvf_host" ]; then
    echo "✅ xvf_host binary found"
    echo "   Size: $(ls -lh "$JETSON_DIR/xvf_host" | awk '{print $5}')"
    echo "   Permissions: $(ls -l "$JETSON_DIR/xvf_host" | awk '{print $1}')"
    echo ""
    
    # Try to get help/usage info
    echo "📖 xvf_host help/usage:"
    "$JETSON_DIR/xvf_host" --help 2>&1 | head -30 || "$JETSON_DIR/xvf_host" -h 2>&1 | head -30 || echo "  (no help available)"
    echo ""
fi

echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""
echo "Repository location: $XVF3800_REPO_DIR"
echo "Jetson directory: $JETSON_DIR"
echo ""
echo "💡 Look for:"
echo "   - README files with initialization instructions"
echo "   - Shell scripts for device setup"
echo "   - udev rules for USB device initialization"
echo "   - Systemd service examples"
echo "   - Documentation about boot-time initialization"
echo ""

