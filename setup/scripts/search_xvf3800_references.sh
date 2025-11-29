#!/bin/bash
# Search for all references to XVF3800 4-Mic Array in the system
# Helps debug why sound capture isn't working on boot

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Searching for XVF3800 References${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

LEDGERAI_DIR="${1:-$HOME/LedgerAI}"

# Search patterns
PATTERNS=(
    "XVF3800"
    "XVF3800 4-Mic Array"
    "reSpeaker_XVF3800"
    "Seeed_Studio_reSpeaker_XVF3800"
    "ArrayUAC10"
    "xvf3800"
    "xvf_host"
)

echo -e "${YELLOW}[INFO]${NC} Searching in: $LEDGERAI_DIR"
echo -e "${YELLOW}[INFO]${NC} Also searching system-wide..."
echo ""

# Function to search in files
search_in_files() {
    local pattern="$1"
    local search_path="$2"
    
    echo -e "${GREEN}[SEARCH]${NC} Pattern: '$pattern'"
    echo "----------------------------------------"
    
    # Search in LedgerAI directory
    if [ -d "$search_path" ]; then
        echo "Files in $search_path:"
        grep -r -l -i "$pattern" "$search_path" 2>/dev/null | head -20 || echo "  (none found)"
    fi
    
    # Search system-wide (common locations)
    echo ""
    echo "System files:"
    
    # Systemd services
    if [ -d "/etc/systemd/system" ]; then
        SYSTEMD_MATCHES=$(grep -r -l -i "$pattern" /etc/systemd/system 2>/dev/null || true)
        if [ -n "$SYSTEMD_MATCHES" ]; then
            echo "  Systemd services:"
            echo "$SYSTEMD_MATCHES" | sed 's/^/    /'
        fi
    fi
    
    # User systemd services
    if [ -d "$HOME/.config/systemd/user" ]; then
        USER_SYSTEMD_MATCHES=$(grep -r -l -i "$pattern" "$HOME/.config/systemd/user" 2>/dev/null || true)
        if [ -n "$USER_SYSTEMD_MATCHES" ]; then
            echo "  User systemd services:"
            echo "$USER_SYSTEMD_MATCHES" | sed 's/^/    /'
        fi
    fi
    
    # Shell scripts in common locations
    for dir in /usr/local/bin /usr/bin /opt /home; do
        if [ -d "$dir" ]; then
            SCRIPTS=$(find "$dir" -maxdepth 3 -type f -name "*.sh" -exec grep -l -i "$pattern" {} \; 2>/dev/null | head -5 || true)
            if [ -n "$SCRIPTS" ]; then
                echo "  Scripts in $dir:"
                echo "$SCRIPTS" | sed 's/^/    /'
            fi
        fi
    done
    
    echo ""
}

# Search for each pattern
for pattern in "${PATTERNS[@]}"; do
    search_in_files "$pattern" "$LEDGERAI_DIR"
done

# Also search for xvf_host specifically
echo -e "${GREEN}[SEARCH]${NC} Looking for xvf_host binary..."
echo "----------------------------------------"
if command -v xvf_host >/dev/null 2>&1; then
    echo "  Found: $(which xvf_host)"
elif [ -f "$HOME/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host" ]; then
    echo "  Found: $HOME/reSpeaker_XVF3800_USB_4MIC_ARRAY/host_control/jetson/xvf_host"
else
    echo "  (not found)"
fi
echo ""

# Check PulseAudio configuration
echo -e "${GREEN}[SEARCH]${NC} PulseAudio configuration..."
echo "----------------------------------------"
if command -v pactl >/dev/null 2>&1; then
    echo "PulseAudio sources matching XVF3800:"
    pactl list short sources 2>/dev/null | grep -i "XVF3800\|reSpeaker\|ArrayUAC10" || echo "  (none found)"
    echo ""
    echo "PulseAudio sinks matching XVF3800:"
    pactl list short sinks 2>/dev/null | grep -i "XVF3800\|reSpeaker\|ArrayUAC10" || echo "  (none found)"
else
    echo "  PulseAudio not available"
fi
echo ""

# Check ALSA configuration
echo -e "${GREEN}[SEARCH]${NC} ALSA configuration..."
echo "----------------------------------------"
if [ -f "$HOME/.asoundrc" ]; then
    echo "  Found: $HOME/.asoundrc"
    grep -i "XVF3800\|reSpeaker\|ArrayUAC10" "$HOME/.asoundrc" || echo "    (no matches)"
fi
if [ -f "/etc/asound.conf" ]; then
    echo "  Found: /etc/asound.conf"
    grep -i "XVF3800\|reSpeaker\|ArrayUAC10" /etc/asound.conf || echo "    (no matches)"
fi
echo ""

# Check USB devices
echo -e "${GREEN}[SEARCH]${NC} USB devices..."
echo "----------------------------------------"
if command -v lsusb >/dev/null 2>&1; then
    lsusb | grep -i "XVF\|reSpeaker\|Seeed" || echo "  (none found)"
else
    echo "  lsusb not available"
fi
echo ""

# Check for running processes
echo -e "${GREEN}[SEARCH]${NC} Running processes..."
echo "----------------------------------------"
ps aux | grep -i "xvf\|reSpeaker\|XVF3800" | grep -v grep || echo "  (none found)"
echo ""

# Check udev rules
echo -e "${GREEN}[SEARCH]${NC} udev rules..."
echo "----------------------------------------"
if [ -d "/etc/udev/rules.d" ]; then
    UDEV_MATCHES=$(grep -r -l -i "XVF3800\|reSpeaker\|xvf" /etc/udev/rules.d 2>/dev/null || true)
    if [ -n "$UDEV_MATCHES" ]; then
        echo "$UDEV_MATCHES" | sed 's/^/  /'
    else
        echo "  (none found)"
    fi
fi
echo ""

# Check environment variables
echo -e "${GREEN}[SEARCH]${NC} Environment variables..."
echo "----------------------------------------"
env | grep -i "XVF\|reSpeaker\|xvf" || echo "  (none found)"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Search Complete${NC}"
echo -e "${BLUE}========================================${NC}"

