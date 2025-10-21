#!/bin/bash
# EHR Integration Toggle Script
# Easily enable/disable FHIR integration with SystmOne

CONFIG_FILE="$(dirname "$0")/.env"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "   EHR INTEGRATION TOGGLE"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}⚠️  No .env file found${NC}"
    echo "Creating new .env file..."
    touch "$CONFIG_FILE"
fi

# Function to get current status
get_status() {
    if grep -q "^EHR_INTEGRATION_ENABLED=true" "$CONFIG_FILE" 2>/dev/null; then
        echo "enabled"
    else
        echo "disabled"
    fi
}

# Function to enable EHR
enable_ehr() {
    echo -e "${BLUE}🏥 Enabling EHR Integration...${NC}"
    
    # Remove any existing EHR_INTEGRATION_ENABLED lines
    grep -v "^EHR_INTEGRATION_ENABLED=" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" 2>/dev/null || true
    mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    
    # Add enabled setting
    echo "EHR_INTEGRATION_ENABLED=true" >> "$CONFIG_FILE"
    
    # Add FHIR URL if not present
    if ! grep -q "^SYSTMONE_FHIR_URL=" "$CONFIG_FILE"; then
        echo "" >> "$CONFIG_FILE"
        echo "# FHIR Server URL" >> "$CONFIG_FILE"
        echo "# Test server (no auth required):" >> "$CONFIG_FILE"
        echo "SYSTMONE_FHIR_URL=https://hapi.fhir.org/baseR4" >> "$CONFIG_FILE"
        echo "# Production (when ready):" >> "$CONFIG_FILE"
        echo "# SYSTMONE_FHIR_URL=https://api.systmone.nhs.uk/fhir" >> "$CONFIG_FILE"
    fi
    
    echo ""
    echo -e "${GREEN}✅ EHR Integration ENABLED${NC}"
    echo ""
    echo "Configuration:"
    echo "  • FHIR calls will be made to SystmOne"
    echo "  • Data will be saved to EHR"
    echo "  • Local JSON files still saved (backup)"
    echo ""
    echo "Server: $(grep "^SYSTMONE_FHIR_URL=" "$CONFIG_FILE" | cut -d'=' -f2)"
    echo ""
}

# Function to disable EHR
disable_ehr() {
    echo -e "${BLUE}💤 Disabling EHR Integration...${NC}"
    
    # Remove any existing EHR_INTEGRATION_ENABLED lines
    grep -v "^EHR_INTEGRATION_ENABLED=" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" 2>/dev/null || true
    mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    
    # Add disabled setting
    echo "EHR_INTEGRATION_ENABLED=false" >> "$CONFIG_FILE"
    
    echo ""
    echo -e "${GREEN}✅ EHR Integration DISABLED${NC}"
    echo ""
    echo "Normal Aura mode:"
    echo "  • No FHIR calls made"
    echo "  • Data saved locally only"
    echo "  • No connection to SystmOne"
    echo ""
}

# Function to show status
show_status() {
    local status=$(get_status)
    
    echo "Current Status:"
    echo ""
    
    if [ "$status" == "enabled" ]; then
        echo -e "  ${GREEN}●${NC} EHR Integration: ${GREEN}ENABLED${NC}"
        echo ""
        echo "  FHIR Server: $(grep "^SYSTMONE_FHIR_URL=" "$CONFIG_FILE" | cut -d'=' -f2 || echo 'Not configured')"
        echo ""
        echo "  Data flow:"
        echo "    Patient → Aura Assessment → Local JSON + SystmOne EHR"
    else
        echo -e "  ${RED}○${NC} EHR Integration: ${RED}DISABLED${NC}"
        echo ""
        echo "  Data flow:"
        echo "    Patient → Aura Assessment → Local JSON only"
    fi
    
    echo ""
}

# Main menu
case "${1:-}" in
    on|enable|yes)
        enable_ehr
        ;;
    off|disable|no)
        disable_ehr
        ;;
    status|check)
        show_status
        ;;
    *)
        # Interactive mode
        show_status
        echo "========================================"
        echo ""
        echo "What would you like to do?"
        echo ""
        echo "  1) Enable EHR integration"
        echo "  2) Disable EHR integration"
        echo "  3) Show status only"
        echo "  4) Exit"
        echo ""
        read -p "Enter choice [1-4]: " choice
        
        case $choice in
            1)
                echo ""
                enable_ehr
                ;;
            2)
                echo ""
                disable_ehr
                ;;
            3)
                echo ""
                show_status
                ;;
            4)
                echo ""
                echo "Exiting..."
                echo ""
                exit 0
                ;;
            *)
                echo ""
                echo -e "${RED}Invalid choice${NC}"
                echo ""
                exit 1
                ;;
        esac
        ;;
esac

# Show restart message if Docker is running
if docker ps | grep -q "aura-llm"; then
    echo -e "${YELLOW}⚠️  Docker container is running${NC}"
    echo ""
    echo "To apply changes, restart the container:"
    echo ""
    echo "  cd /Users/rcabello/Documents/GitHub/LedgerAI"
    echo "  docker-compose restart llm"
    echo ""
fi

echo "========================================"
echo ""

