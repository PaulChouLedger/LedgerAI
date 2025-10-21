#!/bin/bash
# ============================================================================
# AURA UNIFIED CONFIGURATION MANAGER
# Easy management of all Aura settings from one place
# ============================================================================

CONFIG_FILE=".env"
EXAMPLE_FILE=".env.example"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "   $1"
    echo "========================================================================"
    echo ""
}

get_config_value() {
    local key=$1
    if [ -f "$CONFIG_FILE" ]; then
        grep "^${key}=" "$CONFIG_FILE" | cut -d'=' -f2- || echo ""
    else
        echo ""
    fi
}

set_config_value() {
    local key=$1
    local value=$2
    
    if [ ! -f "$CONFIG_FILE" ]; then
        cp "$EXAMPLE_FILE" "$CONFIG_FILE"
        echo "Created $CONFIG_FILE from template"
    fi
    
    # Remove existing line
    grep -v "^${key}=" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" 2>/dev/null || true
    
    # Add new value
    echo "${key}=${value}" >> "${CONFIG_FILE}.tmp"
    
    # Sort by section (preserve comments)
    mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    
    echo -e "${GREEN}✅ Updated: ${key}=${value}${NC}"
}

# ============================================================================
# Display Functions
# ============================================================================

show_all_settings() {
    print_header "CURRENT AURA CONFIGURATION"
    
    echo -e "${BOLD}🏥 EHR INTEGRATION${NC}"
    local ehr_enabled=$(get_config_value "EHR_INTEGRATION_ENABLED")
    if [ "$ehr_enabled" == "true" ]; then
        echo -e "  ${GREEN}●${NC} Enabled"
        echo "  FHIR Server: $(get_config_value 'SYSTMONE_FHIR_URL')"
    else
        echo -e "  ${RED}○${NC} Disabled"
    fi
    echo ""
    
    echo -e "${BOLD}🧠 LLM MODELS${NC}"
    echo "  Complex Model:     $(get_config_value 'MODEL_PATH' | sed 's|.*/||')"
    echo "  Complex Context:   $(get_config_value 'N_CTX')"
    echo "  Simple Model:      $(get_config_value 'SIMPLE_MODEL_PATH' | sed 's|.*/||')"
    echo "  Simple Context:    $(get_config_value 'SIMPLE_N_CTX')"
    echo "  Temperature:       $(get_config_value 'LLM_TEMPERATURE')"
    echo ""
    
    echo -e "${BOLD}📚 RAG SEARCH${NC}"
    echo "  Threshold:     $(get_config_value 'RAG_THRESHOLD')"
    echo "  Top K:         $(get_config_value 'RAG_TOP_K')"
    echo "  Phonetic:      $(get_config_value 'RAG_USE_PHONETIC_MATCHING')"
    echo ""
    
    echo -e "${BOLD}🔊 TEXT-TO-SPEECH${NC}"
    local api_key=$(get_config_value 'ELEVENLABS_API_KEY')
    if [ -n "$api_key" ] && [ "$api_key" != "your_elevenlabs_api_key_here" ]; then
        echo -e "  ${GREEN}✅ API Key configured${NC}"
    else
        echo -e "  ${RED}❌ API Key not set${NC}"
    fi
    echo ""
    
    echo -e "${BOLD}🐛 DEBUGGING${NC}"
    echo "  Debug Mode:    $(get_config_value 'DEBUG_MODE')"
    echo "  Log Level:     $(get_config_value 'LOG_LEVEL')"
    echo ""
    
    echo "========================================================================"
}

# ============================================================================
# Toggle Functions
# ============================================================================

toggle_ehr() {
    local action=$1
    
    if [ "$action" == "on" ]; then
        set_config_value "EHR_INTEGRATION_ENABLED" "true"
        echo ""
        echo -e "${GREEN}✅ EHR Integration ENABLED${NC}"
        echo ""
        echo "FHIR calls will be made to: $(get_config_value 'SYSTMONE_FHIR_URL')"
        echo "Data will be saved to SystmOne EHR"
        echo ""
    else
        set_config_value "EHR_INTEGRATION_ENABLED" "false"
        echo ""
        echo -e "${GREEN}✅ EHR Integration DISABLED${NC}"
        echo ""
        echo "Normal Aura mode (local data only)"
        echo ""
    fi
    
    show_restart_message
}

# ============================================================================
# Configuration Menus
# ============================================================================

configure_ehr() {
    print_header "EHR CONFIGURATION"
    
    echo "Current Status: $(get_config_value 'EHR_INTEGRATION_ENABLED')"
    echo ""
    echo "1) Enable EHR integration"
    echo "2) Disable EHR integration"
    echo "3) Change FHIR server URL"
    echo "4) Back to main menu"
    echo ""
    read -p "Choice [1-4]: " choice
    
    case $choice in
        1) toggle_ehr on ;;
        2) toggle_ehr off ;;
        3)
            echo ""
            echo "Current: $(get_config_value 'SYSTMONE_FHIR_URL')"
            echo ""
            echo "Common options:"
            echo "  1) https://hapi.fhir.org/baseR4 (test server)"
            echo "  2) https://api.systmone.nhs.uk/fhir (production)"
            echo "  3) Custom URL"
            echo ""
            read -p "Choice [1-3]: " url_choice
            case $url_choice in
                1) set_config_value "SYSTMONE_FHIR_URL" "https://hapi.fhir.org/baseR4" ;;
                2) set_config_value "SYSTMONE_FHIR_URL" "https://api.systmone.nhs.uk/fhir" ;;
                3)
                    read -p "Enter FHIR URL: " custom_url
                    set_config_value "SYSTMONE_FHIR_URL" "$custom_url"
                    ;;
            esac
            show_restart_message
            ;;
        4) return ;;
    esac
}

configure_llm() {
    print_header "LLM MODEL CONFIGURATION"
    
    echo "Complex Model:   $(get_config_value 'MODEL_PATH' | sed 's|.*/||')"
    echo "Complex Context: $(get_config_value 'N_CTX')"
    echo ""
    echo "Simple Model:    $(get_config_value 'SIMPLE_MODEL_PATH' | sed 's|.*/||')"
    echo "Simple Context:  $(get_config_value 'SIMPLE_N_CTX')"
    echo ""
    echo "Temperature:     $(get_config_value 'LLM_TEMPERATURE')"
    echo ""
    echo "1) Change complex model path"
    echo "2) Change complex model context size"
    echo "3) Change simple model path"
    echo "4) Change simple model context size"
    echo "5) Adjust temperature"
    echo "6) Back to main menu"
    echo ""
    read -p "Choice [1-6]: " choice
    
    case $choice in
        1)
            read -p "Enter complex model path: " model_path
            set_config_value "MODEL_PATH" "$model_path"
            show_restart_message
            ;;
        2)
            echo ""
            echo "Common values: 4096, 8192, 16384, 32768"
            read -p "Enter complex model context size: " ctx
            set_config_value "N_CTX" "$ctx"
            show_restart_message
            ;;
        3)
            read -p "Enter simple model path: " model_path
            set_config_value "SIMPLE_MODEL_PATH" "$model_path"
            show_restart_message
            ;;
        4)
            echo ""
            echo "Common values: 2048, 4096, 8192"
            read -p "Enter simple model context size: " ctx
            set_config_value "SIMPLE_N_CTX" "$ctx"
            show_restart_message
            ;;
        5)
            read -p "Enter temperature (0.0-1.0): " temp
            set_config_value "LLM_TEMPERATURE" "$temp"
            show_restart_message
            ;;
        6) return ;;
    esac
}

configure_rag() {
    print_header "RAG SEARCH CONFIGURATION"
    
    echo "Threshold (current): $(get_config_value 'RAG_THRESHOLD')"
    echo "Top K (current):     $(get_config_value 'RAG_TOP_K')"
    echo ""
    echo "1) Adjust threshold (0.0 = loose, 1.0 = strict)"
    echo "2) Change Top K (number of results)"
    echo "3) Toggle phonetic matching"
    echo "4) Back to main menu"
    echo ""
    read -p "Choice [1-4]: " choice
    
    case $choice in
        1)
            read -p "Enter threshold (0.0-1.0): " threshold
            set_config_value "RAG_THRESHOLD" "$threshold"
            show_restart_message
            ;;
        2)
            read -p "Enter Top K (1-10): " topk
            set_config_value "RAG_TOP_K" "$topk"
            show_restart_message
            ;;
        3)
            local current=$(get_config_value 'RAG_USE_PHONETIC_MATCHING')
            if [ "$current" == "true" ]; then
                set_config_value "RAG_USE_PHONETIC_MATCHING" "false"
            else
                set_config_value "RAG_USE_PHONETIC_MATCHING" "true"
            fi
            show_restart_message
            ;;
        4) return ;;
    esac
}

# ============================================================================
# Utility Functions
# ============================================================================

show_restart_message() {
    echo ""
    echo -e "${YELLOW}⚠️  To apply changes, restart Docker containers:${NC}"
    echo ""
    echo "  docker-compose restart"
    echo ""
    echo "Or restart specific container:"
    echo "  docker-compose restart llm     (for LLM/EHR changes)"
    echo "  docker-compose restart rag     (for RAG changes)"
    echo ""
}

edit_file() {
    if [ -f "$CONFIG_FILE" ]; then
        ${EDITOR:-nano} "$CONFIG_FILE"
    else
        cp "$EXAMPLE_FILE" "$CONFIG_FILE"
        echo "Created $CONFIG_FILE from template"
        ${EDITOR:-nano} "$CONFIG_FILE"
    fi
}

# ============================================================================
# Main Menu
# ============================================================================

main_menu() {
    while true; do
        show_all_settings
        
        echo "QUICK ACTIONS:"
        echo ""
        echo "  1) Toggle EHR (on/off)"
        echo "  2) Configure EHR settings"
        echo "  3) Configure LLM models"
        echo "  4) Configure RAG search"
        echo "  5) Edit .env file directly"
        echo "  6) Restart Docker containers"
        echo "  7) Exit"
        echo ""
        read -p "Enter choice [1-7]: " choice
        
        case $choice in
            1)
                # Show current state and ask what to do
                local current=$(get_config_value 'EHR_INTEGRATION_ENABLED')
                echo ""
                if [ "$current" == "true" ]; then
                    echo "EHR is currently: ENABLED"
                    echo ""
                    read -p "Turn it OFF? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_ehr off
                    fi
                else
                    echo "EHR is currently: DISABLED"
                    echo ""
                    read -p "Turn it ON? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_ehr on
                    fi
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            2)
                configure_ehr
                read -p "Press Enter to continue..."
                ;;
            3)
                configure_llm
                read -p "Press Enter to continue..."
                ;;
            4)
                configure_rag
                read -p "Press Enter to continue..."
                ;;
            5)
                edit_file
                ;;
            6)
                echo ""
                echo "Restarting Docker containers..."
                docker-compose restart
                echo ""
                echo -e "${GREEN}✅ Containers restarted${NC}"
                read -p "Press Enter to continue..."
                ;;
            7)
                echo ""
                echo "Goodbye!"
                echo ""
                exit 0
                ;;
            *)
                echo ""
                echo -e "${RED}Invalid choice${NC}"
                sleep 1
                ;;
        esac
    done
}

# ============================================================================
# Command Line Arguments
# ============================================================================

case "${1:-}" in
    show|status)
        show_all_settings
        ;;
    ehr)
        case "${2:-}" in
            on|enable) toggle_ehr on ;;
            off|disable) toggle_ehr off ;;
            *) configure_ehr ;;
        esac
        ;;
    edit)
        edit_file
        ;;
    *)
        # No arguments = interactive menu
        main_menu
        ;;
esac

