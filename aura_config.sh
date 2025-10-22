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
    echo -e "${BOLD}🌡️ TEMPERATURE CONTROLS${NC}"
    echo "  Simple Questions:  $(get_config_value 'LLM_TEMPERATURE_SIMPLE')"
    echo "  Complex Questions: $(get_config_value 'LLM_TEMPERATURE_COMPLEX')"
    echo "  Normalization:     $(get_config_value 'LLM_TEMPERATURE_NORMALIZATION')"
    echo "  Creative:         $(get_config_value 'LLM_TEMPERATURE_CREATIVE')"
    echo "  Analysis:         $(get_config_value 'LLM_TEMPERATURE_ANALYSIS')"
    echo ""
    
    echo -e "${BOLD}📚 RAG SEARCH${NC}"
    local rag_enabled=$(get_config_value 'RAG_ENABLED')
    if [ "$rag_enabled" == "true" ]; then
        echo -e "  ${GREEN}●${NC} Mode:          GPU FAISS (fast, separate container)"
    else
        echo -e "  ${YELLOW}○${NC} Mode:          CPU FAISS (slow, built into LLM)"
    fi
    echo "  Threshold:     $(get_config_value 'RAG_THRESHOLD')"
    echo "  Top K:         $(get_config_value 'RAG_TOP_K')"
    echo "  Phonetic:      $(get_config_value 'RAG_USE_PHONETIC_MATCHING')"
    echo ""
    
    echo -e "${BOLD}🔊 TEXT-TO-SPEECH${NC}"
    local api_key=$(get_config_value 'ELEVENLABS_API_KEY')
    local voice_id=$(get_config_value 'ELEVENLABS_VOICE_ID')
    if [ -n "$api_key" ] && [ "$api_key" != "your_elevenlabs_api_key_here" ]; then
        echo -e "  ${GREEN}✅ API Key configured${NC}"
        if [ -n "$voice_id" ] && [ "$voice_id" != "default" ]; then
            echo "  Voice ID:      $voice_id"
        else
            echo "  Voice ID:      default"
        fi
    else
        echo -e "  ${RED}❌ API Key not set${NC}"
    fi
    echo ""
    
    echo -e "${BOLD}💬 TELEGRAM BOT${NC}"
    local tg_token=$(get_config_value 'TELEGRAM_BOT_TOKEN')
    if [ -n "$tg_token" ] && [ "$tg_token" != "your_telegram_bot_token" ]; then
        echo -e "  ${GREEN}✅ Bot token configured${NC}"
    else
        echo -e "  ${YELLOW}○${NC} Not configured (optional)"
    fi
    echo ""
    
    echo -e "${BOLD}🔐 NHS/FHIR CREDENTIALS${NC}"
    local nhs_client_id=$(get_config_value 'NHS_CLIENT_ID')
    local nhs_client_secret=$(get_config_value 'NHS_CLIENT_SECRET')
    if [ -n "$nhs_client_id" ] && [ -n "$nhs_client_secret" ]; then
        echo -e "  ${GREEN}✅ NHS credentials configured${NC}"
        echo "  Client ID:     ${nhs_client_id:0:20}..."
    else
        echo -e "  ${YELLOW}○${NC} Not configured (needed for production)"
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
    echo "🌡️ GRANULAR TEMPERATURE CONTROLS:"
    echo "  Simple Questions:  $(get_config_value 'LLM_TEMPERATURE_SIMPLE')"
    echo "  Complex Questions: $(get_config_value 'LLM_TEMPERATURE_COMPLEX')"
    echo "  Normalization:     $(get_config_value 'LLM_TEMPERATURE_NORMALIZATION')"
    echo "  Creative:         $(get_config_value 'LLM_TEMPERATURE_CREATIVE')"
    echo "  Analysis:         $(get_config_value 'LLM_TEMPERATURE_ANALYSIS')"
    echo ""
    echo "1) Change complex model path"
    echo "2) Change complex model context size"
    echo "3) Change simple model path"
    echo "4) Change simple model context size"
    echo "5) Adjust global temperature"
    echo "6) Configure granular temperatures"
    echo "7) Back to main menu"
    echo ""
    read -p "Choice [1-7]: " choice
    
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
        6)
            configure_temperature
            ;;
        7) return ;;
    esac
}

configure_temperature() {
    print_header "GRANULAR TEMPERATURE CONFIGURATION"
    
    echo "Current Temperature Settings:"
    echo ""
    echo "  Simple Questions:  $(get_config_value 'LLM_TEMPERATURE_SIMPLE') (L, C, T, S, O, D)"
    echo "  Complex Questions: $(get_config_value 'LLM_TEMPERATURE_COMPLEX') (A, R)"
    echo "  Normalization:     $(get_config_value 'LLM_TEMPERATURE_NORMALIZATION') (text processing)"
    echo "  Creative:         $(get_config_value 'LLM_TEMPERATURE_CREATIVE') (creative responses)"
    echo "  Analysis:         $(get_config_value 'LLM_TEMPERATURE_ANALYSIS') (analysis tasks)"
    echo ""
    echo "1) Set simple questions temperature (0.0-1.0)"
    echo "2) Set complex questions temperature (0.0-1.0)"
    echo "3) Set normalization temperature (0.0-1.0)"
    echo "4) Set creative temperature (0.0-1.0)"
    echo "5) Set analysis temperature (0.0-1.0)"
    echo "6) Set all to conservative (0.1)"
    echo "7) Set all to balanced (0.3)"
    echo "8) Set all to creative (0.6)"
    echo "9) Back to LLM menu"
    echo ""
    read -p "Choice [1-9]: " choice
    
    case $choice in
        1)
            read -p "Enter simple questions temperature (0.0-1.0): " temp
            set_config_value "LLM_TEMPERATURE_SIMPLE" "$temp"
            show_restart_message
            ;;
        2)
            read -p "Enter complex questions temperature (0.0-1.0): " temp
            set_config_value "LLM_TEMPERATURE_COMPLEX" "$temp"
            show_restart_message
            ;;
        3)
            read -p "Enter normalization temperature (0.0-1.0): " temp
            set_config_value "LLM_TEMPERATURE_NORMALIZATION" "$temp"
            show_restart_message
            ;;
        4)
            read -p "Enter creative temperature (0.0-1.0): " temp
            set_config_value "LLM_TEMPERATURE_CREATIVE" "$temp"
            show_restart_message
            ;;
        5)
            read -p "Enter analysis temperature (0.0-1.0): " temp
            set_config_value "LLM_TEMPERATURE_ANALYSIS" "$temp"
            show_restart_message
            ;;
        6)
            set_config_value "LLM_TEMPERATURE_SIMPLE" "0.1"
            set_config_value "LLM_TEMPERATURE_COMPLEX" "0.1"
            set_config_value "LLM_TEMPERATURE_NORMALIZATION" "0.1"
            set_config_value "LLM_TEMPERATURE_CREATIVE" "0.1"
            set_config_value "LLM_TEMPERATURE_ANALYSIS" "0.1"
            echo ""
            echo -e "${GREEN}✅ All temperatures set to conservative (0.1)${NC}"
            show_restart_message
            ;;
        7)
            set_config_value "LLM_TEMPERATURE_SIMPLE" "0.3"
            set_config_value "LLM_TEMPERATURE_COMPLEX" "0.3"
            set_config_value "LLM_TEMPERATURE_NORMALIZATION" "0.3"
            set_config_value "LLM_TEMPERATURE_CREATIVE" "0.3"
            set_config_value "LLM_TEMPERATURE_ANALYSIS" "0.3"
            echo ""
            echo -e "${GREEN}✅ All temperatures set to balanced (0.3)${NC}"
            show_restart_message
            ;;
        8)
            set_config_value "LLM_TEMPERATURE_SIMPLE" "0.6"
            set_config_value "LLM_TEMPERATURE_COMPLEX" "0.6"
            set_config_value "LLM_TEMPERATURE_NORMALIZATION" "0.6"
            set_config_value "LLM_TEMPERATURE_CREATIVE" "0.6"
            set_config_value "LLM_TEMPERATURE_ANALYSIS" "0.6"
            echo ""
            echo -e "${GREEN}✅ All temperatures set to creative (0.6)${NC}"
            show_restart_message
            ;;
        9) return ;;
    esac
}

configure_rag() {
    print_header "RAG SEARCH CONFIGURATION"
    
    local rag_enabled=$(get_config_value 'RAG_ENABLED')
    
    echo "Current Settings:"
    echo ""
    if [ "$rag_enabled" == "true" ]; then
        echo -e "  RAG Mode:     ${GREEN}GPU FAISS${NC} (fast, separate container)"
    else
        echo -e "  RAG Mode:     ${YELLOW}CPU FAISS${NC} (slow, built into LLM)"
    fi
    echo "  Threshold:    $(get_config_value 'RAG_THRESHOLD')"
    echo "  Top K:        $(get_config_value 'RAG_TOP_K')"
    echo "  Phonetic:     $(get_config_value 'RAG_USE_PHONETIC_MATCHING')"
    echo ""
    echo "1) Toggle RAG mode (GPU vs CPU)"
    echo "2) Adjust threshold (0.0 = loose, 1.0 = strict)"
    echo "3) Change Top K (number of results)"
    echo "4) Toggle phonetic matching"
    echo "5) Back to main menu"
    echo ""
    read -p "Choice [1-5]: " choice
    
    case $choice in
        1)
            echo ""
            local current=$(get_config_value 'RAG_ENABLED')
            if [ "$current" == "true" ]; then
                echo "Currently: GPU FAISS (separate RAG container)"
                echo ""
                read -p "Switch to CPU FAISS (built into LLM)? (y/n): " answer
                if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                    set_config_value "RAG_ENABLED" "false"
                    echo ""
                    echo -e "${GREEN}✅ Switched to CPU FAISS${NC}"
                    echo ""
                    echo "Benefits:"
                    echo "  • No separate RAG container needed"
                    echo "  • Simpler setup"
                    echo ""
                    echo "Drawbacks:"
                    echo "  • Slower searches (~500ms vs ~50ms)"
                    echo "  • Limited scalability"
                    echo ""
                    show_restart_message
                fi
            else
                echo "Currently: CPU FAISS (built into LLM container)"
                echo ""
                read -p "Switch to GPU FAISS (separate RAG container)? (y/n): " answer
                if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                    set_config_value "RAG_ENABLED" "true"
                    echo ""
                    echo -e "${GREEN}✅ Switched to GPU FAISS${NC}"
                    echo ""
                    echo "Benefits:"
                    echo "  • Faster searches (~50ms vs ~500ms)"
                    echo "  • Better scalability"
                    echo "  • Optimized for production"
                    echo ""
                    echo "Note: Requires GPU and separate RAG container"
                    echo ""
                    show_restart_message
                fi
            fi
            ;;
        2)
            read -p "Enter threshold (0.0-1.0): " threshold
            set_config_value "RAG_THRESHOLD" "$threshold"
            show_restart_message
            ;;
        3)
            read -p "Enter Top K (1-10): " topk
            set_config_value "RAG_TOP_K" "$topk"
            show_restart_message
            ;;
        4)
            local current=$(get_config_value 'RAG_USE_PHONETIC_MATCHING')
            if [ "$current" == "true" ]; then
                set_config_value "RAG_USE_PHONETIC_MATCHING" "false"
            else
                set_config_value "RAG_USE_PHONETIC_MATCHING" "true"
            fi
            show_restart_message
            ;;
        5) return ;;
    esac
}

configure_tts() {
    print_header "TEXT-TO-SPEECH CONFIGURATION"
    
    local api_key=$(get_config_value 'ELEVENLABS_API_KEY')
    local voice_id=$(get_config_value 'ELEVENLABS_VOICE_ID')
    
    echo "Current Settings:"
    echo ""
    if [ -n "$api_key" ] && [ "$api_key" != "your_elevenlabs_api_key_here" ]; then
        echo -e "  API Key:  ${GREEN}✅ Configured${NC}"
        echo "  Voice ID: ${voice_id:-default}"
    else
        echo -e "  API Key:  ${RED}❌ Not set${NC}"
        echo "  Voice ID: ${voice_id:-default}"
    fi
    echo ""
    echo "1) Set ElevenLabs API key"
    echo "2) Set voice ID (optional)"
    echo "3) Clear API key"
    echo "4) Back to main menu"
    echo ""
    read -p "Choice [1-4]: " choice
    
    case $choice in
        1)
            echo ""
            echo "Get your API key from: https://elevenlabs.io/"
            echo ""
            read -p "Enter ElevenLabs API key: " api_key
            if [ -n "$api_key" ]; then
                set_config_value "ELEVENLABS_API_KEY" "$api_key"
                echo ""
                echo -e "${GREEN}✅ API key saved${NC}"
                echo ""
                echo "Note: No container restart needed for TTS changes"
            fi
            ;;
        2)
            echo ""
            echo "Common voice IDs:"
            echo "  - Leave blank for default"
            echo "  - Or enter specific voice ID from ElevenLabs"
            echo ""
            read -p "Enter voice ID (or press Enter for default): " voice_id
            if [ -n "$voice_id" ]; then
                set_config_value "ELEVENLABS_VOICE_ID" "$voice_id"
            else
                set_config_value "ELEVENLABS_VOICE_ID" "default"
            fi
            echo ""
            echo -e "${GREEN}✅ Voice ID saved${NC}"
            ;;
        3)
            set_config_value "ELEVENLABS_API_KEY" "your_elevenlabs_api_key_here"
            echo ""
            echo -e "${GREEN}✅ API key cleared${NC}"
            ;;
        4) return ;;
    esac
}

configure_telegram() {
    print_header "TELEGRAM BOT CONFIGURATION"
    
    local tg_token=$(get_config_value 'TELEGRAM_BOT_TOKEN')
    
    echo "Current Settings:"
    echo ""
    if [ -n "$tg_token" ] && [ "$tg_token" != "your_telegram_bot_token" ]; then
        echo -e "  Bot Token: ${GREEN}✅ Configured${NC}"
        echo "  Token:     ${tg_token:0:20}..."
    else
        echo -e "  Bot Token: ${RED}❌ Not set${NC}"
    fi
    echo ""
    echo "1) Set Telegram bot token"
    echo "2) Clear bot token"
    echo "3) Back to main menu"
    echo ""
    read -p "Choice [1-3]: " choice
    
    case $choice in
        1)
            echo ""
            echo "How to get a Telegram bot token:"
            echo "  1. Open Telegram and search for @BotFather"
            echo "  2. Send /newbot and follow instructions"
            echo "  3. Copy the token you receive"
            echo ""
            read -p "Enter Telegram bot token: " tg_token
            if [ -n "$tg_token" ]; then
                set_config_value "TELEGRAM_BOT_TOKEN" "$tg_token"
                echo ""
                echo -e "${GREEN}✅ Telegram bot token saved${NC}"
                echo ""
                echo "Note: Restart is only needed if running Telegram bot service"
            fi
            ;;
        2)
            set_config_value "TELEGRAM_BOT_TOKEN" "your_telegram_bot_token"
            echo ""
            echo -e "${GREEN}✅ Telegram bot token cleared${NC}"
            ;;
        3) return ;;
    esac
}

configure_nhs_fhir() {
    print_header "NHS/FHIR CREDENTIALS CONFIGURATION"
    
    local client_id=$(get_config_value 'NHS_CLIENT_ID')
    local client_secret=$(get_config_value 'NHS_CLIENT_SECRET')
    local redirect_uri=$(get_config_value 'NHS_REDIRECT_URI')
    
    echo "Current Settings:"
    echo ""
    if [ -n "$client_id" ] && [ -n "$client_secret" ]; then
        echo -e "  ${GREEN}✅ NHS credentials configured${NC}"
        echo "  Client ID:     ${client_id:0:30}..."
        echo "  Client Secret: ${client_secret:0:10}...***"
        echo "  Redirect URI:  ${redirect_uri:-Not set}"
    else
        echo -e "  ${RED}❌ NHS credentials not set${NC}"
        echo ""
        echo "  These are needed for NHS production EHR access"
        echo "  Get them from: https://digital.nhs.uk/developer"
    fi
    echo ""
    echo "1) Set NHS Client ID"
    echo "2) Set NHS Client Secret"
    echo "3) Set NHS Redirect URI"
    echo "4) Clear all NHS credentials"
    echo "5) Back to main menu"
    echo ""
    read -p "Choice [1-5]: " choice
    
    case $choice in
        1)
            echo ""
            read -p "Enter NHS Client ID: " client_id
            if [ -n "$client_id" ]; then
                set_config_value "NHS_CLIENT_ID" "$client_id"
                echo ""
                echo -e "${GREEN}✅ NHS Client ID saved${NC}"
            fi
            ;;
        2)
            echo ""
            read -sp "Enter NHS Client Secret: " client_secret
            echo ""
            if [ -n "$client_secret" ]; then
                set_config_value "NHS_CLIENT_SECRET" "$client_secret"
                echo ""
                echo -e "${GREEN}✅ NHS Client Secret saved${NC}"
            fi
            ;;
        3)
            echo ""
            echo "Example: https://your-app.nhs.uk/callback"
            read -p "Enter NHS Redirect URI: " redirect_uri
            if [ -n "$redirect_uri" ]; then
                set_config_value "NHS_REDIRECT_URI" "$redirect_uri"
                echo ""
                echo -e "${GREEN}✅ NHS Redirect URI saved${NC}"
            fi
            ;;
        4)
            set_config_value "NHS_CLIENT_ID" ""
            set_config_value "NHS_CLIENT_SECRET" ""
            set_config_value "NHS_REDIRECT_URI" ""
            echo ""
            echo -e "${GREEN}✅ All NHS credentials cleared${NC}"
            ;;
        5) return ;;
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
        echo "  5) Configure TTS (ElevenLabs)"
        echo "  6) Configure Telegram bot"
        echo "  7) Configure NHS/FHIR credentials"
        echo "  8) Edit .env file directly"
        echo "  9) Restart Docker containers"
        echo "  0) Exit"
        echo ""
        read -p "Enter choice [0-9]: " choice
        
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
                configure_tts
                read -p "Press Enter to continue..."
                ;;
            6)
                configure_telegram
                read -p "Press Enter to continue..."
                ;;
            7)
                configure_nhs_fhir
                read -p "Press Enter to continue..."
                ;;
            8)
                edit_file
                ;;
            9)
                echo ""
                echo "Restarting Docker containers..."
                docker-compose restart
                echo ""
                echo -e "${GREEN}✅ Containers restarted${NC}"
                read -p "Press Enter to continue..."
                ;;
            0)
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

