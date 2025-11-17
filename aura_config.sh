#!/bin/bash
# ============================================================================
# AURA UNIFIED CONFIGURATION MANAGER
# Easy management of all Aura settings from one place
# ============================================================================
#
# NOTE: .env file now only contains API keys and basic settings.
# LLM/RAG settings are hardcoded in scripts or managed via Settings Dialog.
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
        cp "$EXAMPLE_FILE" "$CONFIG_FILE" 2>/dev/null || touch "$CONFIG_FILE"
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
    
    echo -e "${BOLD}📝 NOTE${NC}"
    echo -e "  ${CYAN}ℹ️  .env file now only contains API keys and basic settings${NC}"
    echo -e "  ${CYAN}ℹ️  LLM/RAG settings are hardcoded in scripts or managed via Settings Dialog${NC}"
    echo ""
    
    echo -e "${BOLD}🔊 TEXT-TO-SPEECH${NC}"
    local api_key=$(get_config_value 'ELEVENLABS_API_KEY')
    local voice_id=$(get_config_value 'ELEVENLABS_VOICE_ID')
    if [ -n "$api_key" ] && [ "$api_key" != "your_elevenlabs_api_key_here" ] && [ "$api_key" != "" ]; then
        echo -e "  ${GREEN}✅ API Key configured${NC}"
        if [ -n "$voice_id" ] && [ "$voice_id" != "default" ] && [ "$voice_id" != "" ]; then
            echo "  Voice ID:      $voice_id"
        else
            echo "  Voice ID:      default"
        fi
    else
        echo -e "  ${RED}❌ API Key not set${NC}"
        echo -e "  ${YELLOW}⚠️  TTS will not work without API key${NC}"
    fi
    echo -e "  ${CYAN}Volume:${NC} Controlled via Settings Dialog in GUI"
    echo ""

    echo -e "${BOLD}🎤 WAKE WORD DETECTION${NC}"
    echo -e "  ${GREEN}✅ OpenWakeWord${NC} (no API key required)"
    echo "  Status:      Works natively on ARM64/Jetson"
    echo "  Install:     pip install openwakeword"
    echo "  GitHub:      https://github.com/dscripka/openWakeWord"
    echo ""
    
    echo -e "${BOLD}💬 TELEGRAM BOT${NC}"
    local tg_token=$(get_config_value 'TELEGRAM_BOT_TOKEN')
    if [ -n "$tg_token" ] && [ "$tg_token" != "your_telegram_bot_token" ] && [ "$tg_token" != "" ]; then
        echo -e "  ${GREEN}✅ Bot token configured${NC}"
    else
        echo -e "  ${YELLOW}○${NC} Not configured (optional)"
    fi
    echo ""
    
    echo -e "${BOLD}🔄 GITHUB OTA UPDATES${NC}"
    local gh_token=$(get_config_value 'GITHUB_TOKEN')
    if [ -n "$gh_token" ] && [ "$gh_token" != "your_github_token_here" ] && [ "$gh_token" != "" ]; then
        echo -e "  ${GREEN}✅ GitHub token configured${NC}"
        echo "  Token:       ${gh_token:0:20}..."
    else
        echo -e "  ${YELLOW}○${NC} Not configured (optional for OTA updates)"
    fi
    echo ""
    
    echo -e "${BOLD}🔐 NHS/FHIR CREDENTIALS${NC}"
    local nhs_client_id=$(get_config_value 'NHS_CLIENT_ID')
    local nhs_client_secret=$(get_config_value 'NHS_CLIENT_SECRET')
    if [ -n "$nhs_client_id" ] && [ -n "$nhs_client_secret" ]; then
        echo -e "  ${GREEN}✅ NHS credentials configured${NC}"
        echo "  Client ID:     ${nhs_client_id:0:20}..."
    else
        echo -e "  ${YELLOW}○${NC} Not configured (needed for production EHR)"
    fi
    echo ""
    
    echo -e "${BOLD}⚙️  OTHER SETTINGS${NC}"
    echo -e "  ${CYAN}LLM Mode (Medical/Generic):${NC} Managed via Settings Dialog"
    echo -e "  ${CYAN}LLM Model Selection:${NC} Managed via Settings Dialog"
    echo -e "  ${CYAN}RAG Mode (CPU/GPU/OFF):${NC} Managed via Settings Dialog"
    echo -e "  ${CYAN}LLM Parameters:${NC} Hardcoded in container_rest.py (top-level variables)"
    echo ""
    
    echo "========================================================================"
}

# ============================================================================
# Configuration Menus
# ============================================================================

configure_tts() {
    print_header "TEXT-TO-SPEECH CONFIGURATION"
    
    local api_key=$(get_config_value 'ELEVENLABS_API_KEY')
    local voice_id=$(get_config_value 'ELEVENLABS_VOICE_ID')
    
    echo "Current Settings:"
    echo ""
    if [ -n "$api_key" ] && [ "$api_key" != "your_elevenlabs_api_key_here" ] && [ "$api_key" != "" ]; then
        echo -e "  API Key:  ${GREEN}✅ Configured${NC}"
        echo "  Voice ID: ${voice_id:-default}"
    else
        echo -e "  API Key:  ${RED}❌ Not set${NC}"
        echo "  Voice ID: ${voice_id:-default}"
    fi
    echo -e "  Volume:   ${CYAN}Controlled via Settings Dialog in GUI${NC}"
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
                echo "Note: No restart needed for TTS changes"
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
            set_config_value "ELEVENLABS_API_KEY" ""
            echo ""
            echo -e "${GREEN}✅ API key cleared${NC}"
            ;;
        4) return ;;
    esac
}

configure_wake_word() {
    print_header "WAKE WORD DETECTION CONFIGURATION (OPENWAKEWORD)"
    
    echo "OpenWakeWord Wake Word Detection:"
    echo "  • ✅ Fully open source (Apache 2.0)"
    echo "  • ✅ No API keys required"
    echo "  • ✅ Works natively on ARM64/Jetson"
    echo "  • ✅ Simple installation: pip install openwakeword"
    echo "  • ✅ Lightweight and efficient"
    echo "  • ✅ Easy custom training (~100-200 samples)"
    echo ""
    echo "Installation:"
    echo "  pip install openwakeword"
    echo ""
    echo "Custom Training:"
    echo "  https://github.com/dscripka/openWakeWord#training-custom-models"
    echo ""
    echo "GitHub:"
    echo "  https://github.com/dscripka/openWakeWord"
            echo ""
    echo "Note: Wake word detection is controlled via Settings Dialog in GUI"
    echo "      (Settings → AI Model Settings → Wake Word toggle)"
    echo ""
    read -p "Press Enter to return to main menu..."
}

configure_picovoice() {
    # Alias for configure_wake_word - Picovoice is the company, Porcupine is the product
    configure_wake_word
}

configure_telegram() {
    print_header "TELEGRAM BOT CONFIGURATION"
    
    local tg_token=$(get_config_value 'TELEGRAM_BOT_TOKEN')
    
    echo "Current Settings:"
    echo ""
    if [ -n "$tg_token" ] && [ "$tg_token" != "your_telegram_bot_token" ] && [ "$tg_token" != "" ]; then
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
            set_config_value "TELEGRAM_BOT_TOKEN" ""
            echo ""
            echo -e "${GREEN}✅ Telegram bot token cleared${NC}"
            ;;
        3) return ;;
    esac
}

configure_github() {
    print_header "GITHUB OTA UPDATE CONFIGURATION"
    
    local gh_token=$(get_config_value 'GITHUB_TOKEN')
    
    echo "Current Settings:"
    echo ""
    if [ -n "$gh_token" ] && [ "$gh_token" != "your_github_token_here" ] && [ "$gh_token" != "" ]; then
        echo -e "  GitHub Token: ${GREEN}✅ Configured${NC}"
        echo "  Token:        ${gh_token:0:20}..."
    else
        echo -e "  GitHub Token: ${RED}❌ Not set${NC}"
    fi
    echo ""
    echo "This token is used for over-the-air updates from GitHub."
    echo "It allows authentication with private repositories and higher API rate limits."
    echo ""
    echo "1) Set GitHub personal access token"
    echo "2) Clear GitHub token"
    echo "3) Back to main menu"
    echo ""
    read -p "Choice [1-3]: " choice
    
    case $choice in
        1)
            echo ""
            echo "How to get a GitHub personal access token:"
            echo "  1. Go to https://github.com/settings/tokens"
            echo "  2. Click 'Generate new token' → 'Generate new token (classic)'"
            echo "  3. Give it a name (e.g., 'Aura OTA Updates')"
            echo "  4. Select scopes: 'repo' (for private repos) or 'public_repo' (for public only)"
            echo "  5. Click 'Generate token' and copy it"
            echo ""
            read -p "Enter GitHub personal access token: " gh_token
            if [ -n "$gh_token" ]; then
                set_config_value "GITHUB_TOKEN" "$gh_token"
                echo ""
                echo -e "${GREEN}✅ GitHub token saved${NC}"
                echo ""
                echo "Note: This token is used by the Settings dialog for OTA updates"
                echo "No restart needed - token is read when updating"
            fi
            ;;
        2)
            set_config_value "GITHUB_TOKEN" ""
            echo ""
            echo -e "${GREEN}✅ GitHub token cleared${NC}"
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

edit_file() {
    if [ -f "$CONFIG_FILE" ]; then
        ${EDITOR:-nano} "$CONFIG_FILE"
    else
        cp "$EXAMPLE_FILE" "$CONFIG_FILE" 2>/dev/null || touch "$CONFIG_FILE"
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
        echo "  1) Configure TTS (ElevenLabs)"
        echo "  2) Configure Wake Word (Porcupine/Picovoice)"
        echo "  3) Configure Telegram bot"
        echo "  4) Configure GitHub OTA updates"
        echo "  5) Configure NHS/FHIR credentials"
        echo "  a) Edit .env file directly"
        echo "  0) Exit"
        echo ""
        echo -e "${CYAN}ℹ️  Note: LLM/RAG settings are managed via Settings Dialog in Aura GUI${NC}"
        echo ""
        read -p "Enter choice [0-5a]: " choice
        
        case $choice in
            1)
                configure_tts
                read -p "Press Enter to continue..."
                ;;
            2)
                configure_wake_word
                read -p "Press Enter to continue..."
                ;;
            3)
                configure_telegram
                read -p "Press Enter to continue..."
                ;;
            4)
                configure_github
                read -p "Press Enter to continue..."
                ;;
            5)
                configure_nhs_fhir
                read -p "Press Enter to continue..."
                ;;
            a|A)
                edit_file
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
    tts)
        configure_tts
        ;;
    wake|porcupine|picovoice)
        configure_wake_word
        ;;
    telegram)
        configure_telegram
        ;;
    github)
        configure_github
        ;;
    nhs|fhir)
        configure_nhs_fhir
        ;;
    edit)
        edit_file
        ;;
    *)
        # No arguments = interactive menu
        main_menu
        ;;
esac
