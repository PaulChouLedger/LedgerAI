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
    
    echo -e "${BOLD}🧠 LLM CONTAINER MODE${NC}"
    local medical_mode=$(get_config_value "USE_MEDICAL_MODE")
    if [ "$medical_mode" == "true" ]; then
        echo -e "  ${GREEN}●${NC} Medical Mode (symptom assessment, adaptive diagnostics)"
        local enabled_categories=$(get_config_value "ENABLED_MEDICAL_CATEGORIES")
        if [ -z "$enabled_categories" ]; then
            enabled_categories="GI (default)"
        fi
        echo "  Enabled Categories: $enabled_categories"
        echo "  Available: GI, CARDIO, DERM, GU, GYN, MSK, NEURO, PULMONARY, RENAL"
        
        # Show Advanced Medical Navigator toggle status
        local navigator_on=$(get_config_value "USE_MEDICAL_NAVIGATOR")
        if [ "$navigator_on" == "true" ]; then
            echo -e "  ${CYAN}🔀 Advanced Navigator: ${GREEN}ENABLED${NC} (pure LLM mode)"
        else
            echo -e "  ${CYAN}🔀 Advanced Navigator: ${YELLOW}DISABLED${NC} (using Adaptive Diagnostic Engine)"
        fi
    else
        echo -e "  ${YELLOW}○${NC} Generic Mode (general conversation, RAG Q&A)"
    fi
    echo ""
    
    echo -e "${BOLD}🧠 LLM MODEL${NC}"
    local navigator_on=$(get_config_value "USE_MEDICAL_NAVIGATOR")
    if [ "$navigator_on" == "true" ]; then
        echo -e "  ${CYAN}📋 Medical Navigator is ENABLED - Task-specific parameters shown below${NC}"
    fi
    echo "  Model:         $(get_config_value 'SIMPLE_MODEL_PATH' | sed 's|.*/||')"
    echo "  Context:       $(get_config_value 'SIMPLE_N_CTX')"
    echo "  Chat Format:   $(get_config_value 'SIMPLE_CHAT_FORMAT')"
    if [ "$navigator_on" == "true" ]; then
        echo "  Temperature:   $(get_config_value 'LLM_TEMPERATURE_SIMPLE') (questions)"
    else
        echo "  Temperature:   $(get_config_value 'LLM_TEMPERATURE_SIMPLE')"
    fi
    echo "  Top P:         $(get_config_value 'LLM_TOP_P')"
    echo "  Top K:         $(get_config_value 'LLM_TOP_K')"
    echo "  Repeat Penalty:$(get_config_value 'LLM_REPEAT_PENALTY')"
    echo "  Presence Pen.: $(get_config_value 'LLM_PRESENCE_PENALTY')"
    echo "  Frequency Pen.:$(get_config_value 'LLM_FREQUENCY_PENALTY')"
    if [ "$navigator_on" == "true" ]; then
        echo "  Num Predict:   $(get_config_value 'LLM_NUM_PREDICT') (questions)"
    else
        echo "  Num Predict:   $(get_config_value 'LLM_NUM_PREDICT')"
    fi
    echo "  Stop (CSV):    $(get_config_value 'LLM_STOP')"
    
    # Show task-specific parameters if Medical Navigator is enabled
    if [ "$navigator_on" == "true" ]; then
        echo ""
        echo -e "${BOLD}  📋 Medical Navigator Task-Specific Settings${NC}"
        echo "  Temperature (Questions):     $(get_config_value 'LLM_TEMPERATURE_SIMPLE' || echo '0.4')"
        echo "  Temperature (Empathetic):    $(get_config_value 'LLM_TEMPERATURE_EMPATHETIC' || echo '0.4')"
        echo "  Temperature (Summary):       $(get_config_value 'LLM_TEMPERATURE_SUMMARY' || echo '0.25')"
        echo "  Max Tokens (Questions):      $(get_config_value 'LLM_NUM_PREDICT' || echo '120')"
        echo "  Max Tokens (Empathetic):     $(get_config_value 'LLM_MAX_TOKENS_EMPATHETIC' || echo '80')"
        echo "  Max Tokens (Chronicity):     $(get_config_value 'LLM_MAX_TOKENS_CHRONICITY' || echo '60')"
        echo "  Max Tokens (Summary):        $(get_config_value 'LLM_MAX_TOKENS_SUMMARY' || echo '220')"
        echo "  ${YELLOW}Note: JSON scoring uses temperature=0.0 (deterministic, hardcoded)${NC}"
    fi
    echo ""
    
    echo -e "${BOLD}📚 RAG SEARCH${NC}"
    local rag_mode=$(get_config_value 'RAG_MODE')
    if [ "$rag_mode" == "GPU" ]; then
        echo -e "  ${GREEN}●${NC} Mode:          GPU (RAG container - port 11435)"
    else
        echo -e "  ${YELLOW}○${NC} Mode:          CPU (FAISS in LLM containers - ports 11434/11436)"
    fi
    echo "  Threshold:     $(get_config_value 'RAG_THRESHOLD')"
    echo "  Top K:         $(get_config_value 'RAG_TOP_K')"
    echo "  Phonetic:      $(get_config_value 'RAG_USE_PHONETIC_MATCHING')"
    echo ""
    
    echo -e "${BOLD}🔊 TEXT-TO-SPEECH${NC}"
    local api_key=$(get_config_value 'ELEVENLABS_API_KEY')
    local voice_id=$(get_config_value 'ELEVENLABS_VOICE_ID')
    local tts_limit=$(get_config_value 'TTS_TOKEN_LIMIT')
    local tts_volume=$(get_config_value 'TTS_VOLUME')
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
    if [ -n "$tts_limit" ]; then
        echo "  Token limit:  $tts_limit"
    else
        echo "  Token limit:  (not set)"
    fi
    if [ -n "$tts_volume" ]; then
        echo "  Volume:       $tts_volume%"
    else
        echo "  Volume:       (not set)"
    fi
    echo ""

    local activation_keywords=$(get_config_value 'ACTIVATION_KEYWORDS')
    local activation_window=$(get_config_value 'ACTIVATION_WINDOW_SECONDS')
    local activation_cooldown=$(get_config_value 'ACTIVATION_COOLDOWN_SECONDS')
    local memory_dir=$(get_config_value 'CONVERSATION_MEMORY_DIR')
    local memory_persist_every=$(get_config_value 'CONVERSATION_MEMORY_PERSIST_EVERY')
    local memory_max_entries=$(get_config_value 'CONVERSATION_MEMORY_MAX_ENTRIES')
    local memory_top_k=$(get_config_value 'CONVERSATION_MEMORY_TOP_K')
    local memory_min_score=$(get_config_value 'CONVERSATION_MEMORY_MIN_SCORE')

    echo -e "${BOLD}🎙️ VOICE ACTIVATION & MEMORY${NC}"
    echo "  Keywords:       ${activation_keywords:-hey aura}"
    echo "  Window (sec):   ${activation_window:-15}"
    echo "  Cooldown (sec): ${activation_cooldown:-3}"
    echo "  Memory dir:     ${memory_dir:-data/learning/conversation_memory}"
    echo "  Persist every:  ${memory_persist_every:-10}"
    echo "  Max entries:    ${memory_max_entries:-5000}"
    echo "  Top K recall:   ${memory_top_k:-3}"
    echo "  Min score:      ${memory_min_score:-0.35}"
    echo ""
    
    echo -e "${BOLD}💬 TELEGRAM BOT${NC}"
    local tg_token=$(get_config_value 'TELEGRAM_BOT_TOKEN')
    if [ -n "$tg_token" ] && [ "$tg_token" != "your_telegram_bot_token" ]; then
        echo -e "  ${GREEN}✅ Bot token configured${NC}"
    else
        echo -e "  ${YELLOW}○${NC} Not configured (optional)"
    fi
    echo ""
    
    echo -e "${BOLD}🔄 GITHUB OTA UPDATES${NC}"
    local gh_token=$(get_config_value 'GITHUB_TOKEN')
    if [ -n "$gh_token" ] && [ "$gh_token" != "your_github_token_here" ]; then
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
        echo -e "  ${YELLOW}○${NC} Not configured (needed for production)"
    fi
    echo ""
    
    echo -e "${BOLD}🐛 DEBUGGING${NC}"
    echo "  Debug Mode:    $(get_config_value 'DEBUG_MODE')"
    echo "  Log Level:     $(get_config_value 'LOG_LEVEL')"
    echo ""
    
    echo -e "${BOLD}🤖 MACHINE LEARNING${NC}"
    local ml_enabled=$(get_config_value "ENABLE_ML_LEARNING")
    if [ "$ml_enabled" == "true" ]; then
        echo -e "  ${GREEN}●${NC} Enabled - Learning new synonyms and guideline terms"
    else
        echo -e "  ${RED}○${NC} Disabled - No learning from interactions"
    fi
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

toggle_medical_mode() {
    local action=$1
    
    if [ "$action" == "on" ]; then
        set_config_value "USE_MEDICAL_MODE" "true"
        echo ""
        echo -e "${GREEN}✅ Medical Mode ENABLED${NC}"
        echo ""
        echo "Container will handle:"
        echo "  • Medical symptom assessment"
        echo "  • Adaptive diagnostic engine"
        echo "  • OLDCARTS-based questioning"
        echo "  • Guideline matching"
        echo ""
    else
        set_config_value "USE_MEDICAL_MODE" "false"
        echo ""
        echo -e "${GREEN}✅ Generic Mode ENABLED${NC}"
        echo ""
        echo "Container will handle:"
        echo "  • General conversation"
        echo "  • RAG-powered document Q&A"
        echo "  • Flexible LLM interactions"
        echo ""
    fi
    
    show_restart_message
}

toggle_ml_learning() {
    local action=$1
    
    if [ "$action" == "on" ]; then
        set_config_value "ENABLE_ML_LEARNING" "true"
        echo ""
        echo -e "${GREEN}✅ ML Learning ENABLED${NC}"
        echo ""
        echo "System will:"
        echo "  • Record successful term matches"
        echo "  • Track unmatched patient responses"
        echo "  • Generate suggestions for new synonyms"
        echo "  • Generate suggestions for new guideline terms"
        echo ""
        echo "Review suggestions with: python3 ml/review_suggestions.py"
        echo ""
    else
        set_config_value "ENABLE_ML_LEARNING" "false"
        echo ""
        echo -e "${GREEN}✅ ML Learning DISABLED${NC}"
        echo ""
        echo "No learning from interactions"
        echo "System will not record or suggest new terms"
        echo ""
    fi
    
    show_restart_message
}

toggle_medical_navigator() {
    local action=$1
    
    if [ "$action" == "on" ]; then
        set_config_value "USE_MEDICAL_NAVIGATOR" "true"
        echo ""
        echo -e "${GREEN}✅ Advanced Medical Navigator ENABLED${NC}"
        echo ""
        echo "System will use:"
        echo "  • Pure LLM-based medical assistant"
        echo "  • Natural, human-like conversations"
        echo "  • Context-aware responses"
        echo "  • Simple, clean implementation"
        echo ""
        echo "Features:"
        echo "  • LLM-powered for all interactions"
        echo "  • Natural conversation flow"
        echo "  • Simple session management"
        echo "  • Designed to grow with features over time"
        echo ""
    else
        set_config_value "USE_MEDICAL_NAVIGATOR" "false"
        echo ""
        echo -e "${GREEN}✅ Advanced Medical Navigator DISABLED${NC}"
        echo ""
        echo "System will use:"
        echo "  • Adaptive Diagnostic Engine (default)"
        echo "  • Guideline-based questioning"
        echo "  • Structured OLDCARTS flow"
        echo "  • Rule-based anatomical extraction"
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
    
    local navigator_on=$(get_config_value "USE_MEDICAL_NAVIGATOR")
    
    echo "Model:         $(get_config_value 'SIMPLE_MODEL_PATH' | sed 's|.*/||')"
    echo "Context:       $(get_config_value 'SIMPLE_N_CTX')"
    echo "Temperature:   $(get_config_value 'LLM_TEMPERATURE_SIMPLE')"
    echo ""
    
    if [ "$navigator_on" == "true" ]; then
        echo -e "${CYAN}Medical Navigator is ENABLED - Task-specific parameters available${NC}"
        echo ""
    fi
    
    echo "General Settings (used by all LLM tasks):"
    echo "  1) Change model path"
    echo "  2) Change context size"
    echo "  3) Adjust temperature (default for question generation)"
    echo "  4) Set top_p (nucleus sampling)"
    echo "  5) Set top_k (top-k sampling)"
    echo "  6) Set repeat_penalty (penalty for repetition)"
    echo "  7) Set presence_penalty (encourages diversity)"
    echo "  8) Set frequency_penalty (penalty for word repetition)"
    echo "  9) Set num_predict (default max tokens for questions)"
    echo " 10) Set stop sequences (CSV)"
    echo " 11) Set chat format (e.g., llama-3, qwen2, qwen2.5)"
    
    if [ "$navigator_on" == "true" ]; then
        echo ""
        echo "Medical Navigator Task-Specific Settings:"
        echo "  (JSON scoring always uses temperature=0.0 for deterministic output)"
        echo " 12) Set temperature for empathetic statements (default: 0.4)"
        echo " 13) Set temperature for summaries (default: 0.25, lower = more accurate)"
        echo " 14) Set max tokens for empathetic statements (default: 80)"
        echo " 15) Set max tokens for chronicity questions (default: 60)"
        echo " 16) Set max tokens for summaries (default: 220)"
        echo " 17) Back to main menu"
    else
        echo " 12) Back to main menu"
    fi
    echo ""
    
    if [ "$navigator_on" == "true" ]; then
        read -p "Choice [1-17]: " choice
    else
        read -p "Choice [1-12]: " choice
    fi
    
    case $choice in
        1)
            read -p "Enter model path: " model_path
            set_config_value "SIMPLE_MODEL_PATH" "$model_path"
            show_restart_message
            ;;
        2)
            echo ""
            echo "Common values: 2048, 4096, 8192"
            read -p "Enter context size: " ctx
            set_config_value "SIMPLE_N_CTX" "$ctx"
            show_restart_message
            ;;
        3)
            echo ""
            echo "Current temperature: $(get_config_value 'LLM_TEMPERATURE_SIMPLE')"
            if [ "$navigator_on" == "true" ]; then
                echo "Used for: Question generation (default)"
                echo "Note: JSON scoring uses temperature=0.0 (hardcoded, deterministic)"
            fi
            read -p "Enter temperature (0.0-1.0): " temp
            if [ -n "$temp" ]; then
                set_config_value "LLM_TEMPERATURE_SIMPLE" "$temp"
                show_restart_message
            fi
            ;;
        4)
            read -p "Enter top_p (0.0-1.0): " v
            set_config_value "LLM_TOP_P" "$v"; show_restart_message ;;
        5)
            read -p "Enter top_k (integer): " v
            set_config_value "LLM_TOP_K" "$v"; show_restart_message ;;
        6)
            read -p "Enter repeat_penalty (>=1.0): " v
            set_config_value "LLM_REPEAT_PENALTY" "$v"; show_restart_message ;;
        7)
            read -p "Enter presence_penalty: " v
            set_config_value "LLM_PRESENCE_PENALTY" "$v"; show_restart_message ;;
        8)
            read -p "Enter frequency_penalty: " v
            set_config_value "LLM_FREQUENCY_PENALTY" "$v"; show_restart_message ;;
        9)
            echo ""
            echo "Current num_predict: $(get_config_value 'LLM_NUM_PREDICT')"
            if [ "$navigator_on" == "true" ]; then
                echo "Used for: Question generation (default max tokens)"
            fi
            read -p "Enter num_predict (max tokens): " v
            if [ -n "$v" ]; then
                set_config_value "LLM_NUM_PREDICT" "$v"
                show_restart_message
            fi
            ;;
        10)
            echo ""
            echo "Current stop sequences: $(get_config_value 'LLM_STOP')"
            echo ""
            echo "Examples:"
            echo "  - ChatML (Nemotron, Qwen): <|im_end|>,</s>"
            echo "  - Llama-3: </s>"
            echo "  - Double newline: \\n\\n"
            echo ""
            read -p "Enter stop sequences as CSV: " v
            set_config_value "LLM_STOP" "$v"; show_restart_message ;;
        11)
            echo ""
            echo "Current chat format: $(get_config_value 'SIMPLE_CHAT_FORMAT')"
            read -p "Enter chat format (e.g., llama-3, qwen2, qwen2.5): " cf
            if [ -n "$cf" ]; then
                set_config_value "SIMPLE_CHAT_FORMAT" "$cf"
                show_restart_message
            fi
            ;;
        12)
            if [ "$navigator_on" == "true" ]; then
                echo ""
                echo "Current empathetic temperature: $(get_config_value 'LLM_TEMPERATURE_EMPATHETIC' || echo '0.4')"
                echo "Used for: Empathetic statements and responses"
                read -p "Enter temperature for empathetic statements (0.0-1.0): " temp
                if [ -n "$temp" ]; then
                    set_config_value "LLM_TEMPERATURE_EMPATHETIC" "$temp"
                    show_restart_message
                fi
            else
                return
            fi
            ;;
        13)
            if [ "$navigator_on" == "true" ]; then
                echo ""
                echo "Current summary temperature: $(get_config_value 'LLM_TEMPERATURE_SUMMARY' || echo '0.25')"
                echo "Used for: History summaries (lower = more accurate)"
                read -p "Enter temperature for summaries (0.0-1.0): " temp
                if [ -n "$temp" ]; then
                    set_config_value "LLM_TEMPERATURE_SUMMARY" "$temp"
                    show_restart_message
                fi
            else
                echo ""
                echo -e "${YELLOW}⚠️  Medical Navigator is not enabled.${NC}"
                echo "Enable Medical Navigator to configure task-specific parameters."
                echo ""
                read -p "Press Enter to continue..."
            fi
            ;;
        14)
            if [ "$navigator_on" == "true" ]; then
                echo ""
                echo "Current empathetic max tokens: $(get_config_value 'LLM_MAX_TOKENS_EMPATHETIC' || echo '80')"
                echo "Used for: Empathetic statements"
                read -p "Enter max tokens for empathetic statements: " tokens
                if [ -n "$tokens" ] && [[ "$tokens" =~ ^[0-9]+$ ]]; then
                    set_config_value "LLM_MAX_TOKENS_EMPATHETIC" "$tokens"
                    show_restart_message
                else
                    echo -e "${RED}Invalid value. Please enter a positive integer.${NC}"
                    sleep 1
                fi
            else
                echo ""
                echo -e "${YELLOW}⚠️  Medical Navigator is not enabled.${NC}"
                echo "Enable Medical Navigator to configure task-specific parameters."
                echo ""
                read -p "Press Enter to continue..."
            fi
            ;;
        15)
            if [ "$navigator_on" == "true" ]; then
                echo ""
                echo "Current chronicity max tokens: $(get_config_value 'LLM_MAX_TOKENS_CHRONICITY' || echo '60')"
                echo "Used for: Chronicity questions"
                read -p "Enter max tokens for chronicity questions: " tokens
                if [ -n "$tokens" ] && [[ "$tokens" =~ ^[0-9]+$ ]]; then
                    set_config_value "LLM_MAX_TOKENS_CHRONICITY" "$tokens"
                    show_restart_message
                else
                    echo -e "${RED}Invalid value. Please enter a positive integer.${NC}"
                    sleep 1
                fi
            else
                echo ""
                echo -e "${YELLOW}⚠️  Medical Navigator is not enabled.${NC}"
                echo "Enable Medical Navigator to configure task-specific parameters."
                echo ""
                read -p "Press Enter to continue..."
            fi
            ;;
        16)
            if [ "$navigator_on" == "true" ]; then
                echo ""
                echo "Current summary max tokens: $(get_config_value 'LLM_MAX_TOKENS_SUMMARY' || echo '220')"
                echo "Used for: History summaries"
                read -p "Enter max tokens for summaries: " tokens
                if [ -n "$tokens" ] && [[ "$tokens" =~ ^[0-9]+$ ]]; then
                    set_config_value "LLM_MAX_TOKENS_SUMMARY" "$tokens"
                    show_restart_message
                else
                    echo -e "${RED}Invalid value. Please enter a positive integer.${NC}"
                    sleep 1
                fi
            else
                echo ""
                echo -e "${YELLOW}⚠️  Medical Navigator is not enabled.${NC}"
                echo "Enable Medical Navigator to configure task-specific parameters."
                echo ""
                read -p "Press Enter to continue..."
            fi
            ;;
        17)
            if [ "$navigator_on" == "true" ]; then
                return
            else
                echo ""
                echo -e "${YELLOW}⚠️  Medical Navigator is not enabled.${NC}"
                echo "Option 17 is only available when Medical Navigator is enabled."
                echo ""
                read -p "Press Enter to continue..."
            fi
            ;;
        *)
            echo ""
            echo -e "${RED}Invalid choice${NC}"
            sleep 1
            ;;
    esac
}

configure_medical_mode() {
    print_header "LLM CONTAINER MODE CONFIGURATION"
    
    local current=$(get_config_value 'USE_MEDICAL_MODE')
    
    echo "Current Mode:"
    if [ "$current" == "true" ]; then
        echo -e "  ${GREEN}Medical Mode${NC} - Symptom assessment, adaptive diagnostics"
        echo ""
        echo "  Endpoints available:"
        echo "    • /chat-medical - Medical conversations"
        echo "    • /chat-tg - Routes to medical mode"
        echo "    • /chat-tts - Routes to medical mode (streaming)"
    else
        echo -e "  ${YELLOW}Generic Mode${NC} - General conversation, RAG Q&A"
        echo ""
        echo "  Endpoints available:"
        echo "    • /chat-generic - Generic conversations"
        echo "    • /chat-tg - Routes to generic mode"
        echo "    • /chat-tts - Routes to generic mode (streaming)"
    fi
    echo ""
    echo "1) Enable Medical Mode"
    echo "2) Enable Generic Mode"
    echo "3) Back to main menu"
    echo ""
    read -p "Choice [1-3]: " choice
    
    case $choice in
        1) toggle_medical_mode on ;;
        2) toggle_medical_mode off ;;
        3) return ;;
    esac
}

configure_medical_categories() {
    print_header "MEDICAL CATEGORIES CONFIGURATION"
    
    local current=$(get_config_value 'ENABLED_MEDICAL_CATEGORIES')
    if [ -z "$current" ]; then
        current="GI (default)"
    fi
    
    echo "Current Enabled Categories: $current"
    echo ""
    echo "Available Categories:"
    echo "  • GI         - Gastrointestinal (curated)"
    echo "  • CARDIO     - Cardiovascular"
    echo "  • DERM       - Dermatology"
    echo "  • GU         - Genitourinary"
    echo "  • GYN        - Gynecological"
    echo "  • MSK        - Musculoskeletal"
    echo "  • NEURO      - Neurological"
    echo "  • PULMONARY  - Pulmonary"
    echo "  • RENAL      - Renal"
    echo ""
    echo "Note: Only curated categories (GI) are recommended for use."
    echo "      Other categories may have incomplete guidelines."
    echo ""
    echo "1) Set enabled categories (comma-separated, e.g., 'GI' or 'GI,CARDIO')"
    echo "2) Clear (use all categories - not recommended)"
    echo "3) Back to main menu"
    echo ""
    read -p "Choice [1-3]: " choice
    
    case $choice in
        1)
            echo ""
            echo "Enter categories (comma-separated, e.g., 'GI' or 'GI,CARDIO'):"
            echo "Available: GI, CARDIO, DERM, GU, GYN, MSK, NEURO, PULMONARY, RENAL"
            read -p "Categories: " categories
            if [ -n "$categories" ]; then
                set_config_value "ENABLED_MEDICAL_CATEGORIES" "$categories"
                echo ""
                echo -e "${GREEN}✅ Updated enabled categories to: $categories${NC}"
                echo ""
                echo "The system will now only load guidelines from these categories."
                echo "Other categories will be ignored."
                show_restart_message
            else
                echo "No categories entered. Keeping current setting."
            fi
            ;;
        2)
            echo ""
            read -p "Clear enabled categories (load all)? This is not recommended. (y/n): " answer
            if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                # Remove the line from .env
                if [ -f "$CONFIG_FILE" ]; then
                    grep -v "^ENABLED_MEDICAL_CATEGORIES=" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" 2>/dev/null || true
                    mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
                fi
                echo ""
                echo -e "${YELLOW}⚠️  Cleared enabled categories - will load all categories${NC}"
                echo ""
                echo "This is not recommended as only GI is curated."
                show_restart_message
            fi
            ;;
        3)
            return ;;
    esac
}

configure_rag() {
    print_header "RAG SEARCH CONFIGURATION"
    
    local rag_mode=$(get_config_value 'RAG_MODE')
    
    echo "Current Settings:"
    echo ""
    if [ "$rag_mode" == "GPU" ]; then
        echo -e "  RAG Mode:     ${GREEN}GPU${NC} (RAG container - port 11435)"
    else
        echo -e "  RAG Mode:     ${YELLOW}CPU${NC} (FAISS in LLM containers)"
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
            local current=$(get_config_value 'RAG_MODE')
            if [ "$current" == "GPU" ]; then
                echo "Currently: RAG_MODE=GPU (RAG container - port 11435)"
                echo ""
                read -p "Switch to RAG_MODE=CPU (FAISS in LLM containers)? (y/n): " answer
                if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                    set_config_value "RAG_MODE" "CPU"
                    echo ""
                    echo -e "${GREEN}✅ Switched to RAG_MODE=CPU${NC}"
                    echo ""
                    echo "RAG is still enabled, but now using:"
                    echo "  • CPU FAISS within LLM containers (ports 11434/11436)"
                    echo "  • No separate RAG container needed"
                    echo "  • Simpler setup, no network calls"
                    echo "  • Direct file processing in LLM containers"
                    echo ""
                    show_restart_message
                fi
            else
                echo "Currently: RAG_MODE=CPU (FAISS in LLM containers)"
                echo ""
                read -p "Switch to RAG_MODE=GPU (separate RAG container)? (y/n): " answer
                if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                    set_config_value "RAG_MODE" "GPU"
                    echo ""
                    echo -e "${GREEN}✅ Switched to RAG_MODE=GPU${NC}"
                    echo ""
                    echo "RAG is still enabled, but now using:"
                    echo "  • Separate RAG container (port 11435)"
                    echo "  • GPU-accelerated FAISS"
                    echo "  • Faster for large batches"
                    echo "  • Better scalability"
                    echo ""
                    echo "Note: Requires GPU and RAG container to be running"
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
    echo "3) Set TTS token limit (tokens per chunk)"
    echo "4) Set TTS volume (0-100%)"
    echo "5) Clear API key"
    echo "6) Back to main menu"
    echo ""
    read -p "Choice [1-6]: " choice
    
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
            echo ""
            echo "Current token limit: ${tts_limit:-not set}"
            read -p "Enter TTS token limit (positive integer): " tts_limit_new
            if [[ "$tts_limit_new" =~ ^[0-9]+$ ]] && [ "$tts_limit_new" -gt 0 ]; then
                set_config_value "TTS_TOKEN_LIMIT" "$tts_limit_new"
                echo -e "${GREEN}✅ TTS token limit saved${NC}"
                echo ""
                echo "Note: Restart speaker service to apply"
            else
                echo -e "${RED}Invalid value. Please enter a positive integer.${NC}"
            fi
            ;;
        4)
            echo ""
            echo "Current volume: ${tts_volume:-not set}"
            read -p "Enter TTS volume (0-100): " tts_volume_new
            if [[ "$tts_volume_new" =~ ^[0-9]+$ ]] && [ "$tts_volume_new" -ge 0 ] && [ "$tts_volume_new" -le 100 ]; then
                set_config_value "TTS_VOLUME" "$tts_volume_new"
                echo -e "${GREEN}✅ TTS volume saved${NC}"
                echo ""
                echo "Note: Restart speaker service to apply"
            else
                echo -e "${RED}Invalid value. Please enter an integer 0-100.${NC}"
            fi
            ;;
        5)
            set_config_value "ELEVENLABS_API_KEY" "your_elevenlabs_api_key_here"
            echo ""
            echo -e "${GREEN}✅ API key cleared${NC}"
            ;;
        6) return ;;
    esac
}

configure_voice_activation() {
    print_header "VOICE ACTIVATION & CONVERSATION MEMORY"

    local keywords=$(get_config_value 'ACTIVATION_KEYWORDS')
    local window=$(get_config_value 'ACTIVATION_WINDOW_SECONDS')
    local cooldown=$(get_config_value 'ACTIVATION_COOLDOWN_SECONDS')
    local memory_dir=$(get_config_value 'CONVERSATION_MEMORY_DIR')
    local persist_every=$(get_config_value 'CONVERSATION_MEMORY_PERSIST_EVERY')
    local max_entries=$(get_config_value 'CONVERSATION_MEMORY_MAX_ENTRIES')
    local top_k=$(get_config_value 'CONVERSATION_MEMORY_TOP_K')
    local min_score=$(get_config_value 'CONVERSATION_MEMORY_MIN_SCORE')

    [ -z "$keywords" ] && keywords="hey aura"
    [ -z "$window" ] && window="15"
    [ -z "$cooldown" ] && cooldown="3"
    [ -z "$memory_dir" ] && memory_dir="data/learning/conversation_memory"
    [ -z "$persist_every" ] && persist_every="10"
    [ -z "$max_entries" ] && max_entries="5000"
    [ -z "$top_k" ] && top_k="3"
    [ -z "$min_score" ] && min_score="0.35"

    echo "Current Settings:"
    echo ""
    echo "  Activation keywords:     $keywords"
    echo "  Activation window (sec): $window"
    echo "  Activation cooldown (sec): $cooldown"
    echo "  Memory directory:        $memory_dir"
    echo "  Persist every (entries): $persist_every"
    echo "  Max entries stored:      $max_entries"
    echo "  Memory Top K recall:     $top_k"
    echo "  Memory minimum score:    $min_score"
    echo ""

    echo "1) Set activation keywords (comma-separated)"
    echo "2) Set activation window (seconds)"
    echo "3) Set activation cooldown (seconds)"
    echo "4) Set memory directory"
    echo "5) Set persistence interval (entries)"
    echo "6) Set memory max entries"
    echo "7) Set memory Top K recall"
    echo "8) Set memory minimum score"
    echo "9) Back to main menu"
    echo ""
    read -p "Choice [1-9]: " choice

    case $choice in
        1)
            echo ""
            read -p "Enter activation keywords (comma-separated): " val
            if [ -n "$val" ]; then
                set_config_value "ACTIVATION_KEYWORDS" "$val"
            else
                set_config_value "ACTIVATION_KEYWORDS" "hey aura"
            fi
            show_restart_message
            ;;
        2)
            read -p "Enter activation window in seconds (e.g., 15): " val
            if [ -n "$val" ]; then
                set_config_value "ACTIVATION_WINDOW_SECONDS" "$val"
                show_restart_message
            fi
            ;;
        3)
            read -p "Enter activation cooldown in seconds (e.g., 3): " val
            if [ -n "$val" ]; then
                set_config_value "ACTIVATION_COOLDOWN_SECONDS" "$val"
                show_restart_message
            fi
            ;;
        4)
            read -p "Enter memory directory path: " val
            if [ -n "$val" ]; then
                set_config_value "CONVERSATION_MEMORY_DIR" "$val"
                show_restart_message
            fi
            ;;
        5)
            read -p "Persist to disk after how many entries (e.g., 10): " val
            if [ -n "$val" ]; then
                set_config_value "CONVERSATION_MEMORY_PERSIST_EVERY" "$val"
                show_restart_message
            fi
            ;;
        6)
            read -p "Maximum number of transcript entries to retain (e.g., 5000): " val
            if [ -n "$val" ]; then
                set_config_value "CONVERSATION_MEMORY_MAX_ENTRIES" "$val"
                show_restart_message
            fi
            ;;
        7)
            read -p "Number of memory snippets to recall (Top K, e.g., 3): " val
            if [ -n "$val" ]; then
                set_config_value "CONVERSATION_MEMORY_TOP_K" "$val"
                show_restart_message
            fi
            ;;
        8)
            read -p "Minimum similarity score for recall (0.0-1.0, e.g., 0.35): " val
            if [ -n "$val" ]; then
                set_config_value "CONVERSATION_MEMORY_MIN_SCORE" "$val"
                show_restart_message
            fi
            ;;
        9) return ;;
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

configure_github() {
    print_header "GITHUB OTA UPDATE CONFIGURATION"
    
    local gh_token=$(get_config_value 'GITHUB_TOKEN')
    
    echo "Current Settings:"
    echo ""
    if [ -n "$gh_token" ] && [ "$gh_token" != "your_github_token_here" ]; then
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
            set_config_value "GITHUB_TOKEN" "your_github_token_here"
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
        echo "  3) Toggle LLM Mode (Medical/Generic)"
        echo "  4) Configure LLM Mode settings"
        echo "  5) Configure Medical Categories (guidelines)"
        echo "  6) Configure LLM models"
        echo "  7) Configure RAG search"
        echo "  8) Configure TTS (ElevenLabs)"
        echo "  9) Configure Telegram bot"
        echo " 10) Configure GitHub OTA updates"
        echo " 11) Configure NHS/FHIR credentials"
        echo " 12) Toggle ML Learning (on/off)"
        echo " 13) Toggle Medical Navigator (on/off)"
        echo " 14) Configure Voice Activation & Memory"
        echo "  a) Edit .env file directly"
        echo "  b) Restart Docker containers"
        echo "  0) Exit"
        echo ""
        read -p "Enter choice [0-14ab]: " choice
        
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
                # Show current state and ask what to do
                local current=$(get_config_value 'USE_MEDICAL_MODE')
                echo ""
                if [ "$current" == "true" ]; then
                    echo "LLM Mode is currently: MEDICAL"
                    echo ""
                    read -p "Switch to GENERIC mode? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_medical_mode off
                    fi
                else
                    echo "LLM Mode is currently: GENERIC"
                    echo ""
                    read -p "Switch to MEDICAL mode? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_medical_mode on
                    fi
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            4)
                configure_medical_mode
                read -p "Press Enter to continue..."
                ;;
            5)
                configure_medical_categories
                read -p "Press Enter to continue..."
                ;;
            6)
                configure_llm
                read -p "Press Enter to continue..."
                ;;
            7)
                configure_rag
                read -p "Press Enter to continue..."
                ;;
            8)
                configure_tts
                read -p "Press Enter to continue..."
                ;;
            9)
                configure_telegram
                read -p "Press Enter to continue..."
                ;;
            10)
                configure_github
                read -p "Press Enter to continue..."
                ;;
            11)
                configure_nhs_fhir
                read -p "Press Enter to continue..."
                ;;
            12)
                # Show current state and ask what to do
                local current=$(get_config_value 'ENABLE_ML_LEARNING')
                echo ""
                if [ "$current" == "true" ]; then
                    echo "ML Learning is currently: ENABLED"
                    echo ""
                    read -p "Turn it OFF? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_ml_learning off
                    fi
                else
                    echo "ML Learning is currently: DISABLED"
                    echo ""
                    read -p "Turn it ON? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_ml_learning on
                    fi
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            13)
                # Show current state and ask what to do
                local current=$(get_config_value 'USE_MEDICAL_NAVIGATOR')
                echo ""
                if [ "$current" == "true" ]; then
                    echo "Advanced Medical Navigator is currently: ENABLED"
                    echo ""
                    echo "Using: Pure LLM-based medical assistant"
                    echo ""
                    read -p "Switch to Adaptive Diagnostic Engine? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_medical_navigator off
                    fi
                else
                    echo "Advanced Medical Navigator is currently: DISABLED"
                    echo ""
                    echo "Using: Adaptive Diagnostic Engine (default)"
                    echo ""
                    read -p "Enable Advanced Medical Navigator? (y/n): " answer
                    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
                        toggle_medical_navigator on
                    fi
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            14)
                configure_voice_activation
                read -p "Press Enter to continue..."
                ;;
            a|A)
                edit_file
                ;;
            b|B)
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
    mode|medical)
        case "${2:-}" in
            on|enable|medical) toggle_medical_mode on ;;
            off|disable|generic) toggle_medical_mode off ;;
            *) configure_medical_mode ;;
        esac
        ;;
    ml|learning)
        case "${2:-}" in
            on|enable) toggle_ml_learning on ;;
            off|disable) toggle_ml_learning off ;;
            *) 
                echo "Usage: $0 ml [on|off]"
                echo "  on  - Enable ML learning"
                echo "  off - Disable ML learning"
                ;;
        esac
        ;;
    navigator|medical-navigator)
        case "${2:-}" in
            on|enable) toggle_medical_navigator on ;;
            off|disable) toggle_medical_navigator off ;;
            *) 
                echo "Usage: $0 navigator [on|off]"
                echo "  on  - Enable Advanced Medical Navigator (pure LLM mode)"
                echo "  off - Disable Advanced Navigator (use Adaptive Diagnostic Engine)"
                ;;
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

