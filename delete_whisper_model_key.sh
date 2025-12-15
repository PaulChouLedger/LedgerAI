#!/bin/bash
# Script to delete whisper_model key from app_settings.json on remote server
# Run this on the remote server: bash delete_whisper_model_key.sh

SETTINGS_FILE="$HOME/LedgerAI/data/app_settings.json"

if [ ! -f "$SETTINGS_FILE" ]; then
    echo "❌ Settings file not found: $SETTINGS_FILE"
    exit 1
fi

echo "📝 Removing whisper_model key from $SETTINGS_FILE"
echo ""

# Use Python to safely remove the key
python3 << PYTHON_SCRIPT
import json
import os

settings_path = "$SETTINGS_FILE"

try:
    with open(settings_path, 'r') as f:
        data = json.load(f)
    
    if "whisper_model" in data:
        old_value = data["whisper_model"]
        del data["whisper_model"]
        
        with open(settings_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Deleted whisper_model key (was: {old_value})")
        print(f"✅ Will now use code default: distil-small.en")
    else:
        print("✅ whisper_model key not found - already using default")
        print("✅ Code default: distil-small.en")
    
    print("")
    print("📋 Current settings:")
    with open(settings_path, 'r') as f:
        print(f.read())
        
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
PYTHON_SCRIPT

echo ""
echo "✅ Done! Restart containers to apply changes:"
echo "   docker-compose -f ~/LedgerAI/setup/docker-compose.yml restart whisper"
