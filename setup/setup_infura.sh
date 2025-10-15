#!/bin/bash
# Quick setup script for Infura RPC provider

echo "=================================================="
echo "🔗 Infura Setup for Wallet Integration"
echo "=================================================="
echo ""
echo "This will help you set up Infura for reliable Ethereum access"
echo ""
echo "Steps:"
echo "1. Go to https://infura.io and create a free account"
echo "2. Create a new project"
echo "3. Copy your Project ID"
echo "4. Paste it below"
echo ""
read -p "Enter your Infura Project ID: " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo "❌ No Project ID provided. Exiting."
    exit 1
fi

echo ""
echo "Setting up environment variable..."

# Add to .bashrc or .zshrc
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    echo "" >> "$SHELL_RC"
    echo "# Infura for LedgerAI Wallet Integration" >> "$SHELL_RC"
    echo "export INFURA_URL=\"https://mainnet.infura.io/v3/$PROJECT_ID\"" >> "$SHELL_RC"
    echo "✅ Added to $SHELL_RC"
else
    echo "⚠️ Could not find .bashrc or .zshrc"
fi

# Set for current session
export INFURA_URL="https://mainnet.infura.io/v3/$PROJECT_ID"

echo ""
echo "✅ Infura configured!"
echo ""
echo "Current session: INFURA_URL set"
echo "Future sessions: Will load from $SHELL_RC"
echo ""
echo "You can now run Aura with reliable Ethereum access:"
echo "  cd aura-control"
echo "  python main.py"
echo ""
echo "=================================================="

