#!/bin/bash
# Installation script for Wallet Integration

set -e

echo "=================================================="
echo "🔗 Installing Ethereum Wallet Integration for Aura"
echo "=================================================="
echo ""

# Navigate to aura-control directory
cd "$(dirname "$0")/aura-control"

echo "📦 Installing Web3 dependencies..."
pip install web3>=6.11.0 eth-account>=0.10.0 eth-utils>=2.3.0

echo ""
echo "✅ Installation complete!"
echo ""
echo "=================================================="
echo "📖 Next Steps:"
echo "=================================================="
echo ""
echo "1. (Optional) Set up an Ethereum RPC provider for better reliability:"
echo ""
echo "   Option A - Infura (recommended):"
echo "   - Sign up at https://infura.io"
echo "   - Create a project and copy your Project ID"
echo "   - Run: export INFURA_URL=\"https://mainnet.infura.io/v3/YOUR_PROJECT_ID\""
echo ""
echo "   Option B - Alchemy:"
echo "   - Sign up at https://alchemy.com"
echo "   - Create an app and copy your API key"
echo "   - Run: export ALCHEMY_URL=\"https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY\""
echo ""
echo "2. Launch Aura:"
echo "   cd aura-control"
echo "   python main.py"
echo ""
echo "3. Click the 📊 Analytics button to open the wallet interface"
echo ""
echo "4. Enter your Ethereum wallet address to view your token balance"
echo ""
echo "=================================================="
echo "📚 For detailed documentation, see:"
echo "   WALLET_INTEGRATION_GUIDE.md"
echo "=================================================="

