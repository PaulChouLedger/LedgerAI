# Wallet Integration Guide for LedgerAI Aura

## Overview

LedgerAI Aura now includes Ethereum wallet integration that allows you to:
- Connect to your Ethereum wallet (MetaMask, Coinbase, or any Ethereum address)
- View your balance for a **real DEX-traded token** (0xD1F2586790a5bD6DA1e443441df53aF6EC213D83)
- Track computational token consumption during Aura interactions
- Send real token payments to the client wallet
- Monitor usage based on query complexity

**Important:** This integration uses a real token with actual market value that can be traded on decentralized exchanges.

## Setup

### 1. Install Dependencies

```bash
cd /Users/rcabello/Documents/GitHub/LedgerAI/aura-control
pip install -r requirements.txt
```

This will install:
- `web3>=6.11.0` - Ethereum connectivity
- `eth-account>=0.10.0` - Account utilities
- `eth-utils>=2.3.0` - Ethereum utilities

### 2. Configure Ethereum Provider (Optional but Recommended)

For the best experience, set up an Ethereum RPC provider. You have two options:

#### Option A: Infura (Recommended)

1. Sign up at [infura.io](https://infura.io)
2. Create a new project
3. Copy your project ID
4. Set environment variable:

```bash
export INFURA_URL="https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
```

#### Option B: Alchemy

1. Sign up at [alchemy.com](https://alchemy.com)
2. Create a new app
3. Copy your API key
4. Set environment variable:

```bash
export ALCHEMY_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY"
```

#### Option C: Public Endpoints (Fallback)

If you don't set up a provider, the system will automatically fall back to public endpoints:
- Cloudflare Ethereum Gateway
- Ethereum Public Node

**Note:** Public endpoints may be slower and rate-limited.

## Usage

### Accessing the Wallet Interface

1. Launch Aura as usual:
   ```bash
   python main.py
   ```

2. Once the GUI appears, click the **📊 Analytics** button on the circular interface

3. The Wallet Dialog will open showing:
   - Network connection status
   - Wallet address input field
   - Token balance display
   - ETH balance display
   - Session usage statistics

### Connecting Your Wallet

1. In the wallet dialog, enter your Ethereum wallet address:
   ```
   0x1234567890abcdef1234567890abcdef12345678
   ```

2. Click **🔗 Connect Wallet** or press Enter

3. Your token balance will be displayed:
   ```
   💰 1234.567890 TOKEN
   Ξ 0.123456 ETH
   ```

**Note:** This is a read-only connection. We only query your balance - no transactions or signing required.

## Token Consumption Tracking

### Storage Location

Token usage is stored in:
```
/Users/rcabello/Documents/GitHub/LedgerAI/data/token_usage.json
```

This file contains:
- Total token consumption
- Total tokens paid to client
- Balance owed (usage - paid)
- Last 100 operations history

## How It Works

Every Aura interaction consumes tokens based on computational complexity:

| Operation Type | Base Cost | Multiplier |
|---------------|-----------|------------|
| **Simple Query** | 0.001 tokens | 1x |
| **RAG Query** | 0.005 tokens | Based on document retrieval |
| **Complex Query** | 0.010 tokens | Scaled by prompt length |
| **Transcription** | 0.002 tokens | Per second of audio |
| **TTS Generation** | 0.001 tokens | Per second of speech |

### Examples

- **Simple question**: "What time is it?" → 0.001 tokens
- **Medical RAG query**: "What are the symptoms of diabetes?" → 0.005 tokens
- **Complex analysis**: "Analyze the pathophysiology of heart failure and explain treatment protocols" → 0.020+ tokens
- **Long conversation**: 30-second question → 0.060 tokens (transcription) + query cost

### Viewing Usage

The wallet dialog displays:
- **Total Usage**: All tokens consumed (saved to disk)
- **Persistent Tracking**: Usage persists across reboots and system restarts
- **Real-time Updates**: Automatically refreshes every 30 seconds
- **Saved Automatically**: Usage is saved after each operation to `~/LedgerAI/data/token_usage.json`
- **Never Resets**: Usage accumulates permanently until manually cleared from disk

## Token Address

**Real DEX-Traded Token:**
```
0xD1F2586790a5bD6DA1e443441df53aF6EC213D83
```

This is a **real ERC-20 token traded on decentralized exchanges** (Ethereum Mainnet). The integration automatically:
- Detects token name and symbol from the blockchain
- Reads token decimals
- Formats balance correctly
- Supports real transactions with actual market value

**Note:** This token has real market value and can be traded on DEX platforms.

## Future Enhancements

### Planned Features

1. **Automatic Balance Deduction**
   - Integrate with smart contract to automatically deduct tokens on usage
   - Real-time balance updates after each interaction

2. **Subscription Tiers**
   - Basic: Simple queries only
   - Pro: RAG-enhanced queries
   - Enterprise: Unlimited complex analysis

3. **Token Purchase Integration**
   - Buy more tokens directly from the interface
   - Integration with DEX for token acquisition

4. **Usage Analytics**
   - Historical usage graphs
   - Query type breakdowns
   - Cost optimization suggestions

5. **Multi-User Support**
   - User profiles with individual token balances
   - Family/team token pools
   - Usage limits per user

## Architecture

### Components

1. **`wallet_integration.py`**
   - `WalletManager`: Handles Web3 connections and token queries
   - `TokenUsageTracker`: Tracks computational costs

2. **`wallet_dialog.py`**
   - PyQt5 dialog for wallet UI
   - Real-time balance display
   - Usage statistics

3. **Integration Points**
   - `speaker.py`: Tracks LLM queries and TTS generation
   - `listener.py`: Tracks audio transcription
   - `aura_gui.py`: Analytics button trigger

### Token Tracking Flow

```
User speaks → transcribe() [0.002 tokens/sec]
    ↓
LLM processing → speak_llm_response() [0.001-0.010 tokens]
    ↓
TTS generation → tts_playback_thread() [0.001 tokens/sec]
    ↓
Total Usage Saved to data/token_usage.json
    ↓
Wallet Dialog Shows:
  💳 Total used
  💸 Total paid
  📊 Balance owed (usage - paid)
```

### Resetting Usage (Manual Only)

Usage is designed to accumulate permanently. To reset:

```bash
# Delete the usage file
rm /Users/rcabello/Documents/GitHub/LedgerAI/data/token_usage.json

# Next time Aura starts, usage will begin at 0
```

**Note**: There is no reset button in the UI by design - this ensures accurate tracking of all computational costs.

### How Payments Work

When you send a payment to the client:
1. Transaction is broadcast to Ethereum blockchain
2. Once confirmed, payment is recorded: `total_paid += payment_amount`
3. Balance owed is automatically calculated: `owed = total_usage - total_paid`
4. All values saved to disk immediately

**Example:**
```
Start: 0.000 used, 0.000 paid, 0.000 owed
Use Aura: 0.050 used, 0.000 paid, 0.050 owed
Pay 0.020: 0.050 used, 0.020 paid, 0.030 owed
Use more: 0.075 used, 0.020 paid, 0.055 owed
Pay 0.055: 0.075 used, 0.075 paid, 0.000 owed ✅
```

## Troubleshooting

### "Not connected to Ethereum network"

**Solution:**
1. Check internet connection
2. Set up an Infura or Alchemy provider (see Setup section)
3. Try again - public endpoints may be temporarily unavailable

### "Invalid Ethereum address"

**Solution:**
- Ensure address starts with `0x`
- Address should be 42 characters long
- Use checksum format (capitals matter)

### "Error fetching balance"

**Possible causes:**
1. RPC provider rate limit reached
2. Network connectivity issues
3. Invalid token contract address

**Solution:**
- Wait a few seconds and click **🔄 Refresh**
- Set up a dedicated RPC provider (Infura/Alchemy)

### Token info shows "Unknown Token"

**Solution:**
- The token contract may not implement standard ERC-20 methods
- Check that the token address is correct
- Balance will still be displayed if available

## Security Notes

### What We Do

✅ Read-only wallet queries
✅ No private keys stored
✅ No transaction signing
✅ Local token tracking

### What We DON'T Do

❌ Never ask for private keys
❌ Never sign transactions
❌ Never send tokens
❌ Never store wallet credentials

**Important:** This integration only reads your wallet balance. It cannot make transactions or access your funds.

## API Reference

### WalletManager

```python
from wallet_integration import get_wallet_manager

manager = get_wallet_manager()

# Connect to wallet
manager.connect_wallet("0x1234...")

# Get token balance
balance = manager.get_token_balance()

# Get ETH balance
eth_balance = manager.get_eth_balance()

# Get complete wallet info
info = manager.get_wallet_info()
```

### TokenUsageTracker

```python
from wallet_integration import get_usage_tracker

tracker = get_usage_tracker()

# Record usage
tracker.record_usage('simple_query')
tracker.record_usage('transcription', multiplier=5.0)  # 5 seconds

# Get total usage (persists until app restart)
total = tracker.get_session_usage()
```

## Support

For issues or questions:
1. Check this guide first
2. Review console logs for error messages
3. Ensure all dependencies are installed
4. Verify Ethereum provider connectivity

## License

This integration is part of LedgerAI and follows the same license as the main project.

