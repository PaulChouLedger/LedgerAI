# NFC Wallet Authentication Guide

## ✅ Option 1: Persistent Storage (IMPLEMENTED)

### How It Works:
1. Enter wallet address once
2. Auto-saves to `~/LedgerAI/data/wallet_address.txt`
3. Next time you open wallet dialog: **Big blue "Use Saved Wallet" button**
4. One click to connect!

### Features:
- ✅ Auto-connects on startup if wallet saved
- ✅ Shows "Use Saved Wallet" button for quick access
- ✅ "Clear Saved" button to remove stored wallet
- ✅ Can still enter different wallet manually

### UI Changes:
When saved wallet exists, you'll see:
```
┌──────────────────────────────────┐
│ 💾 Saved Wallet: 0x1234...5678   │
│ ┌──────────┐  ┌──────────┐      │
│ │✅ Use    │  │🗑️ Clear  │      │
│ │  Saved   │  │  Saved   │      │
│ └──────────┘  └──────────┘      │
│                                  │
│    — OR enter different wallet — │
│                                  │
│ Enter your Ethereum wallet...    │
│ [____________________________]   │
│ [🔗 Connect New Wallet]          │
└──────────────────────────────────┘
```

## 🔜 Option 2: NFC Card Authentication (TO IMPLEMENT)

### Hardware Shopping List:

#### NFC Reader Options:

**RECOMMENDED: ACR122U USB NFC Reader**
- **Price:** $15-25
- **Pros:**
  - ✅ Plug-and-play USB
  - ✅ Well-supported on Linux (nfcpy, pyscard)
  - ✅ Reads/writes most NFC tags
  - ✅ No GPIO wiring needed
  - ✅ Works with Jetson Orin Nano
- **Where to buy:**
  - Amazon: "ACR122U NFC Reader"
  - AliExpress: Search "ACR122U"
- **Specs:**
  - Interface: USB
  - Protocols: ISO14443A/B, Mifare, FeliCa
  - Range: ~5cm (perfect for tap)
  
**Alternative: PN532 NFC Module**
- **Price:** $10-20
- **Pros:**
  - ✅ Can use USB or UART/I2C/SPI
  - ✅ Lower power
  - ✅ GPIO integration possible
- **Cons:**
  - ⚠️ More complex setup
  - ⚠️ May need adapter cable for USB
- **Where to buy:**
  - Amazon: "PN532 NFC Module USB"
  - Adafruit: PN532 Breakout Board

#### NFC Cards/Tags:

**RECOMMENDED: NTAG215 Cards**
- **Price:** $8-15 for 10 pack
- **Specs:**
  - Memory: 504 bytes usable
  - Protocol: ISO14443A (Type 2)
  - Rewritable: Yes
  - Durable plastic cards
- **Where to buy:**
  - Amazon: "NTAG215 NFC Cards Blank"
  - Search: "NTAG215 PVC cards 10 pack"

**Alternative: NTAG216 Key Fobs**
- **Price:** $6-12 for 5-10 pack
- **Specs:**
  - Memory: 888 bytes usable
  - Keychain form factor
  - More portable
- **Where to buy:**
  - Amazon: "NTAG216 Key Fobs"

**Budget Option: Sticker Tags**
- **Price:** $5-8 for 10 pack
- **Pros:** Cheap, can stick anywhere
- **Cons:** Less durable, easier to damage

### Total Cost: $20-40

### What to Buy (Recommendation):

```
🛒 Shopping Cart:
├─ 1x ACR122U NFC Reader        $18-25
├─ 1x NTAG215 Cards (10 pack)   $10-15
└─ Optional: NTAG216 Key Fobs   $8-12
                        TOTAL:   $28-52
```

## NFC Implementation Plan:

### Phase 1: Basic NFC Reading
```python
# Libraries needed:
pip3 install nfcpy pyscard

# Test NFC reader
python3 -c "import nfc; clf = nfc.ContactlessFrontend('usb'); print(clf)"
```

### Phase 2: Write Wallet to NFC Card
```python
# Store wallet address on NFC card
# Format: 42-character Ethereum address
# Storage: 42 bytes (NTAG215 has 504 bytes - plenty!)
```

### Phase 3: Auto-Auth on Card Tap
```python
# When card detected:
# 1. Read wallet address from card
# 2. Auto-connect wallet
# 3. Show confirmation (voice + GUI)
# 4. Ready to use!
```

### Phase 4: Multi-User Support
```python
# Different cards for different users:
# - Card A: Rafael's wallet
# - Card B: User 2's wallet
# - Card C: Guest wallet
```

## How NFC Auth Will Work:

### User Experience:

```
1. Open Aura
2. Tap NFC card on reader
   🔊 "Welcome back, Rafael"
3. Auto-connected, balance loaded
4. Start using Aura
```

### Technical Flow:

```
NFC Reader → Detects card → Reads wallet address → 
Auto-connects → Updates GUI → Ready!
```

### Comparison with Manual Entry:

| Method | Steps | Time | User Experience |
|--------|-------|------|-----------------|
| **Manual Entry** | 5 steps | 30-60s | Type 42-char address |
| **Saved Wallet** | 1 click | 2-3s | Click "Use Saved" button |
| **NFC Card** | 1 tap | 1-2s | Tap card, done! |

## NFC Card Setup Process:

### 1. Write Wallet to Card (One-Time)

```python
# Using Python script (will create this)
python3 scripts/write_nfc_wallet.py

# Prompts:
# "Enter wallet address: 0x..."
# "Tap card to write..."
# [Tap card]
# "✅ Wallet written to card!"
```

### 2. Test Card

```python
python3 scripts/read_nfc_wallet.py

# "Tap card..."
# [Tap card]
# "📖 Wallet: 0x1234...5678"
```

### 3. Use in Aura

- Open Aura
- Tap card
- Auto-authenticates!

## Security Considerations:

### NFC Cards:
- ⚠️ **Not encrypted** - anyone with reader can read wallet address
- ✅ **Read-only access** - can't spend tokens, only view balance
- ✅ **No private keys** - just the public address
- ⚠️ **Physical security** - treat like a door key

### Saved Wallet File:
- ⚠️ **Plain text** - stored in ~/LedgerAI/data/wallet_address.txt
- ✅ **No private keys** - just the public address
- ✅ **Read-only access** - can't spend tokens
- 🔒 **File permissions** - readable only by your user

### Both Are Safe Because:
- ✅ Only stores **public address** (not private key)
- ✅ Aura only **reads** balance (can't spend tokens)
- ✅ Like sharing your email address (public info)

## Future NFC Enhancements:

### Phase 1: Basic Auth (Week 1)
- Write wallet address to card
- Read card to connect wallet
- Single user support

### Phase 2: Multi-User (Week 2)
- Different cards for different users
- Store user preferences on card (name, settings)
- Voice greeting: "Welcome back, [name]"

### Phase 3: Advanced Features (Week 3+)
- Usage tracking per user
- Different models/settings per user
- Session history per card

### Phase 4: NFC + Encryption (Future)
- Encrypted wallet storage on card
- PIN-protected access
- Secure multi-user

## When NFC Hardware Arrives:

### Day 1: Setup
1. Plug in ACR122U reader
2. Install libraries: `pip3 install nfcpy pyscard`
3. Test reader: `nfc-list` or `nfc-scan-device`

### Day 2: Write Cards
1. Run card writing script (I'll create it)
2. Write your wallet to 2-3 cards (backup!)
3. Test reading cards

### Day 3: Integration
1. Add NFC detection to wallet dialog
2. Auto-connect on card tap
3. Test multi-user with different cards

### Day 4: Polish
1. Add voice confirmations
2. Add GUI feedback (card detected animation)
3. Error handling (card removed too soon, etc.)

## Files to Create (When NFC Arrives):

```
scripts/
├─ write_nfc_wallet.py     # Write wallet to NFC card
├─ read_nfc_wallet.py      # Read wallet from NFC card
└─ test_nfc_reader.py      # Test NFC reader is working

aura-control/
└─ nfc_auth.py             # NFC authentication module
```

## Immediate Next Steps:

1. ✅ **Persistent storage is ready** - enter wallet once, use forever
2. 🛒 **Order NFC hardware** - ACR122U + NTAG215 cards
3. ⏳ **Wait for delivery** (~3-7 days)
4. 🔧 **I'll implement NFC code** when hardware arrives

## Expected NFC Delivery Timeline:

- **Amazon Prime:** 1-2 days
- **Amazon Standard:** 3-5 days  
- **AliExpress:** 2-4 weeks (cheaper but slower)

**Recommendation:** Get ACR122U from Amazon for faster delivery!

Would you like me to create a placeholder NFC module now that you can test when the hardware arrives? 🚀

