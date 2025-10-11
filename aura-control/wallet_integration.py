# wallet_integration.py — Ethereum Wallet Integration for Aura

import os
import json
from web3 import Web3
from typing import Optional, Dict, Any

class WalletManager:
    """Manages Ethereum wallet connections and token balance tracking"""
    
    # Real DEX-traded token contract address (Ethereum Mainnet)
    # This token has real market value and can be traded on DEX platforms
    TOKEN_ADDRESS = "0xD1F2586790a5bD6DA1e443441df53aF6EC213D83"
    
    # ERC-20 Token ABI (minimal - just what we need for balanceOf and decimals)
    ERC20_ABI = json.loads('''[
        {
            "constant": true,
            "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "balance", "type": "uint256"}],
            "type": "function"
        },
        {
            "constant": true,
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "type": "function"
        },
        {
            "constant": true,
            "inputs": [],
            "name": "symbol",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        },
        {
            "constant": true,
            "inputs": [],
            "name": "name",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        }
    ]''')
    
    def __init__(self):
        # Initialize Web3 with multiple fallback providers
        self.w3: Optional[Web3] = None
        self.connected_address: Optional[str] = None
        self.token_contract = None
        self.token_info: Dict[str, Any] = {}
        
        # Persistent wallet storage
        self.wallet_file = os.path.expanduser("~/LedgerAI/data/wallet_address.txt")
        
        # Initialize connection
        self._init_web3_connection()
        
        # Try to load saved wallet
        self._load_saved_wallet()
    
    def _init_web3_connection(self) -> bool:
        """Initialize Web3 connection with fallback providers"""
        # Try multiple providers in order of preference
        providers = [
            # Infura (recommended - reliable and fast)
            os.getenv("INFURA_URL", "https://mainnet.infura.io/v3/YOUR_INFURA_KEY"),
            
            # Alchemy (alternative)
            os.getenv("ALCHEMY_URL", "https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY"),
            
            # Public endpoints (less reliable, rate-limited)
            "https://cloudflare-eth.com",
            "https://ethereum.publicnode.com",
        ]
        
        for provider_url in providers:
            try:
                if "YOUR_" in provider_url:
                    # Skip placeholder providers
                    continue
                    
                print(f"[Wallet] 🔗 Trying provider: {provider_url[:50]}...")
                self.w3 = Web3(Web3.HTTPProvider(provider_url))
                
                if self.w3.is_connected():
                    print(f"[Wallet] ✅ Connected to Ethereum via {provider_url[:50]}")
                    self._init_token_contract()
                    return True
                else:
                    print(f"[Wallet] ❌ Failed to connect to {provider_url[:50]}")
            except Exception as e:
                print(f"[Wallet] ⚠️ Provider error: {e}")
                continue
        
        print("[Wallet] ❌ Could not connect to any Ethereum provider")
        print("[Wallet] 💡 Set INFURA_URL or ALCHEMY_URL environment variable")
        return False
    
    def _init_token_contract(self):
        """Initialize token contract interface"""
        try:
            if not self.w3:
                return
            
            # Convert address to checksum format
            token_address = Web3.to_checksum_address(self.TOKEN_ADDRESS)
            
            # Create contract instance
            self.token_contract = self.w3.eth.contract(
                address=token_address,
                abi=self.ERC20_ABI
            )
            
            # Fetch token info
            try:
                self.token_info = {
                    'name': self.token_contract.functions.name().call(),
                    'symbol': self.token_contract.functions.symbol().call(),
                    'decimals': self.token_contract.functions.decimals().call(),
                    'address': token_address
                }
                print(f"[Wallet] 📊 Token: {self.token_info['name']} ({self.token_info['symbol']})")
            except Exception as e:
                print(f"[Wallet] ⚠️ Could not fetch token info: {e}")
                self.token_info = {
                    'name': 'Unknown Token',
                    'symbol': 'UNKN',
                    'decimals': 18,
                    'address': token_address
                }
        except Exception as e:
            print(f"[Wallet] ❌ Token contract initialization failed: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Ethereum network"""
        return self.w3 is not None and self.w3.is_connected()
    
    def _save_wallet_address(self, address: str):
        """Save wallet address to file for persistent storage"""
        try:
            os.makedirs(os.path.dirname(self.wallet_file), exist_ok=True)
            with open(self.wallet_file, 'w') as f:
                f.write(address)
            print(f"[Wallet] 💾 Saved wallet address for next session")
        except Exception as e:
            print(f"[Wallet] ⚠️ Could not save wallet: {e}")
    
    def _load_saved_wallet(self):
        """Load previously saved wallet address"""
        try:
            if os.path.exists(self.wallet_file):
                with open(self.wallet_file, 'r') as f:
                    saved_address = f.read().strip()
                
                if saved_address and self.is_connected():
                    print(f"[Wallet] 🔄 Found saved wallet: {saved_address[:6]}...{saved_address[-4:]}")
                    if self.connect_wallet(saved_address, auto_save=False):
                        print(f"[Wallet] ✅ Auto-connected to saved wallet")
                        return True
        except Exception as e:
            print(f"[Wallet] ⚠️ Could not load saved wallet: {e}")
        return False
    
    def get_saved_wallet(self) -> Optional[str]:
        """Get saved wallet address without connecting"""
        try:
            if os.path.exists(self.wallet_file):
                with open(self.wallet_file, 'r') as f:
                    return f.read().strip()
        except:
            pass
        return None
    
    def clear_saved_wallet(self):
        """Remove saved wallet address"""
        try:
            if os.path.exists(self.wallet_file):
                os.remove(self.wallet_file)
                print("[Wallet] 🗑️ Cleared saved wallet")
                return True
        except Exception as e:
            print(f"[Wallet] ⚠️ Could not clear saved wallet: {e}")
        return False
    
    def connect_wallet(self, address: str, auto_save: bool = True) -> bool:
        """Connect to a wallet address (for read-only operations)"""
        try:
            if not self.is_connected():
                print("[Wallet] ❌ Not connected to Ethereum network")
                return False
            
            # Validate and normalize address
            checksum_address = Web3.to_checksum_address(address)
            
            # Verify it's a valid address
            if not self.w3.is_address(checksum_address):
                print(f"[Wallet] ❌ Invalid Ethereum address: {address}")
                return False
            
            self.connected_address = checksum_address
            print(f"[Wallet] ✅ Connected to wallet: {checksum_address}")
            
            # Save wallet address for next session (unless disabled)
            if auto_save:
                self._save_wallet_address(checksum_address)
            
            return True
            
        except Exception as e:
            print(f"[Wallet] ❌ Wallet connection error: {e}")
            return False
    
    def get_token_balance(self, address: Optional[str] = None) -> Optional[float]:
        """Get token balance for an address"""
        try:
            if not self.is_connected():
                print("[Wallet] ❌ Not connected to Ethereum network")
                return None
            
            if not self.token_contract:
                print("[Wallet] ❌ Token contract not initialized")
                return None
            
            # Use connected address if none provided
            target_address = address or self.connected_address
            if not target_address:
                print("[Wallet] ❌ No wallet address provided")
                return None
            
            # Ensure checksum format
            target_address = Web3.to_checksum_address(target_address)
            
            # Get balance in smallest unit
            balance_wei = self.token_contract.functions.balanceOf(target_address).call()
            
            # Convert to human-readable format
            decimals = self.token_info.get('decimals', 18)
            balance = balance_wei / (10 ** decimals)
            
            print(f"[Wallet] 💰 Balance: {balance:.6f} {self.token_info.get('symbol', 'tokens')}")
            return balance
            
        except Exception as e:
            print(f"[Wallet] ❌ Balance query error: {e}")
            return None
    
    def get_eth_balance(self, address: Optional[str] = None) -> Optional[float]:
        """Get ETH balance for an address"""
        try:
            if not self.is_connected():
                return None
            
            target_address = address or self.connected_address
            if not target_address:
                return None
            
            target_address = Web3.to_checksum_address(target_address)
            balance_wei = self.w3.eth.get_balance(target_address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            
            return float(balance_eth)
            
        except Exception as e:
            print(f"[Wallet] ❌ ETH balance error: {e}")
            return None
    
    def get_wallet_info(self, address: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive wallet information"""
        target_address = address or self.connected_address
        
        if not target_address or not self.is_connected():
            return {
                'connected': False,
                'address': None,
                'eth_balance': None,
                'token_balance': None,
                'token_info': {}
            }
        
        return {
            'connected': True,
            'address': target_address,
            'eth_balance': self.get_eth_balance(target_address),
            'token_balance': self.get_token_balance(target_address),
            'token_info': self.token_info
        }
    
    def format_address(self, address: str) -> str:
        """Format address for display (shortened)"""
        if not address:
            return "Not connected"
        return f"{address[:6]}...{address[-4:]}"


# Singleton instance
_wallet_manager: Optional[WalletManager] = None

def get_wallet_manager() -> WalletManager:
    """Get or create wallet manager singleton"""
    global _wallet_manager
    if _wallet_manager is None:
        _wallet_manager = WalletManager()
    return _wallet_manager


# Token usage tracking for computational cost
class TokenUsageTracker:
    """Track token consumption based on computational complexity
    
    Usage persists across reboots and is saved to disk.
    No reset functionality to maintain accurate total usage tracking.
    """
    
    # Cost per operation type (in tokens)
    COSTS = {
        'simple_query': 0.001,      # Basic Q&A
        'rag_query': 0.005,         # RAG-enhanced query
        'complex_query': 0.010,     # Complex reasoning
        'transcription': 0.002,     # Per second of audio
        'tts_generation': 0.001,    # Per second of speech
    }
    
    def __init__(self):
        # File to store persistent usage data
        self.usage_file = os.path.expanduser("~/LedgerAI/data/token_usage.json")
        self.total_usage = 0.0
        self.total_paid = 0.0  # Track how much has been paid to client
        self.operation_history = []
        
        # Client wallet address
        self.client_wallet = "0x9F8081892c87DDAeD07D0bBD76CC2bd7fF6eE4c2"
        
        # Load saved usage data
        self._load_usage()
    
    def _load_usage(self):
        """Load usage data from file"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    self.total_usage = data.get('total_usage', 0.0)
                    self.total_paid = data.get('total_paid', 0.0)
                    self.operation_history = data.get('operation_history', [])
                    print(f"[TokenUsage] 💾 Loaded saved usage: {self.total_usage:.6f} tokens")
                    print(f"[TokenUsage] 💰 Total paid: {self.total_paid:.6f} tokens")
                    print(f"[TokenUsage] 📊 Balance owed: {self.get_balance_owed():.6f} tokens")
            else:
                print(f"[TokenUsage] 🆕 Starting fresh usage tracking")
        except Exception as e:
            print(f"[TokenUsage] ⚠️ Failed to load usage data: {e}")
            self.total_usage = 0.0
            self.total_paid = 0.0
            self.operation_history = []
    
    def _save_usage(self):
        """Save usage data to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
            
            # Save to file
            data = {
                'total_usage': self.total_usage,
                'total_paid': self.total_paid,
                'operation_history': self.operation_history[-100:],  # Keep last 100 operations
                'client_wallet': self.client_wallet
            }
            
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            print(f"[TokenUsage] ⚠️ Failed to save usage data: {e}")
    
    def record_usage(self, operation_type: str, multiplier: float = 1.0):
        """Record token usage for an operation"""
        cost = self.COSTS.get(operation_type, 0.001) * multiplier
        self.total_usage += cost
        
        self.operation_history.append({
            'type': operation_type,
            'cost': cost,
            'multiplier': multiplier
        })
        
        print(f"[TokenUsage] 💳 {operation_type}: {cost:.6f} tokens (total: {self.total_usage:.6f})")
        
        # Save to disk after each operation
        self._save_usage()
        
        return cost
    
    def get_session_usage(self) -> float:
        """Get total token usage (persists across reboots)"""
        return self.total_usage
    
    def record_payment(self, amount: float):
        """Record a payment made to client"""
        self.total_paid += amount
        print(f"[TokenUsage] 💸 Payment recorded: {amount:.6f} tokens (total paid: {self.total_paid:.6f})")
        self._save_usage()
    
    def get_balance_owed(self) -> float:
        """Get balance owed to client (usage - paid)"""
        return max(0, self.total_usage - self.total_paid)
    
    def get_total_paid(self) -> float:
        """Get total paid to client"""
        return self.total_paid


# Singleton instance
_usage_tracker: Optional[TokenUsageTracker] = None

def get_usage_tracker() -> TokenUsageTracker:
    """Get or create usage tracker singleton"""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = TokenUsageTracker()
    return _usage_tracker

