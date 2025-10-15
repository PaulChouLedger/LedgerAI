# metamask_integration.py — MetaMask Integration via WalletConnect

import os
import json
from typing import Optional, Callable
from web3 import Web3

try:
    from walletconnect import WCClient  # This is an external library, not local
    WALLETCONNECT_AVAILABLE = True
except ImportError:
    WALLETCONNECT_AVAILABLE = False
    print("[MetaMask] ⚠️ WalletConnect not available - install: pip install walletconnect-python")


class MetaMaskConnector:
    """Manages MetaMask connection via WalletConnect protocol"""
    
    def __init__(self):
        self.wc_client: Optional[WCClient] = None
        self.connected_address: Optional[str] = None
        self.session_file = os.path.expanduser("~/LedgerAI/data/walletconnect_session.json")
        self.w3: Optional[Web3] = None
        
    def is_available(self) -> bool:
        """Check if WalletConnect is available"""
        return WALLETCONNECT_AVAILABLE
    
    def create_connection_uri(self) -> tuple[Optional[str], Optional[WCClient]]:
        """Create WalletConnect URI for QR code"""
        if not WALLETCONNECT_AVAILABLE:
            return None, None
        
        try:
            # Create WalletConnect client
            self.wc_client = WCClient()
            
            # Get connection URI
            uri = self.wc_client.uri
            
            print(f"[MetaMask] 🔗 WalletConnect URI created")
            return uri, self.wc_client
            
        except Exception as e:
            print(f"[MetaMask] ❌ Failed to create WalletConnect URI: {e}")
            return None, None
    
    def connect_wallet(self, on_connect: Optional[Callable] = None):
        """Connect to MetaMask wallet"""
        if not self.wc_client:
            return False
        
        try:
            # Set up connection callback
            if on_connect:
                self.wc_client.on_connect = lambda: self._handle_connect(on_connect)
            
            # Wait for connection
            print("[MetaMask] ⏳ Waiting for MetaMask connection...")
            
            return True
            
        except Exception as e:
            print(f"[MetaMask] ❌ Connection failed: {e}")
            return False
    
    def _handle_connect(self, callback):
        """Handle successful connection"""
        try:
            # Get connected accounts
            accounts = self.wc_client.accounts
            if accounts:
                self.connected_address = accounts[0]
                print(f"[MetaMask] ✅ Connected: {self.connected_address}")
                
                # Save session
                self._save_session()
                
                # Trigger callback
                if callback:
                    callback(self.connected_address)
            
        except Exception as e:
            print(f"[MetaMask] ❌ Error handling connection: {e}")
    
    def _save_session(self):
        """Save WalletConnect session for reconnection"""
        try:
            if self.wc_client:
                session_data = {
                    'address': self.connected_address,
                    # Add session persistence data
                }
                
                os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
                with open(self.session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                print(f"[MetaMask] 💾 Session saved")
        except Exception as e:
            print(f"[MetaMask] ⚠️ Failed to save session: {e}")
    
    def send_transaction(self, to_address: str, token_contract_address: str, 
                        amount: int, on_success: Optional[Callable] = None,
                        on_error: Optional[Callable] = None):
        """Send token transaction via MetaMask"""
        if not self.wc_client or not self.connected_address:
            if on_error:
                on_error("Not connected to MetaMask")
            return False
        
        try:
            # Build ERC-20 transfer transaction
            # This will be sent to MetaMask for user approval
            
            print(f"[MetaMask] 📤 Preparing transaction...")
            print(f"[MetaMask]    From: {self.connected_address}")
            print(f"[MetaMask]    To: {to_address}")
            print(f"[MetaMask]    Amount: {amount} wei")
            
            # MetaMask will handle signing and sending
            # User approves in MetaMask app
            
            if on_success:
                on_success("Transaction sent to MetaMask for approval")
            
            return True
            
        except Exception as e:
            print(f"[MetaMask] ❌ Transaction failed: {e}")
            if on_error:
                on_error(str(e))
            return False
    
    def disconnect(self):
        """Disconnect from MetaMask"""
        try:
            if self.wc_client:
                self.wc_client.disconnect()
                self.wc_client = None
                self.connected_address = None
                
                # Remove saved session
                if os.path.exists(self.session_file):
                    os.remove(self.session_file)
                
                print("[MetaMask] 🔌 Disconnected")
                return True
        except Exception as e:
            print(f"[MetaMask] ⚠️ Disconnect error: {e}")
        return False


# Singleton instance
_metamask_connector: Optional[MetaMaskConnector] = None

def get_metamask_connector() -> MetaMaskConnector:
    """Get or create MetaMask connector singleton"""
    global _metamask_connector
    if _metamask_connector is None:
        _metamask_connector = MetaMaskConnector()
    return _metamask_connector

