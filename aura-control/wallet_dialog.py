# wallet_dialog.py — Wallet Connection Dialog for Aura

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from wallet_integration import get_wallet_manager, get_usage_tracker

class WalletDialog(QDialog):
    """Dialog for connecting wallet and viewing token balance"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wallet_manager = get_wallet_manager()
        self.usage_tracker = get_usage_tracker()
        
        self.setWindowTitle("Aura Token Wallet")
        self.setModal(False)  # Allow interaction with main window
        self.setMinimumSize(600, 500)
        
        # Apply dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QGroupBox {
                background-color: #2a2a2a;
                border: 2px solid #4D94D9;
                border-radius: 10px;
                margin-top: 10px;
                padding: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #4D94D9;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #ffffff;
                font-size: 12pt;
            }
            QLineEdit {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 2px solid #4D94D9;
                border-radius: 5px;
                padding: 8px;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #4D94D9;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5DA4E9;
            }
            QPushButton:pressed {
                background-color: #3D84C9;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
            QTextEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 2px solid #4D94D9;
                border-radius: 5px;
                padding: 8px;
                font-size: 10pt;
            }
        """)
        
        self.setup_ui()
        self.update_connection_status()
        
        # Auto-refresh timer for balance updates
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_balance)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
    
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # === Title ===
        title = QLabel("🔗 Ethereum Wallet Connection")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # === Connection Status ===
        status_group = QGroupBox("Network Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("⏳ Checking connection...")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # === Wallet Address Input ===
        wallet_group = QGroupBox("Wallet Address")
        wallet_layout = QVBoxLayout()
        
        instructions = QLabel("Enter your Ethereum wallet address:")
        instructions.setStyleSheet("color: #aaaaaa; font-size: 10pt;")
        wallet_layout.addWidget(instructions)
        
        # Address input with default placeholder
        self.address_input = QLineEdit()
        self.address_input.setPlaceholderText("0x1234567890abcdef1234567890abcdef12345678")
        self.address_input.returnPressed.connect(self.connect_wallet)
        wallet_layout.addWidget(self.address_input)
        
        # Connect button
        connect_btn = QPushButton("🔗 Connect Wallet")
        connect_btn.clicked.connect(self.connect_wallet)
        wallet_layout.addWidget(connect_btn)
        
        wallet_group.setLayout(wallet_layout)
        layout.addWidget(wallet_group)
        
        # === Balance Display ===
        balance_group = QGroupBox("Token Balance")
        balance_layout = QVBoxLayout()
        
        self.balance_label = QLabel("Not connected")
        self.balance_label.setAlignment(Qt.AlignCenter)
        balance_font = QFont()
        balance_font.setPointSize(18)
        balance_font.setBold(True)
        self.balance_label.setFont(balance_font)
        balance_layout.addWidget(self.balance_label)
        
        self.eth_balance_label = QLabel("ETH: --")
        self.eth_balance_label.setAlignment(Qt.AlignCenter)
        self.eth_balance_label.setStyleSheet("color: #aaaaaa; font-size: 11pt;")
        balance_layout.addWidget(self.eth_balance_label)
        
        self.address_display = QLabel("")
        self.address_display.setAlignment(Qt.AlignCenter)
        self.address_display.setStyleSheet("color: #4D94D9; font-size: 10pt;")
        balance_layout.addWidget(self.address_display)
        
        balance_group.setLayout(balance_layout)
        layout.addWidget(balance_group)
        
        # === Token Usage Stats ===
        usage_group = QGroupBox("Session Usage")
        usage_layout = QVBoxLayout()
        
        usage_info = QLabel("Tokens consumed this session based on computational complexity:")
        usage_info.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        usage_layout.addWidget(usage_info)
        
        self.usage_label = QLabel("💳 0.000000 tokens used")
        self.usage_label.setAlignment(Qt.AlignCenter)
        usage_font = QFont()
        usage_font.setPointSize(14)
        usage_font.setBold(True)
        self.usage_label.setFont(usage_font)
        self.usage_label.setStyleSheet("color: #FFD700;")
        usage_layout.addWidget(self.usage_label)
        
        # Reset button
        reset_btn = QPushButton("🔄 Reset Session Usage")
        reset_btn.clicked.connect(self.reset_usage)
        usage_layout.addWidget(reset_btn)
        
        usage_group.setLayout(usage_layout)
        layout.addWidget(usage_group)
        
        # === Action Buttons ===
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_balance)
        button_layout.addWidget(refresh_btn)
        
        close_btn = QPushButton("✖ Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # === Token Info Footer ===
        token_info = QLabel(f"Token: {self.wallet_manager.TOKEN_ADDRESS[:10]}...{self.wallet_manager.TOKEN_ADDRESS[-8:]}")
        token_info.setAlignment(Qt.AlignCenter)
        token_info.setStyleSheet("color: #666666; font-size: 9pt;")
        layout.addWidget(token_info)
        
        self.setLayout(layout)
    
    def update_connection_status(self):
        """Update network connection status"""
        if self.wallet_manager.is_connected():
            self.status_label.setText("✅ Connected to Ethereum Mainnet")
            self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        else:
            self.status_label.setText("❌ Not connected to Ethereum network")
            self.status_label.setStyleSheet("color: #ff0000; font-weight: bold;")
    
    def connect_wallet(self):
        """Connect to the entered wallet address"""
        address = self.address_input.text().strip()
        
        if not address:
            self.balance_label.setText("❌ Please enter an address")
            self.balance_label.setStyleSheet("color: #ff0000;")
            return
        
        # Show connecting status
        self.balance_label.setText("⏳ Connecting...")
        self.balance_label.setStyleSheet("color: #ffff00;")
        
        # Attempt connection
        if self.wallet_manager.connect_wallet(address):
            self.refresh_balance()
        else:
            self.balance_label.setText("❌ Connection failed")
            self.balance_label.setStyleSheet("color: #ff0000;")
            self.address_display.setText("Invalid address or network error")
    
    def refresh_balance(self):
        """Refresh wallet balance display"""
        if not self.wallet_manager.connected_address:
            return
        
        wallet_info = self.wallet_manager.get_wallet_info()
        
        if wallet_info['connected']:
            # Display token balance
            token_balance = wallet_info.get('token_balance')
            if token_balance is not None:
                token_symbol = wallet_info['token_info'].get('symbol', 'tokens')
                self.balance_label.setText(f"💰 {token_balance:.6f} {token_symbol}")
                self.balance_label.setStyleSheet("color: #00ff00;")
            else:
                self.balance_label.setText("❌ Error fetching balance")
                self.balance_label.setStyleSheet("color: #ff0000;")
            
            # Display ETH balance
            eth_balance = wallet_info.get('eth_balance')
            if eth_balance is not None:
                self.eth_balance_label.setText(f"Ξ {eth_balance:.6f} ETH")
            
            # Display address
            short_addr = self.wallet_manager.format_address(wallet_info['address'])
            self.address_display.setText(f"Address: {short_addr}")
        
        # Update usage stats
        session_usage = self.usage_tracker.get_session_usage()
        self.usage_label.setText(f"💳 {session_usage:.6f} tokens used")
        
        # Update connection status
        self.update_connection_status()
    
    def reset_usage(self):
        """Reset session usage counter"""
        self.usage_tracker.reset_session()
        self.usage_label.setText("💳 0.000000 tokens used")
        print("[WalletDialog] ✅ Session usage reset")
    
    def closeEvent(self, event):
        """Handle dialog close"""
        self.refresh_timer.stop()
        event.accept()

