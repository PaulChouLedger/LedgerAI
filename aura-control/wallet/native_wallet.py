# native_wallet.py — Native Ethereum Wallet for Aura (Circular Display)

import os
import json
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from web3 import Web3
from eth_account import Account


class NativeWalletDialog(QDialog):
    """Native Ethereum wallet integrated into Aura - no browser needed"""
    
    def __init__(self, parent=None, amount=None, to_address=None, token_address=None):
        super().__init__(parent)
        self.amount = amount
        self.to_address = to_address
        self.token_address = token_address
        self.account = None
        
        self.setWindowTitle("Aura Wallet")
        self.setFixedSize(1080, 1080)
        
        if parent:
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 0.95);
                color: white;
                border: none;
                border-radius: 532px;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QLineEdit {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 10px;
                border: none;
                padding: 12px;
                font-size: 11pt;
            }
            QPushButton {
                background-color: rgba(142, 142, 147, 0.2);
                color: #ffffff;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(142, 142, 147, 0.4);
            }
            QPushButton:pressed {
                background-color: rgba(142, 142, 147, 0.6);
            }
        """)
        
        self.setup_ui()
        self.center_dialog()
    
    def setup_ui(self):
        """Setup native wallet UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(120, 90, 120, 90)
        layout.setSpacing(15)
        
        layout.addStretch(1)
        
        # Title
        title = QLabel("🔐 Aura Native Wallet")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        layout.addWidget(title)
        
        # Payment details
        if self.amount and self.to_address:
            amount_label = QLabel(f"Send {self.amount:.2f} tokens")
            amount_label.setFont(QFont("Arial", 20, QFont.Bold))
            amount_label.setAlignment(Qt.AlignCenter)
            amount_label.setStyleSheet("color: #FFD700; margin: 10px;")
            layout.addWidget(amount_label)
            
            to_label = QLabel(f"To: {self.to_address[:14]}...{self.to_address[-10:]}")
            to_label.setAlignment(Qt.AlignCenter)
            to_label.setStyleSheet("color: #34C759; font-size: 13px; font-family: 'Courier New'; font-weight: bold; margin: 5px;")
            layout.addWidget(to_label)
        
        # Private key input
        key_label = QLabel("Enter your private key to sign:")
        key_label.setAlignment(Qt.AlignCenter)
        key_label.setStyleSheet("color: #8e8e93; font-size: 12px; margin: 10px 0 5px 0;")
        layout.addWidget(key_label)
        
        self.private_key_input = QLineEdit()
        self.private_key_input.setPlaceholderText("0x...")
        self.private_key_input.setEchoMode(QLineEdit.Password)
        self.private_key_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 10px;
                padding: 12px;
                font-size: 10pt;
                font-family: 'Courier New';
            }
        """)
        layout.addWidget(self.private_key_input)
        
        # Security note
        note = QLabel("🔒 Your key stays on this device\nNever leaves Aura")
        note.setAlignment(Qt.AlignCenter)
        note.setWordWrap(True)
        note.setStyleSheet("color: #34C759; font-size: 11px; margin: 10px;")
        layout.addWidget(note)
        
        # Warning
        warning = QLabel("⚠️ REAL transaction • Cannot be reversed")
        warning.setAlignment(Qt.AlignCenter)
        warning.setStyleSheet("color: #FF9500; font-size: 12px; font-weight: bold; margin: 10px;")
        layout.addWidget(warning)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
            }
            QPushButton:hover { background-color: #D70015; }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        send_btn = QPushButton("💸 Send Payment")
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
            }
            QPushButton:hover { background-color: #30B350; }
        """)
        send_btn.clicked.connect(self.send_transaction)
        button_layout.addWidget(send_btn)
        
        layout.addLayout(button_layout)
        
        # Info
        info = QLabel("💡 Or use mobile wallet\nwith QR code (no key needed)")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #8e8e93; font-size: 10px; font-style: italic; margin: 5px;")
        layout.addWidget(info)
        
        layout.addStretch(1)
        self.setLayout(layout)
    
    def send_transaction(self):
        """Send transaction using native wallet"""
        private_key = self.private_key_input.text().strip()
        
        if not private_key:
            QMessageBox.warning(self, "Error", "Please enter your private key")
            return
        
        try:
            # Validate private key and create account
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            
            self.account = Account.from_key(private_key)
            
            # Show confirmation
            confirm = QMessageBox.question(
                self,
                "Confirm Transaction",
                f"Send {self.amount:.6f} tokens?\n\n"
                f"From: {self.account.address[:10]}...{self.account.address[-8:]}\n"
                f"To: {self.to_address[:10]}...{self.to_address[-8:]}\n\n"
                f"⚠️ This is a REAL transaction",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm != QMessageBox.Yes:
                return
            
            # Start transaction in background
            self.tx_worker = TransactionWorker(
                private_key=private_key,
                to_address=self.to_address,
                token_address=self.token_address,
                amount=self.amount
            )
            
            self.tx_worker.status_signal.connect(self.on_status)
            self.tx_worker.finished_signal.connect(self.on_finished)
            self.tx_worker.start()
            
            # Show progress
            self.show_progress_dialog()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid private key:\n{str(e)}")
    
    def show_progress_dialog(self):
        """Show transaction progress"""
        self.progress = QMessageBox(self)
        self.progress.setIcon(QMessageBox.Information)
        self.progress.setWindowTitle("Sending Transaction")
        self.progress.setText("⏳ Broadcasting transaction...")
        self.progress.setStandardButtons(QMessageBox.NoButton)
        self.progress.show()
    
    def on_status(self, message):
        """Update status during transaction"""
        if hasattr(self, 'progress'):
            self.progress.setText(message)
    
    def on_finished(self, success, message, tx_hash):
        """Handle transaction completion"""
        if hasattr(self, 'progress'):
            self.progress.close()
        
        if success:
            QMessageBox.information(
                self,
                "Success",
                f"✅ Transaction sent!\n\n"
                f"Hash: {tx_hash[:20]}...\n\n"
                f"View on Etherscan:\n"
                f"https://etherscan.io/tx/{tx_hash}"
            )
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Transaction failed:\n{message}")
    
    def center_dialog(self):
        """Center dialog on screen"""
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - 1080) // 2
        y = (screen.height() - 1080) // 2
        self.move(x, y)
        self.raise_()
        self.activateWindow()


class TransactionWorker(QThread):
    """Background worker for sending transaction"""
    
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, str)  # success, message, tx_hash
    
    def __init__(self, private_key, to_address, token_address, amount):
        super().__init__()
        self.private_key = private_key
        self.to_address = to_address
        self.token_address = token_address
        self.amount = amount
    
    def run(self):
        """Execute transaction"""
        try:
# Set up proper imports for organized structure
import os
import sys

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

            from wallet_integration import get_wallet_manager
            
            wallet_manager = get_wallet_manager()
            
            if not wallet_manager.is_connected():
                self.finished_signal.emit(False, "Not connected to Ethereum", "")
                return
            
            w3 = wallet_manager.w3
            account = Account.from_key(self.private_key)
            
            self.status_signal.emit("📝 Preparing transaction...")
            
            # Get token contract
            token_contract = wallet_manager.token_contract
            decimals = wallet_manager.token_info.get('decimals', 18)
            amount_wei = int(self.amount * (10 ** decimals))
            
            # Build transaction
            transfer_function = token_contract.functions.transfer(
                Web3.to_checksum_address(self.to_address),
                amount_wei
            )
            
            # Get nonce
            nonce = w3.eth.get_transaction_count(account.address)
            
            # Estimate gas
            gas_estimate = transfer_function.estimate_gas({'from': account.address})
            gas_price = w3.eth.gas_price
            
            self.status_signal.emit(f"⛽ Gas: {gas_estimate} @ {w3.from_wei(gas_price, 'gwei')} gwei")
            
            # Build transaction
            transaction = transfer_function.build_transaction({
                'from': account.address,
                'gas': gas_estimate,
                'gasPrice': gas_price,
                'nonce': nonce,
                'chainId': 1
            })
            
            self.status_signal.emit("✍️ Signing transaction...")
            
            # Sign
            signed_txn = w3.eth.account.sign_transaction(transaction, self.private_key)
            
            self.status_signal.emit("📤 Broadcasting to network...")
            
            # Send
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = w3.to_hex(tx_hash)
            
            self.status_signal.emit("⏳ Waiting for confirmation...")
            
            # Wait for receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                self.finished_signal.emit(True, "Transaction confirmed", tx_hash_hex)
            else:
                self.finished_signal.emit(False, "Transaction reverted", tx_hash_hex)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, str(e), "")

