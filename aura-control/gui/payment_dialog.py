# payment_dialog.py — Payment Dialog for Sending Tokens to Client

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
# Set up proper imports for organized structure
import os
import sys

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from wallet.wallet_integration import get_wallet_manager, get_usage_tracker

class PaymentDialog(QDialog):
    """Dialog for sending real DEX-traded tokens back to client wallet
    
    ⚠️ WARNING: This sends REAL tokens with actual market value!
    Transactions are irreversible once confirmed on the blockchain.
    """
    
    # Client wallet address (receives real token payments)
    CLIENT_WALLET = "0xd3c4d619C8515Bc764921209821Ec7A77FC31Ba4"
    
    def __init__(self, parent=None, user_address=None):
        super().__init__(parent)
        self.wallet_manager = get_wallet_manager()
        self.usage_tracker = get_usage_tracker()
        self.user_address = user_address
        
        self.setWindowTitle("Send Payment to Client")
        self.setFixedSize(1080, 1080)
        
        # Modal behavior
        if parent:
            # Use Window flag instead of Dialog to ensure proper z-ordering above parent
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # (No translucent background to preserve readability)
        
        # Circular dark theme - match other dialogs
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
            QTextEdit {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 10px;
                border: none;
                padding: 10px;
                font-size: 10pt;
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
            QPushButton:disabled {
                background-color: rgba(142, 142, 147, 0.3);
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        
        # Initialize opacity to 0 for fade-in animation
        self.setWindowOpacity(0.0)
        
        self.setup_ui()
        self.center_dialog()
    
    def showEvent(self, event):
        """Handle dialog show event with smooth fade-in animation"""
        super().showEvent(event)
        
        # Create smooth fade-in animation
        self.fade_in = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in.setDuration(300)  # Slightly longer for smoother feel
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setEasingCurve(QEasingCurve.InOutCubic)  # Smooth ease-in/out
        self.fade_in.start()
        
        # Ensure dialog is raised and focused
        self.raise_()
        self.activateWindow()
    
    def setup_ui(self):
        """Setup payment dialog UI"""
        layout = QVBoxLayout()
        # Keep all content within white circular border (radius 535px)
        layout.setContentsMargins(130, 110, 130, 110)
        layout.setSpacing(12)  # Tighter spacing
        
        # Vertical centering
        layout.addStretch(1)
        
        # Title (match other dialogs)
        title = QLabel("💸 Send Payment to Client")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 8px;")
        layout.addWidget(title)
        
        # Client address display (bigger for readability)
        client_label = QLabel(f"To: {self.CLIENT_WALLET[:14]}...{self.CLIENT_WALLET[-10:]}")
        client_label.setAlignment(Qt.AlignCenter)
        client_label.setStyleSheet("color: #34C759; font-size: 14px; font-family: 'Courier New'; font-weight: bold; margin: 8px;")
        layout.addWidget(client_label)
        
        # From address (if connected) - bigger for readability
        if self.user_address:
            from_label = QLabel(f"From: {self.user_address[:14]}...{self.user_address[-10:]}")
            from_label.setAlignment(Qt.AlignCenter)
            from_label.setStyleSheet("color: #4D94D9; font-size: 14px; font-family: 'Courier New'; font-weight: bold; margin: 8px;")
            layout.addWidget(from_label)
        
        # Amount input section
        amount_label = QLabel("Amount (tokens):")
        amount_label.setAlignment(Qt.AlignCenter)
        amount_label.setStyleSheet("color: #8e8e93; font-size: 11px; margin: 8px 0 3px 0;")
        layout.addWidget(amount_label)
        
        self.amount_input = QLineEdit()
        self.amount_input.setPlaceholderText("0.000000")
        self.amount_input.setAlignment(Qt.AlignCenter)
        self.amount_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 10px;
                border: none;
                padding: 12px;
                font-size: 14pt;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.amount_input)
        
        # Quick amount buttons based on usage owed
        quick_layout = QHBoxLayout()
        quick_layout.setSpacing(8)
        
        # Get current balance owed
        balance_owed = self.usage_tracker.get_balance_owed()
        
        # Create smart quick payment buttons
        if balance_owed > 0:
            # Show percentage buttons
            quick_amounts = [
                ("25%", balance_owed * 0.25),
                ("50%", balance_owed * 0.50),
                ("75%", balance_owed * 0.75),
                ("100%", balance_owed * 1.00)
            ]
        else:
            # No balance owed, show fixed amounts
            quick_amounts = [
                ("0.1", 0.1),
                ("0.5", 0.5),
                ("1.0", 1.0),
                ("All", 0.0)  # Disabled
            ]
        
        for label, amount in quick_amounts:
            btn = QPushButton(label)
            
            # Highlight "100%" button
            if label == "100%":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #34C759;
                        color: white;
                        padding: 8px;
                        font-size: 11pt;
                        font-weight: bold;
                        min-height: 35px;
                        max-height: 35px;
                        border-radius: 12px;
                    }
                    QPushButton:hover {
                        background-color: #30B350;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(142, 142, 147, 0.3);
                        padding: 8px;
                        font-size: 11pt;
                        min-height: 35px;
                        max-height: 35px;
                        border-radius: 12px;
                    }
                    QPushButton:hover {
                        background-color: rgba(142, 142, 147, 0.5);
                    }
                """)
            
            if amount > 0:
                btn.clicked.connect(lambda checked, a=amount: self.set_amount(a))
            else:
                btn.setEnabled(False)
            
            quick_layout.addWidget(btn)
        
        layout.addLayout(quick_layout)
        
        # Warning message - REAL TOKENS (more compact)
        warning1 = QLabel("⚠️ WARNING: REAL tokens • IRREVERSIBLE")
        warning1.setAlignment(Qt.AlignCenter)
        warning1.setWordWrap(True)
        warning1.setStyleSheet("color: #FF3B30; font-size: 11px; font-weight: bold; margin: 8px;")
        layout.addWidget(warning1)
        
        # Private key input
        key_label = QLabel("Private Key:")
        key_label.setAlignment(Qt.AlignCenter)
        key_label.setStyleSheet("color: #8e8e93; font-size: 11px; margin: 8px 0 3px 0;")
        layout.addWidget(key_label)
        
        self.private_key_input = QLineEdit()
        self.private_key_input.setPlaceholderText("0x... (keep secure)")
        self.private_key_input.setEchoMode(QLineEdit.Password)
        self.private_key_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 10px;
                border: none;
                padding: 10px;
                font-size: 10pt;
                font-family: 'Courier New', monospace;
            }
        """)
        layout.addWidget(self.private_key_input)
        
        # Transaction log
        log_label = QLabel("Transaction Log:")
        log_label.setAlignment(Qt.AlignCenter)
        log_label.setStyleSheet("color: #8e8e93; font-size: 11px; margin: 8px 0 3px 0;")
        layout.addWidget(log_label)
        
        self.tx_log = QTextEdit()
        self.tx_log.setReadOnly(True)
        self.tx_log.setMaximumHeight(100)
        self.tx_log.setMinimumHeight(100)
        self.tx_log.setPlaceholderText("Transaction status will appear here...")
        self.tx_log.setStyleSheet("""
            QTextEdit {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 15px;
                border: none;
                padding: 10px;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.tx_log)
        
        # Action buttons (match other dialogs)
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
                border: none;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B30000;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.send_btn = QPushButton("💸 Send Payment")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #30B350;
            }
            QPushButton:pressed {
                background-color: #2A9D47;
            }
            QPushButton:disabled {
                background-color: rgba(142, 142, 147, 0.3);
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        self.send_btn.clicked.connect(self.send_payment)
        button_layout.addWidget(self.send_btn)
        
        layout.addLayout(button_layout)
        
        # Note about MetaMask (smaller)
        note = QLabel("💡 MetaMask recommended for production")
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet("color: #8e8e93; font-size: 10px; font-style: italic; margin: 5px;")
        layout.addWidget(note)
        
        # Vertical centering
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def set_amount(self, amount: float):
        """Set amount from quick button"""
        self.amount_input.setText(f"{amount:.6f}")
    
    def log_message(self, message: str):
        """Add message to transaction log"""
        self.tx_log.append(message)
    
    def send_payment(self):
        """Send payment transaction"""
        # Get amount
        amount_text = self.amount_input.text().strip()
        if not amount_text:
            QMessageBox.warning(self, "Error", "Please enter an amount")
            return
        
        try:
            amount = float(amount_text)
            if amount <= 0:
                QMessageBox.warning(self, "Error", "Amount must be greater than 0")
                return
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid amount format")
            return
        
        # Get private key
        private_key = self.private_key_input.text().strip()
        if not private_key:
            QMessageBox.warning(self, "Error", 
                              "Private key required to sign transaction.\n\n"
                              "For security, consider using MetaMask integration instead.")
            return
        
        # Confirm transaction with strong warning
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("⚠️ Confirm Real Token Payment")
        msg_box.setText("Send REAL tokens with actual market value?")
        msg_box.setInformativeText(
            f"⚠️ WARNING: This is a REAL transaction!\n\n"
            f"To: {self.CLIENT_WALLET}\n"
            f"Amount: {amount} tokens\n\n"
            f"• Tokens have real market value\n"
            f"• Transaction is IRREVERSIBLE\n"
            f"• Gas fees will apply (paid in ETH)\n"
            f"• Transaction goes to Ethereum Mainnet\n\n"
            f"Are you absolutely sure?"
        )
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
        confirm = msg_box.exec_()
        
        if confirm != QMessageBox.Yes:
            return
        
        # Disable button during transaction
        self.send_btn.setEnabled(False)
        self.log_message(f"🔄 Initiating payment of {amount} tokens...")
        
        # Start transaction in background thread
        self.tx_worker = TransactionWorker(
            wallet_manager=self.wallet_manager,
            from_address=self.user_address,
            to_address=self.CLIENT_WALLET,
            amount=amount,
            private_key=private_key
        )
        
        self.tx_worker.log_signal.connect(self.log_message)
        self.tx_worker.finished_signal.connect(self.on_transaction_finished)
        self.tx_worker.start()
    
    def on_transaction_finished(self, success: bool, message: str, tx_hash: str = None):
        """Handle transaction completion"""
        self.send_btn.setEnabled(True)
        
        if success:
            self.log_message(f"✅ {message}")
            if tx_hash:
                self.log_message(f"📝 Transaction: {tx_hash}")
                self.log_message(f"🔗 View: https://etherscan.io/tx/{tx_hash}")
            
            # Record payment in usage tracker
            amount = float(self.amount_input.text().strip())
            self.usage_tracker.record_payment(amount)
            self.log_message(f"💾 Payment recorded in usage tracker")
            
            QMessageBox.information(
                self,
                "Success",
                f"Payment sent successfully!\n\n"
                f"Transaction: {tx_hash[:20]}...\n\n"
                f"View on Etherscan:\n{tx_hash}\n\n"
                f"Payment recorded: {amount:.6f} tokens"
            )
            
            # Close dialog after successful payment
            self.accept()
        else:
            self.log_message(f"❌ {message}")
            QMessageBox.critical(self, "Error", f"Transaction failed:\n\n{message}")
    
    def closeEvent(self, event):
        """Handle dialog close event with smooth fade-out animation"""
        # Reactivate parent window immediately to prevent freezing
        if self.parent():
            try:
                self.parent().raise_()
                self.parent().activateWindow()
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass
        
        # If already animating or not visible, accept immediately
        if hasattr(self, 'fade_out') and self.fade_out.state() == QPropertyAnimation.Running:
            event.accept()
            return
        
        # Only animate if we're actually closing (not just hiding)
        if event.spontaneous() or not self.isVisible():
            event.accept()
            return
        
        # For modal dialogs opened from home screen, accept immediately to avoid blocking
        if self.isModal() and self.parent():
            event.accept()
            return
        
        # Cancel fade-in if still running
        if hasattr(self, 'fade_in') and self.fade_in.state() == QPropertyAnimation.Running:
            self.fade_in.stop()
        
        # Non-modal: use fade animation
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(250)  # Slightly longer for smoother feel
        self.fade_out.setStartValue(self.windowOpacity())
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.InOutCubic)  # Symmetric easing
        
        # Connect finished signal to actually close the dialog
        def _finalize():
            event.accept()
            # Ensure parent is reactivated after close
            if self.parent():
                try:
                    self.parent().raise_()
                    self.parent().activateWindow()
                    from PyQt5.QtWidgets import QApplication
                    QApplication.processEvents()
                except Exception:
                    pass
        
        self.fade_out.finished.connect(_finalize)
        self.fade_out.start()
        
        # Prevent immediate close
        event.ignore()
        print("[PaymentDialog] 🔄 Closing dialog with fade-out animation...")
    
    def center_dialog(self):
        """Center the dialog on screen"""
        from PyQt5.QtWidgets import QDesktopWidget
        
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - 1080) // 2
        y = (screen.height() - 1080) // 2
        
        self.move(x, y)
        self.raise_()
        self.activateWindow()


class TransactionWorker(QThread):
    """Worker thread for sending blockchain transactions"""
    
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, str)  # success, message, tx_hash
    
    def __init__(self, wallet_manager, from_address, to_address, amount, private_key):
        super().__init__()
        self.wallet_manager = wallet_manager
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
        self.private_key = private_key
    
    def run(self):
        """Execute transaction"""
        try:
            from web3 import Web3
            from eth_account import Account
            
            # Ensure Web3 is connected
            if not self.wallet_manager.is_connected():
                self.finished_signal.emit(False, "Not connected to Ethereum network", "")
                return
            
            w3 = self.wallet_manager.w3
            
            # Validate private key
            try:
                if not self.private_key.startswith('0x'):
                    self.private_key = '0x' + self.private_key
                account = Account.from_key(self.private_key)
                
                # Verify the from_address matches the private key
                if account.address.lower() != self.from_address.lower():
                    self.finished_signal.emit(False, "Private key doesn't match wallet address", "")
                    return
                
            except Exception as e:
                self.finished_signal.emit(False, f"Invalid private key: {str(e)}", "")
                return
            
            self.log_signal.emit(f"📝 Preparing transaction...")
            
            # Get token contract
            token_contract = self.wallet_manager.token_contract
            if not token_contract:
                self.finished_signal.emit(False, "Token contract not initialized", "")
                return
            
            # Convert amount to smallest unit (considering decimals)
            decimals = self.wallet_manager.token_info.get('decimals', 18)
            amount_wei = int(self.amount * (10 ** decimals))
            
            self.log_signal.emit(f"💰 Amount: {self.amount} tokens ({amount_wei} wei)")
            
            # Check balance
            balance = self.wallet_manager.get_token_balance(self.from_address)
            if balance is None or balance < self.amount:
                self.finished_signal.emit(False, f"Insufficient balance (have: {balance}, need: {self.amount})", "")
                return
            
            # Get nonce
            nonce = w3.eth.get_transaction_count(account.address)
            self.log_signal.emit(f"🔢 Nonce: {nonce}")
            
            # Build transaction
            transfer_function = token_contract.functions.transfer(
                Web3.to_checksum_address(self.to_address),
                amount_wei
            )
            
            # Estimate gas
            try:
                gas_estimate = transfer_function.estimate_gas({'from': account.address})
                self.log_signal.emit(f"⛽ Estimated gas: {gas_estimate}")
            except Exception as e:
                self.log_signal.emit(f"⚠️ Gas estimation failed: {e}")
                gas_estimate = 100000  # Fallback
            
            # Get gas price
            gas_price = w3.eth.gas_price
            self.log_signal.emit(f"💵 Gas price: {w3.from_wei(gas_price, 'gwei')} gwei")
            
            # Build transaction dict
            transaction = transfer_function.build_transaction({
                'from': account.address,
                'gas': gas_estimate,
                'gasPrice': gas_price,
                'nonce': nonce,
                'chainId': 1  # Mainnet
            })
            
            self.log_signal.emit(f"✍️ Signing transaction...")
            
            # Sign transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, self.private_key)
            
            self.log_signal.emit(f"📤 Broadcasting transaction...")
            
            # Send transaction
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = w3.to_hex(tx_hash)
            
            self.log_signal.emit(f"⏳ Waiting for confirmation...")
            
            # Wait for receipt (with timeout)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                self.finished_signal.emit(True, "Payment successful", tx_hash_hex)
            else:
                self.finished_signal.emit(False, "Transaction reverted", tx_hash_hex)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, str(e), "")

