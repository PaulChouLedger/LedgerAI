# metamask_payment_dialog.py — MetaMask Payment Dialog

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPixmap
# Set up proper imports for organized structure
import os
import sys

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from wallet.wallet_integration import get_wallet_manager, get_usage_tracker
import qrcode
import io
from PIL import Image

class MetaMaskPaymentDialog(QDialog):
    """Dialog for sending tokens via MetaMask (no private key needed!)"""
    
    CLIENT_WALLET = "0xd3c4d619C8515Bc764921209821Ec7A77FC31Ba4"
    
    def __init__(self, parent=None, user_address=None):
        super().__init__(parent)
        self.wallet_manager = get_wallet_manager()
        self.usage_tracker = get_usage_tracker()
        self.user_address = user_address
        
        self.setWindowTitle("MetaMask Payment")
        self.setFixedSize(1080, 1080)
        
        if parent:
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # (No translucent background to preserve readability)
        
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
        """Setup UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(120, 100, 120, 100)
        layout.setSpacing(15)
        
        layout.addStretch(1)
        
        # Title
        title = QLabel("💸 Pay with MetaMask")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel("Secure payment - no private key needed!")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: #34C759; font-size: 12px; margin: 5px;")
        layout.addWidget(instructions)
        
        # To/From addresses
        client_label = QLabel(f"To: {self.CLIENT_WALLET[:14]}...{self.CLIENT_WALLET[-10:]}")
        client_label.setAlignment(Qt.AlignCenter)
        client_label.setStyleSheet("color: #34C759; font-size: 14px; font-family: 'Courier New'; font-weight: bold; margin: 5px;")
        layout.addWidget(client_label)
        
        if self.user_address:
            from_label = QLabel(f"From: {self.user_address[:14]}...{self.user_address[-10:]}")
            from_label.setAlignment(Qt.AlignCenter)
            from_label.setStyleSheet("color: #4D94D9; font-size: 14px; font-family: 'Courier New'; font-weight: bold; margin: 5px;")
            layout.addWidget(from_label)
        
        # Balance owed display
        balance_owed = self.usage_tracker.get_balance_owed()
        owed_label = QLabel(f"Balance Owed: {balance_owed:.6f} tokens")
        owed_label.setAlignment(Qt.AlignCenter)
        owed_label.setStyleSheet("color: #FF9500; font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(owed_label)
        
        # Amount input
        amount_label = QLabel("Amount to send:")
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
        
        # Quick amount buttons - percentages and full
        quick_layout = QHBoxLayout()
        quick_layout.setSpacing(8)
        
        if balance_owed > 0:
            quick_amounts = [
                ("25%", balance_owed * 0.25),
                ("50%", balance_owed * 0.50),
                ("75%", balance_owed * 0.75),
                ("100%", balance_owed * 1.00)
            ]
        else:
            quick_amounts = [("25%", 0), ("50%", 0), ("75%", 0), ("100%", 0)]
        
        for label, amount in quick_amounts:
            btn = QPushButton(label)
            
            if label == "100%" and balance_owed > 0:
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
        
        # Info about wallet payment
        info = QLabel("🔐 Complete transaction in your wallet app\n(MetaMask, Base, Coinbase, etc.)")
        info.setAlignment(Qt.AlignCenter)
        info.setWordWrap(True)
        info.setStyleSheet("color: #8e8e93; font-size: 11px; margin: 10px; line-height: 1.4;")
        layout.addWidget(info)
        
        # Action buttons
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
        
        send_btn = QPushButton("📱 Show Payment Info")
        send_btn.setStyleSheet("""
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
        """)
        send_btn.clicked.connect(self.send_with_wallet)
        button_layout.addWidget(send_btn)
        
        layout.addLayout(button_layout)
        
        # Note
        note = QLabel("🦊 Requires MetaMask mobile app or browser extension")
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet("color: #8e8e93; font-size: 10px; font-style: italic; margin: 5px;")
        layout.addWidget(note)
        
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def set_amount(self, amount: float):
        """Set amount from quick button"""
        self.amount_input.setText(f"{amount:.6f}")
    
    def send_with_wallet(self):
        """Show payment info for wallet app"""
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
        
        # Show payment details screen
        self.show_payment_details(amount)
    
    
    def build_metamask_deeplink(self, amount: float, token_address: str) -> str:
        """Build MetaMask deep link for token transfer"""
        from web3 import Web3
        
        # Convert amount to wei
        decimals = self.wallet_manager.token_info.get('decimals', 18)
        amount_wei = int(amount * (10 ** decimals))
        
        print(f"[MetaMask] 🔗 Building deep link:")
        print(f"[MetaMask]    Token: {token_address}")
        print(f"[MetaMask]    To: {self.CLIENT_WALLET}")
        print(f"[MetaMask]    Amount: {amount} tokens ({amount_wei} wei)")
        
        # Build ERC-20 transfer calldata
        # Function signature: transfer(address,uint256)
        function_sig = "0xa9059cbb"
        
        # Encode recipient (32 bytes, left-padded)
        recipient_padded = self.CLIENT_WALLET[2:].lower().zfill(64)
        
        # Encode amount (32 bytes, left-padded)
        amount_hex = hex(amount_wei)[2:].zfill(64)
        
        # Complete calldata
        data = function_sig + recipient_padded + amount_hex
        
        print(f"[MetaMask]    Data: {data}")
        
        # Use standard ethereum: URI scheme (most compatible)
        # This should work with MetaMask mobile without chain ID issues
        ethereum_uri = f"ethereum:{token_address}?data={data}"
        
        print(f"[MetaMask] 🔗 Ethereum URI: {ethereum_uri}")
        
        return ethereum_uri
    
    def show_payment_details(self, amount: float):
        """Show payment details screen for manual wallet entry"""
        # Create full-screen payment info dialog
        details_dialog = QDialog(self)
        details_dialog.setWindowTitle("Payment Information")
        details_dialog.setFixedSize(1080, 1080)
        details_dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        details_dialog.setModal(True)
        details_dialog.setStyleSheet("QDialog { background-color: rgba(28, 28, 30, 0.95); border-radius: 532px; }")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(100, 80, 100, 80)
        layout.setSpacing(20)
        layout.addStretch(1)
        
        # Title
        title = QLabel("📱 Send Payment")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; margin: 10px;")
        layout.addWidget(title)
        
        # Amount (HUGE)
        amount_label = QLabel(f"{amount:.2f}")
        amount_label.setFont(QFont("Arial", 48, QFont.Bold))
        amount_label.setAlignment(Qt.AlignCenter)
        amount_label.setStyleSheet("color: #FFD700; margin: 20px;")
        layout.addWidget(amount_label)
        
        token_symbol = self.wallet_manager.token_info.get('symbol', 'tokens')
        symbol_label = QLabel(token_symbol)
        symbol_label.setFont(QFont("Arial", 24, QFont.Bold))
        symbol_label.setAlignment(Qt.AlignCenter)
        symbol_label.setStyleSheet("color: #FFD700; margin-bottom: 20px;")
        layout.addWidget(symbol_label)
        
        # Client address with QR code
        addr_title = QLabel("Send to:")
        addr_title.setAlignment(Qt.AlignCenter)
        addr_title.setStyleSheet("color: #8e8e93; font-size: 14px; margin: 10px;")
        layout.addWidget(addr_title)
        
        # QR Code with full payment info (EIP-681 format)
        try:
            # Build EIP-681 URI with token transfer info
            token_address = self.wallet_manager.TOKEN_ADDRESS
            decimals = self.wallet_manager.token_info.get('decimals', 18)
            amount_wei = int(amount * (10 ** decimals))
            
            # EIP-681 format for ERC-20 transfer:
            # ethereum:{token_contract}/transfer?address={recipient}&uint256={amount_in_wei}
            payment_uri = f"ethereum:{token_address}/transfer?address={self.CLIENT_WALLET}&uint256={amount_wei}"
            
            print(f"[Payment] 🔗 QR code contains: {payment_uri}")
            
            qr = qrcode.QRCode(version=1, box_size=6, border=3)
            qr.add_data(payment_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            qr_label = QLabel()
            qr_label.setAlignment(Qt.AlignCenter)
            qr_label.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            qr_label.setStyleSheet("background-color: white; border-radius: 15px; padding: 15px; margin: 10px;")
            layout.addWidget(qr_label)
        except:
            pass
        
        # Client address (readable)
        addr_label = QLabel(f"{self.CLIENT_WALLET[:22]}\n{self.CLIENT_WALLET[22:]}")
        addr_label.setAlignment(Qt.AlignCenter)
        addr_label.setStyleSheet("color: #34C759; font-size: 13px; font-family: 'Courier New'; font-weight: bold; margin: 10px;")
        layout.addWidget(addr_label)
        
        # Instructions
        instructions = QLabel(
            "📱 In your wallet app\n"
            "(MetaMask, Base, Coinbase, etc.):\n\n"
            "1. Select your token\n"
            "2. Tap 'Send'\n"
            "3. Scan QR or enter address\n"
            f"4. Amount: {amount:.6f}\n"
            "5. Approve transaction\n"
            "6. Click 'Done' below"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #ffffff; font-size: 13px; line-height: 1.8; margin: 15px;")
        layout.addWidget(instructions)
        
        # Done button
        button_layout = QHBoxLayout()
        
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
        cancel_btn.clicked.connect(details_dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        done_btn = QPushButton("✅ Mark as Paid")
        done_btn.setStyleSheet("""
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
        done_btn.clicked.connect(lambda: self.confirm_and_close(details_dialog, amount))
        button_layout.addWidget(done_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch(1)
        
        details_dialog.setLayout(layout)
        
        # Center dialog
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - 1080) // 2
        y = (screen.height() - 1080) // 2
        details_dialog.move(x, y)
        
        details_dialog.exec_()
    
    def confirm_and_close(self, dialog, amount):
        """Confirm payment and record it"""
        confirm = QMessageBox.question(
            dialog,
            "Confirm Payment",
            f"Have you completed the payment of {amount:.6f} tokens?\n\n"
            f"⚠️ Only click Yes if transaction is confirmed in your wallet.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Record payment
            self.usage_tracker.record_payment(amount)
            dialog.accept()
            
            QMessageBox.information(
                dialog,
                "Payment Recorded",
                f"✅ Payment of {amount:.6f} tokens recorded!\n\n"
                f"Thank you for your payment!"
            )
            
            self.accept()
    
    def show_qr_code(self, url: str, amount: float):
        """Show QR code for wallet mobile (deprecated - using payment details now)"""
        try:
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=4)
            qr.add_data(url)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to QPixmap
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            # Create QR dialog
            qr_dialog = QDialog(self)
            qr_dialog.setWindowTitle("Scan with MetaMask")
            qr_dialog.setFixedSize(1080, 1080)
            qr_dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            qr_dialog.setModal(True)
            qr_dialog.setStyleSheet("QDialog { background-color: rgba(28, 28, 30, 0.95); border-radius: 532px; }")
            
            qr_layout = QVBoxLayout()
            qr_layout.setContentsMargins(120, 100, 120, 100)
            qr_layout.addStretch(1)
            
            qr_title = QLabel("📱 Scan with MetaMask Mobile")
            qr_title.setFont(QFont("Arial", 16, QFont.Bold))
            qr_title.setAlignment(Qt.AlignCenter)
            qr_title.setStyleSheet("color: #ffffff; margin: 10px;")
            qr_layout.addWidget(qr_title)
            
            qr_amount = QLabel(f"Amount: {amount:.6f} tokens")
            qr_amount.setAlignment(Qt.AlignCenter)
            qr_amount.setStyleSheet("color: #FFD700; font-size: 14px; font-weight: bold; margin: 5px;")
            qr_layout.addWidget(qr_amount)
            
            qr_label = QLabel()
            qr_label.setAlignment(Qt.AlignCenter)
            qr_label.setPixmap(pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            qr_label.setStyleSheet("background-color: white; border-radius: 20px; padding: 15px; margin: 10px;")
            qr_layout.addWidget(qr_label)
            
            qr_instructions = QLabel("1. Open MetaMask app\n2. Tap scan QR\n3. Approve transaction")
            qr_instructions.setAlignment(Qt.AlignCenter)
            qr_instructions.setStyleSheet("color: #8e8e93; font-size: 12px; margin: 10px;")
            qr_layout.addWidget(qr_instructions)
            
            close_btn = QPushButton("✖ Close")
            close_btn.setStyleSheet("""
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
            close_btn.clicked.connect(qr_dialog.accept)
            qr_layout.addWidget(close_btn, alignment=Qt.AlignCenter)
            
            qr_layout.addStretch(1)
            qr_dialog.setLayout(qr_layout)
            
            # Center QR dialog
            from PyQt5.QtWidgets import QDesktopWidget
            screen = QDesktopWidget().screenGeometry()
            x = (screen.width() - 1080) // 2
            y = (screen.height() - 1080) // 2
            qr_dialog.move(x, y)
            
            qr_dialog.exec_()
            
        except Exception as e:
            print(f"[MetaMask] ❌ QR code error: {e}")
            QMessageBox.warning(self, "Error", f"Failed to generate QR code:\n{str(e)}\n\nInstall: pip install qrcode[pil]")
    
    def record_manual_payment(self, amount: float):
        """Record payment after user confirms it was completed in MetaMask"""
        confirm = QMessageBox.question(
            self,
            "Confirm Payment Completed",
            f"Did you complete the payment of {amount:.6f} tokens in MetaMask?\n\n"
            f"⚠️ Only click Yes if the transaction was confirmed on the blockchain.\n\n"
            f"Check MetaMask activity for confirmation.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Record payment
            self.usage_tracker.record_payment(amount)
            
            QMessageBox.information(
                self,
                "Payment Recorded",
                f"✅ Payment of {amount:.6f} tokens recorded!\n\n"
                f"Your balance owed has been updated."
            )
            
            self.accept()
    
    def closeEvent(self, event):
        """Handle dialog close event with smooth fade-out animation"""
        # If already animating or not visible, accept immediately
        if hasattr(self, 'fade_out') and self.fade_out.state() == QPropertyAnimation.Running:
            event.accept()
            return
        
        # Only animate if we're actually closing (not just hiding)
        if event.spontaneous() or not self.isVisible():
            event.accept()
            return
        
        # Cancel fade-in if still running
        if hasattr(self, 'fade_in') and self.fade_in.state() == QPropertyAnimation.Running:
            self.fade_in.stop()
        
        # Create smooth fade-out animation
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(250)  # Slightly longer for smoother feel
        self.fade_out.setStartValue(self.windowOpacity())
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.InOutCubic)  # Symmetric easing
        
        # Connect finished signal to actually close the dialog
        self.fade_out.finished.connect(lambda: event.accept())
        self.fade_out.start()
        
        # Prevent immediate close
        event.ignore()
        print("[MetaMaskPaymentDialog] 🔄 Closing dialog with fade-out animation...")
    
    def center_dialog(self):
        """Center dialog on screen"""
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - 1080) // 2
        y = (screen.height() - 1080) // 2
        self.move(x, y)
        self.raise_()
        self.activateWindow()

