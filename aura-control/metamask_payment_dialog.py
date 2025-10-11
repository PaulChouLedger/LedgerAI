# metamask_payment_dialog.py — MetaMask Payment Dialog

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from wallet_integration import get_wallet_manager, get_usage_tracker
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
        
        # Info about MetaMask
        info = QLabel("🔐 Transaction will open in MetaMask\nApprove in MetaMask app to complete payment")
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
        
        send_btn = QPushButton("💸 Send via MetaMask")
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
        send_btn.clicked.connect(self.send_with_metamask)
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
    
    def send_with_metamask(self):
        """Initiate MetaMask payment"""
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
        
        # Show MetaMask deep link instructions
        self.show_metamask_instructions(amount)
    
    def show_metamask_instructions(self, amount: float):
        """Show instructions for MetaMask payment"""
        token_address = self.wallet_manager.TOKEN_ADDRESS
        
        # Build MetaMask deep link for token transfer
        # This will open MetaMask with pre-filled transaction
        metamask_url = self.build_metamask_deeplink(amount, token_address)
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Open MetaMask")
        msg.setText("Complete payment in MetaMask")
        msg.setInformativeText(
            f"Amount: {amount:.6f} tokens\n"
            f"To: {self.CLIENT_WALLET}\n\n"
            f"Options:\n\n"
            f"1. MetaMask Mobile:\n"
            f"   - Scan QR code below\n"
            f"   - Approve transaction\n\n"
            f"2. MetaMask Browser:\n"
            f"   - Click 'Open MetaMask'\n"
            f"   - Approve transaction\n\n"
            f"After approval, return here and\n"
            f"click 'Mark as Paid' to record payment."
        )
        
        # Add custom buttons
        msg.addButton("📱 Show QR Code", QMessageBox.ActionRole)
        msg.addButton("🌐 Open MetaMask", QMessageBox.ActionRole)
        msg.addButton("✅ Mark as Paid", QMessageBox.AcceptRole)
        msg.addButton("❌ Cancel", QMessageBox.RejectRole)
        
        result = msg.exec_()
        
        if result == 0:  # Show QR
            self.show_qr_code(metamask_url, amount)
        elif result == 1:  # Open MetaMask
            import webbrowser
            webbrowser.open(metamask_url)
        elif result == 2:  # Mark as Paid
            self.record_manual_payment(amount)
    
    def build_metamask_deeplink(self, amount: float, token_address: str) -> str:
        """Build MetaMask deep link for token transfer (correct format)"""
        from web3 import Web3
        
        # Convert amount to wei
        decimals = self.wallet_manager.token_info.get('decimals', 18)
        amount_wei = int(amount * (10 ** decimals))
        
        print(f"[MetaMask] 🔗 Building deep link:")
        print(f"[MetaMask]    Token: {token_address}")
        print(f"[MetaMask]    To: {self.CLIENT_WALLET}")
        print(f"[MetaMask]    Amount: {amount} tokens ({amount_wei} wei)")
        
        # Correct MetaMask mobile deep link format for ERC-20 tokens:
        # https://metamask.app.link/send/{token_address}@{chain_id}/transfer?address={recipient}&uint256={amount}
        
        metamask_link = (
            f"https://metamask.app.link/send/"
            f"{token_address}@1/"  # @1 = Ethereum mainnet, trailing slash important
            f"transfer?"
            f"address={self.CLIENT_WALLET}&"
            f"uint256={amount_wei}"
        )
        
        print(f"[MetaMask] 🔗 Deep link: {metamask_link}")
        
        return metamask_link
    
    def show_qr_code(self, url: str, amount: float):
        """Show QR code for MetaMask mobile"""
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
    
    def center_dialog(self):
        """Center dialog on screen"""
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - 1080) // 2
        y = (screen.height() - 1080) // 2
        self.move(x, y)
        self.raise_()
        self.activateWindow()

