# wallet_dialog.py — Wallet Connection Dialog for Aura

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QTextEdit, QGroupBox, QMessageBox)
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
        self.setFixedSize(1080, 1080)  # Full screen size to match main window
        
        # Use same window flags as upload dialog for proper modal behavior
        if parent:
            # If we have a parent, use Dialog flag for proper modal behavior
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setModal(True)  # Make it modal to block parent interaction
        else:
            # If no parent, stay on top
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # Apply dark theme styling - match upload dialog's circular design
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 1.0);  /* Solid dark background */
                color: #ffffff;
                border: none;
                border-radius: 536px;  /* Circular screen */
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
        
        # Center the dialog on the actual screen
        self.center_dialog()
        
        # Auto-refresh timer for balance updates
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_balance)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
    
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(120, 100, 120, 100)  # Match upload dialog margins
        layout.setSpacing(20)
        
        # Add top spacer for vertical centering
        layout.addStretch(1)
        
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
        
        # Check if saved wallet exists
        saved_wallet = self.wallet_manager.get_saved_wallet()
        
        if saved_wallet:
            # Show saved wallet option
            saved_label = QLabel(f"💾 Saved Wallet: {saved_wallet[:10]}...{saved_wallet[-8:]}")
            saved_label.setStyleSheet("color: #4D94D9; font-size: 11pt; font-weight: bold;")
            wallet_layout.addWidget(saved_label)
            
            # Quick connect buttons
            button_row = QHBoxLayout()
            
            use_saved_btn = QPushButton("✅ Use Saved Wallet")
            use_saved_btn.setStyleSheet("background-color: #4D94D9; font-weight: bold;")
            use_saved_btn.clicked.connect(lambda: self.connect_saved_wallet(saved_wallet))
            button_row.addWidget(use_saved_btn)
            
            clear_btn = QPushButton("🗑️ Clear Saved")
            clear_btn.clicked.connect(self.clear_saved_wallet)
            button_row.addWidget(clear_btn)
            
            wallet_layout.addLayout(button_row)
            
            # Separator
            separator = QLabel("— OR enter different wallet —")
            separator.setAlignment(Qt.AlignCenter)
            separator.setStyleSheet("color: #666666; font-size: 9pt; margin: 10px 0;")
            wallet_layout.addWidget(separator)
        
        instructions = QLabel("Tap to enter wallet address:")
        instructions.setStyleSheet("color: #aaaaaa; font-size: 10pt;")
        wallet_layout.addWidget(instructions)
        
        # Clickable address display (replaced QLineEdit with button)
        self.address_display_btn = QPushButton("⌨️  Tap to Enter Address")
        self.address_display_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #888888;
                border: 2px solid #4D94D9;
                border-radius: 8px;
                padding: 15px;
                font-size: 11pt;
                text-align: left;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 2px solid #5DA4E9;
            }
            QPushButton:pressed {
                background-color: #5a5a5a;
            }
        """)
        self.address_display_btn.clicked.connect(self.show_keyboard)
        wallet_layout.addWidget(self.address_display_btn)
        
        # Store the entered address
        self.entered_address = ""
        
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
        usage_group = QGroupBox("Token Usage & Payments")
        usage_layout = QVBoxLayout()
        
        usage_info = QLabel("Total tokens consumed based on computational complexity:")
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
        
        # Paid to client
        self.paid_label = QLabel("💸 0.000000 tokens paid to client")
        self.paid_label.setAlignment(Qt.AlignCenter)
        self.paid_label.setStyleSheet("color: #34C759; font-size: 11pt; font-weight: bold;")
        usage_layout.addWidget(self.paid_label)
        
        # Balance owed
        self.owed_label = QLabel("📊 0.000000 tokens owed")
        self.owed_label.setAlignment(Qt.AlignCenter)
        owed_font = QFont()
        owed_font.setPointSize(12)
        owed_font.setBold(True)
        self.owed_label.setFont(owed_font)
        self.owed_label.setStyleSheet("color: #FF9500;")
        usage_layout.addWidget(self.owed_label)
        
        # Note about persistent tracking
        note = QLabel("Usage persists across reboots • Saved to disk")
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet("color: #666666; font-size: 9pt; font-style: italic; margin-top: 5px;")
        usage_layout.addWidget(note)
        
        usage_group.setLayout(usage_layout)
        layout.addWidget(usage_group)
        
        # === Action Buttons ===
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Pay Client button
        pay_client_btn = QPushButton("💸 Pay Client")
        pay_client_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 15px;
                border: none;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #30B350;
            }
            QPushButton:pressed {
                background-color: #2A9D47;
            }
        """)
        pay_client_btn.clicked.connect(self.open_payment_dialog)
        button_layout.addWidget(pay_client_btn)
        
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
        
        # Add bottom spacer for vertical centering
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def update_connection_status(self):
        """Update network connection status"""
        if self.wallet_manager.is_connected():
            self.status_label.setText("✅ Connected to Ethereum Mainnet")
            self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        else:
            self.status_label.setText("❌ Not connected to Ethereum network")
            self.status_label.setStyleSheet("color: #ff0000; font-weight: bold;")
    
    def connect_saved_wallet(self, address: str):
        """Connect using the saved wallet address"""
        if self.wallet_manager.connect_wallet(address, auto_save=False):
            self.balance_label.setText("⏳ Fetching balance...")
            self.refresh_balance()
        else:
            self.balance_label.setText("❌ Connection failed")
    
    def clear_saved_wallet(self):
        """Clear the saved wallet address"""
        if self.wallet_manager.clear_saved_wallet():
            # Reload dialog to update UI
            self.close()
            new_dialog = WalletDialog(self.parent())
            new_dialog.show()
    
    def show_keyboard(self):
        """Show custom keyboard for address entry"""
        from custom_keyboard import CircularKeyboard
        
        # Create keyboard with current address
        keyboard = CircularKeyboard(parent=self, initial_text=self.entered_address)
        
        # Connect signals
        keyboard.text_confirmed.connect(self.on_address_entered)
        keyboard.voice_input_requested.connect(self.handle_voice_input)
        
        # Show keyboard
        keyboard.exec_()
    
    def on_address_entered(self, address: str):
        """Handle address entered from keyboard"""
        self.entered_address = address
        
        # Update button display
        if address:
            # Show shortened address
            display_text = f"{address[:10]}...{address[-8:]}" if len(address) > 20 else address
            self.address_display_btn.setText(display_text)
            self.address_display_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3a3a3a;
                    color: #00ff00;
                    border: 2px solid #4D94D9;
                    border-radius: 8px;
                    padding: 15px;
                    font-size: 11pt;
                    font-family: 'Courier New', monospace;
                    font-weight: bold;
                    text-align: left;
                    min-height: 50px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                    border: 2px solid #5DA4E9;
                }
                QPushButton:pressed {
                    background-color: #5a5a5a;
                }
            """)
        
        print(f"[WalletDialog] 📝 Address entered: {address}")
    
    def handle_voice_input(self):
        """Handle voice input request from keyboard"""
        print("[WalletDialog] 🎤 Voice input requested")
        
        # Import voice recording functionality
        import io
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        import requests
        
        try:
            # Show status
            from PyQt5.QtWidgets import QMessageBox, QProgressDialog
            from PyQt5.QtCore import Qt
            
            # Create progress dialog
            progress = QProgressDialog("🎤 Speak your wallet address...", "Cancel", 0, 0, self)
            progress.setWindowTitle("Voice Input")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setCancelButton(None)  # No cancel during recording
            progress.show()
            
            # Record audio (5 seconds)
            sample_rate = 16000
            duration = 5  # seconds
            
            progress.setLabelText("🎤 Recording... (5 seconds)")
            
            print("[VoiceInput] 🎤 Recording started...")
            audio = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
            sd.wait()  # Wait for recording to finish
            print("[VoiceInput] ✅ Recording complete")
            
            progress.setLabelText("🔄 Transcribing...")
            
            # Convert to WAV format
            wav_io = io.BytesIO()
            sf.write(wav_io, audio, sample_rate, format="WAV")
            wav_io.seek(0)
            
            # Send to Whisper for transcription
            response = requests.post(
                "http://localhost:5000/transcribe",
                files={"audio": ("voice.wav", wav_io, "audio/wav")},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", {}).get("text", "").strip() if isinstance(result.get("text"), dict) else result.get("text", "").strip()
                
                print(f"[VoiceInput] 📝 Transcribed: {text}")
                
                progress.close()
                
                # Clean up the transcribed text - remove spaces and common words
                # Ethereum addresses are typically spoken character by character
                cleaned = text.lower().replace(" ", "").replace("zero", "0").replace("one", "1").replace("two", "2").replace("three", "3").replace("four", "4").replace("five", "5").replace("six", "6").replace("seven", "7").replace("eight", "8").replace("nine", "9")
                
                # Ensure it starts with 0x
                if not cleaned.startswith("0x"):
                    if cleaned.startswith("x"):
                        cleaned = "0" + cleaned
                    else:
                        cleaned = "0x" + cleaned
                
                # Set the text in the keyboard (if still open)
                # Or directly set it
                if cleaned and len(cleaned) >= 10:  # At least 0x + some characters
                    self.entered_address = cleaned[:42]  # Limit to valid address length
                    self.on_address_entered(self.entered_address)
                    
                    QMessageBox.information(self, "Voice Input", 
                                          f"Address captured:\n{self.entered_address}")
                else:
                    QMessageBox.warning(self, "Voice Input", 
                                       f"Could not parse wallet address from: {text}\n\nPlease try again or use keyboard.")
            else:
                progress.close()
                QMessageBox.warning(self, "Error", "Failed to transcribe audio. Please try again.")
                
        except Exception as e:
            print(f"[VoiceInput] ❌ Error: {e}")
            try:
                progress.close()
            except:
                pass
            QMessageBox.warning(self, "Error", f"Voice input failed: {str(e)}")
    
    def connect_wallet(self):
        """Connect to the entered wallet address"""
        address = self.entered_address.strip()
        
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
        total_paid = self.usage_tracker.get_total_paid()
        balance_owed = self.usage_tracker.get_balance_owed()
        
        self.usage_label.setText(f"💳 {session_usage:.6f} tokens used")
        self.paid_label.setText(f"💸 {total_paid:.6f} tokens paid to client")
        self.owed_label.setText(f"📊 {balance_owed:.6f} tokens owed")
        
        # Update connection status
        self.update_connection_status()
    
    
    def center_dialog(self):
        """Center the dialog properly on the screen"""
        from PyQt5.QtWidgets import QDesktopWidget
        
        # Get screen geometry
        screen = QDesktopWidget().screenGeometry()
        print(f"[WalletDialog] 🔍 Screen geometry: {screen.width()}x{screen.height()}")
        
        # Get dialog size (use fixed size since we set it to 1080x1080)
        dialog_width = 1080
        dialog_height = 1080
        
        # Calculate center position
        x = (screen.width() - dialog_width) // 2
        y = (screen.height() - dialog_height) // 2
        
        print(f"[WalletDialog] 📐 Calculated center position: ({x}, {y})")
        
        # Move to center
        self.move(x, y)
        print(f"[WalletDialog] ✅ Dialog centered at ({x}, {y})")
        
        # Verify final position
        final_pos = self.pos()
        print(f"[WalletDialog] 📐 Final dialog position: ({final_pos.x()}, {final_pos.y()})")
        
        # Ensure dialog is visible and active
        self.raise_()
        self.activateWindow()
    
    def open_payment_dialog(self):
        """Open payment dialog to send tokens to client"""
        if not self.wallet_manager.connected_address:
            QMessageBox.warning(self, "Not Connected", 
                              "Please connect your wallet first before making payments.")
            return
        
        try:
            from payment_dialog import PaymentDialog
            
            # Create and show payment dialog
            payment_dialog = PaymentDialog(parent=self, user_address=self.wallet_manager.connected_address)
            result = payment_dialog.exec_()
            
            # Refresh balance after payment
            if result == QMessageBox.Accepted:
                print("[WalletDialog] ✅ Payment completed, refreshing balance")
                self.refresh_balance()
                
        except Exception as e:
            print(f"[WalletDialog] ❌ Error opening payment dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open payment dialog:\n{str(e)}")
    
    def closeEvent(self, event):
        """Handle dialog close"""
        self.refresh_timer.stop()
        print("[WalletDialog] ✅ Dialog closed")
        event.accept()

