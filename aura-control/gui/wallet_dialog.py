# wallet_dialog.py — Wallet Connection Dialog for Aura

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QTextEdit, QGroupBox, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor
import os
import sys

# Add the parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from wallet.wallet_integration import get_wallet_manager, get_usage_tracker

# Import base dialog template
from gui.base_dialog import BaseAuraDialog


class BalanceFetchWorker(QThread):
    """Background worker for fetching balance without blocking UI"""
    
    balance_ready = pyqtSignal(dict)  # Emits wallet_info when ready
    
    def __init__(self, wallet_manager):
        super().__init__()
        self.wallet_manager = wallet_manager
    
    def run(self):
        """Fetch balance in background"""
        try:
            wallet_info = self.wallet_manager.get_wallet_info()
            self.balance_ready.emit(wallet_info)
        except Exception as e:
            print(f"[BalanceFetch] ❌ Error: {e}")
            # Emit empty info on error
            self.balance_ready.emit({
                'connected': False,
                'address': None,
                'eth_balance': None,
                'token_balance': None,
                'token_info': {}
            })


class WalletDialog(BaseAuraDialog):
    """Dialog for connecting wallet and viewing token balance"""
    
    def __init__(self, parent=None):
        # Initialize attributes first (before base class)
        try:
            self.wallet_manager = get_wallet_manager()
            self.usage_tracker = get_usage_tracker()
        except Exception as e:
            print(f"[WalletDialog] ⚠️ Error initializing wallet manager: {e}")
            # Set to None to handle gracefully
            self.wallet_manager = None
            self.usage_tracker = None
        
        # Initialize base dialog (non-modal to avoid blocking)
        super().__init__(
            parent=parent,
            title="Aura Token Wallet",
            size=(1080, 1080),
            modal=False
        )
        
        # Apply dark theme styling - match upload dialog exactly
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 0.95);
                color: white;
                border: none;
                border-radius: 532px;
            }
            QGroupBox {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 15px;
                border: none;
                padding: 10px;
                font-size: 12px;
            }
            QGroupBox::title {
                color: #ffffff;
                font-weight: 600;
            }
            QLabel {
                color: white;
                font-size: 12px;
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
        
        # Defer initial background work until after fade-in
        self._initial_refresh_scheduled = False
        
        # Initialize timers (but don't start them until shown)
        self.balance_refresh_timer = None
        self.usage_refresh_timer = None
    
    def _setup_ui(self):
        """Set up dialog UI (called by base class)"""
        try:
            self.setup_ui()
            if self.wallet_manager:
                self.update_connection_status()
            
            # Show usage stats immediately (instant, no network call)
            if self.usage_tracker:
                self.update_usage_stats()
            
            # Initialize timers (started in _on_show)
            self.balance_refresh_timer = QTimer()
            self.balance_refresh_timer.timeout.connect(self.refresh_balance_async)
            
            self.usage_refresh_timer = QTimer()
            self.usage_refresh_timer.timeout.connect(self.update_usage_stats)
        except Exception as e:
            print(f"[WalletDialog] ❌ Error setting up UI: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_show(self):
        """Block transcription and schedule balance refresh when dialog opens"""
        # Base class already blocks transcription, but we call it explicitly for clarity
        self._block_transcription("Wallet dialog open")
        
        # Start timers now that dialog is shown
        try:
            if self.balance_refresh_timer:
                self.balance_refresh_timer.start(30000)  # Refresh balance every 30 seconds
            if self.usage_refresh_timer:
                self.usage_refresh_timer.start(1000)  # Update usage every 1 second
        except Exception as e:
            print(f"[WalletDialog] ⚠️ Error starting timers: {e}")
        
        # After fade-in completes, start the initial refresh to avoid jank
        if not self._initial_refresh_scheduled:
            self._initial_refresh_scheduled = True
            # Check if fade animation exists (it will for non-modal dialogs)
            if hasattr(self, 'fade_in') and self.fade_in:
                self.fade_in.finished.connect(lambda: QTimer.singleShot(50, self.refresh_balance_async))
            else:
                # If no fade animation (modal with parent), refresh immediately
                QTimer.singleShot(50, self.refresh_balance_async)
    
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout()
        # Keep all content within white circular border (radius 535px)
        # Safe area is ~450px radius from center
        layout.setContentsMargins(130, 110, 130, 110)  # Generous margins for circular screen
        layout.setSpacing(15)  # Tighter spacing to fit everything
        
        # Add top spacer for vertical centering
        layout.addStretch(1)
        
        # === Title === (larger for visibility)
        title = QLabel("💰 Wallet & Token Balance")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        layout.addWidget(title)
        
        # === Connection Status ===
        self.status_label = QLabel("⏳ Checking connection...")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(13)  # Bigger for visibility
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)
        
        # === Wallet Address Input ===
        wallet_group = QGroupBox("Wallet Address")
        wallet_layout = QVBoxLayout()
        wallet_layout.setSpacing(8)  # Tighter spacing
        
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
        
        instructions = QLabel("Tap to enter address:")
        instructions.setStyleSheet("color: #aaaaaa; font-size: 9pt;")  # Smaller
        wallet_layout.addWidget(instructions)
        
        # Clickable address display (replaced QLineEdit with button)
        self.address_display_btn = QPushButton("⌨️  Tap to Enter Address")
        self.address_display_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #888888;
                border: 2px solid #4D94D9;
                border-radius: 8px;
                padding: 12px;
                font-size: 10pt;
                text-align: left;
                min-height: 45px;
                max-height: 45px;
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
        balance_layout.setSpacing(5)  # Compact spacing
        
        self.balance_label = QLabel("Not connected")
        self.balance_label.setAlignment(Qt.AlignCenter)
        balance_font = QFont()
        balance_font.setPointSize(14)  # Smaller to fit better
        balance_font.setBold(True)
        self.balance_label.setFont(balance_font)
        balance_layout.addWidget(self.balance_label)
        
        self.eth_balance_label = QLabel("ETH: --")
        self.eth_balance_label.setAlignment(Qt.AlignCenter)
        self.eth_balance_label.setStyleSheet("color: #aaaaaa; font-size: 10pt;")
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
        usage_layout.setSpacing(5)  # Compact spacing
        
        usage_info = QLabel("Total tokens consumed:")
        usage_info.setAlignment(Qt.AlignCenter)
        usage_info.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        usage_layout.addWidget(usage_info)
        
        self.usage_label = QLabel("💳 0.000000 tokens")
        self.usage_label.setAlignment(Qt.AlignCenter)
        usage_font = QFont()
        usage_font.setPointSize(12)  # Smaller
        usage_font.setBold(True)
        self.usage_label.setFont(usage_font)
        self.usage_label.setStyleSheet("color: #FFD700;")
        usage_layout.addWidget(self.usage_label)
        
        # Paid to client
        self.paid_label = QLabel("💸 0.000000 paid")
        self.paid_label.setAlignment(Qt.AlignCenter)
        self.paid_label.setStyleSheet("color: #34C759; font-size: 10pt; font-weight: bold;")
        usage_layout.addWidget(self.paid_label)
        
        # Balance owed
        self.owed_label = QLabel("📊 0.000000 owed")
        self.owed_label.setAlignment(Qt.AlignCenter)
        owed_font = QFont()
        owed_font.setPointSize(11)  # Smaller
        owed_font.setBold(True)
        self.owed_label.setFont(owed_font)
        self.owed_label.setStyleSheet("color: #FF9500;")
        usage_layout.addWidget(self.owed_label)
        
        usage_group.setLayout(usage_layout)
        layout.addWidget(usage_group)
        
        # === Action Buttons ===
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        # Pay Client button (match upload dialog style)
        pay_client_btn = QPushButton("💸 Pay Client")
        pay_client_btn.setStyleSheet("""
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
        pay_client_btn.clicked.connect(self.open_payment_dialog)
        button_layout.addWidget(pay_client_btn)
        
        # Close button (match upload dialog style)
        close_btn = QPushButton("❌ Close")
        close_btn.setStyleSheet("""
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
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # === Token Info Footer === (match upload dialog style)
        token_info = QLabel(f"Token: {self.wallet_manager.TOKEN_ADDRESS[:8]}...{self.wallet_manager.TOKEN_ADDRESS[-6:]}")
        token_info.setAlignment(Qt.AlignCenter)
        token_info.setStyleSheet("color: #8e8e93; font-size: 11px; margin: 5px;")
        layout.addWidget(token_info)
        
        # Add bottom spacer for vertical centering
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def update_connection_status(self):
        """Update network connection status"""
        if not self.wallet_manager:
            return
        if self.wallet_manager.is_connected():
            self.status_label.setText("✅ Connected to Ethereum Mainnet")
            self.status_label.setStyleSheet("color: #34C759; font-weight: bold; margin: 5px;")
        else:
            self.status_label.setText("❌ Not connected to Ethereum network")
            self.status_label.setStyleSheet("color: #FF3B30; font-weight: bold; margin: 5px;")
    
    def connect_saved_wallet(self, address: str):
        """Connect using the saved wallet address"""
        if self.wallet_manager.connect_wallet(address, auto_save=False):
            self.balance_label.setText("⏳ Fetching balance...")
            self.refresh_balance_async()  # Non-blocking
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
        if not self.wallet_manager:
            QMessageBox.warning(self, "Error", "Wallet manager not available")
            return
        
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
            self.refresh_balance_async()  # Non-blocking
        else:
            self.balance_label.setText("❌ Connection failed")
            self.balance_label.setStyleSheet("color: #ff0000;")
            self.address_display.setText("Invalid address or network error")
    
    def refresh_balance_async(self):
        """Refresh wallet balance in background (non-blocking)"""
        if not self.wallet_manager:
            return
        if not self.wallet_manager.connected_address:
            # Still update usage stats even if no wallet connected
            self.update_usage_stats()
            return
        
        # Show loading state
        self.balance_label.setText("⏳ Loading...")
        self.balance_label.setStyleSheet("color: #8e8e93;")
        
        # Start background fetch
        self.balance_worker = BalanceFetchWorker(self.wallet_manager)
        self.balance_worker.balance_ready.connect(self.on_balance_ready)
        self.balance_worker.start()
    
    def on_balance_ready(self, wallet_info):
        """Handle balance data received from background worker"""
        
        if wallet_info['connected']:
            # Display token balance
            token_balance = wallet_info.get('token_balance')
            if token_balance is not None:
                token_symbol = wallet_info['token_info'].get('symbol', 'tokens')
                
                # Check if balance is actually zero vs query error
                if token_balance == 0.0:
                    self.balance_label.setText(f"💰 0.000000 {token_symbol}")
                    self.balance_label.setStyleSheet("color: #FF9500;")  # Orange for zero balance
                    self.address_display.setText("⚠️ No tokens in wallet (ETH balance may exist)")
                    self.address_display.setStyleSheet("color: #FF9500; font-size: 9pt;")
                else:
                    self.balance_label.setText(f"💰 {token_balance:.6f} {token_symbol}")
                    self.balance_label.setStyleSheet("color: #00ff00;")
            else:
                # RPC error - suggest upgrading to better provider
                self.balance_label.setText("⚠️ RPC Error - Click Refresh")
                self.balance_label.setStyleSheet("color: #FF9500;")
                self.address_display.setText("💡 Tip: Set INFURA_URL for reliable access")
                self.address_display.setStyleSheet("color: #8e8e93; font-size: 9pt;")
            
            # Display ETH balance
            eth_balance = wallet_info.get('eth_balance')
            if eth_balance is not None:
                self.eth_balance_label.setText(f"Ξ {eth_balance:.6f} ETH")
            
            # Display address
            if self.wallet_manager:
                short_addr = self.wallet_manager.format_address(wallet_info['address'])
                self.address_display.setText(f"Address: {short_addr}")
        
        # Update usage stats
        self.update_usage_stats()
        
        # Update connection status
        self.update_connection_status()
    
    def update_usage_stats(self):
        """Update usage statistics display (can be called independently)"""
        if not self.usage_tracker:
            return
        session_usage = self.usage_tracker.get_session_usage()
        total_paid = self.usage_tracker.get_total_paid()
        balance_owed = self.usage_tracker.get_balance_owed()
        
        self.usage_label.setText(f"💳 {session_usage:.6f} tokens")
        self.paid_label.setText(f"💸 {total_paid:.6f} paid")
        self.owed_label.setText(f"📊 {balance_owed:.6f} owed")
    
    def refresh_balance(self):
        """Legacy method - redirects to async version"""
        self.refresh_balance_async()
    
    
    
    def open_payment_dialog(self):
        """Open payment dialog to send tokens to client"""
        if not self.wallet_manager:
            QMessageBox.warning(self, "Error", "Wallet manager not available")
            return
        if not self.wallet_manager.connected_address:
            QMessageBox.warning(self, "Not Connected", 
                              "Please connect your wallet first before making payments.")
            return
        
        # Ask user which payment method they prefer
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Choose Payment Method")
        msg.setText("How would you like to send payment?")
        msg.setInformativeText(
            "💻 Native Wallet (Best for Jetson):\n"
            "  • Built into Aura interface\n"
            "  • Fits circular display\n"
            "  • Enter key once\n\n"
            "📱 Mobile Wallet QR:\n"
            "  • Scan with phone\n"
            "  • Works with any wallet app\n"
            "  • No key needed"
        )
        
        native_btn = msg.addButton("💻 Native Wallet", QMessageBox.AcceptRole)
        mobile_btn = msg.addButton("📱 Mobile QR", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("❌ Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        
        if msg.clickedButton() == native_btn:
            # Use native wallet (integrated into Aura)
            self.open_native_wallet()
        elif msg.clickedButton() == mobile_btn:
            # Use mobile wallet QR code
            self.open_metamask_payment()
    
    def open_native_wallet(self):
        """Open native wallet integrated into Aura"""
        if not self.usage_tracker or not self.wallet_manager:
            QMessageBox.warning(self, "Error", "Wallet manager or usage tracker not available")
            return
        try:
            from wallet.native_wallet import NativeWalletDialog
            
            balance_owed = self.usage_tracker.get_balance_owed()
            token_address = self.wallet_manager.TOKEN_ADDRESS
            
            wallet_dialog = NativeWalletDialog(
                parent=self,
                amount=balance_owed,
                to_address="0xd3c4d619C8515Bc764921209821Ec7A77FC31Ba4",
                token_address=token_address
            )
            result = wallet_dialog.exec_()
            
            if result == QDialog.Accepted:
                # Record payment
                if self.usage_tracker:
                    self.usage_tracker.record_payment(balance_owed)
                print("[WalletDialog] ✅ Native wallet payment completed, refreshing balance")
                self.refresh_balance_async()
                
        except Exception as e:
            print(f"[WalletDialog] ❌ Error opening native wallet: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open native wallet:\n{str(e)}")
    
    def open_metamask_payment(self):
        """Open mobile wallet QR code payment dialog"""
        if not self.wallet_manager:
            QMessageBox.warning(self, "Error", "Wallet manager not available")
            return
        try:
            from gui.metamask_payment_dialog import MetaMaskPaymentDialog
            
            payment_dialog = MetaMaskPaymentDialog(parent=self, user_address=self.wallet_manager.connected_address)
            result = payment_dialog.exec_()
            
            if result == QDialog.Accepted:
                print("[WalletDialog] ✅ Mobile wallet payment recorded, refreshing balance")
                self.refresh_balance_async()
                
        except Exception as e:
            print(f"[WalletDialog] ❌ Error opening mobile wallet payment: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open payment dialog:\n{str(e)}")
    
    def open_direct_payment(self):
        """Open direct payment dialog (private key method)"""
        if not self.wallet_manager:
            QMessageBox.warning(self, "Error", "Wallet manager not available")
            return
        try:
            from gui.payment_dialog import PaymentDialog
            
            payment_dialog = PaymentDialog(parent=self, user_address=self.wallet_manager.connected_address)
            result = payment_dialog.exec_()
            
            if result == QDialog.Accepted:
                print("[WalletDialog] ✅ Direct payment completed, refreshing balance")
                self.refresh_balance_async()
                
        except Exception as e:
            print(f"[WalletDialog] ❌ Error opening direct payment: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open payment dialog:\n{str(e)}")
    
    def _on_close(self):
        """Cleanup when dialog closes (called by base class)"""
        # Stop timers first
        try:
            if self.balance_refresh_timer:
                self.balance_refresh_timer.stop()
            if self.usage_refresh_timer:
                self.usage_refresh_timer.stop()
        except Exception:
            pass
        
        # Gracefully stop background worker if running
        try:
            if hasattr(self, "balance_worker") and self.balance_worker and self.balance_worker.isRunning():
                try:
                    self.balance_worker.balance_ready.disconnect(self.on_balance_ready)
                except Exception:
                    pass
                self.balance_worker.quit()
                self.balance_worker.wait(1000)
                self.balance_worker = None
        except Exception:
            pass
        
        # Ensure dialog is non-modal during shutdown
        try:
            self.setModal(False)
            self.setWindowModality(Qt.NonModal)
        except Exception:
            pass
        
        print("[WalletDialog] 🔄 Closing dialog with fade-out animation...")

