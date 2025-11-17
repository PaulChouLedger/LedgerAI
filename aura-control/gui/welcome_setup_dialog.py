# welcome_setup_dialog.py — Welcome Setup Dialog for WiFi Connection
# This dialog appears before main.py fully loads to ensure WiFi is connected
# before TTS initializes (TTS requires WiFi for ElevenLabs API)

import os
import sys
import subprocess
import time
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QListWidget, QListWidgetItem, 
                             QMessageBox, QInputDialog, QLineEdit, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
import shutil

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class WiFiScanThread(QThread):
    """Thread to scan for WiFi networks without blocking UI"""
    networks_found = pyqtSignal(list)
    scan_error = pyqtSignal(str)
    
    def run(self):
        """Scan for available WiFi networks"""
        try:
            # Trigger rescan
            subprocess.run(['nmcli', 'device', 'wifi', 'rescan'], 
                         capture_output=True, timeout=10)
            time.sleep(3)  # Wait for scan to complete
            
            # Get networks
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY,IN-USE', 'device', 'wifi', 'list'],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode != 0:
                self.scan_error.emit("Failed to scan WiFi networks")
                return
            
            networks = []
            seen_ssids = set()
            
            for line in result.stdout.strip().split('\n'):
                if not line or ':' not in line:
                    continue
                
                parts = line.split(':')
                if len(parts) < 3:
                    continue
                
                ssid = parts[0].strip()
                if not ssid or ssid in seen_ssids:
                    continue
                seen_ssids.add(ssid)
                
                signal_str = parts[1].strip() if len(parts) > 1 else "0"
                security = parts[2].strip() if len(parts) > 2 else "Open"
                in_use = parts[3].strip() if len(parts) > 3 else ""
                
                try:
                    signal = int(signal_str) if signal_str.isdigit() else 0
                except:
                    signal = 0
                
                connected = in_use == "*" or "connected" in in_use.lower()
                
                networks.append({
                    'ssid': ssid,
                    'signal': signal,
                    'security': security,
                    'connected': connected
                })
            
            networks.sort(key=lambda x: (-x['signal'], x['ssid']))
            
            if networks:
                self.networks_found.emit(networks)
            else:
                self.scan_error.emit("No WiFi networks found")
                
        except subprocess.TimeoutExpired:
            self.scan_error.emit("WiFi scan timed out")
        except Exception as e:
            self.scan_error.emit(f"Error scanning WiFi: {str(e)}")

class WelcomeSetupDialog(QDialog):
    """Welcome setup dialog that appears before main.py fully loads"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        print("[WelcomeSetup] 🔧 Initializing welcome setup dialog...")
        
        self.setWindowTitle("Welcome to Aura - WiFi Setup")
        self.setFixedSize(1080, 1080)
        
        if parent:
            # Use Window flag instead of Dialog to ensure proper z-ordering above parent
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # Ensure resources are freed when closed
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
        # Initialize opacity to 0 for fade-in
        self.setWindowOpacity(0.0)
        
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 1.0);
                color: white;
                border: none;
                border-radius: 536px;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QListWidget {
                background-color: rgba(44, 44, 46, 0.8);
                color: #ffffff;
                border-radius: 15px;
                border: none;
                padding: 10px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 8px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: rgba(0, 122, 255, 0.3);
            }
            QPushButton {
                background-color: rgba(70, 130, 180, 0.25);
                color: #ffffff;
                font-size: 18px;
                font-weight: 600;
                padding: 15px 20px;
                min-height: 50px;
                border-radius: 15px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 0.45);
            }
            QPushButton:pressed {
                background-color: rgba(70, 130, 180, 0.65);
            }
            QPushButton:disabled {
                background-color: rgba(70, 130, 180, 0.1);
                color: #666;
            }
        """)
        
        self.setup_ui()
        self.center_dialog()
        
        # Check WiFi connection status
        self.wifi_connected = False
        self.check_wifi_connection()
        
        # Auto-scan on startup (kept minimal; also re-triggered post fade-in)
        QTimer.singleShot(300, self.scan_wifi)
    
    def showEvent(self, event):
        """Handle dialog show event with smooth fade-in animation"""
        super().showEvent(event)
        
        # Create smooth fade-in animation
        self.fade_in = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in.setDuration(300)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setEasingCurve(QEasingCurve.InOutCubic)
        # Start scanning shortly after fade-in to avoid stutter
        try:
            self.fade_in.finished.connect(lambda: QTimer.singleShot(100, self.scan_wifi))
        except Exception:
            pass
        self.fade_in.start()
        
        self.raise_()
        self.activateWindow()
    
    def closeEvent(self, event):
        """Handle dialog close - prevent closing if WiFi not connected"""
        if not self.wifi_connected:
            reply = QMessageBox.question(
                self,
                "WiFi Not Connected",
                "WiFi is not connected. TTS requires internet connection.\n\n"
                "Do you want to proceed anyway? (TTS will fail)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        
        # Clean up WiFi scan thread to prevent accessing deleted widgets
        if hasattr(self, 'wifi_scan_thread') and self.wifi_scan_thread:
            try:
                # Disconnect all signals to prevent callbacks after deletion
                self.wifi_scan_thread.networks_found.disconnect()
                self.wifi_scan_thread.scan_error.disconnect()
                self.wifi_scan_thread.finished.disconnect()
            except Exception:
                pass
            try:
                if self.wifi_scan_thread.isRunning():
                    self.wifi_scan_thread.quit()
                    self.wifi_scan_thread.wait(1000)  # Wait up to 1 second
            except Exception:
                pass
            self.wifi_scan_thread = None
        
        # Reactivate parent window immediately to prevent freezing
        if self.parent():
            try:
                self.parent().raise_()
                self.parent().activateWindow()
                QApplication.processEvents()
            except Exception:
                pass
        
        # For modal dialogs opened from home screen, accept immediately to avoid blocking
        if self.isModal() and self.parent():
            event.accept()
            return
        
        # Non-modal: use fade animation
        if hasattr(self, 'fade_in') and self.fade_in.state() == QPropertyAnimation.Running:
            self.fade_in.stop()
        
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(250)
        self.fade_out.setStartValue(self.windowOpacity())
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.InOutCubic)
        
        def _finalize():
            event.accept()
            # Ensure parent is reactivated after close
            if self.parent():
                try:
                    self.parent().raise_()
                    self.parent().activateWindow()
                    QApplication.processEvents()
                except Exception:
                    pass
        
        self.fade_out.finished.connect(_finalize)
        self.fade_out.start()
        event.ignore()
    
    def center_dialog(self):
        """Center dialog on screen/parent using available geometry and clamp to visible area"""
        try:
            if self.parent():
                parent_geometry = self.parent().geometry()
                x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
                y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            else:
                screen = QApplication.primaryScreen().availableGeometry()
                x = (screen.width() - self.width()) // 2
                y = (screen.height() - self.height()) // 2
            # Clamp to ensure not cut off on small bars/margins
            x = max(0, x)
            y = max(0, y)
            self.move(x, y)
        except Exception:
            # Fallback
            self.move(0, 0)
    
    def setup_ui(self):
        """Setup the welcome setup UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(120, 100, 120, 100)
        main_layout.setSpacing(20)
        
        main_layout.addStretch(1)
        
        # Welcome title
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        title = QLabel("👋 Welcome to Aura")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 20px;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        main_layout.addLayout(title_layout)
        
        # Instructions
        instructions = QLabel("Connect to WiFi to continue setup\n(TTS requires internet connection)")
        instructions.setFont(QFont("Arial", 16))
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: #aaaaaa; margin: 10px;")
        main_layout.addWidget(instructions)
        
        # WiFi connection status
        self.status_label = QLabel("Checking WiFi connection...")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #ffa500; margin: 10px;")
        main_layout.addWidget(self.status_label)
        
        # WiFi Section
        wifi_label = QLabel("📶 WiFi Networks")
        wifi_label.setFont(QFont("Arial", 18, QFont.Bold))
        wifi_label.setStyleSheet("color: #ffffff; margin-top: 20px;")
        main_layout.addWidget(wifi_label)
        
        # WiFi buttons
        wifi_button_layout = QHBoxLayout()
        wifi_button_layout.setSpacing(15)
        
        self.scan_wifi_btn = QPushButton("🔍 Scan Networks")
        self.scan_wifi_btn.clicked.connect(self.scan_wifi)
        wifi_button_layout.addWidget(self.scan_wifi_btn)
        
        self.connect_wifi_btn = QPushButton("🔗 Connect")
        self.connect_wifi_btn.clicked.connect(self.connect_wifi)
        self.connect_wifi_btn.setEnabled(False)
        wifi_button_layout.addWidget(self.connect_wifi_btn)
        
        main_layout.addLayout(wifi_button_layout)
        
        # WiFi networks list
        self.wifi_list = QListWidget()
        self.wifi_list.setMaximumHeight(200)
        self.wifi_list.itemSelectionChanged.connect(self.on_wifi_selection_changed)
        main_layout.addWidget(self.wifi_list)
        
        # Continue button (only enabled when WiFi connected)
        continue_layout = QHBoxLayout()
        continue_layout.addStretch()
        self.continue_btn = QPushButton("✅ Continue to Aura")
        self.continue_btn.clicked.connect(self.accept)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                font-size: 20px;
                font-weight: 600;
                padding: 18px 40px;
                border-radius: 20px;
                border: none;
                min-width: 250px;
            }
            QPushButton:hover {
                background-color: #28A745;
            }
            QPushButton:pressed {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: rgba(52, 199, 89, 0.3);
                color: #999;
            }
        """)
        self.continue_btn.setEnabled(False)
        continue_layout.addWidget(self.continue_btn)
        continue_layout.addStretch()
        main_layout.addLayout(continue_layout)
        
        main_layout.addStretch(1)
        self.setLayout(main_layout)
        
        # Initialize
        self.wifi_scan_thread = None
        self.selected_wifi = None
    
    def check_wifi_connection(self):
        """Check if WiFi is currently connected"""
        try:
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'DEVICE,TYPE,STATE', 'device', 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':wifi:' in line.lower() and ':connected' in line.lower():
                        self.wifi_connected = True
                        self.status_label.setText("✅ WiFi Connected")
                        self.status_label.setStyleSheet("color: #34C759; margin: 10px;")
                        self.continue_btn.setEnabled(True)
                        return
                
            self.wifi_connected = False
            self.status_label.setText("❌ WiFi Not Connected")
            self.status_label.setStyleSheet("color: #FF3B30; margin: 10px;")
            self.continue_btn.setEnabled(False)
        except Exception as e:
            print(f"[WelcomeSetup] ⚠️ Error checking WiFi: {e}")
            self.status_label.setText("⚠️ Could not check WiFi status")
            self.status_label.setStyleSheet("color: #ffa500; margin: 10px;")
    
    def scan_wifi(self):
        """Scan for available WiFi networks"""
        self.status_label.setText("🔍 Scanning for networks...")
        self.status_label.setStyleSheet("color: #ffa500; margin: 10px;")
        self.scan_wifi_btn.setEnabled(False)
        self.scan_wifi_btn.setText("🔄 Scanning...")
        self.wifi_list.clear()
        self.wifi_list.addItem("Scanning... Please wait...")
        
        self.wifi_scan_thread = WiFiScanThread()
        self.wifi_scan_thread.networks_found.connect(self.on_wifi_networks_found)
        self.wifi_scan_thread.scan_error.connect(self.on_wifi_scan_error)
        self.wifi_scan_thread.finished.connect(self._on_scan_finished)
        self.wifi_scan_thread.start()
    
    def _on_scan_finished(self):
        """Handle WiFi scan thread finished - safely update UI"""
        try:
            # Check if dialog and button still exist before accessing
            # Use try-except to catch RuntimeError if widget was deleted
            if hasattr(self, 'scan_wifi_btn') and self.scan_wifi_btn is not None:
                try:
                    self.scan_wifi_btn.setEnabled(True)
                    self.scan_wifi_btn.setText("🔍 Scan Networks")
                except RuntimeError:
                    # Widget was deleted, ignore silently
                    pass
        except (RuntimeError, AttributeError):
            # Dialog or widget was deleted, ignore silently
            pass
    
    def on_wifi_networks_found(self, networks):
        """Handle WiFi networks found"""
        try:
            # Check if dialog still exists
            if not hasattr(self, 'wifi_list'):
                return
            self.wifi_list.clear()
            
            if not networks:
                self.wifi_list.addItem("No networks found")
                return
            
            for network in networks:
                ssid = network['ssid']
                signal = network['signal']
                security = network['security']
                connected = network.get('connected', False)
                
                signal_display = f"{signal}%" if signal > 0 else "weak"
                item_text = f"{ssid} ({signal_display})"
                
                if connected:
                    item_text = f"● {item_text} (Connected)"
                if security and security != "Open" and security != "--":
                    item_text += f" 🔒 {security}"
                
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, network)
                self.wifi_list.addItem(item)
            
            # Check connection status after scan
            self.check_wifi_connection()
        except (RuntimeError, AttributeError):
            # Dialog was deleted, ignore silently
            pass
    
    def on_wifi_scan_error(self, error):
        """Handle WiFi scan error"""
        try:
            # Check if dialog still exists
            if not hasattr(self, 'wifi_list'):
                return
            self.wifi_list.clear()
            self.wifi_list.addItem(f"Error: {error}")
            QMessageBox.warning(self, "WiFi Scan Error", error)
            self.check_wifi_connection()
        except (RuntimeError, AttributeError):
            # Dialog was deleted, ignore silently
            pass
    
    def on_wifi_selection_changed(self):
        """Handle WiFi network selection"""
        has_selection = len(self.wifi_list.selectedItems()) > 0
        self.connect_wifi_btn.setEnabled(has_selection)
    
    def connect_wifi(self):
        """Connect to selected WiFi network"""
        selected_items = self.wifi_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a WiFi network")
            return
        
        item = selected_items[0]
        network = item.data(Qt.UserRole)
        ssid = network['ssid']
        security = network['security']
        
        # Prompt for password if secured
        password = None
        if security and security != "Open" and "Open" not in security:
            password, ok = QInputDialog.getText(
                self,
                "WiFi Password",
                f"Enter password for {ssid}:",
                QLineEdit.Password
            )
            if not ok:
                return
        
        self.status_label.setText(f"🔗 Connecting to {ssid}...")
        self.status_label.setStyleSheet("color: #ffa500; margin: 10px;")
        self.connect_wifi_btn.setEnabled(False)
        self.connect_wifi_btn.setText("🔄 Connecting...")
        
        # Connect using nmcli, retrying with pkexec if permission is denied
        try:
            base_cmd = ['nmcli', 'device', 'wifi', 'connect', ssid]
            if password:
                base_cmd += ['password', password]

            # Attempt without elevation first
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                error_msg = (result.stderr or result.stdout or "").strip()
                needs_priv = any(term in error_msg.lower() for term in [
                    "permission denied", "not authorized", "authorization failed", "polkit", "not permitted"
                ])

                if needs_priv and shutil.which("pkexec"):
                    # Retry with pkexec to trigger GUI auth prompt
                    pkexec_cmd = ['pkexec'] + base_cmd
                    result2 = subprocess.run(pkexec_cmd, capture_output=True, text=True, timeout=60)
                    if result2.returncode == 0:
                        result = result2
                        error_msg = ""
                    else:
                        # If pkexec failed, use its error
                        error_msg = (result2.stderr or result2.stdout or error_msg).strip()

                if result.returncode != 0:
                    self.status_label.setText("❌ Connection failed")
                    self.status_label.setStyleSheet("color: #FF3B30; margin: 10px;")
                    QMessageBox.warning(self, "Connection Failed", error_msg or "Unknown error")
                    self.check_wifi_connection()
                    return

            # Success
            self.status_label.setText(f"✅ Connected to {ssid}")
            self.status_label.setStyleSheet("color: #34C759; margin: 10px;")
            QMessageBox.information(self, "Success", f"Connected to {ssid}")
            
            # Wait a moment for connection to stabilize
            time.sleep(2)
            self.check_wifi_connection()
            # Refresh network list
            self.scan_wifi()
        except Exception as e:
            self.status_label.setText("❌ Connection error")
            self.status_label.setStyleSheet("color: #FF3B30; margin: 10px;")
            QMessageBox.warning(self, "Error", str(e))
            self.check_wifi_connection()
        finally:
            self.connect_wifi_btn.setText("🔗 Connect")
            self.connect_wifi_btn.setEnabled(len(self.wifi_list.selectedItems()) > 0)

