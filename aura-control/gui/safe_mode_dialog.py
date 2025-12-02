# safe_mode_dialog.py — Safe Mode Dialog for WiFi and OTA Updates
# This dialog provides access to WiFi and OTA updates even if Aura fails to load

import os
import sys
import subprocess
import time
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QListWidget, QListWidgetItem, 
                             QMessageBox, QInputDialog, QLineEdit, QTextEdit,
                             QProgressBar, QTabWidget, QWidget, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import shutil

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from gui.base_dialog import BaseAuraDialog

# Import WiFi and OTA functionality from settings_dialog
try:
    from gui.settings_dialog import WiFiScanThread, WifiSettingsDialog, OTACheckThread, OTAUpdateThread, UpdateDialog
    SETTINGS_AVAILABLE = True
except ImportError as e:
    print(f"[SafeMode] ⚠️ Settings dialog not available: {e}")
    SETTINGS_AVAILABLE = False

class SafeModeDialog(BaseAuraDialog):
    """Safe Mode dialog that provides WiFi and OTA update access"""
    
    def __init__(self, parent=None):
        super().__init__(parent, title="🛡️ Safe Mode - WiFi & Updates", size=(1080, 1080), modal=True)
        print("[SafeMode] 🔧 Initializing safe mode dialog...")
        
        # Add additional styles to base stylesheet
        additional_styles = """
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
                font-size: 14px;
                font-weight: 600;
                padding: 10px 18px;
                min-height: 40px;
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
            QTabWidget::pane {
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                background-color: rgba(28, 28, 30, 0.9);
            }
            QTabBar::tab {
                background-color: rgba(44, 44, 46, 0.8);
                color: white;
                padding: 8px 16px;
                font-size: 13px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: rgba(70, 130, 180, 0.5);
            }
        """
        base_stylesheet = self.styleSheet()
        self.setStyleSheet(base_stylesheet + additional_styles)
        
        # Initialize threads
        self.wifi_scan_thread = None
        self.ota_check_thread = None
        self.ota_update_thread = None
        self._threads_initialized = False
        self._is_closing = False
    
    def _setup_ui(self):
        """Setup UI - called by BaseAuraDialog"""
        self.setup_ui()
    
    def _on_show(self):
        """Override for additional show logic - start threads after dialog is shown"""
        # Only start threads once, after dialog is fully shown
        if not self._threads_initialized and not self._is_closing:
            self._threads_initialized = True
            # Start WiFi scan and OTA check after dialog is shown
            QTimer.singleShot(500, self.scan_wifi)
            QTimer.singleShot(600, self.check_ota_updates)
    
    def closeEvent(self, event):
        """Override closeEvent to ensure we only close this dialog, not the parent"""
        # Set closing flag
        self._is_closing = True
        
        # Call cleanup
        try:
            self._on_close()
        except Exception as e:
            print(f"[SafeMode] ⚠️ Error in cleanup: {e}")
        
        # Accept the close event - this will close only this dialog
        # Don't call super().closeEvent() to avoid any parent dialog interactions
        event.accept()
        
        # Ensure parent is reactivated but not closed
        if self.parent():
            try:
                self.parent().raise_()
                self.parent().activateWindow()
                QApplication.processEvents()
            except (RuntimeError, AttributeError):
                pass
    
    def _on_close(self):
        """Cleanup when dialog closes"""
        # Set flag to prevent new threads from starting
        self._is_closing = True
        
        # Clean up WiFi scan thread
        if hasattr(self, 'wifi_scan_thread') and self.wifi_scan_thread:
            try:
                # Disconnect signals first
                if hasattr(self.wifi_scan_thread, 'networks_found'):
                    try:
                        self.wifi_scan_thread.networks_found.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                if hasattr(self.wifi_scan_thread, 'scan_error'):
                    try:
                        self.wifi_scan_thread.scan_error.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                if hasattr(self.wifi_scan_thread, 'finished'):
                    try:
                        self.wifi_scan_thread.finished.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                
                # Stop thread gracefully
                if self.wifi_scan_thread.isRunning():
                    self.wifi_scan_thread.quit()
                    if not self.wifi_scan_thread.wait(2000):  # Wait up to 2 seconds
                        # Force terminate if it doesn't stop
                        self.wifi_scan_thread.terminate()
                        self.wifi_scan_thread.wait(500)
            except (RuntimeError, AttributeError) as e:
                print(f"[SafeMode] ⚠️ Error cleaning up WiFi thread: {e}")
            except Exception as e:
                print(f"[SafeMode] ⚠️ Unexpected error cleaning up WiFi thread: {e}")
            finally:
                self.wifi_scan_thread = None
        
        # Clean up OTA check thread
        if hasattr(self, 'ota_check_thread') and self.ota_check_thread:
            try:
                if hasattr(self.ota_check_thread, 'check_complete'):
                    try:
                        self.ota_check_thread.check_complete.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                
                if self.ota_check_thread.isRunning():
                    self.ota_check_thread.quit()
                    if not self.ota_check_thread.wait(1000):
                        self.ota_check_thread.terminate()
                        self.ota_check_thread.wait(500)
            except (RuntimeError, AttributeError) as e:
                print(f"[SafeMode] ⚠️ Error cleaning up OTA check thread: {e}")
            except Exception as e:
                print(f"[SafeMode] ⚠️ Unexpected error cleaning up OTA check thread: {e}")
            finally:
                self.ota_check_thread = None
        
        # Clean up OTA update thread
        if hasattr(self, 'ota_update_thread') and self.ota_update_thread:
            try:
                # Disconnect signals first
                if hasattr(self.ota_update_thread, 'update_progress'):
                    try:
                        self.ota_update_thread.update_progress.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                if hasattr(self.ota_update_thread, 'update_complete'):
                    try:
                        self.ota_update_thread.update_complete.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                if hasattr(self.ota_update_thread, 'finished'):
                    try:
                        self.ota_update_thread.finished.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                
                # Stop thread gracefully
                if self.ota_update_thread.isRunning():
                    self.ota_update_thread.quit()
                    if not self.ota_update_thread.wait(2000):
                        self.ota_update_thread.terminate()
                        self.ota_update_thread.wait(500)
            except (RuntimeError, AttributeError) as e:
                print(f"[SafeMode] ⚠️ Error cleaning up OTA update thread: {e}")
            except Exception as e:
                print(f"[SafeMode] ⚠️ Unexpected error cleaning up OTA update thread: {e}")
            finally:
                self.ota_update_thread = None
    
    def setup_ui(self):
        """Setup the safe mode UI with tabs for WiFi and Updates"""
        main_layout = QVBoxLayout()
        # Use base class default margins to ensure content fits within white perimeter
        margins = BaseAuraDialog.get_default_margins()
        main_layout.setContentsMargins(*margins)
        main_layout.setSpacing(BaseAuraDialog.get_default_spacing())
        
        # Title
        title = QLabel("🛡️ Safe Mode")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        main_layout.addWidget(title)
        
        # Description
        description = QLabel("Access WiFi and OTA updates even if Aura fails to load")
        description.setFont(QFont("Arial", 12))
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("color: #aaaaaa; margin: 5px;")
        main_layout.addWidget(description)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # WiFi Tab
        wifi_tab = QWidget()
        wifi_layout = QVBoxLayout()
        wifi_layout.setContentsMargins(15, 15, 15, 15)
        wifi_layout.setSpacing(12)
        
        # WiFi status
        self.wifi_status_label = QLabel("Checking WiFi connection...")
        self.wifi_status_label.setFont(QFont("Arial", 12))
        self.wifi_status_label.setAlignment(Qt.AlignCenter)
        self.wifi_status_label.setStyleSheet("color: #ffa500; margin: 5px;")
        wifi_layout.addWidget(self.wifi_status_label)
        
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
        
        self.disconnect_wifi_btn = QPushButton("🔌 Disconnect")
        self.disconnect_wifi_btn.clicked.connect(self.disconnect_wifi)
        wifi_button_layout.addWidget(self.disconnect_wifi_btn)
        
        wifi_layout.addLayout(wifi_button_layout)
        
        # WiFi networks list - reduced height to fit within circular perimeter
        self.wifi_list = QListWidget()
        self.wifi_list.setMaximumHeight(220)  # Reduced from 250
        self.wifi_list.itemSelectionChanged.connect(self.on_wifi_selection_changed)
        wifi_layout.addWidget(self.wifi_list)
        
        wifi_tab.setLayout(wifi_layout)
        tabs.addTab(wifi_tab, "📶 WiFi")
        
        # OTA Updates Tab
        ota_tab = QWidget()
        ota_layout = QVBoxLayout()
        ota_layout.setContentsMargins(15, 15, 15, 15)
        ota_layout.setSpacing(12)
        
        # Status log - reduced height to fit within circular perimeter
        self.ota_status_log = QTextEdit()
        self.ota_status_log.setMaximumHeight(180)  # Reduced from 200
        self.ota_status_log.setReadOnly(True)
        self.ota_status_log.setStyleSheet("QTextEdit { background-color: rgba(44,44,46,0.8); color: #ffffff; border-radius: 15px; border: none; padding: 10px; font-size: 11px; }")
        ota_layout.addWidget(self.ota_status_log)
        
        # Update button
        self.update_btn = QPushButton("⬇️ Update from GitHub")
        self.update_btn.clicked.connect(self.start_ota_update)
        ota_layout.addWidget(self.update_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #555; border-radius: 8px; background-color: rgba(44,44,46,0.8); color: white; text-align: center; } QProgressBar::chunk { background-color: #007AFF; border-radius: 8px; }")
        ota_layout.addWidget(self.progress_bar)
        
        ota_tab.setLayout(ota_layout)
        tabs.addTab(ota_tab, "🔄 Updates")
        
        main_layout.addWidget(tabs)
        
        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_btn = QPushButton("❌ Close")
        # Simply close the dialog - closeEvent will handle cleanup
        # This ensures we return to the welcome dialog, not close the entire app
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 12px 30px;
                border-radius: 20px;
                border: none;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #D70015;
            }
            QPushButton:pressed {
                background-color: #B30000;
            }
        """)
        close_layout.addWidget(close_btn)
        close_layout.addStretch()
        main_layout.addLayout(close_layout)
        
        self.setLayout(main_layout)
        
        # Check WiFi connection status (synchronous, no thread)
        self.check_wifi_connection()
        
        # OTA check will be started in _on_show after dialog is shown
    
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
                        self.wifi_status_label.setText("✅ WiFi Connected")
                        self.wifi_status_label.setStyleSheet("color: #34C759; margin: 10px;")
                        self._update_disconnect_button()
                        return
                
            self.wifi_status_label.setText("❌ WiFi Not Connected")
            self.wifi_status_label.setStyleSheet("color: #FF3B30; margin: 10px;")
            self._update_disconnect_button()
        except Exception as e:
            print(f"[SafeMode] ⚠️ Error checking WiFi: {e}")
            self.wifi_status_label.setText("⚠️ Could not check WiFi status")
            self.wifi_status_label.setStyleSheet("color: #ffa500; margin: 10px;")
    
    def _update_disconnect_button(self):
        """Update disconnect button state"""
        try:
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'DEVICE,TYPE,STATE', 'device', 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            has_connected_wifi = False
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':wifi:' in line.lower() and ':connected' in line.lower():
                        has_connected_wifi = True
                        break
            
            self.disconnect_wifi_btn.setEnabled(has_connected_wifi)
        except Exception:
            self.disconnect_wifi_btn.setEnabled(False)
    
    def scan_wifi(self):
        """Scan for available WiFi networks"""
        # Check if dialog still exists or is closing
        if self._is_closing or not hasattr(self, 'wifi_status_label') or not hasattr(self, 'wifi_list'):
            return
        
        try:
            self.wifi_status_label.setText("🔍 Scanning for networks...")
            self.wifi_status_label.setStyleSheet("color: #ffa500; margin: 10px;")
            self.scan_wifi_btn.setEnabled(False)
            self.scan_wifi_btn.setText("🔄 Scanning...")
            self.wifi_list.clear()
            self.wifi_list.addItem("Scanning... Please wait...")
            
            # Clean up any existing thread first
            if self.wifi_scan_thread and self.wifi_scan_thread.isRunning():
                try:
                    self.wifi_scan_thread.quit()
                    self.wifi_scan_thread.wait(500)
                except Exception:
                    pass
            
            # Use WiFiScanThread from settings_dialog if available, otherwise use local implementation
            if SETTINGS_AVAILABLE:
                self.wifi_scan_thread = WiFiScanThread(self)  # Parent to dialog
            else:
                # Fallback to local WiFi scan implementation
                self.wifi_scan_thread = LocalWiFiScanThread()
                self.wifi_scan_thread.setParent(self)  # Parent to dialog
            
            self.wifi_scan_thread.networks_found.connect(self.on_wifi_networks_found)
            self.wifi_scan_thread.scan_error.connect(self.on_wifi_scan_error)
            self.wifi_scan_thread.finished.connect(self._on_scan_finished)
            self.wifi_scan_thread.start()
        except (RuntimeError, AttributeError) as e:
            print(f"[SafeMode] ⚠️ Error starting WiFi scan: {e}")
            if hasattr(self, 'wifi_status_label'):
                self.wifi_status_label.setText("❌ Scan failed")
                self.wifi_status_label.setStyleSheet("color: #FF3B30; margin: 10px;")
            if hasattr(self, 'scan_wifi_btn'):
                self.scan_wifi_btn.setEnabled(True)
                self.scan_wifi_btn.setText("🔍 Scan Networks")
    
    def _on_scan_finished(self):
        """Handle WiFi scan thread finished"""
        try:
            if hasattr(self, 'scan_wifi_btn') and self.scan_wifi_btn is not None:
                self.scan_wifi_btn.setEnabled(True)
                self.scan_wifi_btn.setText("🔍 Scan Networks")
        except (RuntimeError, AttributeError):
            pass
    
    def on_wifi_networks_found(self, networks):
        """Handle WiFi networks found"""
        try:
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
            pass
    
    def on_wifi_scan_error(self, error):
        """Handle WiFi scan error"""
        try:
            if not hasattr(self, 'wifi_list'):
                return
            self.wifi_list.clear()
            self.wifi_list.addItem(f"Error: {error}")
            QMessageBox.warning(self, "WiFi Scan Error", error)
            self.check_wifi_connection()
        except (RuntimeError, AttributeError):
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
            try:
                from gui.password_keyboard import PasswordKeyboard
                keyboard = PasswordKeyboard(
                    parent=self, 
                    initial_text="",
                    title=f"WiFi Password - {ssid}"
                )
                if keyboard.exec_() == QDialog.Accepted:
                    password = keyboard.get_text()
                    if not password:
                        QMessageBox.warning(self, "No Password", "Password cannot be empty")
                        return
                else:
                    return
            except ImportError:
                password, ok = QInputDialog.getText(
                    self,
                    "WiFi Password",
                    f"Enter password for {ssid}:",
                    QLineEdit.Password
                )
                if not ok:
                    return
        
        self.wifi_status_label.setText(f"🔗 Connecting to {ssid}...")
        self.wifi_status_label.setStyleSheet("color: #ffa500; margin: 10px;")
        self.connect_wifi_btn.setEnabled(False)
        self.connect_wifi_btn.setText("🔄 Connecting...")
        
        # Connect using nmcli
        try:
            base_cmd = ['nmcli', 'device', 'wifi', 'connect', ssid]
            if password:
                base_cmd += ['password', password]

            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                error_msg = (result.stderr or result.stdout or "").strip()
                needs_priv = any(term in error_msg.lower() for term in [
                    "permission denied", "not authorized", "authorization failed", "polkit", 
                    "not permitted", "insufficient privileges", "unable to create contextual"
                ])

                if needs_priv:
                    try:
                        add_cmd = ['nmcli', 'connection', 'add', 'type', 'wifi', 'con-name', ssid, 'ssid', ssid]
                        if password:
                            add_cmd += ['wifi-sec.key-mgmt', 'wpa-psk', 'wifi-sec.psk', password]
                        else:
                            add_cmd += ['wifi-sec.key-mgmt', 'none']
                        
                        add_result = subprocess.run(add_cmd, capture_output=True, text=True, timeout=30)
                        
                        if add_result.returncode == 0:
                            activate_cmd = ['nmcli', 'connection', 'up', ssid]
                            activate_result = subprocess.run(activate_cmd, capture_output=True, text=True, timeout=30)
                            
                            if activate_result.returncode == 0:
                                result = activate_result
                                error_msg = ""
                            else:
                                if shutil.which("pkexec"):
                                    pkexec_cmd = ['pkexec'] + base_cmd
                                    result2 = subprocess.run(pkexec_cmd, capture_output=True, text=True, timeout=60)
                                    if result2.returncode == 0:
                                        result = result2
                                        error_msg = ""
                                    else:
                                        error_msg = (result2.stderr or result2.stdout or activate_result.stderr or error_msg).strip()
                                else:
                                    error_msg = (activate_result.stderr or activate_result.stdout or error_msg).strip()
                        else:
                            if shutil.which("pkexec"):
                                pkexec_cmd = ['pkexec'] + base_cmd
                                result2 = subprocess.run(pkexec_cmd, capture_output=True, text=True, timeout=60)
                                if result2.returncode == 0:
                                    result = result2
                                    error_msg = ""
                                else:
                                    error_msg = (result2.stderr or result2.stdout or add_result.stderr or error_msg).strip()
                            else:
                                error_msg = (add_result.stderr or add_result.stdout or error_msg).strip()
                    except Exception as e:
                        if shutil.which("pkexec"):
                            try:
                                pkexec_cmd = ['pkexec'] + base_cmd
                                result2 = subprocess.run(pkexec_cmd, capture_output=True, text=True, timeout=60)
                                if result2.returncode == 0:
                                    result = result2
                                    error_msg = ""
                                else:
                                    error_msg = (result2.stderr or result2.stdout or str(e) or error_msg).strip()
                            except:
                                error_msg = (str(e) or error_msg).strip()
                        else:
                            error_msg = (str(e) or error_msg).strip()

                if result.returncode != 0:
                    self.wifi_status_label.setText("❌ Connection failed")
                    self.wifi_status_label.setStyleSheet("color: #FF3B30; margin: 10px;")
                    QMessageBox.warning(self, "Connection Failed", error_msg or "Unknown error")
                    self.check_wifi_connection()
                    return

            # Success
            self.wifi_status_label.setText(f"✅ Connected to {ssid}")
            self.wifi_status_label.setStyleSheet("color: #34C759; margin: 10px;")
            QMessageBox.information(self, "Success", f"Connected to {ssid}")
            
            time.sleep(2)
            self.check_wifi_connection()
            self.scan_wifi()
        except Exception as e:
            self.wifi_status_label.setText("❌ Connection error")
            self.wifi_status_label.setStyleSheet("color: #FF3B30; margin: 10px;")
            QMessageBox.warning(self, "Error", str(e))
            self.check_wifi_connection()
        finally:
            self.connect_wifi_btn.setText("🔗 Connect")
            self.connect_wifi_btn.setEnabled(len(self.wifi_list.selectedItems()) > 0)
    
    def disconnect_wifi(self):
        """Disconnect from current WiFi network"""
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
                        device = line.split(':')[0]
                        if device:
                            disconnect_result = subprocess.run(
                                ['nmcli', 'device', 'disconnect', device],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if disconnect_result.returncode == 0:
                                self.wifi_status_label.setText("🔌 WiFi Disconnected")
                                self.wifi_status_label.setStyleSheet("color: #ffa500; margin: 10px;")
                                QMessageBox.information(self, "Success", "WiFi disconnected")
                                self.check_wifi_connection()
                                self.scan_wifi()
                            else:
                                QMessageBox.warning(self, "Error", "Failed to disconnect WiFi")
                            return
            
            QMessageBox.information(self, "Info", "No WiFi connection to disconnect")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error disconnecting WiFi: {str(e)}")
    
    def check_ota_updates(self):
        """Check for OTA updates"""
        # Check if dialog still exists or is closing
        if self._is_closing or not hasattr(self, 'ota_status_log'):
            return
        
        try:
            # Clean up any existing thread first
            if self.ota_check_thread and self.ota_check_thread.isRunning():
                try:
                    self.ota_check_thread.quit()
                    self.ota_check_thread.wait(500)
                except Exception:
                    pass
            
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            dotenv_path = os.path.join(workspace_root, '.env')
            github_token = None
            try:
                from dotenv import dotenv_values
                env_vars = dotenv_values(dotenv_path)
                github_token = env_vars.get('GITHUB_TOKEN', '')
                if not github_token or github_token == 'your_github_token_here':
                    github_token = None
            except Exception:
                pass
            
            repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            
            if SETTINGS_AVAILABLE:
                self.ota_check_thread = OTACheckThread(repo_path, github_token or '')
            else:
                # Fallback implementation
                self.ota_check_thread = LocalOTACheckThread(repo_path, github_token or '')
            
            # Parent thread to dialog
            self.ota_check_thread.setParent(self)
            self.ota_check_thread.check_complete.connect(self._on_ota_check_complete)
            self.ota_check_thread.start()
        except (RuntimeError, AttributeError) as e:
            print(f"[SafeMode] ⚠️ Error starting OTA check: {e}")
            self._log_ota(f"⚠️ Error checking for updates: {e}")
        except Exception as e:
            print(f"[SafeMode] ⚠️ Unexpected error in OTA check: {e}")
            self._log_ota(f"⚠️ Error checking for updates: {e}")
    
    def _on_ota_check_complete(self, has_updates, commits_behind):
        """Update button text based on update availability"""
        if has_updates:
            self.update_btn.setText(f"🔄 Update Available ({commits_behind} commits) - Update Now")
            self.update_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9500;
                    color: white;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 12px 24px;
                    border-radius: 15px;
                    border: none;
                    min-width: 180px;
                }
                QPushButton:hover {
                    background-color: #E58500;
                }
                QPushButton:pressed {
                    background-color: #CC7500;
                }
            """)
            self._log_ota(f"ℹ️ {commits_behind} update(s) available")
        else:
            self.update_btn.setText("⬇️ Update from GitHub")
            self.update_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(70, 130, 180, 0.25);
                    color: #ffffff;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 12px 24px;
                    border-radius: 15px;
                    border: none;
                    min-width: 180px;
                }
                QPushButton:hover {
                    background-color: rgba(70, 130, 180, 0.45);
                }
                QPushButton:pressed {
                    background-color: rgba(70, 130, 180, 0.65);
                }
            """)
            self._log_ota("ℹ️ System is up to date")
    
    def start_ota_update(self):
        """Start OTA update process"""
        # Check if dialog is closing
        if self._is_closing:
            return
        
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        dotenv_path = os.path.join(workspace_root, '.env')
        github_token = None
        try:
            from dotenv import dotenv_values
            env_vars = dotenv_values(dotenv_path)
            github_token = env_vars.get('GITHUB_TOKEN', '')
            if not github_token or github_token == 'your_github_token_here':
                github_token = None
        except Exception:
            pass
        
        repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self._log_ota("Starting OTA update...")
        self.update_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # Clean up any existing thread first
        if self.ota_update_thread and self.ota_update_thread.isRunning():
            try:
                self.ota_update_thread.quit()
                self.ota_update_thread.wait(500)
            except Exception:
                pass
        
        if SETTINGS_AVAILABLE:
            self.ota_update_thread = OTAUpdateThread(repo_path, github_token or '')
        else:
            # Fallback implementation
            self.ota_update_thread = LocalOTAUpdateThread(repo_path, github_token or '')
        
        # Parent thread to dialog
        self.ota_update_thread.setParent(self)
        self.ota_update_thread.update_progress.connect(lambda m: self._log_ota(m))
        self.ota_update_thread.update_complete.connect(self._on_ota_update_complete)
        self.ota_update_thread.finished.connect(lambda: (self.update_btn.setEnabled(True), self.progress_bar.setVisible(False)))
        self.ota_update_thread.start()
    
    def _on_ota_update_complete(self, success, message, was_updated):
        """Handle OTA update completion"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        if success:
            self._log_ota(f"✅ {message}")
            if was_updated:
                self._log_ota("🔄 Restarting Aura system...")
                try:
                    result = subprocess.run(
                        ['systemctl', 'is-active', '--quiet', 'aura.service'],
                        capture_output=True,
                        timeout=2
                    )
                    is_systemd_service = (result.returncode == 0)
                    
                    if is_systemd_service:
                        try:
                            subprocess.run(['systemctl', 'restart', 'aura.service'], 
                                         capture_output=True, timeout=5)
                            self._log_ota("✅ Service restart initiated")
                        except:
                            try:
                                subprocess.run(['sudo', 'systemctl', 'restart', 'aura.service'],
                                             capture_output=True, timeout=10)
                                self._log_ota("✅ Service restart initiated (with sudo)")
                            except:
                                self._log_ota("⚠️ Could not restart service automatically")
                                QMessageBox.information(
                                    self,
                                    "Update Complete",
                                    f"{message}\n\nPlease restart Aura manually."
                                )
                    else:
                        QMessageBox.information(
                            self,
                            "Update Complete",
                            f"{message}\n\nPlease restart Aura manually."
                        )
                except Exception as e:
                    self._log_ota(f"⚠️ Error restarting: {e}")
                    QMessageBox.information(
                        self,
                        "Update Complete",
                        f"{message}\n\nPlease restart Aura manually."
                    )
            else:
                QMessageBox.information(self, "Update Check", message)
                self.check_ota_updates()
        else:
            self._log_ota(f"❌ {message}")
            QMessageBox.warning(self, "Update Failed", message)
    
    def _log_ota(self, message):
        """Log message to OTA status log"""
        try:
            if hasattr(self, 'ota_status_log'):
                self.ota_status_log.append(f"[Update] {message}")
        except (RuntimeError, AttributeError):
            pass


# Fallback implementations if settings_dialog is not available
class LocalWiFiScanThread(QThread):
    """Local WiFi scan thread implementation"""
    networks_found = pyqtSignal(list)
    scan_error = pyqtSignal(str)
    
    def run(self):
        """Scan for available WiFi networks"""
        try:
            subprocess.run(['nmcli', 'device', 'wifi', 'rescan'], 
                         capture_output=True, timeout=10)
            time.sleep(3)
            
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


class LocalOTACheckThread(QThread):
    """Local OTA check thread implementation"""
    check_complete = pyqtSignal(bool, int)  # has_updates, commits_behind
    
    def __init__(self, repo_path, github_token=''):
        super().__init__()
        self.repo_path = repo_path
        self.github_token = github_token
    
    def run(self):
        """Check for OTA updates"""
        try:
            if not os.path.exists(self.repo_path):
                self.check_complete.emit(False, 0)
                return
            
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
            
            # Configure git remote if token provided
            if self.github_token:
                result = subprocess.run(
                    ['git', 'remote', 'get-url', 'origin'],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    original_url = result.stdout.strip()
                    remote_url = original_url
                    
                    if remote_url.startswith('git@'):
                        remote_url = remote_url.replace('git@github.com:', 'https://github.com/')
                    
                    import re
                    remote_url = re.sub(r'ghp_[A-Za-z0-9]{36,}@', '', remote_url)
                    remote_url = re.sub(r'https://[^@]+@github\.com', 'https://github.com', remote_url)
                    
                    if remote_url.startswith('https://github.com') and self.github_token not in remote_url:
                        url_parts = remote_url.split('://', 1)
                        if len(url_parts) == 2:
                            remote_url = f"{url_parts[0]}://{self.github_token}@{url_parts[1]}"
                            subprocess.run(
                                ['git', 'remote', 'set-url', 'origin', remote_url],
                                cwd=self.repo_path,
                                capture_output=True
                            )
            
            subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'HEAD..origin/{current_branch}'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            commits_behind = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            has_updates = commits_behind > 0
            
            self.check_complete.emit(has_updates, commits_behind)
            
        except Exception as e:
            print(f"[SafeMode] ⚠️ Error checking for updates: {e}")
            self.check_complete.emit(False, 0)


class LocalOTAUpdateThread(QThread):
    """Local OTA update thread implementation"""
    update_progress = pyqtSignal(str)
    update_complete = pyqtSignal(bool, str, bool)  # success, message, was_updated
    
    def __init__(self, repo_path, github_token=''):
        super().__init__()
        self.repo_path = repo_path
        self.github_token = github_token
    
    def run(self):
        """Perform OTA update"""
        try:
            if not os.path.exists(self.repo_path):
                self.update_complete.emit(False, "Repository path not found", False)
                return
            
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
            
            # Configure git remote if token provided
            if self.github_token:
                result = subprocess.run(
                    ['git', 'remote', 'get-url', 'origin'],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    original_url = result.stdout.strip()
                    remote_url = original_url
                    
                    if remote_url.startswith('git@'):
                        remote_url = remote_url.replace('git@github.com:', 'https://github.com/')
                    
                    import re
                    remote_url = re.sub(r'ghp_[A-Za-z0-9]{36,}@', '', remote_url)
                    remote_url = re.sub(r'https://[^@]+@github\.com', 'https://github.com', remote_url)
                    
                    if remote_url.startswith('https://github.com') and self.github_token not in remote_url:
                        url_parts = remote_url.split('://', 1)
                        if len(url_parts) == 2:
                            remote_url = f"{url_parts[0]}://{self.github_token}@{url_parts[1]}"
                            subprocess.run(
                                ['git', 'remote', 'set-url', 'origin', remote_url],
                                cwd=self.repo_path,
                                capture_output=True
                            )
            
            self.update_progress.emit("Fetching latest changes...")
            result = subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.update_complete.emit(False, f"Fetch failed: {result.stderr}", False)
                return
            
            self.update_progress.emit("Checking for updates...")
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'HEAD..origin/{current_branch}'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            commits_behind = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            
            if commits_behind == 0:
                self.update_complete.emit(True, "Already up to date", False)
                return
            
            self.update_progress.emit(f"Found {commits_behind} new commits. Updating...")
            
            result = subprocess.run(
                ['git', 'pull', 'origin', current_branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                self.update_complete.emit(False, f"Pull failed: {result.stderr}", False)
                return
            
            self.update_progress.emit("Update complete!")
            self.update_complete.emit(True, f"Successfully updated {commits_behind} commits", True)
            
        except subprocess.TimeoutExpired:
            self.update_complete.emit(False, "Update timed out", False)
        except Exception as e:
            self.update_complete.emit(False, f"Error: {str(e)}", False)

