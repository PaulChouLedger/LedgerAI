# settings_dialog.py — Settings Dialog for WiFi and OTA Updates

import os
import sys
import subprocess
import json
import re
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QProgressBar,
                             QMessageBox, QListWidget, QListWidgetItem, QInputDialog,
                             QLineEdit, QWidget, QApplication, QGridLayout, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
import threading
from dotenv import dotenv_values
import glob

class WiFiScanThread(QThread):
    """Thread to scan for WiFi networks without blocking UI"""
    networks_found = pyqtSignal(list)
    scan_error = pyqtSignal(str)
    
    def run(self):
        """Scan for available WiFi networks"""
        import time
        
        try:
            # Strategy 1: Try to trigger a fresh scan with all available methods
            # Method 1a: Standard rescan
            try:
                scan_result = subprocess.run(
                    ['nmcli', 'device', 'wifi', 'rescan'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if scan_result.returncode == 0:
                    time.sleep(5)  # Increased wait time for scan to complete
            except:
                pass  # Continue even if rescan fails
            
            # Method 1b: Force rescan on all WiFi devices
            try:
                # Get list of WiFi devices first
                device_result = subprocess.run(
                    ['nmcli', '-t', '-f', 'DEVICE,TYPE', 'device', 'status'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if device_result.returncode == 0:
                    for line in device_result.stdout.strip().split('\n'):
                        if ':wifi' in line.lower():
                            device = line.split(':')[0]
                            if device:
                                subprocess.run(
                                    ['nmcli', 'device', 'wifi', 'rescan', 'ifname', device],
                                    capture_output=True,
                                    timeout=5
                                )
                    time.sleep(5)  # Wait for all scans to complete
            except:
                pass  # Continue even if device-specific rescan fails
            
            # Strategy 2: Get networks with multiple approaches to ensure we get all
            all_networks = []
            
            # Approach 2a: Standard list (includes recently scanned networks)
            try:
                result = subprocess.run(
                    ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY,IN-USE', 'device', 'wifi', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=25
                )
                if result.returncode == 0 and result.stdout:
                    all_networks.extend(result.stdout.strip().split('\n'))
            except:
                pass
            
            # Approach 2b: Force fresh scan and list
            try:
                result_fresh = subprocess.run(
                    ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY,IN-USE', 'device', 'wifi', 'list', '--rescan', 'yes'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result_fresh.returncode == 0 and result_fresh.stdout:
                    all_networks.extend(result_fresh.stdout.strip().split('\n'))
            except:
                pass
            
            # Approach 2c: Fallback - use cached results if fresh scan fails
            if not all_networks:
                try:
                    result_cached = subprocess.run(
                        ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY,IN-USE', 'device', 'wifi', 'list', '--rescan', 'no'],
                        capture_output=True,
                        text=True,
                        timeout=15
                    )
                    if result_cached.returncode == 0 and result_cached.stdout:
                        all_networks.extend(result_cached.stdout.strip().split('\n'))
                except:
                    pass
            
            # If still no networks, try alternative format
            if not all_networks:
                try:
                    result_alt = subprocess.run(
                        ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY', 'device', 'wifi', 'list'],
                        capture_output=True,
                        text=True,
                        timeout=20
                    )
                    if result_alt.returncode == 0 and result_alt.stdout:
                        all_networks.extend(result_alt.stdout.strip().split('\n'))
                except:
                    pass
            
            if not all_networks:
                self.scan_error.emit("No WiFi networks found. Check WiFi is enabled and try again.")
                return
            
            # Parse all network entries
            networks = []
            seen_ssids = set()  # Track seen SSIDs to avoid duplicates
            
            for line in all_networks:
                if not line or ':' not in line:
                    continue
                
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                
                ssid = parts[0].strip() if parts[0] else ""
                signal = parts[1].strip() if len(parts) > 1 else "0"
                security = parts[2].strip() if len(parts) > 2 else "Open"
                in_use = parts[3].strip() if len(parts) > 3 else ""
                
                # Skip empty SSIDs or placeholder values
                if not ssid or ssid == "(none)" or ssid == "none" or ssid == "--":
                    continue
                
                # Skip if already seen (to avoid duplicates)
                if ssid in seen_ssids:
                    continue
                
                try:
                    signal_int = int(signal) if signal.isdigit() else 0
                    # Include ALL networks, even with weak signal (0% or low)
                    # This ensures we show all available networks
                    seen_ssids.add(ssid)
                    
                    networks.append({
                        'ssid': ssid,
                        'signal': signal_int,
                        'security': security if security else "Open",
                        'connected': in_use == '*'
                    })
                except Exception as e:
                    # Skip invalid entries
                    continue
            
            # Sort by signal strength (strongest first), then by SSID
            # Include all networks regardless of signal strength
            networks.sort(key=lambda x: (x['signal'], x['ssid']), reverse=True)
            
            if networks:
                self.networks_found.emit(networks)
            else:
                self.scan_error.emit("Found network entries but none were valid. Check WiFi permissions.")
            
            
        except subprocess.TimeoutExpired:
            self.scan_error.emit("WiFi scan timed out")
        except FileNotFoundError:
            self.scan_error.emit("nmcli not found. Install NetworkManager.")
        except Exception as e:
            self.scan_error.emit(f"Scan error: {str(e)}")

class OTAUpdateThread(QThread):
    """Thread to perform OTA update without blocking UI"""
    update_progress = pyqtSignal(str)
    update_complete = pyqtSignal(bool, str)
    
    def __init__(self, repo_path, github_token):
        super().__init__()
        self.repo_path = repo_path
        self.github_token = github_token
    
    def run(self):
        """Perform git pull with token authentication"""
        try:
            self.update_progress.emit("Checking repository status...")
            
            # Check if we're in a git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.update_complete.emit(False, "Not a git repository")
                return
            
            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
            
            self.update_progress.emit(f"Current branch: {current_branch}")
            
            # Configure git to use token for authentication
            # Set remote URL with token if token provided
            if self.github_token:
                self.update_progress.emit("Configuring authentication...")
                # Get remote URL
                result = subprocess.run(
                    ['git', 'remote', 'get-url', 'origin'],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    original_url = result.stdout.strip()
                    remote_url = original_url
                    
                    # Convert SSH to HTTPS if needed
                    if remote_url.startswith('git@'):
                        # Convert git@github.com:user/repo.git to https://github.com/user/repo.git
                        remote_url = remote_url.replace('git@github.com:', 'https://github.com/')
                    
                    # Clean the URL - remove ALL existing credentials and tokens
                    # Handle cases where token appears multiple times (https://TOKEN@TOKEN@github.com)
                    import re
                    
                    # Remove any tokens in the URL (ghp_* pattern)
                    remote_url = re.sub(r'ghp_[A-Za-z0-9]{36,}@', '', remote_url)
                    
                    # Remove any credentials before github.com (everything between :// and @github.com)
                    remote_url = re.sub(r'https://[^@]+@github\.com', 'https://github.com', remote_url)
                    
                    # If still has @ symbols, take everything after the last @
                    if '@' in remote_url:
                        parts = remote_url.split('@')
                        if len(parts) > 1:
                            # Take the last part (after @) which should be the host/path
                            host_path = parts[-1]
                            # Ensure it starts with https://
                            if not host_path.startswith('https://'):
                                if host_path.startswith('http://'):
                                    host_path = host_path.replace('http://', 'https://')
                                elif host_path.startswith('github.com'):
                                    host_path = f"https://{host_path}"
                                else:
                                    host_path = f"https://{host_path}"
                            remote_url = host_path
                    
                    # Ensure it's a proper HTTPS URL
                    if not remote_url.startswith('https://'):
                        # Try to extract repo path from various formats
                        if 'github.com' in remote_url:
                            # Extract user/repo from various formats
                            if ':' in remote_url:
                                # Format: github.com:user/repo.git
                                repo_part = remote_url.split(':', 1)[1]
                                remote_url = f"https://github.com/{repo_part}"
                            else:
                                # Format: github.com/user/repo
                                if '/' in remote_url:
                                    remote_url = f"https://{remote_url}" if not remote_url.startswith('http') else remote_url
                    
                    # Remove trailing slash if present
                    if remote_url.endswith('/'):
                        remote_url = remote_url[:-1]
                    
                    # Verify URL is clean (no tokens, no duplicate @)
                    if '@' in remote_url and 'github.com' in remote_url:
                        # Still has @, clean it again
                        remote_url = re.sub(r'https://[^@]+@github\.com', 'https://github.com', remote_url)
                    
                    # Now add token to clean URL (only if URL is clean and doesn't already have token)
                    if remote_url.startswith('https://github.com') and self.github_token not in remote_url:
                        # Split URL: https://github.com/user/repo.git
                        url_parts = remote_url.split('://', 1)
                        if len(url_parts) == 2:
                            # Format: https://TOKEN@github.com/user/repo.git
                            remote_url = f"{url_parts[0]}://{self.github_token}@{url_parts[1]}"
                            
                            # Temporarily set remote URL with token
                            subprocess.run(
                                ['git', 'remote', 'set-url', 'origin', remote_url],
                                cwd=self.repo_path,
                                capture_output=True
                            )
                            
                            self.update_progress.emit("Authentication configured")
            
            self.update_progress.emit("Fetching latest changes...")
            result = subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                # Mask token in error messages
                error_msg = self._mask_token(result.stderr, self.github_token)
                self.update_complete.emit(False, f"Fetch failed: {error_msg}")
                return
            
            self.update_progress.emit("Checking for updates...")
            # Check if there are updates
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'HEAD..origin/{current_branch}'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            commits_behind = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            
            if commits_behind == 0:
                self.update_complete.emit(True, "Already up to date")
                return
            
            self.update_progress.emit(f"Found {commits_behind} new commits. Updating...")
            
            # Pull changes
            result = subprocess.run(
                ['git', 'pull', 'origin', current_branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                # Mask token in error messages
                error_msg = self._mask_token(result.stderr, self.github_token)
                self.update_complete.emit(False, f"Pull failed: {error_msg}")
                return
            
            self.update_progress.emit("Update complete!")
            self.update_complete.emit(True, f"Successfully updated {commits_behind} commits")
            
        except subprocess.TimeoutExpired:
            self.update_complete.emit(False, "Update timed out")
        except Exception as e:
            # Mask token in exception messages
            error_msg = self._mask_token(str(e), self.github_token)
            self.update_complete.emit(False, f"Update error: {error_msg}")
    
    def _mask_token(self, text, token=None):
        """Mask GitHub token in text to prevent exposure in error messages"""
        if not text:
            return text
        
        masked_text = text
        
        # Mask specific token if provided
        if token:
            masked_text = masked_text.replace(token, "***")
        
        # Also mask common token patterns (ghp_*, ghp_*@, etc.)
        import re
        # Pattern for GitHub tokens (ghp_ followed by alphanumeric, 36+ chars)
        token_pattern = r'ghp_[A-Za-z0-9]{36,}'
        masked_text = re.sub(token_pattern, '***', masked_text)
        
        # Mask URLs with tokens (https://TOKEN@github.com or https://TOKEN@TOKEN@github.com)
        url_pattern = r'https://[^@\s]+@[^@\s]+@github\.com'
        masked_text = re.sub(url_pattern, 'https://***@github.com', masked_text)
        
        # Also catch single token in URL
        url_pattern2 = r'https://ghp_[A-Za-z0-9]{36,}@github\.com'
        masked_text = re.sub(url_pattern2, 'https://***@github.com', masked_text)
        
        # Generic pattern for any token-looking string in URL
        url_pattern3 = r'https://[A-Za-z0-9_]{20,}@github\.com'
        masked_text = re.sub(url_pattern3, 'https://***@github.com', masked_text)
        
        return masked_text

# === Sub-Dialogs (WiFi / Updates / AI Model) ===

class WifiSettingsDialog(QDialog):
    """Dedicated dialog for WiFi scanning/connect/disconnect"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WiFi Settings")
        self.setFixedSize(1080, 1080)
        if parent:
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("""
            QDialog { background-color: rgba(28,28,30,1.0); color: white; border: none; border-radius: 536px; }
            QLabel { color: white; font-size: 12px; }
        """)
        self.setWindowOpacity(0.0)
        self._setup_ui()
        self._center_dialog()
        self.wifi_scan_thread = None
        self.selected_wifi = None
        self._update_disconnect_button()
    
    def showEvent(self, event):
        super().showEvent(event)
        self.fade_in = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in.setDuration(300)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setEasingCurve(QEasingCurve.InOutCubic)
        self.fade_in.start()
        self.raise_(); self.activateWindow()
    
    def closeEvent(self, event):
        if hasattr(self, 'fade_in') and self.fade_in.state() == QPropertyAnimation.Running:
            self.fade_in.stop()
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(250)
        self.fade_out.setStartValue(self.windowOpacity())
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.InOutCubic)
        self.fade_out.finished.connect(lambda: event.accept())
        self.fade_out.start()
        event.ignore()
    
    def _center_dialog(self):
        if self.parent():
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)
        else:
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
    
    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(120, 100, 120, 100)
        main_layout.setSpacing(20)
        main_layout.addStretch(1)
        
        title = QLabel("📶 WiFi Settings")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        main_layout.addWidget(title)
        
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(100)
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("QTextEdit { background-color: rgba(44,44,46,0.8); color: #ffffff; border-radius: 15px; border: none; padding: 10px; font-size: 11px; }")
        main_layout.addWidget(self.status_log)
        
        wifi_button_layout = QHBoxLayout()
        wifi_button_layout.setSpacing(15)
        self.scan_wifi_btn = QPushButton("🔍 Scan Networks")
        self.scan_wifi_btn.setStyleSheet(SettingsDialog.get_button_style(None))
        self.scan_wifi_btn.clicked.connect(self._scan_wifi)
        wifi_button_layout.addWidget(self.scan_wifi_btn)
        
        self.connect_wifi_btn = QPushButton("🔗 Connect")
        self.connect_wifi_btn.setStyleSheet(SettingsDialog.get_button_style(None))
        self.connect_wifi_btn.clicked.connect(self._connect_wifi)
        self.connect_wifi_btn.setEnabled(False)
        wifi_button_layout.addWidget(self.connect_wifi_btn)
        
        self.disconnect_wifi_btn = QPushButton("🔌 Disconnect")
        self.disconnect_wifi_btn.setStyleSheet(SettingsDialog.get_button_style(None))
        self.disconnect_wifi_btn.clicked.connect(self._disconnect_wifi)
        wifi_button_layout.addWidget(self.disconnect_wifi_btn)
        
        main_layout.addLayout(wifi_button_layout)
        
        self.wifi_list = QListWidget()
        self.wifi_list.setMaximumHeight(180)
        self.wifi_list.setStyleSheet("""
            QListWidget { background-color: rgba(44,44,46,0.8); color: #ffffff; border-radius: 15px; border: none; padding: 10px; font-size: 12px; }
            QListWidget::item { padding: 8px; border-radius: 8px; margin: 2px; }
            QListWidget::item:selected { background-color: rgba(0,122,255,0.3); }
        """)
        self.wifi_list.itemSelectionChanged.connect(self._on_wifi_selection_changed)
        main_layout.addWidget(self.wifi_list)
        
        main_layout.addStretch(1)
        self.setLayout(main_layout)
    
    def _log(self, message): self.status_log.append(f"[WiFi] {message}")
    
    def _scan_wifi(self):
        self._log("Scanning for WiFi networks...")
        self.scan_wifi_btn.setEnabled(False)
        self.scan_wifi_btn.setText("🔄 Scanning...")
        self.wifi_list.clear()
        self.wifi_list.addItem("Scanning... Please wait...")
        self.wifi_scan_thread = WiFiScanThread()
        self.wifi_scan_thread.networks_found.connect(self._on_wifi_networks_found)
        self.wifi_scan_thread.scan_error.connect(self._on_wifi_scan_error)
        self.wifi_scan_thread.finished.connect(lambda: (self.scan_wifi_btn.setEnabled(True), self.scan_wifi_btn.setText("🔍 Scan Networks")))
        self.wifi_scan_thread.start()
    
    def _on_wifi_networks_found(self, networks):
        self.wifi_list.clear()
        if not networks:
            self._log("No networks found. Make sure WiFi is enabled.")
            self.wifi_list.addItem("No networks found")
            return
        self._log(f"Found {len(networks)} network(s)")
        for network in networks:
            ssid = network['ssid']; signal = network['signal']; security = network['security']; connected = network.get('connected', False)
            signal_display = f"{signal}%" if signal > 0 else "weak"
            item_text = f"{ssid} ({signal_display})"
            if connected: item_text = f"● {item_text} (Connected)"
            if security and security != "Open" and security != "--": item_text += f" 🔒 {security}"
            item = QListWidgetItem(item_text); item.setData(Qt.UserRole, network); self.wifi_list.addItem(item)
        self._update_disconnect_button()
    
    def _on_wifi_scan_error(self, error):
        self._log(f"Error: {error}")
        QMessageBox.warning(self, "WiFi Scan Error", error)
    
    def _on_wifi_selection_changed(self):
        self.connect_wifi_btn.setEnabled(len(self.wifi_list.selectedItems()) > 0)
    
    def _connect_wifi(self):
        selected_items = self.wifi_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a WiFi network"); return
        item = selected_items[0]; network = item.data(Qt.UserRole); ssid = network['ssid']; security = network['security']
        password = None
        if security and security != "Open" and "Open" not in security:
            password, ok = QInputDialog.getText(self, "WiFi Password", f"Enter password for {ssid}:", QLineEdit.Password)
            if not ok: return
        self._log(f"Connecting to {ssid}...")
        try:
            cmd = ['nmcli', 'device', 'wifi', 'connect', ssid] + (['password', password] if password else [])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self._log(f"Successfully connected to {ssid}")
                QMessageBox.information(self, "Success", f"Connected to {ssid}")
                self._update_disconnect_button(); self._scan_wifi()
            else:
                error_msg = result.stderr or result.stdout
                self._log(f"Connection failed: {error_msg}")
                QMessageBox.warning(self, "Connection Failed", error_msg)
        except Exception as e:
            self._log(f"Error: {str(e)}"); QMessageBox.warning(self, "Error", str(e))
    
    def _update_disconnect_button(self):
        try:
            result = subprocess.run(['nmcli', '-t', '-f', 'DEVICE,TYPE,STATE', 'device', 'status'], capture_output=True, text=True, timeout=5)
            wifi_connected = False
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':wifi:' in line.lower() and ':connected' in line.lower(): wifi_connected = True; break
            self.disconnect_wifi_btn.setEnabled(wifi_connected)
        except Exception:
            self.disconnect_wifi_btn.setEnabled(False)
    
    def _disconnect_wifi(self):
        reply = QMessageBox.question(self, "Disconnect WiFi", "Disconnect from the current WiFi network?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self._log("Disconnecting from WiFi..."); self.disconnect_wifi_btn.setEnabled(False); self.disconnect_wifi_btn.setText("🔄 Disconnecting...")
        try:
            result = subprocess.run(['nmcli', '-t', '-f', 'DEVICE,TYPE', 'device', 'status'], capture_output=True, text=True, timeout=5)
            wifi_device = None
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':wifi:' in line.lower(): wifi_device = line.split(':')[0]; break
            if wifi_device:
                disconnect_result = subprocess.run(['nmcli', 'device', 'disconnect', wifi_device], capture_output=True, text=True, timeout=10)
            else:
                disconnect_result = subprocess.run(['nmcli', 'device', 'disconnect', 'wifi'], capture_output=True, text=True, timeout=10)
            if disconnect_result.returncode == 0:
                self._log("Successfully disconnected from WiFi"); QMessageBox.information(self, "Success", "Disconnected from WiFi"); self._scan_wifi()
            else:
                error_msg = disconnect_result.stderr or disconnect_result.stdout
                self._log(f"Disconnect failed: {error_msg}"); QMessageBox.warning(self, "Disconnect Failed", error_msg); self._update_disconnect_button()
        except Exception as e:
            self._log(f"Error: {str(e)}"); QMessageBox.warning(self, "Error", str(e)); self._update_disconnect_button()
        finally:
            self.disconnect_wifi_btn.setText("🔌 Disconnect")


class UpdateDialog(QDialog):
    """Dedicated dialog for OTA updates"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Updates")
        self.setFixedSize(1080, 1080)
        if parent:
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint); self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("QDialog { background-color: rgba(28,28,30,1.0); color: white; border: none; border-radius: 536px; } QLabel { color: white; }")
        self.setWindowOpacity(0.0)
        self._setup_ui()
        self._center_dialog()
        self.ota_update_thread = None
    
    def showEvent(self, event):
        super().showEvent(event)
        self.fade_in = QPropertyAnimation(self, b"windowOpacity"); self.fade_in.setDuration(300)
        self.fade_in.setStartValue(0.0); self.fade_in.setEndValue(1.0); self.fade_in.setEasingCurve(QEasingCurve.InOutCubic); self.fade_in.start()
        self.raise_(); self.activateWindow()
    
    def closeEvent(self, event):
        if hasattr(self, 'fade_in') and self.fade_in.state() == QPropertyAnimation.Running: self.fade_in.stop()
        self.fade_out = QPropertyAnimation(self, b"windowOpacity"); self.fade_out.setDuration(250)
        self.fade_out.setStartValue(self.windowOpacity()); self.fade_out.setEndValue(0.0); self.fade_out.setEasingCurve(QEasingCurve.InOutCubic)
        self.fade_out.finished.connect(lambda: event.accept()); self.fade_out.start(); event.ignore()
    
    def _center_dialog(self):
        screen = QApplication.primaryScreen().geometry(); x = (screen.width() - self.width()) // 2; y = (screen.height() - self.height()) // 2; self.move(x, y)
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(); main_layout.setContentsMargins(120, 100, 120, 100); main_layout.setSpacing(20); main_layout.addStretch(1)
        title = QLabel("🔄 Over-the-Air Updates"); title.setFont(QFont("Arial", 18, QFont.Bold)); title.setAlignment(Qt.AlignCenter); title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;"); main_layout.addWidget(title)
        self.status_log = QTextEdit(); self.status_log.setMaximumHeight(120); self.status_log.setReadOnly(True); self.status_log.setStyleSheet("QTextEdit { background-color: rgba(44,44,46,0.8); color: #ffffff; border-radius: 15px; border: none; padding: 10px; font-size: 11px; }"); main_layout.addWidget(self.status_log)
        button_row = QHBoxLayout(); button_row.setSpacing(15)
        self.update_btn = QPushButton("⬇️ Update from GitHub"); self.update_btn.setStyleSheet(SettingsDialog.get_button_style(None)); self.update_btn.clicked.connect(self._start_ota_update); button_row.addWidget(self.update_btn)
        main_layout.addLayout(button_row)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #555; border-radius: 8px; background-color: rgba(44,44,46,0.8); color: white; text-align: center; } QProgressBar::chunk { background-color: #007AFF; border-radius: 8px; }"); main_layout.addWidget(self.progress_bar)
        main_layout.addStretch(1); self.setLayout(main_layout)
    
    def _log(self, message): self.status_log.append(f"[Update] {message}")
    
    def _start_ota_update(self):
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        dotenv_path = os.path.join(workspace_root, '.env')
        github_token = None
        try:
            self._log(f"Checking for .env file at: {dotenv_path}")
            if os.path.exists(dotenv_path):
                self._log(".env file found, loading GITHUB_TOKEN...")
                env_vars = dotenv_values(dotenv_path); github_token = env_vars.get('GITHUB_TOKEN', '')
                if not github_token or github_token == 'your_github_token_here': github_token = None; self._log("GITHUB_TOKEN not set or placeholder")
        except Exception as e:
            self._log(f"Error loading .env: {e}")
        repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self._log("Starting OTA update..."); self.update_btn.setEnabled(False); self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.ota_update_thread = OTAUpdateThread(repo_path, github_token or '')
        self.ota_update_thread.update_progress.connect(lambda m: self._log(m))
        self.ota_update_thread.update_complete.connect(self._on_update_complete)
        self.ota_update_thread.finished.connect(lambda: (self.update_btn.setEnabled(True), self.progress_bar.setVisible(False)))
        self.ota_update_thread.start()
    
    def _on_update_complete(self, success, message):
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(100)
        if success: self._log(f"✅ {message}"); QMessageBox.information(self, "Update Complete", message)
        else: self._log(f"❌ {message}"); QMessageBox.warning(self, "Update Failed", message)


class AIModelSettingsDialog(QDialog):
    """Dedicated dialog for AI model selection and mode toggle"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Model Settings")
        self.setFixedSize(1080, 1080)
        if parent:
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint); self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("QDialog { background-color: rgba(28,28,30,1.0); color: white; border: none; border-radius: 536px; } QLabel { color: white; }")
        self.setWindowOpacity(0.0)
        self._setup_ui(); self._center_dialog()
    
    def showEvent(self, event):
        super().showEvent(event)
        self.fade_in = QPropertyAnimation(self, b"windowOpacity"); self.fade_in.setDuration(300)
        self.fade_in.setStartValue(0.0); self.fade_in.setEndValue(1.0); self.fade_in.setEasingCurve(QEasingCurve.InOutCubic); self.fade_in.start()
        self.raise_(); self.activateWindow()
    
    def closeEvent(self, event):
        if hasattr(self, 'fade_in') and self.fade_in.state() == QPropertyAnimation.Running: self.fade_in.stop()
        self.fade_out = QPropertyAnimation(self, b"windowOpacity"); self.fade_out.setDuration(250)
        self.fade_out.setStartValue(self.windowOpacity()); self.fade_out.setEndValue(0.0); self.fade_out.setEasingCurve(QEasingCurve.InOutCubic)
        self.fade_out.finished.connect(lambda: event.accept()); self.fade_out.start(); event.ignore()
    
    def _center_dialog(self):
        screen = QApplication.primaryScreen().geometry(); x = (screen.width() - self.width()) // 2; y = (screen.height() - self.height()) // 2; self.move(x, y)
    
    def _setup_ui(self):
        layout = QVBoxLayout(); layout.setContentsMargins(120, 100, 120, 100); layout.setSpacing(20); layout.addStretch(1)
        title = QLabel("🧠 AI Model Settings"); title.setFont(QFont("Arial", 18, QFont.Bold)); title.setAlignment(Qt.AlignCenter); title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;"); layout.addWidget(title)
        
        # Mode toggle
        mode_row = QHBoxLayout(); mode_row.setSpacing(12)
        self.mode_generic_btn = QPushButton("Generic"); self.mode_medical_btn = QPushButton("Medical")
        for b in (self.mode_generic_btn, self.mode_medical_btn): b.setCheckable(True); b.setStyleSheet(SettingsDialog.get_button_style(None))
        mode_row.addWidget(self.mode_generic_btn); mode_row.addWidget(self.mode_medical_btn); layout.addLayout(mode_row)
        
        # Model dropdown + restart
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox { background-color: rgba(44,44,46,0.8); color: #ffffff; padding: 8px; border: none; border-radius: 10px; min-height: 36px; }
            QComboBox QAbstractItemView { background-color: #2d2d2d; color: #ffffff; selection-background-color: #4D94D9; }
        """)
        self.restart_llm_btn = QPushButton("🔁 Restart LLM"); self.restart_llm_btn.setStyleSheet(SettingsDialog.get_button_style(None))
        row = QHBoxLayout(); row.setSpacing(10); row.addWidget(self.model_combo, 2); row.addWidget(self.restart_llm_btn, 1); layout.addLayout(row)
        layout.addStretch(1); self.setLayout(layout)
        
        # Load state and populate
        try:
            from core.state import get_llm_mode, get_llm_model
            mode = get_llm_mode(); self.mode_generic_btn.setChecked(mode == "generic"); self.mode_medical_btn.setChecked(mode == "medical")
        except Exception: self.mode_medical_btn.setChecked(True)
        self._populate_models()
        
        def on_mode_clicked():
            sender = self.sender()
            if sender == self.mode_generic_btn:
                self.mode_medical_btn.setChecked(not self.mode_generic_btn.isChecked())
            else:
                self.mode_generic_btn.setChecked(not self.mode_medical_btn.isChecked())
            mode_now = "medical" if self.mode_medical_btn.isChecked() else "generic"
            try:
                from core.state import set_llm_mode; set_llm_mode(mode_now)
            except Exception as e:
                print(f"[ModelSettings] Error saving mode: {e}")
            self._populate_models()
        
        self.mode_generic_btn.clicked.connect(on_mode_clicked)
        self.mode_medical_btn.clicked.connect(on_mode_clicked)
        
        def on_model_changed(index):
            name = self.model_combo.currentText()
            if not name or name.startswith("("): return
            try:
                from core.state import set_llm_model; set_llm_model(name)
            except Exception as e:
                print(f"[ModelSettings] Error saving model: {e}")
            self._prompt_restart()
        self.model_combo.currentIndexChanged.connect(on_model_changed)
        self.restart_llm_btn.clicked.connect(self._prompt_restart)
    
    def _populate_models(self):
        self.model_combo.clear()
        mode_now = "medical" if self.mode_medical_btn.isChecked() else "generic"
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_dir = os.path.join(workspace_root, 'llm-medical-container', 'data', 'models') if mode_now == "medical" else os.path.join(workspace_root, 'llm-container', 'data', 'models')
        paths = sorted(glob.glob(os.path.join(model_dir, "*.gguf"))); display_names = [os.path.basename(p) for p in paths]
        if display_names: self.model_combo.addItems(display_names)
        else: self.model_combo.addItem("(no models found)")
        try:
            from core.state import get_llm_model; saved = get_llm_model()
            if saved:
                base = os.path.basename(saved); idx = self.model_combo.findText(base); 
                if idx >= 0: self.model_combo.setCurrentIndex(idx)
        except Exception: pass
    
    def _prompt_restart(self):
        try:
            mode_now = "medical" if self.mode_medical_btn.isChecked() else "generic"
            service = "llm-medical" if mode_now == "medical" else "llm-generic"
            reply = QMessageBox.question(self, "Restart Required", f"Restart the {service} container now to apply the new model?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes: return
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); setup_dir = os.path.join(workspace_root, 'setup')
            import subprocess
            result = subprocess.run(["bash", "-lc", f"cd '{setup_dir}' && docker compose restart {service}"], capture_output=True, text=True, timeout=60)
            if result.returncode == 0: QMessageBox.information(self, "Restarted", f"{service} restarted successfully.")
            else: QMessageBox.warning(self, "Restart Failed", result.stderr or result.stdout or "Unknown error")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to restart container: {e}")

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        print("[Settings] 🔧 Initializing settings dialog...")
        
        self.setWindowTitle("Settings - AuraVision")
        self.setFixedSize(1080, 1080)
        
        if parent:
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setModal(True)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # (No translucent background to preserve readability)
        
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 28, 30, 1.0);
                color: white;
                border: none;
                border-radius: 536px;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #555;
                color: white;
                padding: 8px;
            }
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #555;
                color: white;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #2d2d2d;
                color: white;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #555;
                color: white;
                padding: 8px;
                border-radius: 4px;
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
    
    def closeEvent(self, event):
        """Handle dialog close event with smooth fade-out animation"""
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
    
    def center_dialog(self):
        """Center dialog on screen"""
        if self.parent():
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)
        else:
            # Center on primary screen
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
    
    def setup_ui(self):
        """Setup simplified settings UI with subsections"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(120, 100, 120, 100)
        main_layout.setSpacing(20)
        main_layout.addStretch(1)
        
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        title = QLabel("⚙️ Settings")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-weight: 600; margin: 10px;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        main_layout.addLayout(title_layout)
        
        row1 = QHBoxLayout()
        row1.setSpacing(15)
        wifi_btn = QPushButton("📶 WiFi Settings")
        wifi_btn.setStyleSheet(self.get_button_style())
        wifi_btn.clicked.connect(self.open_wifi_settings)
        row1.addWidget(wifi_btn)
        update_btn = QPushButton("🔄 Updates")
        update_btn.setStyleSheet(self.get_button_style())
        update_btn.clicked.connect(self.open_update_dialog)
        row1.addWidget(update_btn)
        main_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.setSpacing(15)
        model_btn = QPushButton("🧠 AI Model Settings")
        model_btn.setStyleSheet(self.get_button_style())
        model_btn.clicked.connect(self.open_model_settings)
        row2.addWidget(model_btn)
        main_layout.addLayout(row2)
        
        exit_button_layout = QHBoxLayout()
        exit_button_layout.setSpacing(15)
        exit_button_layout.addStretch()
        self.exit_btn = QPushButton("🚪 Exit to Desktop")
        self.exit_btn.clicked.connect(self.exit_to_desktop)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9500;
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 15px 30px;
                border-radius: 20px;
                border: none;
                min-width: 200px;
            }
            QPushButton:hover { background-color: #E58500; }
            QPushButton:pressed { background-color: #CC7500; }
        """)
        exit_button_layout.addWidget(self.exit_btn)
        exit_button_layout.addStretch()
        main_layout.addLayout(exit_button_layout)
        
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 20px;
                border: none;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #D70015; }
            QPushButton:pressed { background-color: #B30000; }
        """)
        close_layout.addWidget(self.close_btn)
        close_layout.addStretch()
        main_layout.addLayout(close_layout)
        
        main_layout.addStretch(1)
        self.setLayout(main_layout)
    
    def open_wifi_settings(self):
        dlg = WifiSettingsDialog(self)
        dlg.exec_()
    
    def open_update_dialog(self):
        dlg = UpdateDialog(self)
        dlg.exec_()
    
    def open_model_settings(self):
        dlg = AIModelSettingsDialog(self)
        dlg.exec_()

    def _init_llm_controls(self):
        """Add controls for LLM mode and model selection"""
        try:
            # Append below existing content
            layout = self.layout()
            # Section title
            llm_label = QLabel("🧠 AI Model Settings")
            llm_label.setFont(QFont("Arial", 14, QFont.Bold))
            llm_label.setStyleSheet("color: #ffffff; margin-top: 20px;")
            layout.addWidget(llm_label)
            
            # Mode toggle buttons
            mode_row = QHBoxLayout()
            mode_row.setSpacing(12)
            self.mode_generic_btn = QPushButton("Generic")
            self.mode_medical_btn = QPushButton("Medical")
            for b in (self.mode_generic_btn, self.mode_medical_btn):
                b.setCheckable(True)
                b.setStyleSheet(self.get_button_style())
            mode_row.addWidget(self.mode_generic_btn)
            mode_row.addWidget(self.mode_medical_btn)
            layout.addLayout(mode_row)
            
            # Model dropdown + restart button row
            from PyQt5.QtWidgets import QComboBox
            model_row = QHBoxLayout()
            model_row.setSpacing(10)
            self.model_combo = QComboBox()
            self.model_combo.setStyleSheet("""
                QComboBox {
                    background-color: rgba(44,44,46,0.8);
                    color: #ffffff;
                    padding: 8px;
                    border: none;
                    border-radius: 10px;
                    min-height: 36px;
                }
                QComboBox QAbstractItemView {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    selection-background-color: #4D94D9;
                }
            """)
            self.restart_llm_btn = QPushButton("🔁 Restart LLM")
            self.restart_llm_btn.setStyleSheet(self.get_button_style())
            model_row.addWidget(self.model_combo, 2)
            model_row.addWidget(self.restart_llm_btn, 1)
            layout.addLayout(model_row)
            
            # Load initial state
            try:
                from core.state import get_llm_mode, get_llm_model, set_llm_mode, set_llm_model
                mode = get_llm_mode()
                self.mode_generic_btn.setChecked(mode == "generic")
                self.mode_medical_btn.setChecked(mode == "medical")
            except Exception:
                self.mode_medical_btn.setChecked(True)
            
            # Populate models based on mode
            def populate_models():
                self.model_combo.clear()
                mode_now = "medical" if self.mode_medical_btn.isChecked() else "generic"
                # List available .gguf models in corresponding container models folder
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                if mode_now == "medical":
                    model_dir = os.path.join(workspace_root, 'llm-medical-container', 'data', 'models')
                else:
                    model_dir = os.path.join(workspace_root, 'llm-container', 'data', 'models')
                paths = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
                display_names = [os.path.basename(p) for p in paths]
                if display_names:
                    self.model_combo.addItems(display_names)
                else:
                    self.model_combo.addItem("(no models found)")
                # Set current selection to saved model if present
                try:
                    from core.state import get_llm_model
                    saved = get_llm_model()
                    if saved:
                        base = os.path.basename(saved)
                        idx = self.model_combo.findText(base)
                        if idx >= 0:
                            self.model_combo.setCurrentIndex(idx)
                except Exception:
                    pass
            
            populate_models()
            
            # Handlers
            def on_mode_clicked():
                # Make them mutually exclusive
                sender = self.sender()
                if sender == self.mode_generic_btn:
                    self.mode_medical_btn.setChecked(not self.mode_generic_btn.isChecked())
                else:
                    self.mode_generic_btn.setChecked(not self.mode_medical_btn.isChecked())
                mode_now = "medical" if self.mode_medical_btn.isChecked() else "generic"
                try:
                    from core.state import set_llm_mode
                    set_llm_mode(mode_now)
                    self.log_status(f"LLM mode set to: {mode_now}")
                except Exception as e:
                    self.log_status(f"Error saving mode: {e}")
                populate_models()
            
            self.mode_generic_btn.clicked.connect(on_mode_clicked)
            self.mode_medical_btn.clicked.connect(on_mode_clicked)
            
            def _restart_container_for_mode(mode_now: str):
                """Restart the appropriate LLM container"""
                try:
                    from PyQt5.QtWidgets import QMessageBox
                    service = "llm-medical" if mode_now == "medical" else "llm-generic"
                    # Confirm restart
                    reply = QMessageBox.question(
                        self,
                        "Restart Required",
                        f"Restart the {service} container now to apply the new model?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if reply != QMessageBox.Yes:
                        return
                    # Run docker compose restart in setup directory
                    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    setup_dir = os.path.join(workspace_root, 'setup')
                    import subprocess
                    result = subprocess.run(
                        ["bash", "-lc", f"cd '{setup_dir}' && docker compose restart {service}"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        QMessageBox.information(self, "Restarted", f"{service} restarted successfully.")
                    else:
                        QMessageBox.warning(self, "Restart Failed", result.stderr or result.stdout or "Unknown error")
                except Exception as e:
                    try:
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, "Error", f"Failed to restart container: {e}")
                    except:
                        pass

            def on_model_changed(index):
                name = self.model_combo.currentText()
                if not name or name.startswith("("):
                    return
                try:
                    from core.state import set_llm_model
                    set_llm_model(name)
                    self.log_status(f"Selected model: {name}")
                    # Prompt to restart to apply model
                    mode_now = "medical" if self.mode_medical_btn.isChecked() else "generic"
                    _restart_container_for_mode(mode_now)
                except Exception as e:
                    self.log_status(f"Error saving model: {e}")
            
            self.model_combo.currentIndexChanged.connect(on_model_changed)

            # Manual restart button
            def on_restart_clicked():
                mode_now = "medical" if self.mode_medical_btn.isChecked() else "generic"
                _restart_container_for_mode(mode_now)
            self.restart_llm_btn.clicked.connect(on_restart_clicked)
        except Exception as e:
            print(f"[Settings] ⚠️ Failed to initialize LLM controls: {e}")
    
    def get_button_style(self):
        """Get consistent button styling"""
        return """
            QPushButton {
                background-color: rgba(70, 130, 180, 0.25);
                color: #ffffff;
                font-size: 22px;
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
        """
    
    def log_status(self, message):
        """Add message to status log"""
        self.status_log.append(f"[Settings] {message}")
        print(f"[Settings] {message}")
    
    def exit_to_desktop(self):
        """Exit Aura and return to desktop"""
        reply = QMessageBox.question(
            self,
            "Exit to Desktop",
            "Exit Aura and return to the desktop environment?\n\nThis will stop all Aura services.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.log_status("Exiting Aura...")
            
            # Request shutdown via state management
            try:
                import sys
                import os
                # Add parent directory to path to import state module
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from core.state import request_shutdown
                request_shutdown()
                print("[Settings] ✅ Shutdown requested")
            except ImportError as e:
                print(f"[Settings] ⚠️ Could not import request_shutdown: {e}")
            except Exception as e:
                print(f"[Settings] ⚠️ Error requesting shutdown: {e}")
            
            # Close settings dialog first
            self.close()
            
            # Close main GUI window and exit Qt application
            try:
                from gui.aura_gui import close_gui
                close_gui()
                print("[Settings] ✅ GUI closed")
            except ImportError:
                # Fallback: quit Qt application directly
                QApplication.instance().quit()
                print("[Settings] ✅ Application quit (fallback)")
            except Exception as e:
                print(f"[Settings] ⚠️ Error closing GUI: {e}")
                # Fallback: quit Qt application directly
                QApplication.instance().quit()
    
    def scan_wifi(self):
        """Scan for available WiFi networks"""
        self.log_status("Scanning for WiFi networks (this may take 10-15 seconds)...")
        self.scan_wifi_btn.setEnabled(False)
        self.scan_wifi_btn.setText("🔄 Scanning...")
        self.wifi_list.clear()
        self.wifi_list.addItem("Scanning... Please wait...")
        
        self.wifi_scan_thread = WiFiScanThread()
        self.wifi_scan_thread.networks_found.connect(self.on_wifi_networks_found)
        self.wifi_scan_thread.scan_error.connect(self.on_wifi_scan_error)
        self.wifi_scan_thread.finished.connect(lambda: (
            self.scan_wifi_btn.setEnabled(True),
            self.scan_wifi_btn.setText("🔍 Scan Networks")
        ))
        self.wifi_scan_thread.start()
    
    def on_wifi_networks_found(self, networks):
        """Handle WiFi networks found"""
        self.wifi_list.clear()  # Clear the "Scanning..." message
        
        if not networks:
            self.log_status("No networks found. Make sure WiFi is enabled.")
            self.wifi_list.addItem("No networks found")
            return
        
        self.log_status(f"Found {len(networks)} network(s)")
        
        for network in networks:
            ssid = network['ssid']
            signal = network['signal']
            security = network['security']
            connected = network.get('connected', False)
            
            # Show signal strength, even if 0% (might be a hidden network or weak signal)
            signal_display = f"{signal}%" if signal > 0 else "weak"
            item_text = f"{ssid} ({signal_display})"
            
            if connected:
                item_text = f"● {item_text} (Connected)"
            if security and security != "Open" and security != "--":
                item_text += f" 🔒 {security}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, network)
            self.wifi_list.addItem(item)
        
        if networks:
            self.log_status(f"Select a network to connect ({len(networks)} available)")
        
        # Update disconnect button after scan
        self.update_disconnect_button()
    
    def on_wifi_scan_error(self, error):
        """Handle WiFi scan error"""
        self.log_status(f"Error: {error}")
        QMessageBox.warning(self, "WiFi Scan Error", error)
    
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
        
        self.log_status(f"Connecting to {ssid}...")
        
        # Connect using nmcli
        try:
            if password:
                cmd = ['nmcli', 'device', 'wifi', 'connect', ssid, 'password', password]
            else:
                cmd = ['nmcli', 'device', 'wifi', 'connect', ssid]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.log_status(f"Successfully connected to {ssid}")
                QMessageBox.information(self, "Success", f"Connected to {ssid}")
                # Update disconnect button after successful connection
                self.update_disconnect_button()
                # Refresh network list to show updated connection status
                self.scan_wifi()
            else:
                error_msg = result.stderr or result.stdout
                self.log_status(f"Connection failed: {error_msg}")
                QMessageBox.warning(self, "Connection Failed", error_msg)
        except Exception as e:
            self.log_status(f"Error: {str(e)}")
            QMessageBox.warning(self, "Error", str(e))
    
    def update_disconnect_button(self):
        """Update disconnect button state based on WiFi connection status"""
        try:
            # Check if WiFi is connected using nmcli
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'DEVICE,TYPE,STATE', 'device', 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Look for WiFi devices that are connected
                wifi_connected = False
                for line in result.stdout.strip().split('\n'):
                    if ':wifi:' in line.lower() and ':connected' in line.lower():
                        wifi_connected = True
                        break
                
                self.disconnect_wifi_btn.setEnabled(wifi_connected)
                if wifi_connected:
                    self.log_status("WiFi connected - disconnect button enabled")
                else:
                    self.log_status("WiFi not connected - disconnect button disabled")
            else:
                # If nmcli fails, disable disconnect button
                self.disconnect_wifi_btn.setEnabled(False)
        except Exception as e:
            print(f"[Settings] ⚠️ Error checking WiFi status: {e}")
            self.disconnect_wifi_btn.setEnabled(False)
    
    def disconnect_wifi(self):
        """Disconnect from currently connected WiFi network"""
        reply = QMessageBox.question(
            self,
            "Disconnect WiFi",
            "Disconnect from the current WiFi network?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.log_status("Disconnecting from WiFi...")
        self.disconnect_wifi_btn.setEnabled(False)
        self.disconnect_wifi_btn.setText("🔄 Disconnecting...")
        
        try:
            # Get the WiFi device name
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'DEVICE,TYPE', 'device', 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            wifi_device = None
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':wifi:' in line.lower():
                        wifi_device = line.split(':')[0]
                        break
            
            if wifi_device:
                # Disconnect using device name
                disconnect_result = subprocess.run(
                    ['nmcli', 'device', 'disconnect', wifi_device],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if disconnect_result.returncode == 0:
                    self.log_status(f"Successfully disconnected from WiFi")
                    QMessageBox.information(self, "Success", "Disconnected from WiFi")
                    # Refresh network list to show updated connection status
                    self.scan_wifi()
                else:
                    error_msg = disconnect_result.stderr or disconnect_result.stdout
                    self.log_status(f"Disconnect failed: {error_msg}")
                    QMessageBox.warning(self, "Disconnect Failed", error_msg)
                    self.update_disconnect_button()
            else:
                # Fallback: disconnect all WiFi connections
                disconnect_result = subprocess.run(
                    ['nmcli', 'device', 'disconnect', 'wifi'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if disconnect_result.returncode == 0:
                    self.log_status(f"Successfully disconnected from WiFi")
                    QMessageBox.information(self, "Success", "Disconnected from WiFi")
                    self.scan_wifi()
                else:
                    error_msg = disconnect_result.stderr or disconnect_result.stdout
                    self.log_status(f"Disconnect failed: {error_msg}")
                    QMessageBox.warning(self, "Disconnect Failed", error_msg)
                    self.update_disconnect_button()
                    
        except Exception as e:
            self.log_status(f"Error: {str(e)}")
            QMessageBox.warning(self, "Error", str(e))
            self.update_disconnect_button()
        finally:
            self.disconnect_wifi_btn.setText("🔌 Disconnect")
    
    def start_ota_update(self):
        """Start OTA update process"""
        # Try to load token from .env file
        github_token = None
        try:
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            dotenv_path = os.path.join(workspace_root, '.env')
            
            self.log_status(f"Checking for .env file at: {dotenv_path}")
            
            if os.path.exists(dotenv_path):
                self.log_status(".env file found, loading GITHUB_TOKEN...")
                env_vars = dotenv_values(dotenv_path)
                github_token = env_vars.get('GITHUB_TOKEN', '')
                
                if github_token:
                    if github_token == 'your_github_token_here' or github_token.strip() == '':
                        github_token = None
                        self.log_status("GITHUB_TOKEN found but is placeholder or empty")
                    else:
                        # Show first few chars for confirmation (masked)
                        token_preview = f"{github_token[:10]}...{github_token[-4:]}" if len(github_token) > 14 else "***"
                        self.log_status(f"GITHUB_TOKEN loaded from .env ({token_preview})")
                else:
                    self.log_status("GITHUB_TOKEN not found in .env file")
            else:
                self.log_status(f".env file not found at {dotenv_path}")
        except Exception as e:
            # Mask token in error messages (use a helper function)
            error_msg = str(e)
            # Mask GitHub token patterns
            import re
            token_pattern = r'ghp_[A-Za-z0-9]{36,}'
            error_msg = re.sub(token_pattern, '***', error_msg)
            url_pattern = r'https://[^@\s]+@[^@\s]+@github\.com'
            error_msg = re.sub(url_pattern, 'https://***@github.com', error_msg)
            
            self.log_status(f"Error loading token from .env: {error_msg}")
            print(f"[Settings] Could not load token from .env: {error_msg}")
            import traceback
            traceback.print_exc()
        
        # Get repository path (workspace root)
        repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        self.log_status("Starting OTA update...")
        if github_token:
            self.log_status("Using GitHub token from .env file")
        else:
            self.log_status("Updating without authentication (public repos only)")
        
        self.update_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.ota_update_thread = OTAUpdateThread(repo_path, github_token or '')
        self.ota_update_thread.update_progress.connect(self.on_update_progress)
        self.ota_update_thread.update_complete.connect(self.on_update_complete)
        self.ota_update_thread.finished.connect(
            lambda: (self.update_btn.setEnabled(True), self.progress_bar.setVisible(False))
        )
        self.ota_update_thread.start()
    
    def on_update_progress(self, message):
        """Handle update progress messages"""
        self.log_status(message)
    
    def on_update_complete(self, success, message):
        """Handle update completion"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        if success:
            self.log_status(f"✅ {message}")
            QMessageBox.information(self, "Update Complete", message)
        else:
            self.log_status(f"❌ {message}")
            QMessageBox.warning(self, "Update Failed", message)
        
        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
    
    def closeEvent(self, event):
        """Handle dialog close"""
        # Clean up threads
        if self.wifi_scan_thread and self.wifi_scan_thread.isRunning():
            self.wifi_scan_thread.terminate()
        if self.ota_update_thread and self.ota_update_thread.isRunning():
            self.ota_update_thread.terminate()
        
        event.accept()

