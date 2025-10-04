#!/usr/bin/env python3
"""
Screen Calibration Script for AuraVision
This script helps determine the exact screen center and edges for perfect dialog centering.
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QLabel, 
                             QPushButton, QHBoxLayout, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

class ScreenCalibrationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.calibration_points = []
        self.current_step = 0
        self.steps = [
            "🎯 STEP 1: Tap the CENTER of your screen",
            "📍 STEP 2: Tap the TOP edge of your screen", 
            "📍 STEP 3: Tap the RIGHT edge of your screen",
            "📍 STEP 4: Tap the BOTTOM edge of your screen",
            "📍 STEP 5: Tap the LEFT edge of your screen"
        ]
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Screen Calibration - AuraVision")
        self.setFixedSize(1080, 1080)  # Full screen size
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # Position at origin for full screen
        self.move(0, 0)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)
        
        # Title
        title = QLabel("🔧 Screen Calibration")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4CAF50; margin: 20px;")
        layout.addWidget(title)
        
        # Instructions
        self.instructions = QLabel(self.steps[0])
        self.instructions.setFont(QFont("Arial", 16))
        self.instructions.setAlignment(Qt.AlignCenter)
        self.instructions.setStyleSheet("color: white; margin: 20px; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px;")
        layout.addWidget(self.instructions)
        
        # Coordinates display
        self.coords_display = QTextEdit()
        self.coords_display.setMaximumHeight(200)
        self.coords_display.setReadOnly(True)
        self.coords_display.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 10px;
                color: white;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.coords_display)
        
        # Progress
        self.progress = QLabel(f"Progress: {self.current_step + 1}/5")
        self.progress.setFont(QFont("Arial", 14))
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("color: #4CAF50; margin: 10px;")
        layout.addWidget(self.progress)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_calibration)
        button_layout.addWidget(self.reset_btn)
        
        self.calculate_btn = QPushButton("📊 Calculate Results")
        self.calculate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.calculate_btn.clicked.connect(self.calculate_results)
        self.calculate_btn.setEnabled(False)
        button_layout.addWidget(self.calculate_btn)
        
        layout.addLayout(button_layout)
        
        # Results display
        self.results_display = QTextEdit()
        self.results_display.setMaximumHeight(150)
        self.results_display.setReadOnly(True)
        self.results_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                color: #4CAF50;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.results_display)
        
        self.setLayout(layout)
        
        # Set dialog style
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: white;
                border: 5px solid #ff0000;
                border-radius: 540px;
            }
        """)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Add initial instructions
        self.coords_display.append("🔧 Screen Calibration Started")
        self.coords_display.append("Tap the center of your screen first...")
        
    def mousePressEvent(self, event):
        """Capture mouse/touch coordinates"""
        x, y = event.x(), event.y()
        
        # Add to calibration points
        self.calibration_points.append((x, y))
        
        # Display coordinates
        step_name = ["CENTER", "TOP", "RIGHT", "BOTTOM", "LEFT"][self.current_step]
        self.coords_display.append(f"📍 {step_name}: ({x}, {y})")
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step < len(self.steps):
            self.instructions.setText(self.steps[self.current_step])
            self.progress.setText(f"Progress: {self.current_step + 1}/5")
        else:
            self.instructions.setText("✅ Calibration Complete! Click 'Calculate Results'")
            self.calculate_btn.setEnabled(True)
            
        super().mousePressEvent(event)
        
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_points = []
        self.current_step = 0
        self.instructions.setText(self.steps[0])
        self.progress.setText("Progress: 1/5")
        self.calculate_btn.setEnabled(False)
        self.coords_display.clear()
        self.results_display.clear()
        self.coords_display.append("🔧 Screen Calibration Reset")
        self.coords_display.append("Tap the center of your screen first...")
        
    def calculate_results(self):
        """Calculate screen dimensions and center"""
        if len(self.calibration_points) != 5:
            self.results_display.append("❌ Error: Need exactly 5 calibration points")
            return
            
        center_x, center_y = self.calibration_points[0]
        top_x, top_y = self.calibration_points[1]
        right_x, right_y = self.calibration_points[2]
        bottom_x, bottom_y = self.calibration_points[3]
        left_x, left_y = self.calibration_points[4]
        
        # Calculate screen dimensions
        screen_width = right_x - left_x
        screen_height = bottom_y - top_y
        
        # Calculate actual center
        actual_center_x = left_x + (screen_width // 2)
        actual_center_y = top_y + (screen_height // 2)
        
        # Calculate dialog position for perfect centering
        dialog_x = actual_center_x - 540  # 540 is half of 1080
        dialog_y = actual_center_y - 540
        
        # Display results
        self.results_display.clear()
        self.results_display.append("🎯 CALIBRATION RESULTS:")
        self.results_display.append(f"Screen Width: {screen_width}px")
        self.results_display.append(f"Screen Height: {screen_height}px")
        self.results_display.append(f"Actual Center: ({actual_center_x}, {actual_center_y})")
        self.results_display.append(f"Dialog Position: ({dialog_x}, {dialog_y})")
        self.results_display.append("")
        self.results_display.append("💡 Use these coordinates in your dialog positioning code!")
        
        # Print to console for easy copying
        print("\n" + "="*50)
        print("🎯 SCREEN CALIBRATION RESULTS")
        print("="*50)
        print(f"Screen Width: {screen_width}px")
        print(f"Screen Height: {screen_height}px")
        print(f"Actual Center: ({actual_center_x}, {actual_center_y})")
        print(f"Dialog Position: ({dialog_x}, {dialog_y})")
        print("="*50)

def main():
    app = QApplication(sys.argv)
    dialog = ScreenCalibrationDialog()
    dialog.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
