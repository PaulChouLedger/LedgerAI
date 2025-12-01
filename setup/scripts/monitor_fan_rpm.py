#!/usr/bin/env python3
"""
Real-time Jetson Fan RPM Monitor
Logs fan speed (RPM) and temperature to console for testing cooling methods.

Usage:
    python3 setup/scripts/monitor_fan_rpm.py
    python3 setup/scripts/monitor_fan_rpm.py --interval 1  # Update every 1 second
    python3 setup/scripts/monitor_fan_rpm.py --csv output.csv  # Log to CSV file
"""

import time
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path

def get_fan_rpm():
    """
    Get current Jetson fan RPM.
    Tries multiple methods to find actual RPM reading.
    """
    # Method 1: Check for rpm_measured file (most Jetson models)
    rpm_paths = [
        "/sys/devices/pwm-fan/rpm_measured",
        "/sys/devices/pwm-fan/rpm",
        "/sys/class/hwmon/hwmon0/fan1_input",
        "/sys/class/hwmon/hwmon1/fan1_input",
        "/sys/class/hwmon/hwmon2/fan1_input",
    ]
    
    for path in rpm_paths:
        try:
            result = subprocess.run(
                ["cat", path],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                rpm = int(result.stdout.strip())
                if rpm > 0:  # Valid RPM reading
                    return rpm, path
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            continue
    
    # Method 2: Try sensors command (if available)
    try:
        result = subprocess.run(
            ["sensors"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Parse sensors output for fan RPM
            for line in result.stdout.splitlines():
                if "fan" in line.lower() and "rpm" in line.lower():
                    # Extract RPM value (format: "fan1:        3000 RPM")
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i > 0:
                            rpm = int(part)
                            return rpm, "sensors"
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    
    return None, None

def get_fan_pwm():
    """Get fan PWM value (0-255) as fallback if RPM not available"""
    try:
        result = subprocess.run(
            ["cat", "/sys/devices/pwm-fan/target_pwm"],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except:
        pass
    return None

def get_temperature():
    """Get Jetson CPU/GPU temperature"""
    temp_paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ]
    
    temps = []
    for path in temp_paths:
        try:
            result = subprocess.run(
                ["cat", path],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                temp_millidegrees = int(result.stdout.strip())
                temp_celsius = temp_millidegrees / 1000.0
                temps.append(temp_celsius)
        except:
            continue
    
    if temps:
        return max(temps)  # Return highest temperature
    return None

def get_power_mode():
    """Get current Jetson power mode"""
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Power Mode" in line or "NV Power Mode" in line:
                    return line.strip()
    except:
        pass
    return None

def format_rpm(rpm):
    """Format RPM value with appropriate units"""
    if rpm is None:
        return "N/A"
    if rpm >= 1000:
        return f"{rpm/1000:.1f}K"
    return str(rpm)

def main():
    parser = argparse.ArgumentParser(description="Monitor Jetson fan RPM in real-time")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Update interval in seconds (default: 1.0)")
    parser.add_argument("--csv", type=str, default=None,
                       help="Log to CSV file (e.g., --csv fan_log.csv)")
    parser.add_argument("--no-header", action="store_true",
                       help="Don't print header row")
    args = parser.parse_args()
    
    # Open CSV file if specified
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, 'w')
        csv_file.write("timestamp,rpm,pwm,temp_celsius,power_mode\n")
        print(f"[Fan Monitor] 📝 Logging to: {args.csv}")
    
    try:
        # Print header
        if not args.no_header:
            print("\n" + "="*80)
            print("  🔄 JETSON FAN RPM MONITOR - Real-time Cooling Test")
            print("="*80)
            print(f"  Update interval: {args.interval}s")
            print(f"  Press Ctrl+C to stop\n")
            print(f"{'Time':<12} {'RPM':<12} {'PWM':<8} {'Temp (°C)':<12} {'Power Mode':<20}")
            print("-" * 80)
        
        # Initial detection
        rpm, rpm_source = get_fan_rpm()
        if rpm is None:
            print("[Fan Monitor] ⚠️  RPM reading not available, using PWM as fallback")
            print("[Fan Monitor] 💡 RPM may be available via: sensors command")
            print()
        
        # Main monitoring loop
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Get readings
            rpm, rpm_source = get_fan_rpm()
            pwm = get_fan_pwm()
            temp = get_temperature()
            power_mode = get_power_mode()
            
            # Format power mode (shorten if too long)
            if power_mode:
                power_mode_short = power_mode.split(":")[-1].strip()[:18]
            else:
                power_mode_short = "N/A"
            
            # Print to console
            if rpm is not None:
                rpm_str = f"{rpm:>6} RPM"
            elif pwm is not None:
                rpm_str = f"{pwm:>3} PWM"
            else:
                rpm_str = "N/A"
            
            temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"
            pwm_str = f"{pwm}" if pwm is not None else "N/A"
            
            print(f"{timestamp:<12} {rpm_str:<12} {pwm_str:<8} {temp_str:<12} {power_mode_short:<20}")
            
            # Write to CSV if specified
            if csv_file:
                csv_file.write(f"{datetime.now().isoformat()},{rpm or ''},{pwm or ''},{temp or ''},{power_mode_short}\n")
                csv_file.flush()
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n[Fan Monitor] ⏹️  Monitoring stopped")
        if csv_file:
            csv_file.close()
            print(f"[Fan Monitor] ✅ Data saved to: {args.csv}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Fan Monitor] ❌ Error: {e}")
        if csv_file:
            csv_file.close()
        sys.exit(1)

if __name__ == "__main__":
    main()

