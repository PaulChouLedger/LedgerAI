#!/usr/bin/env python3
"""
Command-Line Screen Calibration Script for AuraVision
This script helps determine the exact screen center and edges for perfect dialog centering.
"""

import sys
import os

def get_calibration_points():
    """Get calibration points from user input"""
    print("🔧 Screen Calibration for AuraVision")
    print("=" * 50)
    print("This script will help you determine the exact screen center and edges.")
    print("You'll need to provide coordinates by touching/clicking on your screen.")
    print("=" * 50)
    
    calibration_points = []
    
    # Step 1: Center
    print("\n🎯 STEP 1: CENTER")
    print("Touch/click the CENTER of your screen")
    try:
        center_input = input("Enter center coordinates (x,y): ").strip()
        center_x, center_y = map(int, center_input.split(','))
        calibration_points.append((center_x, center_y))
        print(f"✅ Center recorded: ({center_x}, {center_y})")
    except ValueError:
        print("❌ Invalid input. Please enter coordinates as 'x,y' (e.g., '540,540')")
        return None
    
    # Step 2: Top
    print("\n📍 STEP 2: TOP EDGE")
    print("Touch/click the TOP edge of your screen")
    try:
        top_input = input("Enter top edge coordinates (x,y): ").strip()
        top_x, top_y = map(int, top_input.split(','))
        calibration_points.append((top_x, top_y))
        print(f"✅ Top edge recorded: ({top_x}, {top_y})")
    except ValueError:
        print("❌ Invalid input. Please enter coordinates as 'x,y'")
        return None
    
    # Step 3: Right
    print("\n📍 STEP 3: RIGHT EDGE")
    print("Touch/click the RIGHT edge of your screen")
    try:
        right_input = input("Enter right edge coordinates (x,y): ").strip()
        right_x, right_y = map(int, right_input.split(','))
        calibration_points.append((right_x, right_y))
        print(f"✅ Right edge recorded: ({right_x}, {right_y})")
    except ValueError:
        print("❌ Invalid input. Please enter coordinates as 'x,y'")
        return None
    
    # Step 4: Bottom
    print("\n📍 STEP 4: BOTTOM EDGE")
    print("Touch/click the BOTTOM edge of your screen")
    try:
        bottom_input = input("Enter bottom edge coordinates (x,y): ").strip()
        bottom_x, bottom_y = map(int, bottom_input.split(','))
        calibration_points.append((bottom_x, bottom_y))
        print(f"✅ Bottom edge recorded: ({bottom_x}, {bottom_y})")
    except ValueError:
        print("❌ Invalid input. Please enter coordinates as 'x,y'")
        return None
    
    # Step 5: Left
    print("\n📍 STEP 5: LEFT EDGE")
    print("Touch/click the LEFT edge of your screen")
    try:
        left_input = input("Enter left edge coordinates (x,y): ").strip()
        left_x, left_y = map(int, left_input.split(','))
        calibration_points.append((left_x, left_y))
        print(f"✅ Left edge recorded: ({left_x}, {left_y})")
    except ValueError:
        print("❌ Invalid input. Please enter coordinates as 'x,y'")
        return None
    
    return calibration_points

def calculate_results(calibration_points):
    """Calculate screen dimensions and center"""
    if len(calibration_points) != 5:
        print("❌ Error: Need exactly 5 calibration points")
        return
    
    center_x, center_y = calibration_points[0]
    top_x, top_y = calibration_points[1]
    right_x, right_y = calibration_points[2]
    bottom_x, bottom_y = calibration_points[3]
    left_x, left_y = calibration_points[4]
    
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
    print("\n" + "=" * 50)
    print("🎯 CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Screen Width: {screen_width}px")
    print(f"Screen Height: {screen_height}px")
    print(f"Actual Center: ({actual_center_x}, {actual_center_y})")
    print(f"Dialog Position: ({dialog_x}, {dialog_y})")
    print("=" * 50)
    
    # Generate code snippet
    print("\n💡 CODE TO USE IN YOUR DIALOG:")
    print("-" * 30)
    print(f"self.move({dialog_x}, {dialog_y})")
    print("-" * 30)
    
    # Save results to file
    results_file = "screen_calibration_results.txt"
    with open(results_file, "w") as f:
        f.write("Screen Calibration Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Screen Width: {screen_width}px\n")
        f.write(f"Screen Height: {screen_height}px\n")
        f.write(f"Actual Center: ({actual_center_x}, {actual_center_y})\n")
        f.write(f"Dialog Position: ({dialog_x}, {dialog_y})\n")
        f.write("\nCode to use:\n")
        f.write(f"self.move({dialog_x}, {dialog_y})\n")
    
    print(f"\n📄 Results saved to: {results_file}")

def main():
    """Main function"""
    try:
        calibration_points = get_calibration_points()
        if calibration_points:
            calculate_results(calibration_points)
        else:
            print("❌ Calibration failed. Please try again.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n❌ Calibration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
