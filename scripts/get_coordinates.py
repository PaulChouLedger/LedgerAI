#!/usr/bin/env python3
"""
Simple coordinate helper script
Use this to get coordinates from the upload dialog touch events
"""

def analyze_coordinates():
    """Analyze coordinates from touch events"""
    print("🔧 Coordinate Analysis Helper")
    print("=" * 40)
    print("Enter the coordinates you got from touching the upload dialog:")
    print("(Format: x,y for each coordinate)")
    print("=" * 40)
    
    coordinates = []
    
    # Get multiple touch points
    while True:
        coord_input = input("\nEnter coordinates (x,y) or 'done' to finish: ").strip()
        if coord_input.lower() == 'done':
            break
        
        try:
            x, y = map(int, coord_input.split(','))
            coordinates.append((x, y))
            print(f"✅ Recorded: ({x}, {y})")
        except ValueError:
            print("❌ Invalid format. Use 'x,y' (e.g., '540,540')")
    
    if len(coordinates) < 2:
        print("❌ Need at least 2 coordinates to analyze")
        return
    
    # Analyze coordinates
    print("\n📊 ANALYSIS:")
    print("-" * 20)
    
    # Find min/max values
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    print(f"X range: {min_x} to {max_x} (width: {max_x - min_x})")
    print(f"Y range: {min_y} to {max_y} (height: {max_y - min_y})")
    
    # Calculate center
    center_x = min_x + (max_x - min_x) // 2
    center_y = min_y + (max_y - min_y) // 2
    
    print(f"Calculated center: ({center_x}, {center_y})")
    
    # Calculate dialog position
    dialog_x = center_x - 540
    dialog_y = center_y - 540
    
    print(f"Dialog position: ({dialog_x}, {dialog_y})")
    
    # Show all coordinates
    print("\n📍 ALL COORDINATES:")
    for i, (x, y) in enumerate(coordinates, 1):
        print(f"  {i}. ({x}, {y})")

if __name__ == "__main__":
    analyze_coordinates()
