#!/bin/bash

# Comprehensive script to fix ALL uv pip commands in jetson-containers
# Searches for any file containing uv pip commands and fixes them

set -e

JETSON_CONTAINERS_DIR="$HOME/jetson-containers"
BACKUP_DIR="/tmp/jetson_containers_complete_backup_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/tmp/jetson_containers_fix_log_$(date +%Y%m%d_%H%M%S).txt"

echo "🔧 Comprehensive jetson-containers registry fix..."
echo "📁 Jetson containers directory: $JETSON_CONTAINERS_DIR"
echo "💾 Backup directory: $BACKUP_DIR"
echo "📝 Log file: $LOG_FILE"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Initialize log file
echo "Jetson Containers Registry Fix Log - $(date)" > "$LOG_FILE"
echo "===============================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to fix a single file
fix_file() {
    local file="$1"
    local relative_path="${file#$JETSON_CONTAINERS_DIR/}"
    
    echo "🔍 Processing: $relative_path" | tee -a "$LOG_FILE"
    
    # Create backup
    cp "$file" "$BACKUP_DIR/${relative_path//\//_}"
    
    # Count original uv pip commands
    local original_count=$(grep -c "uv pip" "$file" 2>/dev/null || echo "0")
    
    # Replace all uv pip commands
    sed -i 's/uv pip install/python -m pip install --index-url "https:\/\/pypi.org\/simple"/g' "$file"
    sed -i 's/uv pip show/python -m pip show/g' "$file"
    sed -i 's/uv pip list/python -m pip list/g' "$file"
    sed -i 's/uv pip freeze/python -m pip freeze/g' "$file"
    sed -i 's/uv pip uninstall/python -m pip uninstall/g' "$file"
    sed -i 's/uv pip wheel/python -m pip wheel/g' "$file"
    sed -i 's/uv pip download/python -m pip download/g' "$file"
    sed -i 's/uv pip check/python -m pip check/g' "$file"
    sed -i 's/uv pip config/python -m pip config/g' "$file"
    sed -i 's/uv pip cache/python -m pip cache/g' "$file"
    
    # Count remaining uv pip commands (should be 0)
    local remaining_count=$(grep -c "uv pip" "$file" 2>/dev/null || echo "0")
    
    if [ "$remaining_count" -eq 0 ]; then
        echo "✅ Fixed $original_count uv pip commands in: $relative_path" | tee -a "$LOG_FILE"
    else
        echo "⚠️  Warning: $remaining_count uv pip commands still remain in: $relative_path" | tee -a "$LOG_FILE"
        echo "   Remaining commands:" | tee -a "$LOG_FILE"
        grep -n "uv pip" "$file" | tee -a "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
}

# Search for ALL files containing uv pip commands
echo "🔍 Searching for ALL files containing 'uv pip' commands..."
echo ""

# Find all files (any type) that contain uv pip commands
files_with_uv_pip=$(find "$JETSON_CONTAINERS_DIR" -type f \( -name "*.sh" -o -name "Dockerfile*" -o -name "*.py" -o -name "*.yml" -o -name "*.yaml" -o -name "*.txt" -o -name "*.md" \) -exec grep -l "uv pip" {} \; 2>/dev/null)

if [ -z "$files_with_uv_pip" ]; then
    echo "✅ No files found containing 'uv pip' commands!"
    echo "🎉 All registry issues have been resolved!"
    exit 0
fi

echo "📋 Found $(echo "$files_with_uv_pip" | wc -l) files containing 'uv pip' commands:"
echo "$files_with_uv_pip" | sed 's|^|   |'
echo ""

# Process each file
echo "$files_with_uv_pip" | while read -r file; do
    if [ -f "$file" ]; then
        fix_file "$file"
    fi
done

echo ""
echo "🎉 Comprehensive fix complete!"
echo ""
echo "📊 Summary:"
echo "   - Backup created at: $BACKUP_DIR"
echo "   - All uv pip commands replaced with python -m pip using standard PyPI"
echo "   - Detailed log saved to: $LOG_FILE"
echo ""
echo "📋 Files processed:"
echo "$files_with_uv_pip" | sed 's|^|   |'
echo ""
echo "🚀 You can now try building jetson-containers packages:"
echo "   jetson-containers build faiss_lite"
echo ""
echo "🔍 To verify the fix worked:"
echo "   grep -r 'uv pip' $JETSON_CONTAINERS_DIR/packages/ || echo 'No uv pip commands found!'"
echo ""
echo "🔄 To restore original files if needed:"
echo "   # Find backup files: ls $BACKUP_DIR"
echo "   # Restore: cp $BACKUP_DIR/[filename] [original_path]"
