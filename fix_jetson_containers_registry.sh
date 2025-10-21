#!/bin/bash

# Script to fix all uv pip install commands in jetson-containers packages
# This replaces uv pip install with python -m pip install using standard PyPI registry

set -e

JETSON_CONTAINERS_DIR="$HOME/jetson-containers"
BACKUP_DIR="/tmp/jetson_containers_backup_$(date +%Y%m%d_%H%M%S)"

echo "🔧 Fixing jetson-containers registry issues..."
echo "📁 Jetson containers directory: $JETSON_CONTAINERS_DIR"
echo "💾 Backup directory: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Function to fix a single install.sh file
fix_install_script() {
    local file="$1"
    local relative_path="${file#$JETSON_CONTAINERS_DIR/}"
    
    echo "🔍 Processing: $relative_path"
    
    # Create backup
    cp "$file" "$BACKUP_DIR/${relative_path//\//_}"
    
    # Replace uv pip install commands
    sed -i 's/uv pip install/python -m pip install --index-url "https:\/\/pypi.org\/simple"/g' "$file"
    
    # Replace uv pip show commands
    sed -i 's/uv pip show/python -m pip show/g' "$file"
    
    # Replace uv pip list commands
    sed -i 's/uv pip list/python -m pip list/g' "$file"
    
    # Replace uv pip freeze commands
    sed -i 's/uv pip freeze/python -m pip freeze/g' "$file"
    
    # Replace uv pip uninstall commands
    sed -i 's/uv pip uninstall/python -m pip uninstall/g' "$file"
    
    echo "✅ Fixed: $relative_path"
}

# Find all install.sh files in the packages directory
echo "🔍 Searching for install.sh files..."

find "$JETSON_CONTAINERS_DIR/packages" -name "install.sh" -type f | while read -r file; do
    # Check if the file contains uv pip commands
    if grep -q "uv pip" "$file"; then
        fix_install_script "$file"
    else
        echo "⏭️  Skipping (no uv pip commands): ${file#$JETSON_CONTAINERS_DIR/}"
    fi
done

echo ""
echo "🎉 Fix complete!"
echo "📊 Summary:"
echo "   - Backup created at: $BACKUP_DIR"
echo "   - All uv pip commands replaced with python -m pip using standard PyPI"
echo ""
echo "🚀 You can now try building jetson-containers packages again:"
echo "   jetson-containers build faiss_lite"
echo ""
echo "🔄 To restore original files if needed:"
echo "   # Find the backup file: ls $BACKUP_DIR"
echo "   # Restore: cp $BACKUP_DIR/[filename] [original_path]"
