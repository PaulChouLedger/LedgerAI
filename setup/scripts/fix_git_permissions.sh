#!/bin/bash
# Fix Git repository permissions
# Run this if you get "insufficient permission for adding an object to repository database" errors
# This typically happens when files were created/modified by root (sudo) or Docker containers

LEDGERAI_DIR="${LEDGERAI_DIR:-$HOME/LedgerAI}"
AURA_USER="${AURA_USER:-$(whoami)}"

echo "🔧 Fixing Git repository permissions..."
echo "Repository: $LEDGERAI_DIR"
echo "User: $AURA_USER"

if [ ! -d "$LEDGERAI_DIR/.git" ]; then
    echo "❌ Not a git repository: $LEDGERAI_DIR"
    exit 1
fi

# Fix ownership of .git directory and all files
echo "📝 Fixing ownership of .git directory..."
sudo chown -R "$AURA_USER:$AURA_USER" "$LEDGERAI_DIR/.git"

# Fix permissions on .git directory (read/write for owner, read for group/others)
echo "📝 Fixing permissions on .git directory..."
chmod -R u+rwX,go+rX "$LEDGERAI_DIR/.git"

# Fix ownership of all files in repository (in case some were created by root/Docker)
echo "📝 Fixing ownership of repository files..."
sudo chown -R "$AURA_USER:$AURA_USER" "$LEDGERAI_DIR"

# Ensure .git/objects directory is writable
if [ -d "$LEDGERAI_DIR/.git/objects" ]; then
    echo "📝 Ensuring .git/objects is writable..."
    chmod -R u+rwX "$LEDGERAI_DIR/.git/objects"
fi

# Ensure .git/index is writable
if [ -f "$LEDGERAI_DIR/.git/index" ]; then
    echo "📝 Ensuring .git/index is writable..."
    chmod u+rw "$LEDGERAI_DIR/.git/index"
fi

echo "✅ Git permissions fixed!"
echo ""
echo "You can now run 'git pull' without sudo."
