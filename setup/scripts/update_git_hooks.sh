#!/bin/bash
# Quick script to update git hooks from template
# Run this after git pull to ensure hooks are up to date

LEDGERAI_DIR="${LEDGERAI_DIR:-$HOME/LedgerAI}"
GIT_HOOKS_DIR="$LEDGERAI_DIR/.git/hooks"
POST_MERGE_TEMPLATE="$LEDGERAI_DIR/setup/scripts/git-hooks/post-merge"

if [ ! -d "$GIT_HOOKS_DIR" ]; then
    echo "❌ Git hooks directory not found: $GIT_HOOKS_DIR"
    exit 1
fi

if [ ! -f "$POST_MERGE_TEMPLATE" ]; then
    echo "❌ Post-merge template not found: $POST_MERGE_TEMPLATE"
    exit 1
fi

# Update post-merge hook
cp "$POST_MERGE_TEMPLATE" "$GIT_HOOKS_DIR/post-merge"
chmod +x "$GIT_HOOKS_DIR/post-merge"
echo "✅ Git post-merge hook updated"

