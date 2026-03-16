#!/bin/bash
# Cron wrapper for daily brief generation.
# Install with:
#   crontab -e
#   0 6 * * * /home/paul/LedgerAI/farsight-server/cron_daily_briefs.sh >> /home/paul/LedgerAI/data/briefings/cron.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env for SMTP credentials
if [ -f "$REPO_DIR/.env" ]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

echo ""
echo "========================================"
echo "Daily Brief Cron — $(date)"
echo "========================================"

# Check Farsight is running
if ! curl -sf http://localhost:11435/health > /dev/null 2>&1; then
    echo "[cron] Farsight server not running — cannot generate briefs"
    exit 1
fi

cd "$SCRIPT_DIR"
python3 daily_briefs.py

echo "[cron] Done at $(date)"
