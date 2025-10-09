#!/bin/bash
#
# Quick Whisper Container Rebuild Script
#

echo ""
echo "================================================================================"
echo "  🔧 WHISPER CONTAINER REBUILD"
echo "================================================================================"
echo ""

cd ~/LedgerAI

echo "[1/3] Stopping Whisper container..."
docker compose stop whisper-container
echo "     ✅ Stopped"

echo "[2/3] Rebuilding Whisper container..."
docker compose build whisper-container
echo "     ✅ Built"

echo "[3/3] Starting Whisper container..."
docker compose up -d whisper-container
echo "     ✅ Started"

echo ""
echo "================================================================================"
echo "  ✅ WHISPER CONTAINER REBUILD COMPLETE"
echo "================================================================================"
echo ""
echo "  Test name guidance:"
echo "    1. Say: 'My name is Rafael'"
echo "    2. Look for: [Listener] 👤 User name detected: 'Rafael'"
echo "    3. Say: 'Who is Rafael?'"
echo "    4. Look for: [Whisper] 🎯 Using name guidance: 'Rafael'"
echo ""
echo "  Check logs:"
echo "    docker logs -f whisper-container"
echo ""
echo "================================================================================"
echo ""

