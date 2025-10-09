#!/bin/bash
#
# Quick RAG Container Rebuild Script
#
# Rebuilds only the RAG container with latest code changes
#

echo ""
echo "================================================================================"
echo "  🔧 RAG CONTAINER REBUILD"
echo "================================================================================"
echo ""

cd ~/LedgerAI

echo "[1/3] Stopping RAG container..."
docker compose stop rag-container
echo "     ✅ Stopped"

echo "[2/3] Rebuilding RAG container..."
docker compose build rag-container
echo "     ✅ Built"

echo "[3/3] Starting RAG container..."
docker compose up -d rag-container
echo "     ✅ Started"

echo ""
echo "================================================================================"
echo "  ✅ RAG CONTAINER REBUILD COMPLETE"
echo "================================================================================"
echo ""
echo "  Check logs:"
echo "    docker logs -f rag-container"
echo ""
echo "  Test readiness:"
echo "    curl http://localhost:11435/ready"
echo ""
echo "================================================================================"
echo ""

