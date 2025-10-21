#!/bin/bash
# Test if models load only once

echo "========================================"
echo "DIAGNOSTIC: Model Loading Test"
echo "========================================"
echo ""

cd /Users/rcabello/Documents/GitHub/LedgerAI

# Stop everything
echo "1. Stopping all containers..."
docker-compose down
docker rm -f $(docker ps -aq --filter name=aura) 2>/dev/null

echo ""
echo "2. Starting ONLY LLM container..."
echo "   (Watch for model loading messages)"
echo ""

# Start just LLM, show logs in real-time
docker-compose up llm 2>&1 | while read line; do
    echo "$line"
    
    # Count model loading messages
    if echo "$line" | grep -q "Loading COMPLEX model"; then
        echo ""
        echo ">>> DETECTED: Complex model loading (count this!)"
        echo ""
    fi
    
    if echo "$line" | grep -q "Loading SIMPLE model"; then
        echo ""
        echo ">>> DETECTED: Simple model loading (count this!)"
        echo ""
    fi
    
    if echo "$line" | grep -q "Running on http"; then
        echo ""
        echo ">>> Flask started - container is ready"
        echo ""
        echo "Press Ctrl+C to stop and see summary"
    fi
done

