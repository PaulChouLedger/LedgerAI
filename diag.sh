#!/bin/bash
# Aura Puck Diagnostic — run this and send the output to Paul
# Usage: bash diag.sh | tee /tmp/diag.txt

echo "========================================="
echo "  AURA PUCK DIAGNOSTIC — $(date)"
echo "  Hostname: $(hostname)"
echo "========================================="
echo ""

echo "--- NETWORK ---"
echo "IP addresses:"
ip -4 addr show | grep inet | grep -v 127.0.0.1 | awk '{print "  " $2}'
echo "Tailscale status:"
tailscale status 2>/dev/null | head -5 || echo "  Tailscale not running"
echo "Internet:"
ping -c 1 -W 3 8.8.8.8 > /dev/null 2>&1 && echo "  OK" || echo "  NO INTERNET"
echo ""

echo "--- SERVICES ---"
for svc in aura4 aura-telegram ollama; do
  status=$(systemctl is-active $svc 2>/dev/null || echo "not found")
  enabled=$(systemctl is-enabled $svc 2>/dev/null || echo "n/a")
  echo "  $svc: $status (enabled: $enabled)"
done
echo ""

echo "--- PROCESSES ---"
echo "  Bot:"
pgrep -af 'bot.py' 2>/dev/null || echo "    not running"
echo "  Ollama:"
pgrep -af 'ollama' 2>/dev/null || echo "    not running"
echo "  Aura main:"
pgrep -af 'aura.py' 2>/dev/null || echo "    not running"
echo "  Whisper:"
pgrep -af 'whisper\|container_rest' 2>/dev/null || echo "    not running"
echo ""

echo "--- GPU ---"
if command -v tegrastats &>/dev/null; then
  timeout 2 tegrastats --interval 1000 2>/dev/null | head -1 || echo "  tegrastats timeout"
elif command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
else
  echo "  No GPU tool found"
fi
echo ""

echo "--- DISK ---"
df -h / | tail -1 | awk '{print "  Root: " $3 " used / " $2 " total (" $5 " full)"}'
df -h /home 2>/dev/null | tail -1 | awk '{print "  Home: " $3 " used / " $2 " total (" $5 " full)"}'
echo ""

echo "--- TEMPS ---"
if [ -d /sys/class/thermal ]; then
  for z in /sys/class/thermal/thermal_zone*/temp; do
    zone=$(basename $(dirname $z))
    temp=$(cat $z 2>/dev/null)
    if [ -n "$temp" ]; then
      echo "  $zone: $((temp/1000))°C"
    fi
  done
else
  echo "  No thermal zones found"
fi
echo ""

echo "--- RECENT LOGS (last 20 lines of TG bot) ---"
tail -20 /tmp/aura-telegram.log 2>/dev/null || echo "  No TG bot log found"
echo ""

echo "--- GIT STATUS ---"
cd ~/Aura4 2>/dev/null || cd ~/LedgerAI 2>/dev/null || echo "  Repo not found"
echo "  Branch: $(git branch --show-current 2>/dev/null)"
echo "  Last commit: $(git log --oneline -1 2>/dev/null)"
echo "  Behind remote: $(git fetch --dry-run 2>&1 | head -3)"
echo ""

echo "========================================="
echo "  DONE — send this output to Paul"
echo "========================================="
