#!/bin/bash
set -euo pipefail

# One-command public launcher:
# 1) Starts app in background if needed
# 2) Enables Tailscale Funnel
# 3) Prints public HTTPS URL

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

if ! command -v tailscale >/dev/null 2>&1; then
  echo "tailscale command not found. Install Tailscale first."
  exit 1
fi

if ! tailscale status >/dev/null 2>&1; then
  echo "Tailscale is not connected. Run: sudo tailscale up"
  exit 1
fi

# Start app in background if not already listening
if ! ss -ltn 2>/dev/null | grep -q ':8501'; then
  echo "Starting app in background..."
  nohup ./run.sh > streamlit.log 2>&1 &

  for _ in $(seq 1 30); do
    if ss -ltn 2>/dev/null | grep -q ':8501'; then
      break
    fi
    sleep 1
  done
fi

if ! ss -ltn 2>/dev/null | grep -q ':8501'; then
  echo "App failed to start on port 8501. Check streamlit.log"
  exit 1
fi

echo "Enabling public HTTPS via Funnel..."
sudo tailscale funnel --bg 8501

echo ""
echo "Funnel status:"
tailscale funnel status

echo ""
echo "If you need to stop public access later:"
echo "sudo tailscale funnel reset"
