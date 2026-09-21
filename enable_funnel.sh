#!/bin/bash
set -e

echo "Enabling public HTTPS access for Exuvia app via Tailscale Funnel..."

if ! command -v tailscale >/dev/null 2>&1; then
    echo "tailscale command not found. Install Tailscale first."
    exit 1
fi

if ! tailscale status >/dev/null 2>&1; then
    echo "Tailscale is not connected. Run: sudo tailscale up"
    exit 1
fi

if ! ss -ltn 2>/dev/null | grep -q ':8501'; then
    echo "App does not appear to be listening on port 8501."
    echo "Start the app first with: ./run.sh"
    exit 1
fi

# Expose the Streamlit port publicly over HTTPS.
sudo tailscale funnel --bg 8501

echo ""
echo "Funnel enabled. Current status:"
tailscale funnel status || true

echo ""
echo "To disable later, run:"
echo "sudo tailscale funnel reset"
