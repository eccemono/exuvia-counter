#!/bin/bash
# Quick start script for Exuvia Counter app

set -euo pipefail

WITH_ML="false"
HEADLESS="false"

for arg in "$@"; do
    case "$arg" in
        --with-ml)
            WITH_ML="true"
            ;;
        --headless)
            HEADLESS="true"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./run.sh [--with-ml] [--headless]"
            exit 1
            ;;
    esac
done

echo "Exuvia Counter - Setup and Run"
echo "================================"
echo "Status: STARTING"
echo ""

# Pick best available Python runtime.
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_CMD="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_CMD="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "Python 3 not found. Install Python 3.11+ first."
    exit 1
fi

echo "Python selected: $($PYTHON_CMD --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv venv
else
    echo "Activating existing virtual environment..."
fi

source venv/bin/activate

PYTHON_BIN="$(pwd)/venv/bin/python"
PIP_BIN="$(pwd)/venv/bin/pip"

echo ""
echo "Installing dependencies..."
if ! "$PIP_BIN" install --disable-pip-version-check --upgrade pip; then
    echo "pip upgrade skipped; continuing with existing pip."
fi
"$PIP_BIN" install -r requirements.txt

if [ "$WITH_ML" = "true" ]; then
    echo ""
    echo "Installing optional ML dependencies..."
    if ! "$PIP_BIN" install -r requirements-ml.txt; then
        echo "Optional ML install failed. App will run without YOLO detection."
    fi
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
    echo "Streamlit is still missing after install."
    echo "Try running manually: $PIP_BIN install streamlit"
    exit 1
fi

echo ""
echo "Setup complete"
echo ""
echo "Status: RUNNING"
echo "Starting app..."
echo ""

if ss -ltn 2>/dev/null | grep -q ':8501'; then
    echo "Port 8501 is already in use. Stopping old Streamlit instance."
    pkill -f "streamlit run app.py" || true
    sleep 2
fi

echo "Access at: http://localhost:8501"

LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$LAN_IP" ]; then
    echo "LAN access: http://$LAN_IP:8501"
fi

if command -v tailscale >/dev/null 2>&1; then
    TS_IP=$(tailscale ip -4 2>/dev/null | head -n 1)
    if [ -n "$TS_IP" ]; then
        echo "Tailscale access: http://$TS_IP:8501"
    fi
fi

echo ""
echo "Press Ctrl+C to stop"
echo "Tip: keep this terminal running; use a second terminal for Funnel commands."
echo ""

STREAMLIT_ARGS=(
    --server.address 0.0.0.0
    --server.port 8501
    --browser.gatherUsageStats false
)

if [ "$HEADLESS" = "true" ]; then
    STREAMLIT_ARGS+=(--server.headless true)
fi

"$PYTHON_BIN" -m streamlit run app.py "${STREAMLIT_ARGS[@]}"
