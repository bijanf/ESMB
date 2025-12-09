#!/bin/bash

# start_dashboard.sh
# Robust startup script for Chronos-ESM Dashboard

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Define paths
PROJECT_ROOT=".."
VENV_PATH="$PROJECT_ROOT/venv"
SERVER_DIR="server"
CLIENT_DIR="client"

# Check for venv
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null
    fi
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "=================================================="
echo "   Chronos-ESM Dashboard Startup"
echo "=================================================="

# 1. Start Backend
echo "[1/3] Starting Backend Server (FastAPI)..."
echo "      Logs will be written to backend.log"

# Kill any existing process on port 8000
fuser -k 8000/tcp > /dev/null 2>&1 || true

cd "$SERVER_DIR"
python app.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 2. Wait for Backend to be Ready
echo "[2/3] Waiting for Backend to initialize..."
echo "      (This may take a minute for JAX compilation)"

# Loop until backend is responsive (returns 200 OK)
MAX_RETRIES=60
COUNT=0
while ! python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/state')" > /dev/null 2>&1; do
    sleep 1
    COUNT=$((COUNT+1))
    echo -ne "      Waiting... ${COUNT}s\r"
    
    # Check if process died
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo ""
        echo "Error: Backend process died unexpectedly."
        echo "Check dashboard/backend.log for details."
        exit 1
    fi

    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo ""
        echo "Error: Timed out waiting for backend."
        exit 1
    fi
done
echo ""
echo "      Backend is READY!"

# 3. Start Frontend
echo "[3/3] Starting Frontend (Vite)..."
cd "$CLIENT_DIR"
npm run dev

# Cleanup is handled by trap when user hits Ctrl+C
