#!/usr/bin/env bash
# ============================================================================
# start-dev.sh — Launch the full ABSA development stack
# ============================================================================

set -e

# Change to the script's directory
cd "$(dirname "$0")"
ROOT_DIR=$(pwd)

# -- Pre-flight checks -------------------------------------------------------

PY="$ROOT_DIR/.venv/bin/python"
if [ ! -f "$PY" ]; then
    echo "Python venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

VITE_BIN="$ROOT_DIR/frontend/node_modules/.bin/vite"
if [ ! -f "$VITE_BIN" ]; then
    echo "Frontend dependencies not found. Run: cd frontend && npm install"
    exit 1
fi

if command -v dotnet >/dev/null 2>&1 && [ -f "$ROOT_DIR/backend-csharp/BackendGateway.csproj" ]; then
    HAS_CSHARP=true
else
    HAS_CSHARP=false
fi

# -- Kill existing processes on target ports ----------------------------------

for PORT in 8000 8001 5173; do
    PID=$(lsof -ti tcp:$PORT) || true
    if [ ! -z "$PID" ]; then
        echo "  Killing existing process on port $PORT (PID $PID)..."
        kill -9 $PID
    fi
done
sleep 0.5

# Handle cleanup on script exit
cleanup() {
    echo ""
    echo "Stopping all services..."
    [ -n "$PID_BACKEND" ] && kill $PID_BACKEND 2>/dev/null || true
    [ -n "$PID_GATEWAY" ] && kill $PID_GATEWAY 2>/dev/null || true
    [ -n "$PID_FRONTEND" ] && kill $PID_FRONTEND 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo ""
echo "[1/3] Starting Python backend (http://127.0.0.1:8000)..."
cd "$ROOT_DIR"
"$PY" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
PID_BACKEND=$!
echo "      Waiting for Python Backend to be ready (Model loading may take ~10 seconds)..."
while ! curl -s http://127.0.0.1:8000/health >/dev/null; do
    sleep 1
done
echo "      OK"

if [ "$HAS_CSHARP" = true ]; then
    echo "[2/3] Starting C# gateway (http://127.0.0.1:8001)..."
    cd "$ROOT_DIR/backend-csharp"
    ASPNETCORE_ENVIRONMENT=Development dotnet run --project BackendGateway.csproj --framework net8.0 --no-restore --urls http://127.0.0.1:8001 &
    PID_GATEWAY=$!
    echo "      OK"
else
    echo "[2/3] Skipping C# gateway (dotnet or project not found)."
fi

if [ "$HAS_CSHARP" = true ]; then
    echo "      Waiting for C# Gateway to be ready on port 8001..."
    while ! curl -s http://127.0.0.1:8001/health >/dev/null; do
        sleep 0.5
    done
fi

echo "[3/3] Starting frontend (http://127.0.0.1:5173)..."
cd "$ROOT_DIR/frontend"
"$VITE_BIN" --host 127.0.0.1 --port 5173 &
PID_FRONTEND=$!
echo "      OK"

echo ""
echo "================================================"
echo "  All services started successfully!"
echo ""
echo "  Frontend:  http://127.0.0.1:5173"
echo "  Gateway:   http://127.0.0.1:8001"
echo "  Backend:   http://127.0.0.1:8000"
echo ""
echo "  Model loading may take ~10 seconds."
echo "  To stop all services, press Ctrl+C here."
echo "================================================"

# Wait for all background processes
wait
