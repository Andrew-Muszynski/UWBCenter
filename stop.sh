#!/usr/bin/env bash
cd "$(dirname "$0")"
PID_FILE=".dashboard.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Dashboard stopped (PID $PID)"
    else
        echo "Process $PID not running"
    fi
    rm -f "$PID_FILE"
else
    echo "No PID file found — dashboard may not be running"
fi
