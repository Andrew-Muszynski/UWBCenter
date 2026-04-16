#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
nohup python3 uwb_dashboard.py > dashboard.log 2>&1 &
echo $! > .dashboard.pid
echo "Dashboard started (PID $(cat .dashboard.pid)) → http://localhost:5050"
