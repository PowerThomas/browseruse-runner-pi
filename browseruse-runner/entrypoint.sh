#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:99

# Start virtual display for Chromium/Playwright.
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp &
XVFB_PID=$!

# Lightweight window manager keeps VNC happy.
fluxbox >/tmp/fluxbox.log 2>&1 &

# VNC server bound to localhost inside the container; host binding is limited via compose.
x11vnc -display :99 -localhost -nopw -shared -forever -rfbport 5900 >/tmp/x11vnc.log 2>&1 &

# noVNC websockets proxy.
websockify --web=/usr/share/novnc/ 7900 localhost:5900 >/tmp/websockify.log 2>&1 &

# Run the FastAPI app.
uvicorn server:app --host 0.0.0.0 --port 8000
