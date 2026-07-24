#!/bin/zsh
# One-click launcher: starts the API and the frontend, opens the app.
# (Double-click in Finder, or run ./start.command)
set -e
cd "$(dirname "$0")"

echo "Starting backend on :8000 ..."
(
  cd backend
  source .venv/bin/activate
  exec uvicorn app.main:app --port 8000
) &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null" EXIT

sleep 2
(sleep 3 && open http://localhost:5173) &

echo "Starting frontend on :5173 ..."
cd frontend
npm run dev
