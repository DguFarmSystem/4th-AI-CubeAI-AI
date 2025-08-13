#!/bin/bash

# Wait for postgres to be ready (simple wait instead of script)
sleep 10

# Start FastAPI app in background
uvicorn app:app --host 0.0.0.0 --port 80 &
APP_PID=$!

# Wait a bit for the app to start
sleep 5

# Run auto-ingest
python startup.py

# Wait for the main app
wait $APP_PID