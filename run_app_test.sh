#!/bin/bash
# Start app in background
python3 app.py > app_output.log 2>&1 &
APP_PID=$!
echo "App started with PID $APP_PID"

# Wait for it to start
sleep 5

# Trigger update
echo "Triggering update..."
curl -X POST -F "game=2024010074" http://localhost:5001/replot

# Check if it's still running
if ps -p $APP_PID > /dev/null; then
    echo "App is still running."
    kill $APP_PID
else
    echo "App crashed!"
    cat app_output.log
fi
