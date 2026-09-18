#!/bin/bash
LOGDIR=/home/aditya/stock_alpha/logs
LATEST=$(ls -t $LOGDIR/*.log 2>/dev/null | head -1)

if [ -z "$LATEST" ]; then
    echo "No logs yet."
else
    echo "Latest log: $LATEST"
    echo "================================="
    tail -50 $LATEST
fi
