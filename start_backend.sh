#!/bin/bash
echo "============================================================"
echo "  HEART RATE ESTIMATION - WEB API SERVER"
echo "============================================================"
echo

cd "$(dirname "$0")/backend"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

echo
echo "Starting server..."
python run_server.py

