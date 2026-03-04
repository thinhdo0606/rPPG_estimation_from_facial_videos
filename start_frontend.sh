#!/bin/bash
echo "============================================================"
echo "  HEART RATE ESTIMATION - WEB FRONTEND"
echo "============================================================"
echo

cd "$(dirname "$0")/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo
echo "Starting development server..."
npm run dev

