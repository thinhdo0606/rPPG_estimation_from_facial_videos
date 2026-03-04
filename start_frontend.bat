@echo off
title Heart Rate Web Frontend
echo ============================================================
echo   HEART RATE ESTIMATION - WEB FRONTEND
echo ============================================================
echo.

cd /d "%~dp0frontend"

if not exist node_modules (
    echo Installing dependencies...
    npm install
)

echo.
echo Starting development server...
npm run dev

pause

