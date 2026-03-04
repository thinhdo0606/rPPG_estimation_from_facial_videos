@echo off
title Heart Rate Web API
echo ============================================================
echo   HEART RATE ESTIMATION - WEB API SERVER
echo ============================================================
echo.

cd /d "%~dp0backend"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting server...
python run_server.py

pause

