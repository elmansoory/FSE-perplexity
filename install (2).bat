@echo off
setlocal
cd /d %~dp0
where python >nul 2>nul
if not %errorlevel%==0 (
    echo Python not found. Please install Python first from python.org and check Add Python to PATH.
    pause
    exit /b 1
)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pause
