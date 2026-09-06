@echo off
setlocal
cd /d %~dp0
where python >nul 2>nul
if %errorlevel%==0 (
    python -m streamlit run app.py
    goto :eof
)
where py >nul 2>nul
if %errorlevel%==0 (
    py -m streamlit run app.py
    goto :eof
)
echo Python is not installed or not found in PATH.
pause
