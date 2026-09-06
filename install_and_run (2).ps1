$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-Python {
    try { python --version > $null 2>&1; return $true } catch { return $false }
}

if (-not (Test-Python)) {
    Write-Host "Python is not installed or not in PATH. Install Python first and enable Add Python to PATH." -ForegroundColor Red
    exit 1
}

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
