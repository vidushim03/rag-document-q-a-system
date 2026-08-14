$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonCommand = $null

if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCommand = "py -3.11"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCommand = "python"
}

if (-not $pythonCommand) {
    Write-Host "Python was not found on PATH." -ForegroundColor Yellow
    Write-Host "Install Python 3.11 or newer from https://www.python.org/downloads/windows/ and enable 'Add python.exe to PATH'." -ForegroundColor Yellow
    exit 1
}

$venvConfig = Join-Path $projectRoot ".venv\pyvenv.cfg"
$hasUnixVenv = (Test-Path $venvConfig) -and ((Get-Content $venvConfig -Raw) -match "/Library/Frameworks|/usr/local/bin|/Users/")

if ($hasUnixVenv) {
    Write-Host "A non-Windows virtual environment was found in .venv." -ForegroundColor Yellow
    Write-Host "Remove .venv and rerun this script to create a Windows-compatible environment." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    Invoke-Expression "$pythonCommand -m venv .venv"
}

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

Write-Host "Installing core dependencies..." -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Optional packages for transformer embeddings can be installed with:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Copy .env.example to .env and add your Groq API key." -ForegroundColor Green
Write-Host "2. Run .\run_app.ps1" -ForegroundColor Green
