$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Windows virtual environment not found. Run .\setup_windows.ps1 first." -ForegroundColor Yellow
    exit 1
}

Set-Location $projectRoot

$gitDirs = @(
    "C:\Program Files\Git\cmd",
    "C:\Program Files\Git\bin"
)
foreach ($dir in $gitDirs) {
    if ((Test-Path $dir) -and ($env:Path -notmatch [regex]::Escape($dir))) {
        $env:Path = "$dir;$env:Path"
    }
}

& $pythonExe -m streamlit run app.py
