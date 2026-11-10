$ErrorActionPreference = 'Stop'
$Port = 8000
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run scripts/setup.ps1 first.' }
if (-not (Test-Path -LiteralPath 'frontend/dist/index.html')) {
    & npm.cmd --prefix frontend run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
}
Write-Host "ScratchForge: http://127.0.0.1:$Port (Ctrl+C to stop)"
& $pythonExe -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
if ($LASTEXITCODE -ne 0) { throw "Studio exited with code $LASTEXITCODE" }
