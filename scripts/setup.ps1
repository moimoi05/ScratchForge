param(
    [string]$Source = (Join-Path $PSScriptRoot '../../GAN_Project'),
    [switch]$Gpu,
    [switch]$DownloadModel,
    [switch]$Dev
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}

if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv')
}
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
Invoke-Checked $pythonExe @('-c', 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"')
Invoke-Checked $pythonExe @('-m', 'pip', 'install', '--upgrade', 'pip==26.3')
$requirements = if ($Dev) { 'requirements-dev.txt' } else { 'requirements.txt' }
Invoke-Checked $pythonExe @('-m', 'pip', 'install', '-r', $requirements)
if ($Gpu) {
    Invoke-Checked $pythonExe @('-m', 'pip', 'install', '-r', 'requirements-gpu.txt')
    Invoke-Checked $pythonExe @('-m', 'pip', 'install', '-r', 'requirements-ai.txt')
    Invoke-Checked $pythonExe @('scripts/check_gpu.py', '--require-cuda')
}
if ($DownloadModel) {
    if (-not $Gpu) { throw 'Use -Gpu -DownloadModel together on the NVIDIA machine.' }
    Invoke-Checked $pythonExe @('scripts/download_model.py')
}
Invoke-Checked $pythonExe @('scripts/import_f971.py', '--source', $Source)
Invoke-Checked $pythonExe @('scripts/build_library.py')
Invoke-Checked $pythonExe @('scripts/verify_dataset.py')
Invoke-Checked 'npm.cmd' @('--prefix', 'frontend', 'ci')
Invoke-Checked 'npm.cmd' @('--prefix', 'frontend', 'run', 'build')
Write-Host 'Setup complete. Start the studio: powershell -ExecutionPolicy Bypass -File scripts/start.ps1'
if ($Gpu -and $DownloadModel) {
    Write-Host 'Next run scripts/check_gpu.py --require-cuda --inference-smoke on real hardware.'
}
