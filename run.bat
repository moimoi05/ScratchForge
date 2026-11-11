@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment is missing. Run scripts\setup.ps1 first.
    pause
    exit /b 1
)

if not exist "frontend\node_modules\vite\bin\vite.js" (
    echo [ERROR] Frontend dependencies are missing. Run: npm.cmd --prefix frontend ci
    pause
    exit /b 1
)

start "ScratchForge Backend" /D "%PROJECT_ROOT%" cmd /k .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
start "ScratchForge Frontend" /D "%PROJECT_ROOT%frontend" cmd /k npm.cmd run dev -- --strictPort

echo Starting ScratchForge backend and frontend...
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo Close the two ScratchForge terminal windows to stop the app.
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:5173

endlocal
