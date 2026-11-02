@echo off
title ML Studio - Frontend Dashboard
echo ========================================================
echo        ML STUDIO - INTERACTIVE FRONTEND APPLICATION
echo ========================================================
echo.

cd /d "%~dp0\frontend"

if not exist "node_modules\" (
    echo [INFO] First-time setup: Installing frontend dependencies...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install npm packages.
        pause
        exit /b 1
    )
)

echo Starting Vite React TypeScript development server...
echo Frontend will be accessible at: http://localhost:5173
echo.
call npm run dev -- --open

pause
