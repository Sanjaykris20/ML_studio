@echo off
title ML Studio - Dual Startup Launcher
echo ========================================================
echo       ML STUDIO - INTERACTIVE MACHINE LEARNING PLATFORM
echo ========================================================
echo.
echo Launching full-stack ML Studio application...
echo - Backend:  FastAPI + SQLite + PyTorch (Port 8000)
echo - Frontend: React + TypeScript + Tailwind (Port 5173)
echo.

:: Start Backend in separate window
echo Starting Backend Server window...
start "ML Studio - Backend (Port 8000)" "%~dp0start_backend.bat"

:: Brief 2-second pause to allow backend init
timeout /t 2 /nobreak >nul

:: Start Frontend in separate window
echo Starting Frontend Application window...
start "ML Studio - Frontend (Port 5173)" "%~dp0start_frontend.bat"

echo.
echo ========================================================
echo Both services launched successfully!
echo - Web Dashboard:  http://localhost:5173
echo - API Docs (Swagger): http://127.0.0.1:8000/docs
echo ========================================================
echo.
pause
