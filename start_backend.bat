@echo off
title ML Studio - Backend Server
echo ========================================================
echo         ML STUDIO - INTERACTIVE BACKEND SERVER
echo ========================================================
echo.

cd /d "%~dp0\backend"

:: Locate Python
set "PYTHON_CMD=python"
where %PYTHON_CMD% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if exist "C:\Users\GURU PRASATH JV\AppData\Local\Python\bin\python.exe" (
        set "PYTHON_CMD=C:\Users\GURU PRASATH JV\AppData\Local\Python\bin\python.exe"
    ) else (
        echo [ERROR] Python not found on PATH. Please install Python 3.10+ or set PATH.
        pause
        exit /b 1
    )
)

echo [1/2] Initializing SQLite database and built-in datasets...
"%PYTHON_CMD%" -c "from app.db.session import init_db; init_db()"
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Database init command reported a warning, proceeding to start server...
)

echo [2/2] Starting High-Availability FastAPI backend on http://127.0.0.1:8000...
echo Swagger UI Documentation: http://127.0.0.1:8000/docs
echo.
"%PYTHON_CMD%" run_server.py

pause
