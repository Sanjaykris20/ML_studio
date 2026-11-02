@echo off
title ML Studio - Project Backup & Export Archive
echo ========================================================
echo         ML STUDIO - PROJECT EXPORT & BACKUP
echo ========================================================
echo.

cd /d "%~dp0"

set "PYTHON_CMD=python"
if exist "C:\Users\GURU PRASATH JV\AppData\Local\Python\bin\python.exe" (
    set "PYTHON_CMD=C:\Users\GURU PRASATH JV\AppData\Local\Python\bin\python.exe"
)

"%PYTHON_CMD%" export_zip.py

echo.
pause
