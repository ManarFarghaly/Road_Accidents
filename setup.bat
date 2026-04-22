@echo off
REM =============================================================================
REM  Road Accidents Project — one-command Windows setup
REM
REM  Run this once after cloning the repo:
REM      setup.bat
REM
REM  What it does:
REM    1. Checks for Python 3.11+
REM    2. Installs all Python dependencies into the system Python
REM    3. Downloads winutils.exe + hadoop.dll into winutils\bin\
REM =============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   Road Accidents Project -- Windows Setup
echo ============================================================
echo.

REM --------------------------------------------------------------------------
REM 1. Check Python
REM --------------------------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.11 or later from https://python.org
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if "%PY_MAJOR%" neq "3" (
    echo [ERROR] Python 3 required. Found: %PY_VER%
    exit /b 1
)
if %PY_MINOR% LSS 11 (
    echo [ERROR] Python 3.11 or later required. Found: %PY_VER%
    exit /b 1
)
echo [OK] Python %PY_VER%

REM --------------------------------------------------------------------------
REM 2. Install dependencies
REM --------------------------------------------------------------------------
echo [..] Installing dependencies (this may take a few minutes) ...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection.
    exit /b 1
)
echo [OK] Dependencies installed.

REM --------------------------------------------------------------------------
REM 3. Download winutils
REM --------------------------------------------------------------------------
echo [..] Downloading winutils.exe and hadoop.dll ...
python scripts\get_winutils.py
if errorlevel 1 (
    echo [ERROR] winutils download failed. Retry with: python scripts\get_winutils.py
    exit /b 1
)

REM --------------------------------------------------------------------------
REM 4. Done
REM --------------------------------------------------------------------------
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   Run the pipeline stages:
echo       python -m src.preprocessing.run
echo.

endlocal
