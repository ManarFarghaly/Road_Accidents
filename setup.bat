@echo off
REM =============================================================================
REM  Road Accidents Project — one-command Windows setup
REM
REM  Run this once after cloning the repo:
REM      setup.bat
REM
REM  What it does:
REM    1. Checks for Python 3.11 / 3.12 (required; 3.14 is NOT supported)
REM    2. Creates a virtual environment in road_env\
REM    3. Installs all Python dependencies from requirements.txt
REM    4. Downloads winutils.exe + hadoop.dll into winutils\bin\
REM    5. Prints next-step instructions
REM =============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   Road Accidents Project — Windows Setup
echo ============================================================
echo.

REM --------------------------------------------------------------------------
REM 1. Check Python version
REM --------------------------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.11 or 3.12 from https://python.org
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if "%PY_MAJOR%" neq "3" (
    echo [ERROR] Python 3 is required. Found: %PY_VER%
    exit /b 1
)
if %PY_MINOR% LSS 11 (
    echo [ERROR] Python 3.11 or later required. Found: %PY_VER%
    exit /b 1
)
echo [OK] Python %PY_VER%

REM --------------------------------------------------------------------------
REM 2. Create virtual environment
REM --------------------------------------------------------------------------
if exist road_env\Scripts\activate.bat (
    echo [OK] Virtual environment already exists — skipping creation.
) else (
    echo [..] Creating virtual environment in road_env\ ...
    python -m venv road_env
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM --------------------------------------------------------------------------
REM 3. Activate venv and install dependencies
REM --------------------------------------------------------------------------
echo [..] Installing dependencies (this may take a few minutes) ...
call road_env\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] pip install failed. Check requirements.txt and your internet connection.
    exit /b 1
)
echo [OK] Dependencies installed.

REM --------------------------------------------------------------------------
REM 4. Download winutils
REM --------------------------------------------------------------------------
echo [..] Downloading winutils.exe and hadoop.dll ...
python scripts\get_winutils.py
if errorlevel 1 (
    echo [ERROR] winutils download failed. Check your internet connection.
    echo         You can retry later with:  python scripts\get_winutils.py
)

REM --------------------------------------------------------------------------
REM 5. Done
REM --------------------------------------------------------------------------
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   Activate the virtual environment:
echo       road_env\Scripts\activate
echo.
echo   Then run the pipeline stages:
echo       python -m src.data.ingest
echo       python -m src.data.validate
echo       python -m src.preprocessing.run
echo.
echo   Or run tests:
echo       python -m tests.test_preprocessing
echo.

endlocal
