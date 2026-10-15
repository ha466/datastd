@echo off
REM DataSTD — Windows Startup Script
REM Powered by audio.cpp (TTS) and llama.cpp (LLM textgen) — zero external daemons needed!
echo.
echo ========================================
echo   DataSTD - Synthetic Voice Dataset
echo ========================================
echo.

REM Check uv is installed
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

REM Setup: download native binaries (audio.cpp + llama-server) and default models
if not exist "bin\audiocpp_cli.exe" (
    echo [SETUP] Downloading audio.cpp and llama.cpp binaries and models...
    uv run python setup.py --platform windows --default
)

if not exist "models" (
    echo [SETUP] Downloading default models...
    uv run python setup.py --models-only --default
)

REM Start the server
echo [START] Launching DataSTD server...
uv run datastd

pause
