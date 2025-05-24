@echo off
chcp 65001 >nul
title Whisper Diarization Pipeline - Windows Setup

echo.
echo =====================================================
echo  Whisper Diarization Pipeline - Windows Setup
echo =====================================================
echo.

echo This script will help set up the system on Windows.
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Administrator privileges required.
    echo.
    echo 1. Right-click on this file
    echo 2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo Checking PowerShell...
powershell -Command "Get-Host" >nul 2>&1
if errorlevel 1 (
    echo PowerShell not found. Update Windows.
    pause
    exit /b 1
)

echo.
echo Choose installation method:
echo.
echo 1. Automatic installation (PowerShell)
echo 2. Download Docker Desktop manually
echo 3. Show instructions
echo.

set /p choice="Your choice (1-3): "

if "%choice%"=="1" goto auto_install
if "%choice%"=="2" goto manual_install
if "%choice%"=="3" goto show_instructions

echo Invalid choice.
pause
exit /b 1

:auto_install
echo.
echo Running automatic installation...
echo.
powershell -ExecutionPolicy Bypass -File setup-complete.ps1
goto end

:manual_install
echo.
echo Manual installation:
echo.
echo 1. Download Docker Desktop:
echo    https://www.docker.com/products/docker-desktop
echo.
echo 2. Install and reboot computer
echo.
echo 3. Start Docker Desktop
echo.
echo 4. Place audio file in input\ folder
echo.
echo 5. Run: run.bat your_audio.wav
echo.
start https://www.docker.com/products/docker-desktop
goto end

:show_instructions
echo.
echo Quick start:
echo.
echo 1. This script will install Docker and WSL2
echo 2. Automatically configure NVIDIA GPU (if available)
echo 3. Build Docker images
echo 4. System will be ready to use
echo.
echo For RTX 3080: get 10-20x acceleration!
echo.
echo After installation:
echo   copy audio.wav input\
echo   run.bat audio.wav
echo.
goto :eof

:end
echo.
echo Done! System configured.
echo.
echo Next steps:
echo 1. Place audio file: copy audio.wav input\
echo 2. Run: run.bat audio.wav
echo 3. Results in: output\
echo.
pause 