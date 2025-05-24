@echo off
REM Whisper Diarization Pipeline - Windows Batch Launcher
REM Simple launcher for Windows without PowerShell

title Whisper Diarization Pipeline

echo.
echo ===========================================
echo  Whisper Diarization Pipeline
echo ===========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker not installed. Install Docker Desktop and try again.
    echo Download from https://www.docker.com/products/docker-desktop
    echo Or run PowerShell as Administrator and use:
    echo    PowerShell -ExecutionPolicy Bypass -File install-docker.ps1
    pause
    exit /b 1
)

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo Docker not running. Start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "input" mkdir input
if not exist "output" mkdir output
if not exist "models" mkdir models

REM Create .env file if it doesn't exist
if not exist ".env" (
    if exist "env.example" (
        copy "env.example" ".env" >nul
        echo Created .env file from template
        echo Edit .env file and add HUGGINGFACE_TOKEN if needed
    )
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    set "GPU_AVAILABLE=false"
    set "PROFILE=cpu"
    set "SERVICE=whisper-diarization-cpu"
    echo NVIDIA GPU not detected - using CPU version
) else (
    set "GPU_AVAILABLE=true"
    set "PROFILE=gpu"
    set "SERVICE=whisper-diarization-gpu"
    echo NVIDIA GPU detected - using GPU version
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    set "COMPOSE_CMD=docker compose"
) else (
    set "COMPOSE_CMD=docker-compose"
)

REM If no arguments provided, show help
if "%~1"=="" (
    echo Showing help...
    echo.
    %COMPOSE_CMD% --profile %PROFILE% run --rm %SERVICE% --help
    echo.
    echo Usage examples:
    echo   run.bat audio.wav                    # Basic processing
    echo   run.bat audio.wav --model large     # Large model
    echo   run.bat audio.wav --max-speakers 5  # Maximum 5 speakers
    echo.
    echo Place audio files in input\ directory
    echo Results will be saved in output\ directory
    echo.
    echo For PowerShell version use: .\run.ps1
    echo For full installation: .\setup-complete.ps1
    pause
    exit /b 0
)

REM Check if input file exists
if not exist "input\%~1" (
    echo File input\%~1 not found
    echo Place audio file in input\ directory
    pause
    exit /b 1
)

echo Processing: %~1

REM Build the image first
echo Building Docker image...
%COMPOSE_CMD% --profile %PROFILE% build %SERVICE%

REM Run the pipeline with all arguments
echo Running processing...
%COMPOSE_CMD% --profile %PROFILE% run --rm %SERVICE% %*

if errorlevel 1 (
    echo Execution error
    pause
    exit /b 1
)

echo.
echo Processing completed!
echo Results in output\ directory
echo.
pause 