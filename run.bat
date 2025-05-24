@echo off
REM Whisper Diarization Pipeline - Windows Batch Launcher
REM Простой лаунчер для Windows без PowerShell

title Whisper Diarization Pipeline

echo.
echo ===========================================
echo  🎤 Whisper Diarization Pipeline
echo ===========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker не установлен. Установите Docker Desktop и попробуйте снова.
    echo 💡 Скачайте с https://www.docker.com/products/docker-desktop
    echo 💡 Или запустите PowerShell как администратор и используйте:
    echo    PowerShell -ExecutionPolicy Bypass -File install-docker.ps1
    pause
    exit /b 1
)

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker не запущен. Запустите Docker Desktop и попробуйте снова.
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
        echo ✅ Создан .env файл из шаблона
        echo 💡 Отредактируйте .env файл и добавьте HUGGINGFACE_TOKEN если нужно
    )
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    set "GPU_AVAILABLE=false"
    set "PROFILE=cpu"
    set "SERVICE=whisper-diarization-cpu"
    echo ⚠️  NVIDIA GPU не обнаружена - используем CPU версию
) else (
    set "GPU_AVAILABLE=true"
    set "PROFILE=gpu"
    set "SERVICE=whisper-diarization-gpu"
    echo ✅ NVIDIA GPU обнаружена - используем GPU версию
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
    echo 📖 Показываем справку...
    echo.
    %COMPOSE_CMD% --profile %PROFILE% run --rm %SERVICE% --help
    echo.
    echo Примеры использования:
    echo   run.bat audio.wav                    # Базовая обработка
    echo   run.bat audio.wav --model large     # Большая модель
    echo   run.bat audio.wav --max-speakers 5  # Максимум 5 спикеров
    echo.
    echo 💡 Поместите аудиофайлы в директорию input\
    echo 📁 Результаты будут сохранены в директории output\
    echo.
    echo Для PowerShell версии используйте: .\run.ps1
    echo Для полной установки: .\setup-complete.ps1
    pause
    exit /b 0
)

REM Check if input file exists
if not exist "input\%~1" (
    echo ❌ Файл input\%~1 не найден
    echo 💡 Поместите аудиофайл в директорию input\
    pause
    exit /b 1
)

echo 🎵 Обрабатываем: %~1

REM Build the image first
echo 🔨 Собираем Docker образ...
%COMPOSE_CMD% --profile %PROFILE% build %SERVICE%

REM Run the pipeline with all arguments
echo 🚀 Запускаем обработку...
%COMPOSE_CMD% --profile %PROFILE% run --rm %SERVICE% %*

if errorlevel 1 (
    echo ❌ Ошибка выполнения
    pause
    exit /b 1
)

echo.
echo ✅ Обработка завершена!
echo 📁 Результаты в директории output\
echo.
pause 