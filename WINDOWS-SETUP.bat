@echo off
chcp 65001 >nul
title Whisper Diarization Pipeline - Windows Setup

echo.
echo =====================================================
echo  Whisper Diarization Pipeline - Windows Setup
echo =====================================================
echo.

echo Этот скрипт поможет настроить систему на Windows.
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Требуются права администратора.
    echo.
    echo 1. Щелкните правой кнопкой мыши на этом файле
    echo 2. Выберите "Запуск от имени администратора"
    echo.
    pause
    exit /b 1
)

echo Проверяем PowerShell...
powershell -Command "Get-Host" >nul 2>&1
if errorlevel 1 (
    echo PowerShell не найден. Обновите Windows.
    pause
    exit /b 1
)

echo.
echo Выберите способ установки:
echo.
echo 1. Автоматическая установка (PowerShell)
echo 2. Скачать Docker Desktop вручную
echo 3. Показать инструкции
echo.

set /p choice="Ваш выбор (1-3): "

if "%choice%"=="1" goto auto_install
if "%choice%"=="2" goto manual_install
if "%choice%"=="3" goto show_instructions

echo Неверный выбор.
pause
exit /b 1

:auto_install
echo.
echo Запускаем автоматическую установку...
echo.
powershell -ExecutionPolicy Bypass -File setup-complete.ps1
goto end

:manual_install
echo.
echo Ручная установка:
echo.
echo 1. Скачайте Docker Desktop:
echo    https://www.docker.com/products/docker-desktop
echo.
echo 2. Установите и перезагрузите компьютер
echo.
echo 3. Запустите Docker Desktop
echo.
echo 4. Поместите аудиофайл в папку input\
echo.
echo 5. Запустите: run.bat your_audio.wav
echo.
start https://www.docker.com/products/docker-desktop
goto end

:show_instructions
echo.
echo Быстрый старт:
echo.
echo 1. Этот скрипт установит Docker и WSL2
echo 2. Автоматически настроит NVIDIA GPU (если есть)
echo 3. Соберет Docker образы
echo 4. Система будет готова к использованию
echo.
echo Для RTX 3080: получите 10-20x ускорение!
echo.
echo После установки:
echo   copy audio.wav input\
echo   run.bat audio.wav
echo.
goto :eof

:end
echo.
echo Готово! Система настроена.
echo.
echo Следующие шаги:
echo 1. Поместите аудиофайл: copy audio.wav input\
echo 2. Запустите: run.bat audio.wav
echo 3. Результаты в: output\
echo.
pause 