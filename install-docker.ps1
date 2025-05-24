# Docker and NVIDIA GPU Setup for Windows
# PowerShell script for Windows installation

param(
    [switch]$Force,
    [switch]$SkipNvidia
)

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Write-ColorOutput {
    param([string]$Text, [string]$Color = "White")
    
    $colors = @{
        "Red" = "Red"
        "Green" = "Green" 
        "Yellow" = "Yellow"
        "Blue" = "Cyan"
        "Purple" = "Magenta"
    }
    
    Write-Host $Text -ForegroundColor $colors[$Color]
}

Write-ColorOutput "🐳 Docker and NVIDIA GPU Setup for Windows" "Blue"
Write-ColorOutput "===========================================" "Blue"

# Check administrator privileges
if (-not (Test-Administrator)) {
    Write-ColorOutput "❌ Этот скрипт должен запускаться с правами администратора" "Red"
    Write-ColorOutput "💡 Запустите PowerShell как администратор и попробуйте снова" "Yellow"
    exit 1
}

# Check Windows version
$osVersion = [System.Environment]::OSVersion.Version
if ($osVersion.Major -lt 10) {
    Write-ColorOutput "❌ Требуется Windows 10 или новее" "Red"
    exit 1
}

Write-ColorOutput "✅ Windows $($osVersion.Major).$($osVersion.Minor) обнаружена" "Green"

# Function to check if WSL is available
function Test-WSL {
    try {
        $wslInfo = wsl --status 2>$null
        return $true
    } catch {
        return $false
    }
}

# Function to install WSL2
function Install-WSL2 {
    Write-ColorOutput "📦 Устанавливаем WSL2..." "Blue"
    
    # Enable WSL feature
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    
    # Download and install WSL2 kernel update
    $wslUpdateUrl = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
    $wslUpdatePath = "$env:TEMP\wsl_update_x64.msi"
    
    Write-ColorOutput "📥 Скачиваем WSL2 kernel update..." "Blue"
    Invoke-WebRequest -Uri $wslUpdateUrl -OutFile $wslUpdatePath
    
    Write-ColorOutput "📦 Устанавливаем WSL2 kernel update..." "Blue"
    Start-Process msiexec.exe -Wait -ArgumentList "/i $wslUpdatePath /quiet"
    
    # Set WSL2 as default
    wsl --set-default-version 2
    
    Write-ColorOutput "✅ WSL2 установлен" "Green"
    Write-ColorOutput "💡 Установите Ubuntu из Microsoft Store после перезагрузки" "Yellow"
}

# Function to install Docker Desktop
function Install-DockerDesktop {
    Write-ColorOutput "📦 Устанавливаем Docker Desktop..." "Blue"
    
    # Try winget first
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-ColorOutput "💡 Используем winget для установки..." "Blue"
        try {
            winget install Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
            Write-ColorOutput "✅ Docker Desktop установлен через winget" "Green"
            return
        } catch {
            Write-ColorOutput "⚠️  Ошибка установки через winget, пробуем вручную..." "Yellow"
        }
    }
    
    # Manual download and install
    $dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $dockerPath = "$env:TEMP\DockerDesktopInstaller.exe"
    
    Write-ColorOutput "📥 Скачиваем Docker Desktop..." "Blue"
    Invoke-WebRequest -Uri $dockerUrl -OutFile $dockerPath
    
    Write-ColorOutput "📦 Устанавливаем Docker Desktop..." "Blue"
    Start-Process $dockerPath -Wait -ArgumentList "install --quiet"
    
    Write-ColorOutput "✅ Docker Desktop установлен" "Green"
}

# Function to check NVIDIA GPU
function Test-NvidiaGPU {
    try {
        $nvidiaInfo = nvidia-smi 2>$null
        return $true
    } catch {
        return $false
    }
}

# Function to install NVIDIA drivers
function Install-NvidiaDrivers {
    Write-ColorOutput "🎮 Настройка NVIDIA GPU..." "Blue"
    
    if (Test-NvidiaGPU) {
        Write-ColorOutput "✅ NVIDIA драйверы уже установлены" "Green"
        nvidia-smi
        return
    }
    
    Write-ColorOutput "⚠️  NVIDIA драйверы не найдены" "Yellow"
    Write-ColorOutput "💡 Скачайте актуальные драйверы с https://www.nvidia.com/drivers" "Yellow"
    Write-ColorOutput "💡 Для RTX 3080 выберите GeForce RTX 30 Series" "Yellow"
    
    $openBrowser = Read-Host "Открыть страницу загрузки драйверов? (y/N)"
    if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
        Start-Process "https://www.nvidia.com/drivers"
    }
}

# Main installation flow
Write-ColorOutput "🚀 Начинаем установку..." "Blue"

# Step 1: Check and install WSL2
Write-ColorOutput "📋 Шаг 1: Проверка WSL2..." "Blue"
if (-not (Test-WSL)) {
    if ($Force -or (Read-Host "WSL2 не найден. Установить? (y/N)") -eq "y") {
        Install-WSL2
        $needReboot = $true
    }
} else {
    Write-ColorOutput "✅ WSL2 уже установлен" "Green"
}

# Step 2: Install Docker Desktop
Write-ColorOutput "📋 Шаг 2: Установка Docker Desktop..." "Blue"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    if ($Force -or (Read-Host "Установить Docker Desktop? (y/N)") -eq "y") {
        Install-DockerDesktop
        $needReboot = $true
    }
} else {
    Write-ColorOutput "✅ Docker уже установлен" "Green"
    docker --version
}

# Step 3: NVIDIA GPU setup
if (-not $SkipNvidia) {
    Write-ColorOutput "📋 Шаг 3: Настройка NVIDIA GPU..." "Blue"
    Install-NvidiaDrivers
}

# Step 4: Create project directories
Write-ColorOutput "📋 Шаг 4: Создание директорий проекта..." "Blue"
$projectDirs = @("input", "output", "models")
foreach ($dir in $projectDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-ColorOutput "✅ Создана директория: $dir" "Green"
    }
}

# Step 5: Setup environment file
Write-ColorOutput "📋 Шаг 5: Настройка окружения..." "Blue"
if (-not (Test-Path ".env")) {
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-ColorOutput "✅ Создан файл .env из шаблона" "Green"
        Write-ColorOutput "💡 Отредактируйте .env и добавьте HUGGINGFACE_TOKEN" "Yellow"
    }
}

# Final instructions
Write-ColorOutput "" "White"
Write-ColorOutput "🎉 Установка завершена!" "Green"
Write-ColorOutput "" "White"

if ($needReboot) {
    Write-ColorOutput "💡 Требуется перезагрузка компьютера" "Yellow"
    Write-ColorOutput "💡 После перезагрузки:" "Yellow"
    Write-ColorOutput "   1. Запустите Docker Desktop" "Blue"
    Write-ColorOutput "   2. Установите Ubuntu из Microsoft Store" "Blue"
    Write-ColorOutput "   3. Запустите: ./run.sh your_audio.wav" "Blue"
    
    $rebootNow = Read-Host "Перезагрузить сейчас? (y/N)"
    if ($rebootNow -eq "y" -or $rebootNow -eq "Y") {
        Restart-Computer -Force
    }
} else {
    Write-ColorOutput "💡 Следующие шаги:" "Yellow"
    Write-ColorOutput "   1. Запустите Docker Desktop" "Blue"
    Write-ColorOutput "   2. Поместите аудиофайл в input/" "Blue" 
    Write-ColorOutput "   3. Запустите: ./run.sh your_audio.wav" "Blue"
}

Write-ColorOutput "" "White"
Write-ColorOutput "🚀 Готово к использованию!" "Purple" 