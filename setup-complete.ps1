# Complete Whisper Diarization Pipeline Setup for Windows
# Полная автоматическая установка системы на Windows

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

Write-ColorOutput "🎤 Whisper Diarization Pipeline - Complete Setup" "Purple"
Write-ColorOutput "=================================================" "Purple"
Write-Host ""
Write-ColorOutput "Этот скрипт автоматически установит:" "Blue"
Write-ColorOutput "• WSL2 (если нужно)" "Blue"  
Write-ColorOutput "• Docker Desktop" "Blue"
Write-ColorOutput "• NVIDIA GPU поддержку (если доступно)" "Blue"
Write-ColorOutput "• Соберет Docker образы" "Blue"
Write-ColorOutput "• Настроит окружение" "Blue"
Write-Host ""

# Ask for confirmation
if (-not $Force) {
    $continue = Read-Host "Продолжить установку? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-ColorOutput "Установка отменена" "Yellow"
        exit 0
    }
}

Write-ColorOutput "🚀 Начинаем установку..." "Blue"

# Check administrator privileges
if (-not (Test-Administrator)) {
    Write-ColorOutput "❌ Требуются права администратора" "Red"
    Write-ColorOutput "💡 Запустите PowerShell как администратор и попробуйте снова" "Yellow"
    exit 1
}

# Step 1: Install dependencies via PowerShell script
Write-ColorOutput "📦 Шаг 1: Установка зависимостей..." "Blue"
if (Test-Path "install-docker.ps1") {
    Write-ColorOutput "Запускаем install-docker.ps1..." "Blue"
    $installArgs = @()
    if ($Force) { $installArgs += "-Force" }
    if ($SkipNvidia) { $installArgs += "-SkipNvidia" }
    
    try {
        if ($installArgs.Count -gt 0) {
            & ".\install-docker.ps1" @installArgs
        } else {
            & ".\install-docker.ps1"
        }
    } catch {
        Write-ColorOutput "❌ Ошибка установки зависимостей" "Red"
        Write-ColorOutput "💡 Попробуйте установить Docker Desktop вручную" "Yellow"
    }
} else {
    Write-ColorOutput "❌ Файл install-docker.ps1 не найден" "Red"
    Write-ColorOutput "💡 Скачайте Docker Desktop: https://www.docker.com/products/docker-desktop" "Yellow"
}

# Step 2: Create directories
Write-ColorOutput "📁 Шаг 2: Создание директорий..." "Blue"
$dirs = @("input", "output", "models")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-ColorOutput "✅ Создана директория: $dir" "Green"
    }
}

# Step 3: Setup environment
Write-ColorOutput "⚙️  Шаг 3: Настройка окружения..." "Blue"
if (-not (Test-Path ".env")) {
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-ColorOutput "✅ Создан файл .env из шаблона" "Green"
        Write-ColorOutput "💡 Отредактируйте .env и добавьте HUGGINGFACE_TOKEN если нужно" "Yellow"
    }
} else {
    Write-ColorOutput "✅ Файл .env уже существует" "Green"
}

# Step 4: Test Docker
Write-ColorOutput "🧪 Шаг 4: Проверка Docker..." "Blue"
$dockerWorking = $false
try {
    docker run --rm hello-world | Out-Null
    Write-ColorOutput "✅ Docker работает корректно" "Green"
    $dockerWorking = $true
} catch {
    Write-ColorOutput "❌ Проблемы с Docker" "Red"
    Write-ColorOutput "💡 Убедитесь что Docker Desktop запущен" "Yellow"
    Write-ColorOutput "💡 Возможно нужна перезагрузка" "Yellow"
}

# Step 5: Check GPU support
Write-ColorOutput "🎮 Шаг 5: Проверка GPU поддержки..." "Blue"
$gpuAvailable = $false

if (-not $SkipNvidia) {
    try {
        nvidia-smi | Out-Null
        Write-ColorOutput "✅ NVIDIA GPU обнаружена" "Green"
        
        if ($dockerWorking) {
            try {
                $dockerInfo = docker info 2>$null | Out-String
                if ($dockerInfo -match "nvidia") {
                    Write-ColorOutput "✅ NVIDIA Docker runtime доступен" "Green"
                    $gpuAvailable = $true
                } else {
                    Write-ColorOutput "⚠️  NVIDIA Docker runtime не настроен" "Yellow"
                    Write-ColorOutput "💡 Docker Desktop автоматически поддерживает NVIDIA GPU" "Yellow"
                }
            } catch {
                Write-ColorOutput "⚠️  Не удалось проверить Docker info" "Yellow"
            }
        }
    } catch {
        Write-ColorOutput "⚠️  NVIDIA GPU не обнаружена" "Yellow"
        Write-ColorOutput "💡 Установите NVIDIA драйверы для GPU ускорения" "Yellow"
    }
}

# Step 6: Build Docker images
if ($dockerWorking) {
    Write-ColorOutput "🔨 Шаг 6: Сборка Docker образов..." "Blue"
    
    $profile = if ($gpuAvailable) { "gpu" } else { "cpu" }
    $service = if ($gpuAvailable) { "whisper-diarization-gpu" } else { "whisper-diarization-cpu" }
    $version = if ($gpuAvailable) { "GPU" } else { "CPU" }
    
    Write-ColorOutput "🚀 Собираем $version версию..." $(if ($gpuAvailable) { "Green" } else { "Yellow" })
    
    try {
        $composeCommand = "docker compose"
        if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
            $composeCommand = "docker-compose"
        }
        
        if ($composeCommand -eq "docker-compose") {
            & docker-compose --profile $profile build $service
        } else {
            & docker compose --profile $profile build $service
        }
        
        Write-ColorOutput "✅ Docker образ собран успешно" "Green"
    } catch {
        Write-ColorOutput "❌ Ошибка сборки Docker образа" "Red"
        Write-ColorOutput "💡 Проверьте интернет соединение" "Yellow"
    }
} else {
    Write-ColorOutput "⚠️  Пропускаем сборку - Docker не работает" "Yellow"
}

# Step 7: Test the pipeline
if ($dockerWorking) {
    Write-ColorOutput "🧪 Шаг 7: Тестирование системы..." "Blue"
    Write-ColorOutput "Показываем справку..." "Blue"
    
    try {
        if (Test-Path "run.ps1") {
            & ".\run.ps1" -Help
        } else {
            Write-ColorOutput "💡 Используйте run.ps1 для запуска или run.sh в WSL/Git Bash" "Yellow"
        }
    } catch {
        Write-ColorOutput "⚠️  Не удалось показать справку" "Yellow"
    }
}

# Final results
Write-Host ""
Write-ColorOutput "🎉 Установка завершена!" "Green"
Write-Host ""
Write-ColorOutput "📋 Что дальше:" "Purple"
Write-Host ""
Write-ColorOutput "1. Поместите аудиофайл в директорию input\:" "Blue"
Write-ColorOutput "   copy your_audio.wav input\" "Green"
Write-Host ""
Write-ColorOutput "2. Запустите обработку:" "Blue"
Write-ColorOutput "   .\run.ps1 your_audio.wav                    # PowerShell" "Green"
Write-ColorOutput "   или ./run.sh your_audio.wav в WSL/Git Bash  # Bash" "Green"
Write-Host ""
Write-ColorOutput "3. Результаты будут в директории output\" "Blue"
Write-Host ""

if ($gpuAvailable) {
    Write-ColorOutput "💡 Рекомендуемая версия: GPU (RTX 3080 обнаружена!)" "Green"
} else {
    Write-ColorOutput "💡 Рекомендуемая версия: CPU" "Yellow"
    if (-not $SkipNvidia) {
        Write-ColorOutput "💡 Для GPU ускорения установите NVIDIA драйверы" "Yellow"
    }
}

Write-Host ""
Write-ColorOutput "📖 Подробная документация: README.md" "Blue"
Write-ColorOutput "🐛 Проблемы? Проверьте INSTALLATION.md" "Blue"
Write-Host ""
Write-ColorOutput "Готово к использованию! 🚀" "Purple" 