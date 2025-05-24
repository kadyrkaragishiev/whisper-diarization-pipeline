# Whisper Diarization Pipeline - Windows PowerShell Launcher
# Универсальный лаунчер для Windows с автоопределением GPU

param(
    [string]$AudioFile,
    [string]$Model = "base",
    [string]$CustomModel = "",
    [string]$Output = "output", 
    [string]$Device = "",
    [int]$MinSpeakers = 1,
    [int]$MaxSpeakers = 10,
    [double]$MinSegment = 0.5,
    [string]$AlignmentStrategy = "smart",
    [double]$TimeLimit = 0,
    [switch]$Help
)

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

Write-ColorOutput "🎤 Whisper Diarization Pipeline" "Blue"
Write-ColorOutput "================================" "Blue"

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-ColorOutput "❌ Docker не установлен. Установите Docker Desktop и попробуйте снова." "Red"
    Write-ColorOutput "💡 Запустите: PowerShell -ExecutionPolicy Bypass -File install-docker.ps1" "Yellow"
    exit 1
}

# Check if Docker is running
try {
    docker version | Out-Null
} catch {
    Write-ColorOutput "❌ Docker не запущен. Запустите Docker Desktop и попробуйте снова." "Red"
    exit 1
}

# Check if Docker Compose is available
$dockerComposeAvailable = $false
if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    $dockerComposeAvailable = $true
    $composeCommand = "docker-compose"
} elseif ((docker compose version 2>$null) -and $?) {
    $dockerComposeAvailable = $true
    $composeCommand = "docker compose"
}

if (-not $dockerComposeAvailable) {
    Write-ColorOutput "❌ Docker Compose не найден. Обновите Docker Desktop." "Red"
    exit 1
}

# Function to check NVIDIA GPU availability
function Test-NvidiaGPU {
    try {
        nvidia-smi | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to check NVIDIA Docker runtime
function Test-NvidiaDocker {
    try {
        $dockerInfo = docker info 2>$null | Out-String
        return $dockerInfo -match "nvidia"
    } catch {
        return $false
    }
}

# Create necessary directories
Write-ColorOutput "📁 Создаем необходимые директории..." "Blue"
$dirs = @("input", "output", "models")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-ColorOutput "📝 Создаем .env файл..." "Blue"
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-ColorOutput "💡 Отредактируйте .env файл и добавьте ваш HUGGINGFACE_TOKEN если нужно" "Yellow"
    }
}

# Determine which version to use
$useGPU = $false
$profile = "cpu"
$service = "whisper-diarization-cpu"

if (Test-NvidiaGPU) {
    Write-ColorOutput "✅ NVIDIA GPU обнаружена" "Green"
    
    if (Test-NvidiaDocker) {
        Write-ColorOutput "✅ NVIDIA Docker runtime доступен" "Green"
        $useGPU = $true
        $profile = "gpu"
        $service = "whisper-diarization-gpu"
        Write-ColorOutput "🚀 Запускаем GPU версию" "Green"
    } else {
        Write-ColorOutput "⚠️  NVIDIA Docker runtime не найден" "Yellow"
        Write-ColorOutput "🚀 Запускаем CPU версию" "Yellow"
    }
} else {
    Write-ColorOutput "⚠️  NVIDIA GPU не обнаружена или драйверы не установлены" "Yellow"
    Write-ColorOutput "🚀 Запускаем CPU версию" "Yellow"
}

# Build the image
Write-ColorOutput "🔨 Собираем Docker образ..." "Blue"
try {
    if ($composeCommand -eq "docker-compose") {
        & docker-compose --profile $profile build $service
    } else {
        & docker compose --profile $profile build $service
    }
} catch {
    Write-ColorOutput "❌ Ошибка сборки образа" "Red"
    Write-ColorOutput "💡 Проверьте интернет соединение и Docker" "Yellow"
    exit 1
}

# Function to show help
function Show-Help {
    Write-ColorOutput "📖 Показываем справку..." "Blue"
    
    if ($composeCommand -eq "docker-compose") {
        & docker-compose --profile $profile run --rm $service --help
    } else {
        & docker compose --profile $profile run --rm $service --help
    }
    
    Write-Host ""
    Write-ColorOutput "Примеры использования:" "Yellow"
    Write-ColorOutput "  .\run.ps1 audio.wav                           # Базовая обработка" "Green"
    Write-ColorOutput "  .\run.ps1 audio.wav -Model large             # Использовать большую модель" "Green"
    Write-ColorOutput "  .\run.ps1 audio.wav -MaxSpeakers 5           # Максимум 5 спикеров" "Green"
    Write-ColorOutput "  .\run.ps1 audio.wav -CustomModel path/model  # Кастомная модель" "Green"
    Write-Host ""
    Write-ColorOutput "💡 Поместите аудиофайлы в директорию input\" "Blue"
    Write-ColorOutput "📁 Результаты будут сохранены в директории output\" "Blue"
}

# Function to run the pipeline
function Start-Pipeline {
    param([string]$AudioFile, [string[]]$Arguments)
    
    $inputPath = "input\$AudioFile"
    if (-not (Test-Path $inputPath)) {
        Write-ColorOutput "❌ Файл $inputPath не найден" "Red"
        Write-ColorOutput "💡 Поместите аудиофайл в директорию input\" "Yellow"
        exit 1
    }
    
    Write-ColorOutput "🎵 Обрабатываем: $AudioFile" "Green"
    
    # Build docker run arguments
    $dockerArgs = @($AudioFile)
    $dockerArgs += $Arguments
    
    try {
        if ($composeCommand -eq "docker-compose") {
            & docker-compose --profile $profile run --rm $service @dockerArgs
        } else {
            & docker compose --profile $profile run --rm $service @dockerArgs
        }
    } catch {
        Write-ColorOutput "❌ Ошибка выполнения" "Red"
        exit 1
    }
}

# Main logic
if ($Help -or (-not $AudioFile -and $args.Count -eq 0)) {
    Show-Help
} else {
    # Build arguments array
    $arguments = @()
    
    if ($Model -ne "base") { $arguments += "--model"; $arguments += $Model }
    if ($CustomModel) { $arguments += "--custom-model"; $arguments += $CustomModel }
    if ($Output -ne "output") { $arguments += "--output"; $arguments += $Output }
    if ($Device) { $arguments += "--device"; $arguments += $Device }
    if ($MinSpeakers -ne 1) { $arguments += "--min-speakers"; $arguments += $MinSpeakers }
    if ($MaxSpeakers -ne 10) { $arguments += "--max-speakers"; $arguments += $MaxSpeakers }
    if ($MinSegment -ne 0.5) { $arguments += "--min-segment"; $arguments += $MinSegment }
    if ($AlignmentStrategy -ne "smart") { $arguments += "--alignment-strategy"; $arguments += $AlignmentStrategy }
    if ($TimeLimit -gt 0) { $arguments += "--time-limit"; $arguments += $TimeLimit }
    
    # If no audio file provided but arguments exist, use first argument as audio file
    if (-not $AudioFile -and $args.Count -gt 0) {
        $AudioFile = $args[0]
        # Add remaining arguments
        for ($i = 1; $i -lt $args.Count; $i++) {
            $arguments += $args[$i]
        }
    }
    
    if ($AudioFile) {
        Start-Pipeline -AudioFile $AudioFile -Arguments $arguments
    } else {
        Show-Help
    }
} 