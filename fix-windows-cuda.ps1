# Quick fix for Windows CUDA issues
# PowerShell script to resolve Docker CUDA problems

param(
    [switch]$Force,
    [switch]$UseStable
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

Write-ColorOutput "Whisper Diarization Pipeline - CUDA Fix" "Purple"
Write-ColorOutput "=======================================" "Purple"
Write-Host ""

# Step 1: Check Docker
Write-ColorOutput "Step 1: Checking Docker..." "Blue"
try {
    docker --version | Out-Host
    Write-ColorOutput "Docker OK" "Green"
} catch {
    Write-ColorOutput "Docker not found - install Docker Desktop first" "Red"
    Write-ColorOutput "Download: https://www.docker.com/products/docker-desktop" "Yellow"
    exit 1
}

# Step 2: Check NVIDIA GPU
Write-ColorOutput "Step 2: Checking NVIDIA GPU..." "Blue"
try {
    nvidia-smi | Out-Host
    Write-ColorOutput "NVIDIA GPU detected" "Green"
} catch {
    Write-ColorOutput "NVIDIA GPU not detected" "Yellow"
    Write-ColorOutput "Will use CPU version" "Yellow"
    $UseCPU = $true
}

# Step 3: Fix Docker Compose version warning
Write-ColorOutput "Step 3: Fixing docker-compose.yml..." "Blue"
$dockerCompose = Get-Content "docker-compose.yml" -Raw
if ($dockerCompose -match "version:") {
    Write-ColorOutput "Removing obsolete version attribute..." "Yellow"
    # Already fixed in previous step
}
Write-ColorOutput "docker-compose.yml fixed" "Green"

# Step 4: Test CUDA image availability
Write-ColorOutput "Step 4: Testing CUDA image..." "Blue"
if (-not $UseCPU) {
    try {
        Write-ColorOutput "Pulling CUDA image..." "Blue"
        docker pull nvidia/cuda:12.1.0-devel-ubuntu22.04
        Write-ColorOutput "CUDA image available" "Green"
    } catch {
        Write-ColorOutput "CUDA image pull failed, trying stable version..." "Yellow"
        $UseStable = $true
    }
}

# Step 5: Build appropriate version
Write-ColorOutput "Step 5: Building Docker image..." "Blue"
if ($UseCPU) {
    Write-ColorOutput "Building CPU version..." "Yellow"
    docker compose --profile cpu build whisper-diarization-cpu
} elseif ($UseStable) {
    Write-ColorOutput "Building stable GPU version..." "Blue"
    # Use the stable dockerfile
    docker build -f Dockerfile.gpu-stable -t whisper-pipeline-gpu-stable .
} else {
    Write-ColorOutput "Building latest GPU version..." "Green"
    docker compose --profile gpu build whisper-diarization-gpu
}

# Step 6: Test the build
Write-ColorOutput "Step 6: Testing build..." "Blue"
if ($UseCPU) {
    docker compose --profile cpu run --rm whisper-diarization-cpu --help
} elseif ($UseStable) {
    docker run --rm whisper-pipeline-gpu-stable --help
} else {
    docker compose --profile gpu run --rm whisper-diarization-gpu --help
}

Write-Host ""
Write-ColorOutput "Fix completed!" "Green"
Write-Host ""
Write-ColorOutput "To run pipeline:" "Purple"
if ($UseCPU) {
    Write-ColorOutput "  .\run.ps1 -CPU your_audio.wav" "Green"
} else {
    Write-ColorOutput "  .\run.ps1 your_audio.wav" "Green"
}
Write-Host "" 