# Complete Whisper Diarization Pipeline Setup for Windows
# Full automatic system installation on Windows

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
        "White" = "White"
    }
    
    $actualColor = if ($colors.ContainsKey($Color)) { $colors[$Color] } else { "White" }
    Write-Host $Text -ForegroundColor $actualColor
}

Write-ColorOutput "Whisper Diarization Pipeline - Complete Setup" "Purple"
Write-ColorOutput "=================================================" "Purple"
Write-Host ""
Write-ColorOutput "This script will automatically install:" "Blue"
Write-ColorOutput "• WSL2 (if needed)" "Blue"  
Write-ColorOutput "• Docker Desktop" "Blue"
Write-ColorOutput "• NVIDIA GPU support (if available)" "Blue"
Write-ColorOutput "• Build Docker images" "Blue"
Write-ColorOutput "• Setup environment" "Blue"
Write-Host ""

# Ask for confirmation
if (-not $Force) {
    $continue = Read-Host "Continue with installation? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-ColorOutput "Installation cancelled" "Yellow"
        exit 0
    }
}

Write-ColorOutput "Starting installation..." "Blue"

# Check administrator privileges
if (-not (Test-Administrator)) {
    Write-ColorOutput "Administrator privileges required" "Red"
    Write-ColorOutput "Run PowerShell as Administrator and try again" "Yellow"
    exit 1
}

# Step 1: Install dependencies via PowerShell script
Write-ColorOutput "Step 1: Installing dependencies..." "Blue"
if (Test-Path "install-docker.ps1") {
    Write-ColorOutput "Running install-docker.ps1..." "Blue"
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
        Write-ColorOutput "Error installing dependencies: $($_.Exception.Message)" "Red"
        Write-ColorOutput "Detailed error: $($_.Exception)" "Yellow"
        Write-ColorOutput "Try installing Docker Desktop manually" "Yellow"
        Write-ColorOutput "Download from: https://www.docker.com/products/docker-desktop" "Blue"
    }
} else {
    Write-ColorOutput "File install-docker.ps1 not found" "Red"
    Write-ColorOutput "Download Docker Desktop: https://www.docker.com/products/docker-desktop" "Yellow"
}

# Step 2: Create directories
Write-ColorOutput "Step 2: Creating directories..." "Blue"
$dirs = @("input", "output", "models")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-ColorOutput "Created directory: $dir" "Green"
    }
}

# Step 3: Setup environment
Write-ColorOutput "Step 3: Setting up environment..." "Blue"
if (-not (Test-Path ".env")) {
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-ColorOutput "Created .env file from template" "Green"
        Write-ColorOutput "Edit .env and add HUGGINGFACE_TOKEN if needed" "Yellow"
    }
} else {
    Write-ColorOutput ".env file already exists" "Green"
}

# Step 4: Test Docker
Write-ColorOutput "Step 4: Testing Docker..." "Blue"
$dockerWorking = $false
try {
    docker run --rm hello-world | Out-Null
    Write-ColorOutput "Docker working correctly" "Green"
    $dockerWorking = $true
} catch {
    Write-ColorOutput "Docker issues detected" "Red"
    Write-ColorOutput "Make sure Docker Desktop is running" "Yellow"
    Write-ColorOutput "You may need to reboot" "Yellow"
}

# Step 5: Check GPU support
Write-ColorOutput "Step 5: Checking GPU support..." "Blue"
$gpuAvailable = $false

if (-not $SkipNvidia) {
    try {
        nvidia-smi | Out-Null
        Write-ColorOutput "NVIDIA GPU detected" "Green"
        
        if ($dockerWorking) {
            try {
                $dockerInfo = docker info 2>$null | Out-String
                if ($dockerInfo -match "nvidia") {
                    Write-ColorOutput "NVIDIA Docker runtime available" "Green"
                    $gpuAvailable = $true
                } else {
                    Write-ColorOutput "NVIDIA Docker runtime not configured" "Yellow"
                    Write-ColorOutput "Docker Desktop automatically supports NVIDIA GPU" "Yellow"
                }
            } catch {
                Write-ColorOutput "Could not check Docker info" "Yellow"
            }
        }
    } catch {
        Write-ColorOutput "NVIDIA GPU not detected" "Yellow"
        Write-ColorOutput "Install NVIDIA drivers for GPU acceleration" "Yellow"
    }
}

# Step 6: Build Docker images
if ($dockerWorking) {
    Write-ColorOutput "Step 6: Building Docker images..." "Blue"
    
    $profile = if ($gpuAvailable) { "gpu" } else { "cpu" }
    $service = if ($gpuAvailable) { "whisper-diarization-gpu" } else { "whisper-diarization-cpu" }
    $version = if ($gpuAvailable) { "GPU" } else { "CPU" }
    
    Write-ColorOutput "Building $version version..." $(if ($gpuAvailable) { "Green" } else { "Yellow" })
    
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
        
        Write-ColorOutput "Docker image built successfully" "Green"
    } catch {
        Write-ColorOutput "Error building Docker image" "Red"
        Write-ColorOutput "Check internet connection" "Yellow"
    }
} else {
    Write-ColorOutput "Skipping build - Docker not working" "Yellow"
}

# Step 7: Test the pipeline
if ($dockerWorking) {
    Write-ColorOutput "Step 7: Testing system..." "Blue"
    Write-ColorOutput "Showing help..." "Blue"
    
    try {
        if (Test-Path "run.ps1") {
            & ".\run.ps1" -Help
        } else {
            Write-ColorOutput "Use run.ps1 for launch or run.sh in WSL/Git Bash" "Yellow"
        }
    } catch {
        Write-ColorOutput "Could not show help" "Yellow"
    }
}

# Final results
Write-Host ""
Write-ColorOutput "Installation completed!" "Green"
Write-Host ""
Write-ColorOutput "Next steps:" "Purple"
Write-Host ""
Write-ColorOutput "1. Place audio file in input directory:" "Blue"
Write-ColorOutput "   copy your_audio.wav input\" "Green"
Write-Host ""
Write-ColorOutput "2. Run processing:" "Blue"
Write-ColorOutput "   .\run.ps1 your_audio.wav                    # PowerShell" "Green"
Write-ColorOutput "   or ./run.sh your_audio.wav in WSL/Git Bash  # Bash" "Green"
Write-Host ""
Write-ColorOutput "3. Results will be in output\" "Blue"
Write-Host ""

if ($gpuAvailable) {
    Write-ColorOutput "Recommended version: GPU (RTX 3080 detected!)" "Green"
} else {
    Write-ColorOutput "Recommended version: CPU" "Yellow"
    if (-not $SkipNvidia) {
        Write-ColorOutput "Install NVIDIA drivers for GPU acceleration" "Yellow"
    }
}

Write-Host ""
Write-ColorOutput "Documentation: README.md" "Blue"
Write-ColorOutput "Issues? Check INSTALLATION.md" "Blue"
Write-Host ""
Write-ColorOutput "Ready to use!" "Purple" 