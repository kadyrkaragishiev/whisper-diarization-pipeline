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

Write-ColorOutput "Docker and NVIDIA GPU Setup for Windows" "Blue"
Write-ColorOutput "===========================================" "Blue"

# Check administrator privileges
if (-not (Test-Administrator)) {
    Write-ColorOutput "This script must be run with administrator privileges" "Red"
    Write-ColorOutput "Run PowerShell as Administrator and try again" "Yellow"
    exit 1
}

# Check Windows version
$osVersion = [System.Environment]::OSVersion.Version
if ($osVersion.Major -lt 10) {
    Write-ColorOutput "Windows 10 or newer required" "Red"
    exit 1
}

Write-ColorOutput "Windows $($osVersion.Major).$($osVersion.Minor) detected" "Green"

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
    Write-ColorOutput "Installing WSL2..." "Blue"
    
    # Enable WSL feature
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    
    # Download and install WSL2 kernel update
    $wslUpdateUrl = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
    $wslUpdatePath = "$env:TEMP\wsl_update_x64.msi"
    
    Write-ColorOutput "Downloading WSL2 kernel update..." "Blue"
    Invoke-WebRequest -Uri $wslUpdateUrl -OutFile $wslUpdatePath
    
    Write-ColorOutput "Installing WSL2 kernel update..." "Blue"
    Start-Process msiexec.exe -Wait -ArgumentList "/i $wslUpdatePath /quiet"
    
    # Set WSL2 as default
    wsl --set-default-version 2
    
    Write-ColorOutput "WSL2 installed" "Green"
    Write-ColorOutput "Install Ubuntu from Microsoft Store after reboot" "Yellow"
}

# Function to install Docker Desktop
function Install-DockerDesktop {
    Write-ColorOutput "Installing Docker Desktop..." "Blue"
    
    # Try winget first
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-ColorOutput "Using winget for installation..." "Blue"
        try {
            winget install Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
            Write-ColorOutput "Docker Desktop installed via winget" "Green"
            return
        } catch {
            Write-ColorOutput "Error installing via winget, trying manual..." "Yellow"
        }
    }
    
    # Manual download and install
    $dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $dockerPath = "$env:TEMP\DockerDesktopInstaller.exe"
    
    Write-ColorOutput "Downloading Docker Desktop..." "Blue"
    Invoke-WebRequest -Uri $dockerUrl -OutFile $dockerPath
    
    Write-ColorOutput "Installing Docker Desktop..." "Blue"
    Start-Process $dockerPath -Wait -ArgumentList "install --quiet"
    
    Write-ColorOutput "Docker Desktop installed" "Green"
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
    Write-ColorOutput "Setting up NVIDIA GPU..." "Blue"
    
    if (Test-NvidiaGPU) {
        Write-ColorOutput "NVIDIA drivers already installed" "Green"
        nvidia-smi
        return
    }
    
    Write-ColorOutput "NVIDIA drivers not found" "Yellow"
    Write-ColorOutput "Download latest drivers from https://www.nvidia.com/drivers" "Yellow"
    Write-ColorOutput "For RTX 3080 select GeForce RTX 30 Series" "Yellow"
    
    $openBrowser = Read-Host "Open driver download page? (y/N)"
    if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
        Start-Process "https://www.nvidia.com/drivers"
    }
}

# Main installation flow
Write-ColorOutput "Starting installation..." "Blue"

# Step 1: Check and install WSL2
Write-ColorOutput "Step 1: Checking WSL2..." "Blue"
if (-not (Test-WSL)) {
    if ($Force -or (Read-Host "WSL2 not found. Install? (y/N)") -eq "y") {
        Install-WSL2
        $needReboot = $true
    }
} else {
    Write-ColorOutput "WSL2 already installed" "Green"
}

# Step 2: Install Docker Desktop
Write-ColorOutput "Step 2: Installing Docker Desktop..." "Blue"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    if ($Force -or (Read-Host "Install Docker Desktop? (y/N)") -eq "y") {
        Install-DockerDesktop
        $needReboot = $true
    }
} else {
    Write-ColorOutput "Docker already installed" "Green"
    docker --version
}

# Step 3: NVIDIA GPU setup
if (-not $SkipNvidia) {
    Write-ColorOutput "Step 3: Setting up NVIDIA GPU..." "Blue"
    Install-NvidiaDrivers
}

# Step 4: Create project directories
Write-ColorOutput "Step 4: Creating project directories..." "Blue"
$projectDirs = @("input", "output", "models")
foreach ($dir in $projectDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-ColorOutput "Created directory: $dir" "Green"
    }
}

# Step 5: Setup environment file
Write-ColorOutput "Step 5: Setting up environment..." "Blue"
if (-not (Test-Path ".env")) {
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-ColorOutput "Created .env file from template" "Green"
        Write-ColorOutput "Edit .env and add HUGGINGFACE_TOKEN" "Yellow"
    }
}

# Final instructions
Write-ColorOutput "" "White"
Write-ColorOutput "Installation completed!" "Green"
Write-ColorOutput "" "White"

if ($needReboot) {
    Write-ColorOutput "Computer reboot required" "Yellow"
    Write-ColorOutput "After reboot:" "Yellow"
    Write-ColorOutput "   1. Start Docker Desktop" "Blue"
    Write-ColorOutput "   2. Install Ubuntu from Microsoft Store" "Blue"
    Write-ColorOutput "   3. Run: ./run.sh your_audio.wav" "Blue"
    
    $rebootNow = Read-Host "Reboot now? (y/N)"
    if ($rebootNow -eq "y" -or $rebootNow -eq "Y") {
        Restart-Computer -Force
    }
} else {
    Write-ColorOutput "Next steps:" "Yellow"
    Write-ColorOutput "   1. Start Docker Desktop" "Blue"
    Write-ColorOutput "   2. Place audio file in input/" "Blue" 
    Write-ColorOutput "   3. Run: ./run.sh your_audio.wav" "Blue"
}

Write-ColorOutput "" "White"
Write-ColorOutput "Ready to use!" "Purple" 