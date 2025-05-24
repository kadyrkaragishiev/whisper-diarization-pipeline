# Quick rebuild and test script for Windows
# Пересборка и тест после исправления main.py

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

Write-ColorOutput "Rebuild and Test - Whisper Diarization Pipeline" "Purple"
Write-ColorOutput "==============================================" "Purple"
Write-Host ""

# Function to load environment variables from .env file
function Load-EnvFile {
    param([string]$FilePath = ".env")
    
    if (Test-Path $FilePath) {
        Write-ColorOutput "Loading environment variables from $FilePath..." "Blue"
        Get-Content $FilePath | ForEach-Object {
            if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)\s*$") {
                $name = $matches[1].Trim()
                $value = $matches[2].Trim()
                # Remove quotes if present
                $value = $value -replace '^["'']|["'']$', ''
                
                # Set environment variable for current session
                Set-Item -Path "env:$name" -Value $value
                Write-ColorOutput "  Set $name" "Green"
            }
        }
    } else {
        Write-ColorOutput "No .env file found at $FilePath" "Yellow"
    }
}

# Load environment variables from .env file
Load-EnvFile

# Step 1: Check Docker
Write-ColorOutput "Step 1: Checking Docker..." "Blue"
try {
    docker --version | Out-Host
    Write-ColorOutput "Docker OK" "Green"
} catch {
    Write-ColorOutput "Docker not available" "Red"
    exit 1
}

# Step 2: Rebuild image
Write-ColorOutput "Step 2: Rebuilding Docker image..." "Blue"
try {
    Write-ColorOutput "Building GPU version..." "Blue"
    docker compose --profile gpu build whisper-diarization-gpu --no-cache
    Write-ColorOutput "Build completed" "Green"
} catch {
    Write-ColorOutput "Build failed, trying CPU version..." "Yellow"
    docker compose --profile cpu build whisper-diarization-cpu --no-cache
}

# Step 3: Check input files
Write-ColorOutput "Step 3: Checking input files..." "Blue"
$inputFiles = Get-ChildItem -Path "input\" -File -ErrorAction SilentlyContinue
if ($inputFiles) {
    Write-ColorOutput "Available files in input/:" "Green"
    foreach ($file in $inputFiles) {
        Write-ColorOutput "  • $($file.Name)" "Green"
    }
} else {
    Write-ColorOutput "No files found in input/" "Red"
    exit 1
}

# Step 4: Test with first available file
$testFile = $inputFiles[0].Name
Write-ColorOutput "Step 4: Testing with: $testFile" "Blue"

try {
    Write-ColorOutput "Running test..." "Blue"
    .\run.ps1 $testFile
    Write-ColorOutput "Test completed!" "Green"
} catch {
    Write-ColorOutput "Test failed" "Red"
    Write-ColorOutput "Error: $($_.Exception.Message)" "Yellow"
}

Write-Host ""
Write-ColorOutput "Script completed!" "Purple" 