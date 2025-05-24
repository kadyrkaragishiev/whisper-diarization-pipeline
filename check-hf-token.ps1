# HuggingFace Token Diagnostic and Setup Script
# Диагностика и настройка HuggingFace токена

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

Write-ColorOutput "HuggingFace Token Diagnostics" "Purple"
Write-ColorOutput "============================" "Purple"
Write-Host ""

# Step 1: Check environment variable
Write-ColorOutput "Step 1: Checking environment variable..." "Blue"
$envToken = $env:HUGGINGFACE_TOKEN
if ($envToken) {
    if ($envToken -eq "HF_TOKEN" -or $envToken.Length -lt 10) {
        Write-ColorOutput "Environment variable exists but seems to be placeholder: $envToken" "Yellow"
    } else {
        $maskedToken = $envToken.Substring(0, [Math]::Min(10, $envToken.Length)) + "..."
        Write-ColorOutput "Environment variable found: $maskedToken" "Green"
    }
} else {
    Write-ColorOutput "Environment variable HUGGINGFACE_TOKEN not set" "Red"
}

# Step 2: Check .env file
Write-ColorOutput "Step 2: Checking .env file..." "Blue"
if (Test-Path ".env") {
    $envFileContent = Get-Content ".env" -ErrorAction SilentlyContinue
    $hfTokenLine = $envFileContent | Where-Object { $_ -match "^HUGGINGFACE_TOKEN=" }
    
    if ($hfTokenLine) {
        $fileToken = ($hfTokenLine -split "=", 2)[1]
        if ($fileToken -eq "HF_TOKEN" -or $fileToken.Length -lt 10) {
            Write-ColorOutput ".env file exists but token is placeholder: $fileToken" "Yellow"
        } else {
            $maskedFileToken = $fileToken.Substring(0, [Math]::Min(10, $fileToken.Length)) + "..."
            Write-ColorOutput ".env file contains token: $maskedFileToken" "Green"
        }
    } else {
        Write-ColorOutput ".env file exists but no HUGGINGFACE_TOKEN found" "Yellow"
    }
} else {
    Write-ColorOutput ".env file not found" "Red"
}

# Step 3: Check env.example
Write-ColorOutput "Step 3: Checking env.example..." "Blue"
if (Test-Path "env.example") {
    Write-ColorOutput "env.example found" "Green"
} else {
    Write-ColorOutput "env.example not found" "Red"
}

# Step 4: Docker environment check
Write-ColorOutput "Step 4: Checking Docker environment..." "Blue"
try {
    $dockerInfo = docker compose config 2>$null
    if ($dockerInfo -match "HUGGINGFACE_TOKEN") {
        Write-ColorOutput "Docker Compose will pass HUGGINGFACE_TOKEN to container" "Green"
    } else {
        Write-ColorOutput "HUGGINGFACE_TOKEN not configured in Docker Compose" "Red"
    }
} catch {
    Write-ColorOutput "Cannot check Docker Compose configuration" "Yellow"
}

# Step 5: Recommendations
Write-Host ""
Write-ColorOutput "Recommendations:" "Purple"
Write-Host ""

$needsSetup = $false

if (-not $envToken -or $envToken -eq "HF_TOKEN") {
    Write-ColorOutput "1. Set environment variable:" "Blue"
    Write-ColorOutput '   $env:HUGGINGFACE_TOKEN="hf_your_actual_token_here"' "Green"
    Write-ColorOutput "   OR permanently in Windows:" "Blue"
    Write-ColorOutput '   setx HUGGINGFACE_TOKEN "hf_your_actual_token_here"' "Green"
    Write-Host ""
    $needsSetup = $true
}

if (-not (Test-Path ".env") -or (-not $hfTokenLine) -or $fileToken -eq "HF_TOKEN") {
    Write-ColorOutput "2. Setup .env file:" "Blue"
    if (-not (Test-Path ".env")) {
        Write-ColorOutput "   copy env.example .env" "Green"
    }
    Write-ColorOutput "   Edit .env file and replace HF_TOKEN with your actual token" "Green"
    Write-Host ""
    $needsSetup = $true
}

if (-not $needsSetup) {
    Write-ColorOutput "✅ HuggingFace token configured correctly!" "Green"
    Write-Host ""
    Write-ColorOutput "Test the configuration:" "Blue"
    Write-ColorOutput "   .\run.ps1 your_audio_file.wav" "Green"
} else {
    Write-ColorOutput "🔑 Get your HuggingFace token:" "Blue"
    Write-ColorOutput "   1. Go to https://huggingface.co/settings/tokens" "Yellow"
    Write-ColorOutput "   2. Create a new token with 'Read' permissions" "Yellow"
    Write-ColorOutput "   3. Accept pyannote/speaker-diarization-3.1 license" "Yellow"
    Write-ColorOutput "   4. Use one of the setup methods above" "Yellow"
    Write-Host ""
    
    # Offer automatic setup
    $autoSetup = Read-Host "Setup .env file automatically? Enter your HF token (or press Enter to skip)"
    if ($autoSetup -and $autoSetup.Length -gt 10) {
        try {
            if (-not (Test-Path ".env")) {
                Copy-Item "env.example" ".env"
                Write-ColorOutput "Created .env from template" "Green"
            }
            
            $envContent = Get-Content ".env"
            $newContent = $envContent -replace "HUGGINGFACE_TOKEN=.*", "HUGGINGFACE_TOKEN=$autoSetup"
            $newContent | Set-Content ".env"
            
            Write-ColorOutput "✅ .env file updated with your token!" "Green"
            Write-ColorOutput "Rebuild Docker image to apply changes:" "Blue"
            Write-ColorOutput "   .\rebuild-and-test.ps1" "Green"
            
        } catch {
            Write-ColorOutput "❌ Error updating .env file: $($_.Exception.Message)" "Red"
        }
    }
}

Write-Host ""
Write-ColorOutput "Current token sources priority:" "Blue"
Write-ColorOutput "1. Command line: --hf-token your_token" "Yellow"
Write-ColorOutput "2. Environment variable: HUGGINGFACE_TOKEN" "Yellow"
Write-ColorOutput "3. .env file: HUGGINGFACE_TOKEN=your_token" "Yellow" 