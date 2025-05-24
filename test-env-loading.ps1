# Test script to verify .env loading functionality

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

Write-ColorOutput "Testing .env file loading..." "Purple"
Write-Host ""

Write-ColorOutput "Before loading:" "Yellow"
$beforeToken = $env:HUGGINGFACE_TOKEN
if ($beforeToken) {
    $maskedBefore = $beforeToken.Substring(0, [Math]::Min(10, $beforeToken.Length)) + "..."
    Write-ColorOutput "HUGGINGFACE_TOKEN: $maskedBefore" "Blue"
} else {
    Write-ColorOutput "HUGGINGFACE_TOKEN: Not set" "Red"
}

Load-EnvFile

Write-Host ""
Write-ColorOutput "After loading:" "Yellow"
$afterToken = $env:HUGGINGFACE_TOKEN
if ($afterToken) {
    $maskedAfter = $afterToken.Substring(0, [Math]::Min(10, $afterToken.Length)) + "..."
    Write-ColorOutput "HUGGINGFACE_TOKEN: $maskedAfter" "Green"
} else {
    Write-ColorOutput "HUGGINGFACE_TOKEN: Still not set" "Red"
}

Write-Host ""
Write-ColorOutput "Test completed!" "Purple" 