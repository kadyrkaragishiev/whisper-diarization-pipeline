# Test PowerShell script to verify Write-ColorOutput function
# Простой тест функции цветного вывода

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

Write-ColorOutput "Test PowerShell Functions" "Purple"
Write-ColorOutput "=========================" "Purple"
Write-Host ""

Write-ColorOutput "Testing colors:" "Blue"
Write-ColorOutput "• Red text" "Red"
Write-ColorOutput "• Green text" "Green"  
Write-ColorOutput "• Yellow text" "Yellow"
Write-ColorOutput "• Blue text" "Blue"
Write-ColorOutput "• Purple text" "Purple"
Write-ColorOutput "• White text" "White"

Write-Host ""
Write-ColorOutput "Function test completed!" "Green" 