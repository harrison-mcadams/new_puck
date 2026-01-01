$ErrorActionPreference = "SilentlyContinue"
$parent = "C:\Users\harri\.antigravity\bin"
$new = "bin_disabled"
$target = "C:\Users\harri\.antigravity\bin_disabled"

Write-Host "Starting Kill/Rename Loop..."
$end = (Get-Date).AddSeconds(10)
while ((Get-Date) -lt $end) {
    Stop-Process -Name pyrefly -Force
    Rename-Item -LiteralPath $parent -NewName $new -Force
    
    if (Test-Path $target) {
        Write-Host "SUCCESS: Directory renamed to $target"
        exit 0
    }
    Start-Sleep -Milliseconds 10
}
Write-Host "FAILURE: Could not rename directory."
