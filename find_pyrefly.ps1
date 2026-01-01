$ErrorActionPreference = "SilentlyContinue"
$files = Get-ChildItem -Path "C:\Users\harri" -Filter "pyrefly.exe" -Recurse -Force
foreach ($file in $files) {
    Write-Host "FOUND: $($file.FullName)"
}
