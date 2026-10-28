param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    throw "CMake not found. Install CMake and add it to PATH before running this script."
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $repoRoot "build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

Write-Host "Configuring project..."
cmake -S $repoRoot -B $buildDir -G "Ninja" -DCMAKE_BUILD_TYPE=$BuildType

Write-Host "Building project..."
cmake --build $buildDir --config $BuildType

$exePath = Join-Path $buildDir "bin\MinecraftCVAssistant.exe"
if (Test-Path $exePath) {
    Write-Host "SUCCESS: EXE generated at $exePath"
} else {
    $exePath = Join-Path $buildDir "MinecraftCVAssistant.exe"
    if (Test-Path $exePath) {
        Write-Host "SUCCESS: EXE generated at $exePath"
    } else {
        throw "Build finished without producing MinecraftCVAssistant.exe. Check CMake logs."
    }
}
