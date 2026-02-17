param(
  [string]$RepoRoot = $PSScriptRoot,
  [string]$Distro = "Ubuntu",
  [string]$WslBinPath = "/home/admmin/bin/llama-swap",
  [string]$WslConfigPath = "/home/admmin/llama-swap/config.yaml",
  [switch]$SkipUIBuild,
  [switch]$SkipGoBuild,
  [switch]$SkipRestart
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Step([string]$msg) {
  Write-Host "`n=== $msg ===" -ForegroundColor Cyan
}

if (-not (Test-Path $RepoRoot)) {
  throw "Repo root not found: $RepoRoot"
}

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path
$drive = [char]::ToLower($resolvedRepoRoot[0])
$rest = $resolvedRepoRoot.Substring(2).Replace('\', '/')
$wslRepoRoot = "/mnt/$drive$rest"

$uiDir = Join-Path $RepoRoot "ui-svelte"
$linuxBin = Join-Path $RepoRoot "build/llama-swap-linux-amd64"
$startScript = Join-Path $RepoRoot "start-tbg-services-wsl.ps1"
$wslSourceBin = "$wslRepoRoot/build/llama-swap-linux-amd64"

Step "Build UI (vite)"
if (-not $SkipUIBuild) {
  if (-not (Test-Path $uiDir)) {
    throw "UI directory not found: $uiDir"
  }
  Push-Location $uiDir
  try {
    npm run build
  } finally {
    Pop-Location
  }
} else {
  Write-Host "Skipped UI build." -ForegroundColor Yellow
}

Step "Build Go Linux binary"
if (-not $SkipGoBuild) {
  Push-Location $RepoRoot
  try {
    $env:GOOS = "linux"
    $env:GOARCH = "amd64"
    go build -o build/llama-swap-linux-amd64 .
  } finally {
    Remove-Item Env:GOOS -ErrorAction SilentlyContinue
    Remove-Item Env:GOARCH -ErrorAction SilentlyContinue
    Pop-Location
  }
} else {
  Write-Host "Skipped Go build." -ForegroundColor Yellow
}

if (-not (Test-Path $linuxBin)) {
  throw "Linux binary not found: $linuxBin"
}

Step "Install binary into WSL"
& wsl -d $Distro -- bash -lc "install -m 755 '$wslSourceBin' '$WslBinPath'"

Step "Kill running llama-swap/playwright in WSL"
& wsl -d $Distro -- bash -lc 'pkill -f llama-swap || true; pkill -f playwright-mcp || true'

if (-not $SkipRestart) {
  Step "Restart services"
  if (Test-Path $startScript) {
    powershell -ExecutionPolicy Bypass -File $startScript
  } else {
    Write-Host "start-tbg-services-wsl.ps1 not found. Starting llama-swap directly..." -ForegroundColor Yellow
    & wsl -d $Distro -- bash -lc "nohup '$WslBinPath' --config '$WslConfigPath' --listen 0.0.0.0:8080 >/tmp/llama-swap.log 2>&1 &"
  }
} else {
  Write-Host "Skipped restart." -ForegroundColor Yellow
}

Step "Verify WSL status"
& wsl -d $Distro -- bash -lc "'$WslBinPath' -version; ss -ltnp | sed -n '1p;/:8080/p;/:8931/p'; curl -sS http://127.0.0.1:8080/health || true; echo; pgrep -af 'llama-swap|playwright-mcp' || true"

Step "Done"
Write-Host "WSL update/restart flow completed." -ForegroundColor Green
