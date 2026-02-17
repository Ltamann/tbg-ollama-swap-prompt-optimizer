param(
  [string]$RepoRoot = $PSScriptRoot,
  [switch]$RunTests,
  [switch]$RunBinaryVersion
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Step([string]$msg) {
  Write-Host "`n=== $msg ===" -ForegroundColor Cyan
}

if (-not (Test-Path $RepoRoot)) {
  throw "Repo root not found: $RepoRoot"
}

$winBin = Join-Path $RepoRoot "build/llama-swap-windows-amd64.exe"

Push-Location $RepoRoot
try {
  if ($RunTests) {
    Step "Run Go tests"
    go test ./...
  }

  Step "Build Windows binary"
  Remove-Item Env:GOOS -ErrorAction SilentlyContinue
  Remove-Item Env:GOARCH -ErrorAction SilentlyContinue
  go build -o build/llama-swap-windows-amd64.exe .

  if (-not (Test-Path $winBin)) {
    throw "Windows binary not found after build: $winBin"
  }

  Step "Verify binary"
  Get-Item $winBin | Select-Object FullName,Length,LastWriteTime | Format-List

  if ($RunBinaryVersion) {
    Step "Run binary version"
    & $winBin -version
  }

  Step "Done"
  Write-Host "Windows build completed." -ForegroundColor Green
}
finally {
  Pop-Location
}
