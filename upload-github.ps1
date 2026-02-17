param(
  [string]$RepoRoot = $PSScriptRoot,
  [string]$Branch = "main",
  [string]$Remote = "origin",
  [string]$Message = "update",
  [switch]$AddAll,
  [switch]$PullRebase,
  [switch]$SetUpstream
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Step([string]$msg) {
  Write-Host "`n=== $msg ===" -ForegroundColor Cyan
}

function Require-GitRepo([string]$path) {
  if (-not (Test-Path $path)) {
    throw "Repo root not found: $path"
  }
  $inside = git -C $path rev-parse --is-inside-work-tree 2>$null
  if ($inside -ne "true") {
    throw "Not a git repository: $path"
  }
}

Require-GitRepo $RepoRoot

Step "Repo status"
$branchName = git -C $RepoRoot rev-parse --abbrev-ref HEAD
Write-Host "Current branch: $branchName"

git -C $RepoRoot status --short

if ($PullRebase) {
  Step "Fetch and rebase"
  git -C $RepoRoot fetch $Remote
  git -C $RepoRoot rebase "$Remote/$Branch"
}

if ($AddAll) {
  Step "Stage changes"
  git -C $RepoRoot add -A
}

# Check if there is anything staged for commit.
$staged = git -C $RepoRoot diff --cached --name-only
if (-not $staged) {
  Step "No staged changes"
  Write-Host "Nothing to commit."
} else {
  Step "Commit"
  git -C $RepoRoot commit -m $Message
}

Step "Push"
if ($SetUpstream) {
  git -C $RepoRoot push -u $Remote $Branch
} else {
  git -C $RepoRoot push $Remote $Branch
}

Step "Done"
Write-Host "Upload flow completed." -ForegroundColor Green
