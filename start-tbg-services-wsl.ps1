$ErrorActionPreference = "Stop"

$distro = "Ubuntu"
$llamaSwapBin = "/home/admmin/bin/llama-swap"
$llamaSwapConfig = "/home/admmin/llama-swap/config.yaml"
$llamaSwapLog = "/tmp/llama-swap.log"
$playwrightMcpBin = "/home/admmin/.local/npm-global/bin/playwright-mcp"
$playwrightLog = "/tmp/playwright-mcp.log"
$playwrightBrowser = "/home/admmin/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome"

Write-Host ""
Write-Host "==========================================================="
Write-Host "  TBG (O)llama Swap + Playwright MCP (WSL One-Click Start)"
Write-Host "==========================================================="
Write-Host ""

Write-Host "[1/4] Stopping previous instances..."
wsl -d $distro -- bash -lc "pkill -9 -f 'llama-swap' || true; pkill -9 -f 'playwright-mcp' || true; pkill -9 -f 'llama-server' || true" | Out-Null

Write-Host "[2/4] Starting llama-swap on 0.0.0.0:8080 (new window)..."
$llamaCmd = "$llamaSwapBin --config $llamaSwapConfig --listen 0.0.0.0:8080 2>&1 | tee -a $llamaSwapLog"
Start-Process -FilePath "wsl.exe" -ArgumentList "-d $distro -- bash -lc `"$llamaCmd`"" | Out-Null

Write-Host "[3/4] Starting Playwright MCP on 0.0.0.0:8931 (new window)..."
$mcpCmd = "$playwrightMcpBin --host 0.0.0.0 --port 8931 --headless --browser chrome --executable-path $playwrightBrowser 2>&1 | tee -a $playwrightLog"
Start-Process -FilePath "wsl.exe" -ArgumentList "-d $distro -- bash -lc `"$mcpCmd`"" | Out-Null

Write-Host "[4/4] Verifying status..."
wsl -d $distro -- bash -lc "for i in 1 2 3 4 5 6 7 8; do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done; echo '--- listeners ---'; ss -ltnp | sed -n '1p;/:8080/p;/:8931/p'; echo '--- health ---'; curl -sS http://127.0.0.1:8080/health || true; echo; echo '--- processes ---'; pgrep -af 'llama-swap|playwright-mcp' || true; echo '--- recent logs ---'; echo '[llama-swap]'; tail -n 10 $llamaSwapLog 2>/dev/null || true; echo '[playwright-mcp]'; tail -n 10 $playwrightLog 2>/dev/null || true"

Write-Host ""
Write-Host "Done. Open:"
Write-Host "  http://localhost:8080/ui/"
Write-Host ""
