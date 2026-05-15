# ============================================================================
# start-dev.ps1 — Launch the full ABSA development stack
# ============================================================================
#
# Usage:  .\start-dev.ps1        (from repo root)
#
# This script starts three services, each in its own terminal window:
#   1. Python FastAPI backend   (http://127.0.0.1:8000)
#   2. C# ASP.NET Core gateway  (http://127.0.0.1:8001)  [optional]
#   3. Vite React frontend      (http://127.0.0.1:5173)
#
# The script automatically kills any existing processes on the target ports
# before starting, so it is safe to run multiple times.
# ============================================================================

$ErrorActionPreference = "Stop"
$root = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location $root

# -- Pre-flight checks -------------------------------------------------------

$py = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Python venv not found. Run: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$nodeModulesBin = "$root\frontend\node_modules\.bin"
if (-not (Test-Path "$nodeModulesBin\vite.cmd")) {
    Write-Error "Frontend dependencies not found. Run: cd frontend && npm install"
    exit 1
}

$dotnetExe = Get-Command "dotnet" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
$hasCsharp = ($null -ne $dotnetExe) -and (Test-Path "$root\backend-csharp\BackendGateway.csproj")

# -- Kill existing processes on target ports ----------------------------------

foreach ($port in @(8000, 8001, 5173)) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        if ($c.OwningProcess -gt 0) {
            Write-Host "  Killing existing process on port $port (PID $($c.OwningProcess))..." -ForegroundColor Yellow
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}
Start-Sleep -Milliseconds 500

# -- 1) Python FastAPI Backend ------------------------------------------------

Write-Host ""
Write-Host "[1/3] Starting Python backend (http://127.0.0.1:8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; & '$py' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
) -WindowStyle Normal
Write-Host "      OK" -ForegroundColor Green

# -- 2) C# Gateway (optional) ------------------------------------------------

if ($hasCsharp) {
    Write-Host "[2/3] Starting C# gateway (http://127.0.0.1:8001)..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$env:ASPNETCORE_ENVIRONMENT='Development'; Set-Location '$root\backend-csharp'; & '$dotnetExe' run --project '$root\backend-csharp\BackendGateway.csproj' --framework net8.0 --no-restore --urls http://127.0.0.1:8001"
    ) -WindowStyle Normal
    Write-Host "      OK" -ForegroundColor Green
} else {
    Write-Host "[2/3] Skipping C# gateway (dotnet or project not found)." -ForegroundColor Yellow
}

# -- 3) Vite Frontend --------------------------------------------------------

Write-Host "[3/3] Starting frontend (http://127.0.0.1:5173)..." -ForegroundColor Cyan
$viteCmd = "$nodeModulesBin\vite.cmd"
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; & '$viteCmd' --host 127.0.0.1 --port 5173"
) -WindowStyle Normal
Write-Host "      OK" -ForegroundColor Green

# -- Summary ------------------------------------------------------------------

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  All services started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend:  http://127.0.0.1:5173" -ForegroundColor White
Write-Host "  Gateway:   http://127.0.0.1:8001" -ForegroundColor White
Write-Host "  Backend:   http://127.0.0.1:8000" -ForegroundColor White
Write-Host ""
Write-Host "  Model loading may take ~10 seconds." -ForegroundColor DarkGray
Write-Host "  To stop: close the opened terminal windows." -ForegroundColor DarkGray
Write-Host "================================================" -ForegroundColor Green