$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @(
    (Join-Path $projectRoot ".venv-win\Scripts\python.exe"),
    (Join-Path $projectRoot "..\fgds\.venv-win\Scripts\python.exe")
)
$pythonExe = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonExe) {
    throw "Python environment not found. Create .venv-win or keep the training environment in ..\fgds\.venv-win."
}

if (-not $env:MODEL_PATH) {
    $modelCandidates = @(
        (Join-Path $projectRoot "runs\combined_galar_v1\best.pt"),
        (Join-Path $projectRoot "..\fgds\runs\combined_galar_v1\best.pt")
    )
    $modelPath = $modelCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $modelPath) {
        throw "Checkpoint best.pt not found. Set MODEL_PATH to the trained checkpoint."
    }
    $env:MODEL_PATH = (Resolve-Path -LiteralPath $modelPath).Path
}

Set-Location -LiteralPath $projectRoot
& $pythonExe -m uvicorn web_demo.app:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
