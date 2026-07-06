# Lab session launcher (Windows)
# Usage:
#   .\env.ps1          load env + start MLflow + Jupyter Lab
#   . .\env.ps1        same (Jupyter runs in this terminal)
#   .\env.ps1 -EnvOnly load env vars only (no servers)

param(
    [switch]$EnvOnly
)

$Root = $PSScriptRoot
$VenvScripts = Join-Path $Root ".venv\Scripts"
$Hadoop = Join-Path $Root "tools\hadoop"
$LogDir = Join-Path $Root ".logs"
$JupyterPort = 8888
$MlflowPort = 5050

function Test-PortListening {
    param([int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $Port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-PortListening {
    param(
        [int]$Port,
        [int]$TimeoutSec = 20
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortListening -Port $Port) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Start-MlflowUi {
    param(
        [string]$BackendUri,
        [int]$Port
    )
    if (Test-PortListening -Port $Port) {
        Write-Host "  MLflow UI already running: http://127.0.0.1:$Port"
        return $true
    }

    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $logFile = Join-Path $LogDir "mlflow-ui.log"
    if (Test-Path $logFile) { Remove-Item $logFile -Force }

    $pythonExe = Join-Path $VenvScripts "python.exe"
    $args = @(
        "-m", "mlflow", "ui",
        "--backend-store-uri", $BackendUri,
        "--host", "127.0.0.1",
        "--port", "$Port"
    )

    Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $args `
        -WorkingDirectory $Root `
        -WindowStyle Minimized `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $logFile | Out-Null

    if (Wait-PortListening -Port $Port) {
        Write-Host "  MLflow UI ready: http://127.0.0.1:$Port"
        return $true
    }

    Write-Host "  MLflow UI failed to start on port $Port."
    if (Test-Path $logFile) {
        Write-Host "  Last log lines:"
        Get-Content $logFile -Tail 8 | ForEach-Object { Write-Host "    $_" }
    }
    return $false
}

if (-not (Test-Path (Join-Path $VenvScripts "python.exe"))) {
    Write-Error "Missing .venv. Run setup.bat first."
    exit 1
}

$env:VIRTUAL_ENV = Join-Path $Root ".venv"
$pathParts = $env:PATH -split ';' | Where-Object { $_ -and ($_ -ne $VenvScripts) }
$env:PATH = "$VenvScripts;" + ($pathParts -join ';')

Remove-Item Env:JAVA_TOOL_OPTIONS -ErrorAction SilentlyContinue

$env:HF_HOME = Join-Path $Root "hf_cache"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_VERBOSITY = "error"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONWARNINGS = "ignore::UserWarning,ignore::FutureWarning"
$env:PYSPARK_PYTHON = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
$env:SPARK_LOCAL_IP = "127.0.0.1"

# MLflow 3.x: filesystem ./mlruns store is blocked unless opted in. Use SQLite instead.
$MlrunsDir = Join-Path $Root "mlruns"
New-Item -ItemType Directory -Force -Path $MlrunsDir | Out-Null
$MlflowDb = Join-Path $MlrunsDir "mlflow.db"
$DbUri = "sqlite:///$($MlflowDb.Replace('\', '/'))"
$env:MLFLOW_TRACKING_URI = $DbUri
$env:MLFLOW_UI_PORT = "$MlflowPort"

$Jdk17 = "C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot"
if (Test-Path $Jdk17) {
    $env:JAVA_HOME = $Jdk17
    $env:PATH = "$Jdk17\bin;" + $env:PATH
}

if (Test-Path (Join-Path $Hadoop "bin\winutils.exe")) {
    $env:HADOOP_HOME = $Hadoop
}

Write-Host "Environment loaded:"
Write-Host "  Venv:    $env:VIRTUAL_ENV"
Write-Host "  Python:  $(python --version 2>&1)"
Write-Host "  HF_HOME: $env:HF_HOME"
if ($env:HADOOP_HOME) { Write-Host "  HADOOP_HOME: $env:HADOOP_HOME" }
Write-Host "  MLFLOW_TRACKING_URI: $env:MLFLOW_TRACKING_URI"

if ($EnvOnly) {
    Write-Host ""
    Write-Host "Env only (-EnvOnly). Start servers with: .\env.ps1"
    return
}

Write-Host ""
Write-Host "Starting services..."
Start-MlflowUi -BackendUri $env:MLFLOW_TRACKING_URI -Port $MlflowPort | Out-Null

$jupyterExe = Join-Path $VenvScripts "jupyter.exe"
if (-not (Test-Path $jupyterExe)) {
    Write-Error "jupyter not found in .venv. Re-run setup.bat."
    exit 1
}

if (Test-PortListening -Port $JupyterPort) {
    Write-Host "  Jupyter already running: http://127.0.0.1:$JupyterPort/lab"
    Write-Host "  Close the other Jupyter window or pick a free port."
    exit 0
}

Write-Host ""
Write-Host "Jupyter Lab: http://127.0.0.1:$JupyterPort/lab"
Write-Host "MLflow UI:   http://127.0.0.1:$MlflowPort"
Write-Host "Kernel: Python 3.11 (labs)  |  Notebooks: labs/"
Write-Host "Press Ctrl+C to stop Jupyter (MLflow keeps running in its own window)."
Write-Host ""

Set-Location $Root
& $jupyterExe lab --ip=127.0.0.1 --port=$JupyterPort
