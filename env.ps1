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
$JupyterPort = 8888
$MlflowPort = 5000

function Test-PortListening {
    param([int]$Port)
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try {
        $listener.Start()
        $listener.Stop()
        return $false
    } catch {
        return $true
    }
}

function Start-BackgroundProcess {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$ArgumentList,
        [int]$Port,
        [string]$Url
    )
    if (Test-PortListening -Port $Port) {
        Write-Host "  $Name already running: $Url"
        return
    }
    Start-Process `
        -FilePath $Exe `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $Root `
        -WindowStyle Minimized | Out-Null
    Start-Sleep -Seconds 2
    Write-Host "  $Name started: $Url"
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
$env:MLFLOW_TRACKING_URI = Join-Path $Root "mlruns"
$env:PYSPARK_PYTHON = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
$env:SPARK_LOCAL_IP = "127.0.0.1"
$env:PYTHONWARNINGS = "ignore::UserWarning,ignore::FutureWarning"

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

$mlflowExe = Join-Path $VenvScripts "mlflow.exe"
if (Test-Path $mlflowExe) {
    Start-BackgroundProcess `
        -Name "MLflow UI" `
        -Exe $mlflowExe `
        -ArgumentList @(
            "ui",
            "--backend-store-uri", $env:MLFLOW_TRACKING_URI,
            "--host", "127.0.0.1",
            "--port", "$MlflowPort"
        ) `
        -Port $MlflowPort `
        -Url "http://127.0.0.1:$MlflowPort"
} else {
    Write-Host "  MLflow not installed; skipping UI."
}

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
Write-Host "Kernel: Python 3.11 (labs)  |  Notebooks: labs/"
Write-Host "Press Ctrl+C to stop Jupyter (MLflow keeps its own window)."
Write-Host ""

Set-Location $Root
& $jupyterExe lab --ip=127.0.0.1 --port=$JupyterPort
