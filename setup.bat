@echo off
REM Advanced AI course: one-time setup (Windows)
cd /d "%~dp0"

echo === Advanced AI: Course Setup ===
python --version || (echo Python 3.10+ required & exit /b 1)

if not exist .venv (
    echo Creating venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel

echo Installing PyTorch (CUDA 12.1 when available)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing requirements...
pip install -r requirements.txt

echo Registering Jupyter kernel...
python configure_kernel.py

echo Prefetching bundled datasets and sample images...
python download_datasets.py --minimal

echo.
echo Optional: JDK 17 for Spark lab (Module 18)...
winget install -e --id Microsoft.OpenJDK.17 --accept-package-agreements --accept-source-agreements 2>nul

echo.
echo === Setup complete ===
echo Start labs:  .\env.ps1
echo Env only:    .\env.ps1 -EnvOnly
echo.
echo Large datasets (MNIST, COCO128, HF models) download on first lab run.
echo Full prefetch: python download_datasets.py
echo See datasets\README.md and notes\00-environment-setup.pdf for details.
