@echo off
REM One-time lab environment setup (Windows)
cd /d "%~dp0"

echo === Lab Environment Setup ===
python --version || (echo Python 3.10+ required & exit /b 1)

if not exist .venv (
    echo Creating venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel

echo Installing PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing requirements...
pip install -r requirements.txt

echo Registering Jupyter kernel...
python -m ipykernel install --user --name lab-env --display-name "Python 3.11 (labs)"

echo.
echo Optional: JDK 17 for Spark lab (Module 18)...
winget install -e --id Microsoft.OpenJDK.17 --accept-package-agreements --accept-source-agreements 2>nul

echo.
echo === Setup complete ===
echo Start labs:  .\env.ps1
echo Env only:    .\env.ps1 -EnvOnly
echo.
echo Large datasets (MNIST, COCO128, HF models) download on first lab run.
echo See datasets\README.md and notes\00-environment-setup.pdf for details.
