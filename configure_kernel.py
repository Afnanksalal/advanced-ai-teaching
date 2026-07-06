"""Configure Jupyter kernel + IPython startup for portable lab notebooks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from runtime_env import default_mlflow_uri, find_java_home, hadoop_home, project_root

KERNEL_NAME = "lab-env"


def venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def kernel_dir() -> Path:
    if sys.platform == "win32":
        return Path.home() / "AppData" / "Roaming" / "jupyter" / "kernels" / KERNEL_NAME
    return Path.home() / ".local" / "share" / "jupyter" / "kernels" / KERNEL_NAME


def kernel_env(root: Path) -> dict[str, str]:
    hh = hadoop_home()
    env = {
        "HF_HOME": str(root / "hf_cache"),
        "HF_HUB_CACHE": str(root / "hf_cache" / "hub"),
        "HF_DATASETS_CACHE": str(root / "hf_cache" / "datasets"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_VERBOSITY": "error",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MLFLOW_TRACKING_URI": default_mlflow_uri(root),
        "MLFLOW_UI_PORT": "5050",
        "PYSPARK_PYTHON": str(venv_python(root)),
        "PYSPARK_DRIVER_PYTHON": str(venv_python(root)),
        "SPARK_LOCAL_IP": "127.0.0.1",
        "PYTHONWARNINGS": "ignore::UserWarning,ignore::FutureWarning",
    }
    java_home = find_java_home()
    if java_home is not None:
        env["JAVA_HOME"] = str(java_home)
    winutils = hh / "bin" / ("winutils.exe" if sys.platform == "win32" else "winutils")
    if winutils.exists():
        env["HADOOP_HOME"] = str(hh)
    return env


def write_ipython_startup(root: Path) -> Path:
    startup = Path.home() / ".ipython" / "profile_default" / "startup"
    startup.mkdir(parents=True, exist_ok=True)
    path = startup / "00-lab-env.py"
    path.write_text(
        f"""# Auto-loaded by the lab Jupyter kernel
import sys
ROOT = r"{root}"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from runtime_env import configure_runtime
configure_runtime(quiet=True)
""",
        encoding="utf-8",
    )
    return path


def main() -> int:
    root = project_root()
    python_exe = venv_python(root)
    if not python_exe.exists():
        print(f"Missing venv python: {python_exe}")
        print("Run setup.bat (Windows) or ./setup.sh (Linux/macOS) first.")
        return 1

    kdir = kernel_dir()
    kdir.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [
            str(python_exe),
            "-Xfrozen_modules=off",
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Python 3.11 (labs)",
        "language": "python",
        "env": kernel_env(root),
        "metadata": {"debugger": True},
    }
    (kdir / "kernel.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    startup = write_ipython_startup(root)
    print(f"  OK  kernel: {kdir / 'kernel.json'}")
    print(f"  OK  ipython startup: {startup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
