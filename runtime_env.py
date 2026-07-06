"""Portable runtime bootstrap for all lab notebooks (tracked in git)."""
from __future__ import annotations

import logging
import os
import sys
import urllib.request
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGURED = False

SAMPLE_IMAGE_URLS = {
    "street.jpg": "https://ultralytics.com/images/bus.jpg",
    "objects.jpg": "https://ultralytics.com/images/zidane.jpg",
}


def project_root() -> Path:
    return ROOT


def hadoop_home() -> Path:
    return ROOT / "tools" / "hadoop"


def default_mlflow_uri(root: Path | None = None) -> str:
    """SQLite store compatible with MLflow 3.x (same default as env.ps1)."""
    base = root or project_root()
    db_dir = base / "mlruns"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "mlflow.db"
    return "sqlite:///" + str(db_path.resolve()).replace("\\", "/")


def _java_bin_name() -> str:
    return "java.exe" if sys.platform == "win32" else "java"


def find_java_home() -> Path | None:
    env = os.environ.get("JAVA_HOME")
    if env:
        candidate = Path(env)
        if (candidate / "bin" / _java_bin_name()).exists():
            return candidate

    candidates: list[Path] = []
    if sys.platform == "win32":
        candidates.extend(
            [
                Path(r"C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot"),
                Path(r"C:\Program Files\Microsoft\jdk-17"),
                Path(r"C:\Program Files\Eclipse Adoptium\jdk-17"),
            ]
        )
        for entry in Path(r"C:\Program Files\Microsoft").glob("jdk-*"):
            candidates.append(entry)
    elif sys.platform == "darwin":
        candidates.extend(
            Path("/Library/Java/JavaVirtualMachines").glob("*.jdk/Contents/Home")
        )
    else:
        candidates.extend(
            [
                Path("/usr/lib/jvm/java-17-openjdk"),
                Path("/usr/lib/jvm/java-17-openjdk-amd64"),
                Path("/usr/lib/jvm/java-17"),
            ]
        )

    for candidate in candidates:
        if (candidate / "bin" / _java_bin_name()).exists():
            return candidate
    return None


def _java_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def configure_runtime(*, quiet: bool = True) -> None:
    """Idempotent bootstrap for notebooks and setup scripts."""
    global CONFIGURED
    if CONFIGURED:
        return

    hf_home = ROOT / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MLFLOW_TRACKING_URI", default_mlflow_uri())
    os.environ.setdefault("MLFLOW_UI_PORT", "5050")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    java_home = find_java_home()
    if java_home is not None:
        os.environ.setdefault("JAVA_HOME", str(java_home))
        java_bin = str(java_home / "bin")
        if java_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")

    hh = hadoop_home()
    bin_dir = hh / "bin"
    winutils = bin_dir / ("winutils.exe" if sys.platform == "win32" else "winutils")
    if winutils.exists():
        os.environ.setdefault("HADOOP_HOME", str(hh))
        if str(bin_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    if quiet:
        warnings.filterwarnings("ignore", message=".*symlinks.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*Repo card metadata.*")
        warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger("pyspark").setLevel(logging.ERROR)
        try:
            import datasets as ds

            ds.disable_progress_bar()
        except Exception:
            pass

    CONFIGURED = True


def ensure_sample_images(force: bool = False) -> Path:
    """Download default JPGs for vision labs if missing."""
    configure_runtime(quiet=True)
    dest = ROOT / "datasets" / "sample_images"
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLE_IMAGE_URLS.items():
        path = dest / name
        if path.exists() and not force:
            continue
        try:
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded sample image: {path.name}")
        except Exception as exc:
            print(f"WARN: could not download {name} from {url}: {exc}")
    return dest


def load_sst2():
    """Load SST-2 without GLUE / card-metadata warnings."""
    configure_runtime()
    from datasets import load_dataset

    return load_dataset("SetFit/sst2")


def create_spark(app_name: str = "spark-local", master: str = "local[1]"):
    """Create a local SparkSession with cross-platform-friendly settings."""
    configure_runtime()
    from pyspark.sql import SparkSession

    hh = hadoop_home()
    bin_dir = hh / "bin"
    log4j = _java_path(ROOT / "tools" / "spark-log4j2.properties")
    extra_java = (
        "-XX:+IgnoreUnrecognizedVMOptions "
        f"-Dlog4j2.configurationFile=file:///{log4j} "
        f"-Dlog4j.configuration=file:///{log4j}"
    )
    winutils = bin_dir / "winutils.exe"
    if winutils.exists():
        extra_java += (
            f" -Dhadoop.home.dir={_java_path(hh)}"
            f" -Djava.library.path={_java_path(bin_dir)}"
        )

    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.extraJavaOptions", extra_java)
        .config("spark.executor.extraJavaOptions", extra_java)
    )
    if winutils.exists():
        builder = builder.config("spark.hadoop.home.dir", _java_path(hh))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def stop_spark(spark) -> None:
    """Stop Spark without JVM shutdown crashes on some Windows + Java versions."""
    try:
        if spark is not None:
            spark.sparkContext.stop()
    except Exception:
        pass
