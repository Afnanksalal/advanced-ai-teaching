#!/usr/bin/env bash
# Lab session launcher (Linux / macOS)
# Usage:
#   ./env.sh              load env + start MLflow + Jupyter Lab
#   source ./env.sh       same (Jupyter runs in this terminal)
#   ./env.sh --env-only   load env vars only (no servers)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_BIN="$ROOT/.venv/bin"
LOG_DIR="$ROOT/.logs"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
MLFLOW_PORT="${MLFLOW_PORT:-5050}"
ENV_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --env-only) ENV_ONLY=1 ;;
  esac
done

if [ ! -x "$VENV_BIN/python" ]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$VENV_BIN:$PATH"
unset JAVA_TOOL_OPTIONS 2>/dev/null || true

export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export PYSPARK_PYTHON="$VENV_BIN/python"
export PYSPARK_DRIVER_PYTHON="$VENV_BIN/python"
export SPARK_LOCAL_IP=127.0.0.1

mkdir -p "$ROOT/mlruns"
MLFLOW_DB="$ROOT/mlruns/mlflow.db"
export MLFLOW_TRACKING_URI="sqlite:///${MLFLOW_DB//\\//}"
export MLFLOW_UI_PORT="$MLFLOW_PORT"

if [ -z "${JAVA_HOME:-}" ]; then
  for candidate in /usr/lib/jvm/java-17-openjdk /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/java-17; do
    if [ -x "$candidate/bin/java" ]; then
      export JAVA_HOME="$candidate"
      export PATH="$JAVA_HOME/bin:$PATH"
      break
    fi
  done
fi

if [ -d "$ROOT/tools/hadoop/bin" ] && [ -f "$ROOT/tools/hadoop/bin/winutils" ]; then
  export HADOOP_HOME="$ROOT/tools/hadoop"
fi

echo "Environment loaded:"
echo "  Venv:    $VIRTUAL_ENV"
echo "  Python:  $(python --version 2>&1)"
echo "  HF_HOME: $HF_HOME"
echo "  MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"

if [ "$ENV_ONLY" -eq 1 ]; then
  echo
  echo "Env only (--env-only). Start servers with: ./env.sh"
  return 0 2>/dev/null || exit 0
fi

port_open() {
  python - <<PY
import socket
s = socket.socket()
try:
    s.connect(("127.0.0.1", int("$1")))
    print("yes")
except OSError:
    print("no")
finally:
    s.close()
PY
}

mkdir -p "$LOG_DIR"
MLFLOW_LOG="$LOG_DIR/mlflow-ui.log"

if [ "$(port_open "$MLFLOW_PORT")" = "yes" ]; then
  echo "  MLflow UI already running: http://127.0.0.1:$MLFLOW_PORT"
else
  : > "$MLFLOW_LOG"
  nohup python -m mlflow ui \
    --backend-store-uri "$MLFLOW_TRACKING_URI" \
    --host 127.0.0.1 \
    --port "$MLFLOW_PORT" \
    >>"$MLFLOW_LOG" 2>&1 &
  sleep 2
  if [ "$(port_open "$MLFLOW_PORT")" = "yes" ]; then
    echo "  MLflow UI ready: http://127.0.0.1:$MLFLOW_PORT"
  else
    echo "  MLflow UI failed to start. See $MLFLOW_LOG"
  fi
fi

if [ "$(port_open "$JUPYTER_PORT")" = "yes" ]; then
  echo "  Jupyter already running: http://127.0.0.1:$JUPYTER_PORT/lab"
  exit 0
fi

echo
echo "Jupyter Lab: http://127.0.0.1:$JUPYTER_PORT/lab"
echo "MLflow UI:   http://127.0.0.1:$MLFLOW_PORT"
echo "Kernel: Python 3.11 (labs)  |  Notebooks: labs/"
echo "Press Ctrl+C to stop Jupyter."
echo

cd "$ROOT"
exec jupyter lab --ip=127.0.0.1 --port="$JUPYTER_PORT"
