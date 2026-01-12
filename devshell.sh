#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG (edit if needed) ---
ENV_NAME="${ENV_NAME:-ontocodex}"
CONDA_BASE="${CONDA_BASE:-/opt/anaconda3}"   # <-- your conda root
PY_VER="${PY_VER:-3.11}"
# ------------------------------

CONDA_BIN="$CONDA_BASE/bin/conda"
if [[ ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: conda not found at: $CONDA_BIN"
  echo "Set CONDA_BASE to your conda install, e.g.: CONDA_BASE=$HOME/miniconda3"
  exit 1
fi

# Always start from a clean PATH so broken shell init files don't matter.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$CONDA_BASE/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Create env if missing (idempotent).
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[devshell] Creating conda env: $ENV_NAME"
  if [[ -f "environment.yml" ]]; then
    "$CONDA_BIN" env create -n "$ENV_NAME" -f environment.yml
  else
    "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PY_VER"
  fi
fi

# OPTIONAL: ensure editable install if pyproject.toml exists
if [[ -f "pyproject.toml" ]]; then
  echo "[devshell] Installing package editable mode"
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -U pip
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e .
fi

echo ""
echo "[devshell] Repo: $REPO_ROOT"
echo "[devshell] Conda: $CONDA_BASE"
echo "[devshell] Env  : $ENV_NAME"
echo ""

# Launch a shell with env variables set (without conda activate).
# This avoids conda’s shell integration entirely.
exec "$CONDA_BIN" run -n "$ENV_NAME" bash --noprofile --norc
