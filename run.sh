#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ontocodex}"
CONDA_BASE="${CONDA_BASE:-/opt/anaconda3}"

CONDA_BIN="$CONDA_BASE/bin/conda"
if [[ ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: conda not found at: $CONDA_BIN"
  exit 1
fi

export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$CONDA_BASE/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

exec "$CONDA_BIN" run -n "$ENV_NAME" "$@"
