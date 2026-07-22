#!/usr/bin/env bash
# One-command hierarchical minimal-circuit discovery on Qwen3.
#
#   git clone <repo> && cd <repo>
#   bash packages/mwrl-circuits/scripts/run_cluster.sh            # default: Qwen3-1.7B
#   bash packages/mwrl-circuits/scripts/run_cluster.sh 8b         # Qwen3-8B
#   bash packages/mwrl-circuits/scripts/run_cluster.sh 0.6b       # quick real-model check
#   bash packages/mwrl-circuits/scripts/run_cluster.sh <config.yaml>
#
# Creates an isolated venv, installs the mwrl core + mwrl-circuits packages, downloads
# the model into a repo-local HF cache on first run, and writes a JSON report under runs/.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"
CONFIGS="packages/mwrl-circuits/configs"

ARG="${1:-1.7b}"
case "$ARG" in
  0.6b|0p6b) CONFIG="$CONFIGS/qwen3_0p6b.yaml" ;;
  1.7b|1p7b) CONFIG="$CONFIGS/qwen3_1p7b.yaml" ;;
  8b)        CONFIG="$CONFIGS/qwen3_8b.yaml" ;;
  *.yaml)    CONFIG="$ARG" ;;
  *) echo "unknown target '$ARG' (use: 0.6b | 1.7b | 8b | <config.yaml>)"; exit 1 ;;
esac

export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYBIN="${PYTHON:-python3}"
if [ ! -d ".venv_llm" ]; then
  echo "[setup] creating .venv_llm"
  "$PYBIN" -m venv .venv_llm
fi
# shellcheck disable=SC1091
source .venv_llm/bin/activate
python -m pip install -q -U pip
echo "[setup] installing mwrl core + mwrl-circuits"
python -m pip install -q -e packages/mwrl -e packages/mwrl-circuits

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="runs/circuits/${ARG//./}_${STAMP}"
echo "[run] target=$ARG  output=$OUT"
python -m mwrl_circuits.run --config "$CONFIG" --output "$OUT"
echo "[ok] report: $OUT/report.json"
