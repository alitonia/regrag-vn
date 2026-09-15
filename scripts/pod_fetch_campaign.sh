#!/usr/bin/env bash
# pod_fetch_campaign.sh — pull campaign outputs from the HF artifacts repo
# into the git repo. Works at any time, including AFTER pod termination.
# Usage: bash scripts/pod_fetch_campaign.sh <run_id>     e.g. pod_20260913_1410
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ID="${1:?Usage: bash scripts/pod_fetch_campaign.sh <run_id>}"
HF_REPO="${HF_ARTIFACTS_REPO:-hunopapa/regrag-artifacts}"

HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || hf auth token 2>/dev/null \
  || .venv/bin/python -c 'from huggingface_hub import get_token; print(get_token())' 2>/dev/null || true)"
[ -n "$HF_TOKEN" ] || { echo "[FATAL] no local HF token"; exit 1; }
export HF_TOKEN
command -v hf >/dev/null || { echo "[FATAL] hf CLI not on PATH"; exit 1; }

STAGE="$(mktemp -d)"
rm -rf "$STAGE/.cache" 2>/dev/null || true
hf download "$HF_REPO" --repo-type dataset \
  --include "regrag/outputs/${RUN_ID}/*" --local-dir "$STAGE"

SRC="$STAGE/regrag/outputs/$RUN_ID"
[ -d "$SRC" ] || { echo "[FATAL] nothing fetched for run $RUN_ID"; exit 1; }

DEST="pod_artifacts/${RUN_ID}"
mkdir -p "$DEST"
cp -r "$SRC"/. "$DEST"/
N=$(find "$DEST" -type f | wc -l)
echo "[FETCH] $N files -> $DEST"
ls -la "$DEST"

# Latest results also land in the eval area for analysis
if [ -d "$SRC/results" ]; then
  mkdir -p data/eval
  cp -f "$SRC"/results/generations.json "$SRC"/results/generations_meta.jsonl data/eval/ 2>/dev/null || true
  echo "[FETCH] generations.json + meta refreshed in data/eval/"
fi
rm -rf "$STAGE"
