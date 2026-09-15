#!/usr/bin/env bash
# pod_hf_push_campaign.sh — stage code + inputs for the RunPod campaign and
# push them to the private HF artifacts repo. Pod-free: run any time, bills
# nothing. Idempotent; re-run after any local code change.
# Layout: regrag/code/<repo subset>, regrag/inputs/<data files>.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_ARTIFACTS_REPO:-hunopapa/regrag-artifacts}"
STAGE="$(mktemp -d)/regrag"
mkdir -p "$STAGE/code" "$STAGE/inputs"

# --- code: the whole harness package + campaign scripts (flat import closure)
tar cf - --exclude='__pycache__' regrag scripts | tar xf - -C "$STAGE/code"

# --- inputs: gold CSV + both chunk corpora (driver reproduces the notebook's
#     candidate order: corpus_chunks.json is rejected on duplicate ids ->
#     tier1, exactly like the banked Colab rows)
mkdir -p "$STAGE/inputs/processed_chunks"
cp data/gold/bank_qa_data.csv "$STAGE/inputs/"
cp data/processed_chunks/corpus_chunks.json data/processed_chunks/tier1_chunks.json \
   "$STAGE/inputs/processed_chunks/"

# --- local token: token FILE -> hf CLI -> venv python (never bare system import)
HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || hf auth token 2>/dev/null \
  || .venv/bin/python -c 'from huggingface_hub import get_token; print(get_token())' 2>/dev/null || true)"
[ -n "$HF_TOKEN" ] || { echo "[FATAL] no local HF token (file, hf CLI, venv all failed)"; exit 1; }
export HF_TOKEN

command -v hf >/dev/null || { echo "[FATAL] hf CLI not on PATH"; exit 1; }
hf repos create "$HF_REPO" --type dataset --private --exist-ok 2>/dev/null || true

STAGE_PARENT="$(dirname "$STAGE")"
hf upload "$HF_REPO" "$STAGE/code" "regrag/code" --type dataset \
  --commit-message "campaign code $(date -u +%Y-%m-%dT%H:%M)Z"
hf upload "$HF_REPO" "$STAGE/inputs" "regrag/inputs" --type dataset \
  --commit-message "campaign inputs $(date -u +%Y-%m-%dT%H:%M)Z"

N_PY=$(find "$STAGE/code" -name '*.py' | wc -l)
echo "[PUSH] staged $N_PY python files + $(find "$STAGE/inputs" -type f | wc -l) input files -> $HF_REPO"
rm -rf "$STAGE_PARENT"
